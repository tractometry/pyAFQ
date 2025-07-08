import multiprocessing
import warnings
import numpy as np
from tqdm import tqdm
import ray

from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from numba import njit, prange, set_num_threads, config
from numba.core.errors import NumbaPerformanceWarning

import osqp

from dipy.reconst.mcsd import MSDeconvFit
from dipy.reconst.mcsd import MultiShellDeconvModel

from AFQ.utils.stats import chunk_indices


__all__ = ["fit"]


# CPU implementation of primal-dual Interior Point method
# for convex quadratic constrained optimization (QP)
# Specifically, ||Rx-d||_2 where Ax>=b
# parallelized to solve 10s-100s of thousands of QPs
# ultimately used to fit MSMT CSD
@njit(fastmath=True)
def solve_qp(Rt, R_pinv, G, A, At, A_outer, b, x0,
             d, max_iter, tol):
    '''
    Solves 1/2*x^t*G*x+(Rt*d)^t*x given Ax>=b
    In MSMT, G, R, A, b are the same across voxels,
    but there are different d for different voxels.
    This fact is not used currently. So this is a more general
    batched CLS QP solver.

    Let:
    c=(Rt*d)^t
    L = diag(l)
    Y = diag(y)
    mu as centering parameter that tends to 0 with iteration number.

    Set up with interior points, this is reformulated to:
    | G  0 -A.T | |dx|   |0 |   | G*x-A.T*l+c |
    | A -I  0   | |dy| = |0 | - | A*x-y-b     |
    | 0  L  Y   | |dl|   |mu|   | YL          |

    This reduces to:
    |G -A.T| |dx| = | -G*x-A.T*l+c          |
    |A Y\L | |dl|   | -(A*x-y-b) + (-y+mu/l)|
    with dy = A*dx+(A*x-y-b)

    Solving for dx, dl:
    (G+A.T*(Y\L)*A)*dx = -G*x-A.T*l+c
    dl = (Y\L)*(-(A*x-y-b) + (-y+mu/l) - A*dx)

    So, the tricky part is solving for dx.
    However, note that G+A.T*(Y\L)*A is hermitian positive semidefinite.
    So, it can be solved using conjugate gradients.
    This is done every iteration and is the longest part of the calculation.
    '''

    n = A.shape[1]
    m = A.shape[0]

    # First check if naive solution satisfies constraints
    x = R_pinv @ d
    if np.all(A @ x - b >= -tol):
        return True, x

    c = Rt @ d

    x = x0.copy()
    y = np.maximum(np.abs(A @ x - b), 1.0)
    l = np.ones(m) / np.max(y)
    dx = np.zeros(n)
    dy = np.zeros(m)
    dl = np.zeros(m)
    dy_aff = np.zeros(m)
    dl_aff = np.zeros(m)

    rhs1 = np.empty(n)
    rhs2 = np.empty(m)
    Gx = np.empty(n)
    ATl = np.empty(n)
    rhs1l = np.empty(n)
    Zrhs2 = np.empty(m)
    ATZrhs2 = np.empty(n)
    schur = np.empty((n, n))
    temp = np.empty(m)
    tempn1 = np.empty(n)
    tempn2 = np.empty(n)

    tau = 0.95
    shur_regularization = 1e-6
    for ii in range(max_iter):
        # mu = (y @ l) / m
        mu = 0.0
        for i in range(m):
            mu += y[i] * l[i]
        mu /= m

        # Predictor step
        Z = safe_divide_vec(l, y)

        # dy = A @ x - y - b
        for i in range(m):
            ax = 0.0
            for j in range(n):
                ax += A[i, j] * x[j]
            dy[i] = ax - y[i] - b[i]

        # rhs2 = -dy - y
        for i in range(m):
            rhs2[i] = -dy[i] - y[i]

        # rhs1l = -(G @ x - A.T @ l + c)
        # G @ x
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += G[i, j] * x[j]
            Gx[i] = s

        # Then compute A.T @ l
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += At[i, j] * l[j]
            ATl[i] = s

        # Now compute rhs1l = -(Gx - ATl + c)
        for i in range(n):
            rhs1l[i] = -(Gx[i] - ATl[i] + c[i])

        # rhs1 = rhs1l + A.T @ (Z * rhs2)
        # Z * rhs2
        for i in range(m):
            Zrhs2[i] = Z[i] * rhs2[i]

        # A.T @ (Z * rhs2)
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += At[i, j] * Zrhs2[j]
            ATZrhs2[i] = s

        # rhs1 = rhs1l + ATZrhs2
        for i in range(n):
            rhs1[i] = rhs1l[i] + ATZrhs2[i]

        for i in range(n):
            for j in range(i, n):
                s = G[i, j]
                for k in range(m):
                    s += Z[k] * A_outer[i, j, k]
                schur[i, j] = s
                schur[j, i] = s  # fill symmetric

        for i in range(n):
            schur[i, i] += shur_regularization

        L = np.linalg.cholesky(schur)
        dx = cholesky_solve(L, rhs1, tempn1, tempn2)

        # temp = A @ dx
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += A[i, j] * dx[j]
            temp[i] = s

        # dl_aff = Z * (rhs2 - A @ dx)
        for i in range(m):
            dl_aff[i] = Z[i] * (rhs2[i] - temp[i])

        # dy_aff = dy + A @ dx
        for i in range(m):
            dy_aff[i] = dy[i] + temp[i]

        alpha_aff_pri = 1.0
        alpha_aff_dual = 1.0
        for i in range(m):
            if dy_aff[i] < 0:
                alpha_aff_pri = min(alpha_aff_pri, -y[i] / dy_aff[i])
            if dl_aff[i] < 0:
                alpha_aff_dual = min(alpha_aff_dual, -l[i] / dl_aff[i])
        alpha_aff = min(tau * alpha_aff_pri, tau * alpha_aff_dual)

        # mu_aff = ((y + alpha_aff * dy_aff) @ (l + alpha_aff * dl_aff)) / m
        mu_aff = 0.0
        for i in range(m):
            mu_aff += (y[i] + alpha_aff * dy_aff[i]) \
                * (l[i] + alpha_aff * dl_aff[i])
        mu_aff /= m

        sigma = (mu_aff / mu) ** 3
        mu = sigma * mu

        # rhs2 = -dy + (-y + mu / l)
        rhs2 = -dy + (-y + safe_divide_vec(mu, l))

        # Zrhs2 = Z * rhs2
        for i in range(m):
            Zrhs2[i] = Z[i] * rhs2[i]

        # ATZrhs2 = A.T @ Zrhs2
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += At[i, j] * Zrhs2[j]
            ATZrhs2[i] = s

        # rhs1 = rhs1l + ATZrhs2
        for i in range(n):
            rhs1[i] = rhs1l[i] + ATZrhs2[i]

        dx = cholesky_solve(L, rhs1, tempn1, tempn2)

        # temp = A @ dx
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += A[i, j] * dx[j]
            temp[i] = s

        # dl = Z * (rhs2 - A @ dx)
        for i in range(m):
            dl[i] = Z[i] * (rhs2[i] - temp[i])

        # dy += A @ dx
        for i in range(m):
            dy[i] += temp[i]

        beta = 1.0
        sigma = 1.0
        for i in range(m):
            if dy[i] < 0 and dy[i] * sigma < -y[i]:
                sigma = -y[i] / dy[i]
            if dl[i] < 0 and dl[i] * beta < -l[i]:
                beta = -l[i] / dl[i]
        beta *= tau
        sigma *= tau
        alpha = min(beta, sigma)

        x += alpha * dx
        y += alpha * dy
        l += alpha * dl

        if alpha * np.dot(dx, dx) < n * tol:
            if np.all(A @ x - b >= -tol):
                return True, x

    return False, x


@njit(fastmath=True, inline='always')
def cholesky_solve(L, b, y, x):
    n = L.shape[0]

    # Forward substitution: L y = b
    for i in range(n):
        s = 0.0
        for j in range(i):
            s += L[i, j] * y[j]
        y[i] = (b[i] - s) / L[i, i]

    # Backward substitution: L.T x = y
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += L[j, i] * x[j]
        x[i] = (y[i] - s) / L[i, i]

    return x


@njit(fastmath=True)
def safe_divide_vec(numer, denom):
    safe_divisor = 1e-6  # Used to avoid division by zero
    return numer / np.where(
        np.abs(denom) < safe_divisor,
        np.sign(denom) * safe_divisor, denom)


def find_analytic_center(A, b, x0):
    """Find the analytic center using scipy.optimize.minimize"""
    cons = {'type': 'ineq', 'fun': lambda x: A @ x - b}
    result = minimize(
        lambda x: 1e-3 * np.linalg.norm(x)**2 - np.sum(np.log(A @ x - b)),
        x0,
        constraints=cons,
        method='SLSQP'
    )
    return result.x


@njit(fastmath=True)
def _process_slice(slice_data, slice_mask,
                   Rt, R_pinv,
                   G, A, At, A_outer, b, x0,
                   max_iter, tol):
    results = np.zeros(
        slice_data.shape[:2] + (A.shape[1],),
        dtype=np.float64)

    for j in prange(slice_data.shape[0]):
        for k in range(slice_data.shape[1]):
            if slice_mask[j, k]:
                success, result = solve_qp(
                    Rt, R_pinv, G, A, At, A_outer, b,
                    x0,
                    slice_data[j, k],
                    max_iter, tol)

                if success:
                    results[j, k] = result
                else:
                    results[j, k] = np.zeros(A.shape[1])
            else:
                results[j, k] = np.zeros(A.shape[1])
    return results


def _fit(self, data, mask=None, max_iter=1e3, tol=1e-6,
         use_osqppy=True,
         numba_threading_layer="workqueue",
         n_threads=None, n_cpus=None):
    # Note cholesky is ~50% slower but more robust
    if n_threads is not None:
        set_num_threads(n_threads)

    if numba_threading_layer != "default":
        config.THREADING_LAYER = numba_threading_layer

    if n_cpus is None:
        n_cpus = multiprocessing.cpu_count() - 1

    og_data_shape = data.shape
    if len(data.shape) < 4:
        data = data.reshape((1,) * (4 - data.ndim) + data.shape)

    m, n = self.fitter._reg.shape
    coeff = np.zeros((*data.shape[:3], n), dtype=np.float64)
    if mask is None:
        mask = np.ones(data.shape[:3], dtype=bool)

    R = np.ascontiguousarray(self.fitter._X, dtype=np.float64)
    A = np.ascontiguousarray(self.fitter._reg, dtype=np.float64)
    b = np.zeros(A.shape[0], dtype=np.float64)

    # Normalize constraints
    for i in range(A.shape[0]):
        A[i] /= np.linalg.norm(A[i])

    A_outer = np.empty((n, n, m), dtype=np.float64)
    for k in range(m):
        for i in range(n):
            for j in range(n):
                A_outer[i, j, k] = A[k, i] * A[k, j]

    Q = R.T @ R
    if use_osqppy:
        if n_cpus > 1:
            ray.init(ignore_reinit_error=True)

            data_id = ray.put(data)
            mask_id = ray.put(mask)
            Q_id = ray.put(Q)
            A_id = ray.put(A)
            b_id = ray.put(b)
            R_id = ray.put(R)

            @ray.remote(
                num_cpus=n_cpus)
            def process_batch_remote(batch_indices, data, mask,
                                     Q, A, b, R):
                from scipy.sparse import csr_matrix
                import osqp
                import numpy as np

                m = osqp.OSQP()
                m.setup(
                    P=csr_matrix(Q), A=csr_matrix(A), l=b,
                    u=None, q=None,
                    verbose=False)
                return_values = np.zeros(
                    (len(batch_indices),) + data.shape[1:3] + (A.shape[1],),
                    dtype=np.float64)
                for i, ii in enumerate(batch_indices):
                    for jj in range(data.shape[1]):
                        for kk in range(data.shape[2]):
                            if mask[ii, jj, kk]:
                                c = np.dot(-R.T, data[ii, jj, kk])
                                m.update(q=c)
                                results = m.solve()
                                return_values[i, jj, kk] = results.x
                return return_values

            # Launch tasks in chunks
            all_indices = list(range(data.shape[0]))
            indices_chunked = list(chunk_indices(all_indices, n_cpus * 2))
            futures = [
                process_batch_remote.remote(batch, data_id, mask_id,
                                            Q_id, A_id,
                                            b_id, R_id)
                for batch in indices_chunked
            ]

            # Collect and assign results
            for batch, future in zip(
                    indices_chunked, tqdm(futures)):
                results = ray.get(future)
                for i, ii in enumerate(batch):
                    coeff[ii] = results[i]
        else:
            m = osqp.OSQP()
            m.setup(
                P=csr_matrix(Q), A=csr_matrix(A), l=b,
                u=None, q=None,
                verbose=False, adaptive_rho=True)
            for ii in tqdm(range(data.shape[0])):
                for jj in range(data.shape[1]):
                    for kk in range(data.shape[2]):
                        if mask[ii, jj, kk]:
                            c = np.dot(-R.T, data[ii, jj, kk])
                            m.update(q=c)
                            results = m.solve()
                            coeff[ii, jj, kk] = results.x
        coeff = coeff.reshape(og_data_shape[:-1] + (n,))
        return MSDeconvFit(self, coeff, None)
    else:
        x0 = np.linalg.pinv(A) @ np.ones(A.shape[0])
        x0 = find_analytic_center(A, b, x0)

        Rt = -R.T
        R_pinv = np.linalg.pinv(R).astype(np.float64)
        data = np.ascontiguousarray(data, dtype=np.float64)

        Rt = np.ascontiguousarray(Rt, dtype=np.float64)
        R_pinv = np.ascontiguousarray(R_pinv, dtype=np.float64)
        Q = np.ascontiguousarray(Q, dtype=np.float64)
        A = np.ascontiguousarray(A, dtype=np.float64)
        At = np.ascontiguousarray(A.T, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        x0 = np.ascontiguousarray(x0, dtype=np.float64)

        warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

        if n_cpus > 1:
            ray.init(ignore_reinit_error=True)

            data_id = ray.put(data)
            mask_id = ray.put(mask)
            Rt_id = ray.put(Rt)
            R_pinv_id = ray.put(R_pinv)
            Q_id = ray.put(Q)
            A_id = ray.put(A)
            At_id = ray.put(At)
            A_outer_id = ray.put(A_outer)
            b_id = ray.put(b)
            x0_id = ray.put(x0)

            @ray.remote(
                num_cpus=n_cpus,
                runtime_env={"env_vars": {
                    "NUMBA_THREADING_LAYER": f"{numba_threading_layer}"}})
            def process_batch_remote(batch_indices, data, mask, Rt, R_pinv,
                                     Q, A, At, A_outer, b, x0, max_iter, tol):
                return [_process_slice(
                    data[ii], mask[ii], Rt, R_pinv, Q, A, At,
                    A_outer, b, x0, max_iter, tol
                ) for ii in batch_indices]

            # Launch tasks in chunks
            all_indices = list(range(data.shape[0]))
            futures = [
                process_batch_remote.remote(batch, data_id, mask_id,
                                            Rt_id, R_pinv_id,
                                            Q_id, A_id, At_id,
                                            A_outer_id, b_id, x0_id,
                                            max_iter, tol)
                for batch in chunk_indices(all_indices, n_cpus * 2)
            ]

            # Collect and assign results
            for batch, future in zip(
                    chunk_indices(all_indices, n_cpus * 2), tqdm(futures)):
                results = ray.get(future)
                for i, ii in enumerate(batch):
                    coeff[ii] = results[i]
        else:
            for ii in tqdm(range(data.shape[0])):
                coeff[ii] = _process_slice(
                    data[ii], mask[ii],
                    Rt,
                    R_pinv,
                    Q,
                    A,
                    At,
                    A_outer,
                    b,
                    x0,
                    max_iter, tol)

        coeff = coeff.reshape(og_data_shape[:-1] + (n,))
        return MSDeconvFit(self, coeff, None)


MultiShellDeconvModel.fit = _fit

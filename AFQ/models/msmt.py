import multiprocessing
import numpy as np

from scipy.optimize import minimize

from numba import njit, prange, config, set_num_threads
from tqdm import tqdm
import ray

from dipy.reconst.mcsd import MSDeconvFit
from dipy.reconst.mcsd import MultiShellDeconvModel

from AFQ.utils.stats import chunk_indices


config.THREADING_LAYER = 'workqueue'


__all__ = ["fit"]


# CPU implementation of primal-dual Interior Point method
# for convex quadratic constrained optimization (QP)
# Specifically, ||Rx-d||_2 where Ax>=b
# parallelized to solve 10s-100s of thousands of QPs
# ultimately used to fit MSMT CSD
@njit(fastmath=True)
def solve_qp(Rt, R_pinv, G, A, b, x0,
             d, max_iter, tol, use_chol):
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
    y = np.ones(m)
    l = np.ones(m)
    dx = np.zeros(n)
    dy = np.zeros(m)
    dl = np.zeros(m)

    tau = 0.9
    shur_regularization = 1e-6
    for _ in range(max_iter):
        mu = (y @ l) / m

        # Predictor step
        Z = safe_divide_vec(l, y)
        dy = A @ x - y - b
        rhs2 = -dy - y
        rhs1l = -(G @ x - A.T @ l + c)
        rhs1 = rhs1l + A.T @ (Z * rhs2)

        schur = G + A.T @ np.diag(Z) @ A
        schur += np.eye(schur.shape[0]) * shur_regularization
        if use_chol:
            L = np.linalg.cholesky(schur)
            dx = cholesky_solve(L, rhs1)
        else:
            schur_diag = np.diag(schur)
            dx = conjugate_gradient_precond(
                schur, rhs1, dx,
                schur_diag, 5, 1e-4 * tol)
        dl_aff = Z * (rhs2 - A @ dx)
        dy_aff = dy + A @ dx

        alpha_aff_pri = 1.0
        alpha_aff_dual = 1.0
        for ii in range(m):
            if dy_aff[ii] < 0:
                alpha_aff_pri = min(alpha_aff_pri, -y[ii] / dy_aff[ii])
            if dl_aff[ii] < 0:
                alpha_aff_dual = min(alpha_aff_dual, -l[ii] / dl_aff[ii])
        alpha_aff = min(tau * alpha_aff_pri, tau * alpha_aff_dual)

        mu_aff = ((y + alpha_aff * dy_aff) @ (l + alpha_aff * dl_aff)) / m
        sigma = (mu_aff / mu) ** 3
        mu = sigma * mu

        # Main corrector step
        rhs2 = -dy + (-y + safe_divide_vec(mu, l))
        rhs1 = rhs1l + A.T @ (Z * rhs2)

        if use_chol:
            dx = cholesky_solve(L, rhs1)
        else:
            dx = conjugate_gradient_precond(
                schur, rhs1, dx,
                schur_diag, max_iter, max(tol, 1e-2 * mu))
        dl = Z * (rhs2 - A @ dx)
        dy += A @ dx

        beta = 1.0
        sigma = 1.0
        for ii in range(m):
            if dy[ii] < 0 and dy[ii] * sigma < -y[ii]:
                sigma = -y[ii] / dy[ii]
            if dl[ii] < 0 and dl[ii] * beta < -l[ii]:
                beta = -l[ii] / dl[ii]
        beta *= tau
        sigma *= tau
        alpha = min(beta, sigma)

        x += alpha * dx
        y += alpha * dy
        l += alpha * dl

        if (alpha * np.sum(np.abs(dx)) < n * tol
            and alpha * np.sum(np.abs(dy)) < m * tol
                and alpha * np.sum(np.abs(dl)) < m * tol):
            if np.all(A @ x - b >= -tol):
                return True, x
            else:
                return False, x0

    return False, x0


@njit(fastmath=True)
def cholesky_solve(L, b):
    n = L.shape[0]

    # Forward substitution: L * y = b
    y = np.zeros_like(b)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Backward substitution: L^T * x = y
    x = np.zeros_like(b)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]

    return x


@njit(fastmath=True)
def conjugate_gradient_precond(A, b, x, diag, max_iter, tol):
    """
    Solves A x = b with Jacobi preconditioner (diag) using Conjugate Gradient.

    Args:
        A : 2D np.ndarray, symmetric positive-definite matrix
        b : 1D np.ndarray, RHS vector
        x : 1D np.ndarray, initial guess (will be modified in-place)
        diag : 1D np.ndarray, diagonal of preconditioner (Jacobi)
        max_iter : int, maximum number of iterations
        tol : float, stopping threshold

    Returns:
        x : Updated solution
    """
    r = b - A @ x
    r /= diag
    p = r.copy()
    rs_old = np.dot(r * r, diag)

    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap / diag
        rs_new = np.dot(r * r, diag)
        if rs_new < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

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


@njit(parallel=True, fastmath=True)
def _process_slice(slice_data, slice_mask,
                   Rt, R_pinv,
                   G, A, b, x0,
                   max_iter, tol, use_chol):
    results = np.zeros(
        slice_data.shape[:2] + (A.shape[1],),
        dtype=np.float64)

    for j in prange(slice_data.shape[0]):
        x_prev = x0.copy()
        for k in range(slice_data.shape[1]):
            if slice_mask[j, k]:
                # previous values warm start next solver
                # More complicated warm starts are slower
                x_prev = np.clip(x_prev, -1.0, 1.0)
                success, x_prev = solve_qp(
                    Rt, R_pinv, G, A, b,
                    x_prev,
                    slice_data[j, k],
                    max_iter, tol, use_chol)

                if success:
                    results[j, k] = x_prev
                else:
                    results[j, k] = np.zeros(A.shape[1])
            else:
                results[j, k] = np.zeros(A.shape[1])
    return results


def _fit(self, data, mask=None, max_iter=1e6, tol=1e-6,
         n_threads=None, n_cpus=1, use_chol=False):
    # Note cholesky is ~50% slower but more robust
    if n_threads is not None:
        set_num_threads(n_threads)

    if n_cpus is None:
        n_cpus = multiprocessing.cpu_count() - 1

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

    Q = R.T @ R
    # Q += 1e-1 * np.eye(n)  # Strong Regularization # TODO
    x0 = np.linalg.pinv(A) @ np.ones(A.shape[0])
    x0 = find_analytic_center(A, b, x0)

    Rt = -R.T
    R_pinv = np.linalg.pinv(R).astype(np.float64)
    data = np.ascontiguousarray(data, dtype=np.float64)

    Rt = np.ascontiguousarray(Rt, dtype=np.float64)
    R_pinv = np.ascontiguousarray(R_pinv, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    x0 = np.ascontiguousarray(x0, dtype=np.float64)

    if n_cpus > 1:
        ray.init(ignore_reinit_error=True)

        data_id = ray.put(data)
        mask_id = ray.put(mask)
        Rt_id = ray.put(Rt)
        R_pinv_id = ray.put(R_pinv)
        Q_id = ray.put(Q)
        A_id = ray.put(A)
        b_id = ray.put(b)
        x0_id = ray.put(x0)

        @ray.remote(num_cpus=n_cpus)
        def process_batch_remote(batch_indices, data, mask, Rt, R_pinv,
                                 Q, A, b, x0, max_iter, tol, use_chol):
            return [_process_slice(
                data[ii], mask[ii], Rt, R_pinv, Q, A, b, x0, max_iter, tol, use_chol
            ) for ii in batch_indices]

        # Launch tasks in chunks
        all_indices = list(range(data.shape[0]))
        futures = [
            process_batch_remote.remote(batch, data_id, mask_id,
                                        Rt_id, R_pinv_id,
                                        Q_id, A_id, b_id, x0_id,
                                        max_iter, tol, use_chol)
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
                b,
                x0,
                max_iter, tol, use_chol)

    return MSDeconvFit(self, coeff, None)


MultiShellDeconvModel.fit = _fit

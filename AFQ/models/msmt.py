import multiprocessing

import numpy as np
import osqp
import ray
from dipy.reconst.mcsd import MSDeconvFit, MultiShellDeconvModel
from scipy.sparse import csr_matrix
from tqdm import tqdm

from AFQ.utils.stats import chunk_indices

__all__ = ["MultiShellDeconvModel"]


def _fit(self, data, mask=None, n_cpus=None):
    """
    Use OSQP to fit the multi-shell spherical deconvolution model.
    """
    if n_cpus is None:
        n_cpus = max(multiprocessing.cpu_count() - 1, 1)

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
            import numpy as np
            import osqp
            from scipy.sparse import csr_matrix

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


MultiShellDeconvModel.fit = _fit

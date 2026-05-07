import numpy as np
import osqp
from dipy.reconst.mcsd import MSDeconvFit, MultiShellDeconvModel
from scipy.sparse import csr_matrix
from tqdm import tqdm

__all__ = ["MultiShellDeconvModel"]


def _fit(self, data, mask=None):
    """
    Use OSQP to fit the multi-shell spherical deconvolution model.
    """
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

    A = csr_matrix(A)
    Q = csr_matrix(Q)

    m = osqp.OSQP()
    m.setup(
        P=Q,
        A=A,
        l=b,
        u=None,
        q=None,
        verbose=False,
        adaptive_rho=True,
    )
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

import numpy as np
import sys
import itertools
import os
import multiprocessing as mp

from cvxopt import matrix
from cvxopt.solvers import options, qp

import ray
from ray.experimental import tqdm_ray

from dipy.reconst import shm
from dipy.reconst import csdeconv as csd
from dipy.core.sphere import HemiSphere


options['show_progress'] = False  # disable cvxopt output
options['maxiters'] = 100    # maximum number of qp iteration
options['abstol'] = 1e-6
options['reltol'] = 1e-6
options['feastol'] = 1e-9

nprocs = mp.cpu_count()
remote_tqdm = ray.remote(tqdm_ray.tqdm)


__all__ = ['AsymConstrainedSphericalDeconvModel']


def _get_weights(vertices, sigma=40):
    '''Computes neighbouring fod weights for asymmetric CSD.

    Vendorized from:
    https://github.com/mabast85/aFOD/blob/master/aFOD/csdeconv/csdeconv.py

    Generates matrix that contains the weight for each point on the
    neighbouring fod based on their distance to the current voxel and
    the angle between the current fod point and the point of the
    neighbouring fod.

    Args:
        vertices: Nx3 numpy array with vertices of the unit sphere.
        sigma: cut-off angle.

    Returns:
        26xN weight matrix as numpy array.
    '''
    neighs = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
    neighs = np.delete(neighs, 13, 0)   # Remove [0, 0, 0]
    d = np.linalg.norm(neighs, ord=2, axis=1)
    deg_mat = np.arccos(np.dot(neighs / d[:, np.newaxis], vertices.T))
    weights = np.exp(-deg_mat / np.deg2rad(sigma))
    # Do not consider vertices that are not aligned with any neighbouring voxel
    weights[deg_mat > np.deg2rad(60)] = 0
    weights = weights / d[:, np.newaxis]    # Account for distance
    # Divide by the vertex-wise weight sum
    weights = weights / np.sum(weights, axis=0)[np.newaxis, :]
    weights[np.isnan(weights)] = 0  # Check for nans
    return weights


class AsymConstrainedSphericalDeconvModel(csd.ConstrainedSphericalDeconvModel):
    '''
    Vendorized from:
    https://github.com/mabast85/aFOD/blob/master/aFOD/csdeconv/csdeconv.py
    '''

    def fit_prev(self, data, **kwargs):
        self.prev_fod = super().fit(data, **kwargs).shm_coeff

    def fit(self, data, **kwargs):
        # if isinstance(self.sphere, HemiSphere): # TODO: is this necessary?
        #     raise ValueError("Asym CSD does not support HemiSphere")

        _w = _get_weights(self.sphere.vertices)
        _X = np.concatenate((self._X, self.B_reg), axis=0)

        if not hasattr(self, 'prev_fod'):
            self.fit_prev(data, engine="ray", **kwargs)

        data = data[..., self._where_dwi]
        fod = np.zeros(
            (*data[..., 0].shape, self.B_reg.shape[1]),
            dtype=np.float32)

        neighs = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
        neighs = np.delete(neighs, 13, 0)   # Remove [0, 0, 0]

        if not ray.is_initialized():
            ray.init()
        data_ref = ray.put(data)
        prev_fod_ref = ray.put(self.prev_fod)

        @ray.remote
        def _ray_fitter(_P, B_reg, neighs, _X, _w, xyz, bar):
            h = matrix(np.zeros(B_reg.shape[0]))
            args = [matrix(_P), 0, matrix(-B_reg), h]
            data = ray.get(data_ref)
            prev_fod = ray.get(prev_fod_ref)

            shm_coeffs = []
            for x, y, z in xyz:
                signal = data[x, y, z, :]

                fNeighs = prev_fod[
                    x + neighs[:, 0],
                    y + neighs[:, 1],
                    z + neighs[:, 2]]
                n_fod = np.diag(np.dot(np.dot(-B_reg, fNeighs.T), _w))
                signal = np.concatenate((signal, n_fod))
                f = np.dot(-_X.T, signal)
                # Using cvxopt
                args[1] = matrix(f)

                # Suppress cvxopt output
                sys.stdout = open(os.devnull, 'w')
                sol = qp(*args)
                sys.stdout = sys.__stdout__

                shm_coeffs.append((
                    x, y, z,
                    np.array(sol['x']).reshape((f.shape[0],)),
                    'optimal' not in sol['status']))
                bar.update.remote(1)
            return shm_coeffs

        # Chunk up indices
        mask = kwargs.get('mask', np.ones_like(data[..., 0], dtype=bool))
        ii = np.where(mask)
        batches = np.array_split(list(zip(*ii)), nprocs)
        bar = remote_tqdm.remote(total=len(ii[0]), desc="Running Asym CSD")
        tasks = []
        for batch in batches:
            task_args = [
                self._P, self.B_reg, neighs,
                _X, _w, batch, bar]
            tasks.append(_ray_fitter.remote(*task_args))

        results = []
        for batch_result in ray.get(tasks):
            results.extend(batch_result)

        suboptimal_count = 0
        for x, y, z, sol, suboptimal in results:
            suboptimal_count += int(suboptimal)
            fod[x, y, z, :] = sol
        print("Suboptimal solutions: %d" % suboptimal_count)
        bar.close.remote()

        return shm.SphHarmFit(self, fod, mask)

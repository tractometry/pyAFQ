import numpy as np
from scipy.stats import mode
from tqdm import tqdm
import logging
from itertools import product

from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
import dipy.core.gradients as dpg
import dipy.tracking.life as dtl

# TODO: remove these in the final version
from dipy.direction.pmf import PmfGen
from dipy.data import default_sphere
from dipy.direction import (
    DeterministicMaximumDirectionGetter)
import nibabel as nib


logger = logging.getLogger('AFQ')


class PrincipalTracking(LocalTracking):
    def __init__(self, data, gtab, n_sls, affine,
                 step_size, brain_mask, stopping_criterion,
                 max_cross=None, maxlen=500, minlen=5,
                 fixedstep=True, return_all=True,
                 unidirectional=False,
                 randomize_forward_direction=False, initial_directions=None):
        """Creates streamlines by using local fiber-tracking.

        Parameters
        ----------
        stopping_criterion : instance of StoppingCriterion
            Identifies endpoints and invalid points to inform tracking.
        step_size : float, optional
            Step size used for tracking.
        max_cross : int or None, optional
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int, optional
            Maximum length of generated streamlines. Longer streamlines will be
            discarted if `return_all=False`.
        minlen : int, optional
            Minimum length of generated streamlines. Shorter streamlines will
            be discarted if `return_all=False`.
        fixedstep : bool, optional
            If true, a fixed stepsize is used, otherwise a variable step size
            is used.
        return_all : bool, optional
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        unidirectional : bool, optional
            If true, the tracking is performed only in the forward direction.
            The seed position will be the first point of all streamlines.
        randomize_forward_direction : bool, optional
            If true, the forward direction is randomized (multiplied by 1
            or -1). Otherwise, the provided forward direction is used.
        initial_directions: array (N, npeaks, 3), optional
            Initial direction to follow from the ``seed`` position. If
            ``max_cross`` is None, one streamline will be generated per peak
            per voxel. If None, `direction_getter.initial_direction` is used.
        """
        self.data = data.copy()

        # determine shell to use
        bvals = dpg.round_bvals(gtab.bvals)
        single_shell_b_idx = np.logical_or(
            bvals == mode(bvals).mode,
            gtab.b0s_mask)
        self.bvecs = gtab.bvecs[single_shell_b_idx]
        self.bvals = bvals[single_shell_b_idx]

        self.gtab = dpg.gradient_table(self.bvals, self.bvecs)
        self.bvecs_nonzero = self.bvecs[~self.gtab.b0s_mask]
        self.bvals_nonzero = self.bvals[~self.gtab.b0s_mask]

        self._refit_models(brain_mask.astype(bool))

        super(PrincipalTracking, self).__init__(
            self.direction_getter, stopping_criterion,
            [], np.eye(4),
            step_size, max_cross=max_cross,
            maxlen=maxlen, minlen=minlen,
            fixedstep=fixedstep,
            return_all=return_all,
            save_seeds=False,
            unidirectional=unidirectional,
            randomize_forward_direction=randomize_forward_direction,
            initial_directions=initial_directions)

        self.n_sls = n_sls
        self.seeds = self._generate_seeds()
        self.out_affine = affine

    def _refit_models(self, mask):
        noise_threshold = 200  # new parameter
        self.data[self.data < noise_threshold] = 0

        # dtf = self.model.fit(self.data, mask=mask)
        # evals = dtf.model_params[..., :3]
        # evecs = dtf.model_params[..., 3:12].reshape(
        #     dtf.model_params.shape[:3] + (3, 3))
        # odf = tensor_odf(evals, evecs, default_sphere, num_batches=1)

        # self.odf[mask] = odf[mask]
        self.direction_getter = DeterministicMaximumDirectionGetter.from_pmf(
            PmfGen(self.data, default_sphere),
            max_angle=60,
            sphere=default_sphere)
        # return dtf

    def _generate_seeds(self):
        for _ in range(self.n_sls):
            yield np.unravel_index(
                self.data[..., ~self.gtab.b0s_mask].argmax(),
                self.data[..., ~self.gtab.b0s_mask].shape)[:3]

    def _life_adjustment(self):
        track = self._generate_tractogram()

        factor = 1  # new parameter

        saving_points = [(self.n_sls * ii) // 10 for ii in range(1, 11)]
        for jj, sl in enumerate(tqdm(track, total=self.n_sls)):
            # calculate LIFE tensors
            if len(sl) > 1:
                tensors = dtl.streamline_tensors(sl)
                sig = np.empty((len(sl), np.sum(~self.gtab.b0s_mask)))
                for ii, tensor in enumerate(tensors):
                    ADC = np.diag(np.linalg.multi_dot([
                        self.bvecs_nonzero, tensor, self.bvecs_nonzero.T]))
                    sig[ii] = np.exp(-self.bvals_nonzero * ADC)
            else:
                logger.info(f"Single point streamline: {sl}")

            # Update data
            affected_voxels = np.zeros(self.data.shape[:3], dtype=bool)
            for ii in range(len(sl)):
                pos_index = np.zeros((3, 2), dtype=int)
                pos_weights = np.zeros((3, 2), dtype=float)
                for kk in range(3):
                    flr = np.floor(sl[ii][kk])
                    rem = sl[ii][kk] - flr

                    pos_index[kk][0] = flr + (flr == -1)
                    pos_index[kk][1] = flr + (
                        flr != (self.data.shape[kk] - 1))
                    pos_weights[kk][0] = 1 - rem
                    pos_weights[kk][1] = rem

                for x_pos, y_pos, z_pos in product(range(2), repeat=3):
                    idx = pos_index[0][x_pos], \
                        pos_index[1][y_pos], \
                        pos_index[2][z_pos]
                    affected_voxels[idx] = True
                    pos_weight = pos_weights[0][x_pos] *\
                        pos_weights[1][y_pos] *\
                        pos_weights[2][z_pos]

                    if len(sl) > 1:
                        self.data[idx + (~self.gtab.b0s_mask,)] -=\
                            sig[ii] * pos_weight * factor
                    else:
                        self.data[idx + (~self.gtab.b0s_mask,)] -=\
                            self.data[idx + (~self.gtab.b0s_mask,)] *\
                            max(factor, 1)

            self._refit_models(affected_voxels)

            if jj in saving_points:
                nib.save(
                    nib.Nifti1Image(
                        self.data,
                        self.out_affine),
                    f"/Users/john/pyafq_scripts/prinicpal_test/{jj}.nii.gz")
            yield sl

    def __iter__(self):
        return utils.transform_tracking_output(
            self._life_adjustment(), self.out_affine)

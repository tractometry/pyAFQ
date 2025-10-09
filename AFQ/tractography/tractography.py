from collections.abc import Iterable
import numpy as np
import nibabel as nib
import logging
from tqdm import tqdm

import dipy.data as dpd
from dipy.reconst.dti import decompose_tensor, from_lower_triangular
from dipy.align import resample
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.stopping_criterion import (ThresholdStoppingCriterion,
                                              CmcStoppingCriterion,
                                              ActStoppingCriterion)
from dipy.reconst import shm

from nibabel.streamlines.tractogram import LazyTractogram

from dipy.tracking.local_tracking import (LocalTracking,
                                          ParticleFilteringTracking)
from AFQ._fixes import tensor_odf
from AFQ.tractography.utils import gen_seeds, get_percentile_threshold


def track(params_file, directions="prob", max_angle=30., sphere=None,
          seed_mask=None, seed_threshold=0.5, thresholds_as_percentages=False,
          n_seeds=2000000, random_seeds=True, rng_seed=None, stop_mask=None,
          stop_threshold=0.5, step_size=0.5, minlen=50, maxlen=250,
          odf_model="CSD", basis_type="descoteaux07", legacy=True,
          tracker="pft", trx=False):
    """
    Tractography

    Parameters
    ----------
    params_file : str, nibabel img.
        Full path to a nifti file containing CSD spherical harmonic
        coefficients, or nibabel img with model params.
    directions : str
        How tracking directions are determined.
        One of: {"det" | "prob"}
        Default: "prob"
    max_angle : float, optional.
        The maximum turning angle in each step. Default: 30
    sphere : Sphere object, optional.
        The discretization of direction getting. default:
        dipy.data.default_sphere.
    seed_mask : array, optional.
        Float or binary mask describing the ROI within which we seed for
        tracking.
        Default to the entire volume (all ones).
    seed_threshold : float, optional.
        A value of the seed_mask above which tracking is seeded.
        Default to 0.
    n_seeds : int or 2D array, optional.
        The seeding density: if this is an int, it is is how many seeds in each
        voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
        array, these are the coordinates of the seeds. Unless random_seeds is
        set to True, in which case this is the total number of random seeds
        to generate within the mask. Default: 2000000
    random_seeds : bool
        Whether to generate a total of n_seeds random seeds in the mask.
        Default: True
    rng_seed : int
        random seed used to generate random seeds if random_seeds is
        set to True. Default: None
    stop_mask : array or str, optional.
        If array: A float or binary mask that determines a stopping criterion
        (e.g. FA).
        If tuple: it contains a sequence that is interpreted as:
        (pve_wm, pve_gm, pve_csf), each item of which is either a string
        (full path) or a nibabel img to be used in particle filtering
        tractography.
        A tuple is required if tracker is set to "pft".
        Defaults to no stopping (all ones).
    stop_threshold : float or tuple, optional.
        If float, this a value of the stop_mask below which tracking is
        terminated (and stop_mask has to be an array).
        If str, "CMC" for Continuous Map Criterion [Girard2014]_.
                "ACT" for Anatomically-constrained tractography [Smith2012]_.
        A string is required if the tracker is set to "pft".
        Defaults to 0 (this means that if no stop_mask is passed,
        we will stop only at the edge of the image).
    thresholds_as_percentages : bool, optional
        Interpret seed_threshold and stop_threshold as percentages of the
        total non-nan voxels in the seed and stop mask to include
        (between 0 and 100), instead of as a threshold on the
        values themselves.
        Default: False
    step_size : float, optional.
        The size of a step (in mm) of tractography. Default: 0.5
    minlen: int, optional
        The miminal length (mm) in a streamline. Default: 20
    maxlen: int, optional
        The miminal length (mm) in a streamline. Default: 250
    odf_model : str or Definition, optional
        Can be either a string or Definition. If a string, it must be one of
        {"DTI", "CSD", "DKI", "GQ", "RUMBA"}. If a Definition, we assume
        it is a definition of a file containing Spherical Harmonics
        coefficients.
        Defaults to use "CSD"
    basis_type : str, optional
        The spherical harmonic basis type used to represent the coefficients.
        One of {"descoteaux07", "tournier07"}. Deafult: "descoteaux07"
    legacy : bool, optional
        Whether to use the legacy implementation of the direction getter.
        See Dipy documentation for more details. Default: True
    tracker : str, optional
        Which strategy to use in tracking. This can be the standard local
        tracking ("local") or Particle Filtering Tracking ([Girard2014]_).
        One of {"local", "pft"}. Default: "pft"
    trx : bool, optional
        Whether to return the streamlines compatible with input to TRX file
        (i.e., as a LazyTractogram class instance).
        Default: False

    Returns
    -------
    list of streamlines ()

    References
    ----------
    .. [Girard2014] Girard, G., Whittingstall, K., Deriche, R., &
        Descoteaux, M. Towards quantitative connectivity analysis: reducing
        tractography biases. NeuroImage, 98, 266-278, 2014.
    .. [Smith2012] Smith RE, Tournier JD, Calamante F, Connelly A.
        Anatomically-constrained tractography: improved diffusion
        MRI streamlines tractography through effective use of anatomical
        information. Neuroimage. 2012 Sep;62(3):1924-38.
        doi: 10.1016/j.neuroimage.2012.06.005. Epub 2012 Jun 13.
    """
    logger = logging.getLogger('AFQ')

    logger.info("Loading Image...")
    if isinstance(params_file, str):
        params_img = nib.load(params_file)
    else:
        params_img = params_file

    model_params = params_img.get_fdata()
    if isinstance(odf_model, str):
        odf_model = odf_model.upper()
    directions = directions.lower()

    # We need to calculate the size of a voxel, so we can transform
    # from mm to voxel units:
    minlen = int(minlen / step_size)
    maxlen = int(maxlen / step_size)

    if seed_mask is None:
        seed_mask = np.ones(params_img.shape[:3])

    seeds = gen_seeds(
        seed_mask, seed_threshold,
        n_seeds, thresholds_as_percentages,
        random_seeds, rng_seed, params_img.affine)

    if sphere is None:
        sphere = dpd.default_sphere

    logger.info("Getting Directions...")
    if directions == "det":
        dg = DeterministicMaximumDirectionGetter
    elif directions == "prob":
        dg = ProbabilisticDirectionGetter
    else:
        raise ValueError(f"Unrecognized direction '{directions}'.")

    logger.debug(f"Using basis type: {basis_type}")
    logger.debug(f"Using legacy DG: {legacy}")

    if odf_model == "DTI" or odf_model == "DKI":
        evals, evecs = decompose_tensor(
            from_lower_triangular(model_params))
        odf = tensor_odf(evals, evecs, sphere)
        dg = dg.from_pmf(odf, max_angle=max_angle, sphere=sphere)
    elif "AODF" in odf_model:
        sh_order = shm.order_from_ncoef(
            model_params.shape[3], full_basis=True)
        pmf = shm.sh_to_sf(
            model_params, sphere,
            sh_order_max=sh_order, full_basis=True)
        pmf[pmf < 0] = 0
        dg = dg.from_pmf(
            np.asarray(pmf, dtype=float),
            max_angle=max_angle, sphere=sphere)
    else:
        dg = dg.from_shcoeff(model_params, max_angle=max_angle, sphere=sphere,
                             basis_type=basis_type, legacy=legacy)

    if stop_mask is None:
        stop_mask = np.ones(params_img.shape[:3])

    if isinstance(stop_threshold, str):
        if not (isinstance(stop_mask, Iterable) and len(stop_mask) == 3):
            raise RuntimeError(
                "You are using CMC/ACT stropping, but did not provide a length "
                "3 iterable for `stop_mask`. "
                "Expected a (pve_wm, pve_gm, pve_csf) tuple.")
        pves = []
        pve_affines = []
        for pve in stop_mask:
            if isinstance(pve, str):
                seg_data = nib.load(pve).get_fdata()
                seg_affine = nib.load(pve).affine
            elif isinstance(pve, nib.Nifti1Image):
                seg_data = pve.get_fdata()
                seg_affine = pve.affine
            else:
                seg_data = pve
                seg_affine = params_img.affine
            pves.append(seg_data)
            pve_affines.append(seg_affine)

        pve_wm_data, pve_gm_data, pve_csf_data = pves
        pve_wm_affine, pve_gm_affine, pve_csf_affine = pve_affines
        pve_wm_data = resample(
            pve_wm_data, model_params[..., 0],
            moving_affine=pve_wm_affine,
            static_affine=params_img.affine).get_fdata()
        pve_gm_data = resample(
            pve_gm_data, model_params[..., 0],
            moving_affine=pve_gm_affine,
            static_affine=params_img.affine).get_fdata()
        pve_csf_data = resample(
            pve_csf_data, model_params[..., 0],
            moving_affine=pve_csf_affine,
            static_affine=params_img.affine).get_fdata()

        if stop_threshold == "CMC":
            stopping_criterion = CmcStoppingCriterion.from_pve(
                pve_wm_data,
                pve_gm_data,
                pve_csf_data,
                step_size=step_size,
                average_voxel_size=np.mean(
                    params_img.header.get_zooms()[:3]))
        elif stop_threshold == "ACT":
            stopping_criterion = ActStoppingCriterion.from_pve(
                pve_wm_data,
                pve_gm_data,
                pve_csf_data)
    else:
        if len(stop_mask) == 3:
            raise RuntimeError(
                "You are not using CMC/ACT stropping, but provided tissue "
                "probability maps in `stop_mask`. Please provide a single "
                "3D array for `stop_mask` or use CMC/ACT")

        if len(np.unique(stop_mask)) <= 2:
            stopping_criterion = ThresholdStoppingCriterion(stop_mask,
                                                            0.5)
        else:
            if thresholds_as_percentages:
                stop_threshold = get_percentile_threshold(
                    stop_mask, stop_threshold)
            stop_mask_copy = np.copy(stop_mask)
            stop_thresh_copy = np.copy(stop_threshold)
            stopping_criterion = ThresholdStoppingCriterion(stop_mask_copy,
                                                            stop_thresh_copy)

    if tracker == "local":
        my_tracker = LocalTracking
    elif tracker == "pft":
        my_tracker = ParticleFilteringTracking
    else:
        raise ValueError(f"Unrecognized tracker '{tracker}'. Must be one of "
                         "{'local', 'pft'}.")

    logger.info(
        f"Tracking with {len(seeds)} seeds, 2 directions per seed...")

    return _tracking(my_tracker, seeds, dg, stopping_criterion, params_img,
                     step_size=step_size, minlen=minlen,
                     maxlen=maxlen, random_seed=rng_seed,
                     trx=trx)


def _tracking(tracker, seeds, dg, stopping_criterion, params_img,
              step_size=0.5, minlen=40, maxlen=200,
              random_seed=None, trx=False):
    """
    Helper function
    """
    if len(seeds.shape) == 1:
        seeds = seeds[None, ...]

    tracker = tqdm(tracker(
        dg,
        stopping_criterion,
        seeds,
        params_img.affine,
        step_size=step_size,
        minlen=minlen,
        maxlen=maxlen,
        return_all=False,
        random_seed=random_seed))

    if trx:
        return LazyTractogram(lambda: tracker,
                              affine_to_rasmm=params_img.affine)
    else:
        return StatefulTractogram(tracker, params_img, Space.RASMM)

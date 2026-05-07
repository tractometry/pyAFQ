import logging
from math import radians
from time import time

import dipy.data as dpd
import nibabel as nib
import numba
import numpy as np
from dipy.align import resample
from dipy.core.sphere import HemiSphere
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.reconst import shm
from dipy.reconst.dti import decompose_tensor, from_lower_triangular
from dipy.tracking.stopping_criterion import ActStoppingCriterion
from dipy.tracking.tracker import (
    deterministic_tracking,
    pft_tracking,
)
from nibabel.streamlines.tractogram import LazyTractogram
from skimage.segmentation import find_boundaries
from tqdm import tqdm

from AFQ._fixes import tensor_odf
from AFQ.tractography.utils import gen_seeds


def track(
    params_file,
    pve,
    n_threads,
    directions="prob",
    max_angle=30.0,
    sphere="repulsion724",
    seed_mask=None,
    seed_threshold=0.5,
    gm_threshold=0.4,
    thresholds_as_percentages=False,
    n_seeds=1e7,
    random_seeds=True,
    rng_seed=None,
    step_size=0.5,
    minlen=20,
    maxlen=500,
    odf_model="CSD_AODF",
    basis_type="descoteaux07",
    legacy=True,
    trx=True,
    jit_backend="numba",
    jit_chunk_size=25000,
):
    """
    Tractography

    Parameters
    ----------
    params_file : str, nibabel img.
        Full path to a nifti file containing CSD spherical harmonic
        coefficients, or nibabel img with model params.
    pve : str, nibabel img
        Full path to a nifti file containing tissue probability maps,
        or nibabel img with tissue probability maps. This should be of the
        order (pve_csf, pve_gm, pve_wm).
    n_threads : int
        The number of threads to use in tracking.
        If 0 or -1, uses all available threads.
    directions : str
        How tracking directions are determined.
        One of: {"det" | "prob" | "pft"}
        pft refers to Particle Filtering Tracking ([Girard2014]_).
        Default: "prob"
    max_angle : float, optional.
        The maximum turning angle in each step. Default: 30
    sphere : str or DIPY Sphere
        The discretization of the ODF. Can be a DIPY Sphere or
        a string name of a DIPY Sphere.
        Default: "repulsion724"
    seed_mask : array, optional.
        Float or binary mask describing the ROI within which we seed for
        tracking.
        Default to the entire volume (all ones).
    seed_threshold : float, optional.
        A value of the seed_mask above which tracking is seeded.
        Default to 0.
    gm_threshold : float, optional.
        A value of the pve_gm_data above which we consider a voxel to be GM
        for the purposes of ACT stopping criterion. Default: 0.4.
    n_seeds : int or 2D array, optional.
        The seeding density: if this is an int, it is is how many seeds in each
        voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
        array, these are the coordinates of the seeds. Unless random_seeds is
        set to True, in which case this is the total number of random seeds
        to generate within the mask. Default: 1e7
    random_seeds : bool
        Whether to generate a total of n_seeds random seeds in the mask.
        Default: True
    rng_seed : int
        random seed used to generate random seeds if random_seeds is
        set to True. Default: None
    thresholds_as_percentages : bool, optional
        Interpret seed_threshold as percentages of the
        total non-nan voxels in the seed mask to include
        (between 0 and 100), instead of as a threshold on the
        values themselves.
        Default: False
    step_size : float, optional.
        The size of a step (in mm) of tractography. Default: 0.5
    minlen: int, optional
        The minimal length (mm) in a streamline. Default: 20
    maxlen: int, optional
        The maximum length (mm) in a streamline. Default: 250
    odf_model : str or Definition, optional
        Can be either a string or Definition. If a string, it must be one of
        {"DTI", "CSD", "DKI", "GQ", "RUMBA", "MSMT_AODF", "CSD_AODF", "MSMTCSD"}.
        If a Definition, we assume it is a definition of a file containing
        Spherical Harmonics coefficients.
        Defaults to use "CSD_AODF"
    basis_type : str, optional
        The spherical harmonic basis type used to represent the coefficients.
        One of {"descoteaux07", "tournier07"}. Default: "descoteaux07"
    legacy : bool, optional
        Whether the legacy SH basis definition should be used.
        See Dipy documentation for more details. Default: True
    trx : bool, optional
        Whether to return the streamlines compatible with input to TRX file
        (i.e., as a LazyTractogram class instance).
        Default: True
    jit_backend : str, optional
        If directions is "prob", the JIT backend to use. One of {"auto",
        "cuda", "metal", "webgpu", or "numba"}. Default: "numba"
    jit_chunk_size : int, optional
        If directions is "prob", the chunk size to use for JIT tracking.
        Default: 25000

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
    logger = logging.getLogger("AFQ")

    logger.info("Loading Image...")
    if isinstance(params_file, str):
        params_img = nib.load(params_file)
    else:
        params_img = params_file

    if isinstance(pve, str):
        pve_img = nib.load(pve)
    pve_data = pve_img.get_fdata()

    model_params = params_img.get_fdata()
    if isinstance(odf_model, str):
        odf_model = odf_model.upper()
    directions = directions.lower()

    if n_threads == -1:
        n_threads = 0

    if seed_mask is None:
        seed_mask = np.ones(params_img.shape[:3])

    if isinstance(sphere, str):
        sphere = dpd.get_sphere(name=sphere)

    if not len(pve_data.shape) == 4 or pve_data.shape[3] != 3:
        raise RuntimeError(
            "For pve, expected pve_data with shape [x, y, z, 3]. "
            f"Instead, got {pve_data.shape}."
        )

    pve_csf_data = pve_data[..., 0]
    pve_gm_data = pve_data[..., 1]
    pve_wm_data = pve_data[..., 2]

    pve_csf_data = resample(
        pve_csf_data,
        model_params[..., 0],
        moving_affine=pve_img.affine,
        static_affine=params_img.affine,
    ).get_fdata()
    pve_gm_data = resample(
        pve_gm_data,
        model_params[..., 0],
        moving_affine=pve_img.affine,
        static_affine=params_img.affine,
    ).get_fdata()
    pve_wm_data = resample(
        pve_wm_data,
        model_params[..., 0],
        moving_affine=pve_img.affine,
        static_affine=params_img.affine,
    ).get_fdata()

    # here we treat wm that borders the edge of the brain mask as gm
    # this is so that streamlines that hit the end of the
    # (presumably masked) fodf are treated as valid
    # (think brain stem)
    brain_mask = np.any(model_params != 0, axis=-1).astype(np.uint8)
    edge = find_boundaries(brain_mask, mode="inner")
    pve_gm_data[edge] = 1.0
    pve_wm_data[edge] = 0.0
    pve_csf_data[edge] = 0.0

    # We relax ACT stopping criterion here to allow streamlines closer
    # to the WM/GM boundary.
    pve_gm_data *= 0.5 / gm_threshold

    stopping_criterion = ActStoppingCriterion.from_pve(
        pve_wm_data, pve_gm_data, pve_csf_data
    )

    if odf_model == "DTI" or odf_model == "DKI":
        evals, evecs = decompose_tensor(from_lower_triangular(model_params))
        odf = tensor_odf(evals, evecs, sphere)
        model_params = shm.sf_to_sh(
            odf, sphere, basis_type=basis_type, legacy=legacy, full_basis=True
        )

    tracking_kwargs = {}
    if directions == "pft" and (odf_model == "DTI" or odf_model == "DKI"):
        tracking_kwargs["sf"] = odf
    else:
        sym_order = (-3.0 + np.sqrt(1.0 + 8.0 * model_params.shape[3])) / 2.0
        if sym_order.is_integer():
            sh_order_max = sym_order
            full_basis = False
        else:
            full_order = np.sqrt(model_params.shape[3]) - 1.0
            sh_order_max = full_order
            full_basis = True
        pmf = shm.sh_to_sf(
            model_params, sphere, sh_order_max=sh_order_max, full_basis=full_basis
        )
        pmf[pmf < 0] = 0
        tracking_kwargs["sf"] = pmf

    if rng_seed is not None:
        tracking_kwargs["random_seed"] = int(rng_seed)
    else:
        tracking_kwargs["random_seed"] = np.random.randint(0, 2**31 - 1)

    if directions == "prob":
        jit_backend = jit_backend.lower()
        if jit_backend == "auto":
            from cuslines import (
                ProbDirectionGetter,
                Tracker,
            )
        elif jit_backend == "cuda":
            from cuslines.cuda_python import (
                ProbDirectionGetter,
                Tracker,
            )
        elif jit_backend == "metal":
            from cuslines.metal import (
                MetalGPUTracker as Tracker,
            )
            from cuslines.metal import (
                MetalProbDirectionGetter as ProbDirectionGetter,
            )
        elif jit_backend == "webgpu":
            from cuslines.webgpu import (
                WebGPUProbDirectionGetter as ProbDirectionGetter,
            )
            from cuslines.webgpu import (
                WebGPUTracker as Tracker,
            )
        elif jit_backend == "numba":
            from cuslines.numba import (
                CPUProbDirectionGetter as ProbDirectionGetter,
            )
            from cuslines.numba import (
                CPUTracker as Tracker,
            )
        else:
            raise ValueError(
                "gpu_backend must be one of 'auto', 'cuda', "
                f"'metal', 'numba', or 'webgpu', not {jit_backend}"
            )

        dg = ProbDirectionGetter()

        seeds = gen_seeds(
            seed_mask,
            seed_threshold,
            n_seeds,
            thresholds_as_percentages,
            random_seeds,
            rng_seed,
            np.eye(4),  # JIT expects seeds in voxel space
        )

        minlen = int(minlen / step_size)
        maxlen = int(maxlen / step_size)

        R = params_img.affine[0:3, 0:3]
        vox_dim = np.mean(np.diag(np.linalg.cholesky(R.T.dot(R))))
        step_size = step_size / vox_dim

        if n_threads != 0:
            old_numba_n_threads = numba.get_num_threads()
            numba.set_num_threads(n_threads)

        with Tracker(
            dg,
            tracking_kwargs["sf"],
            pve_wm_data,
            gm_threshold,
            sphere.vertices,
            sphere.edges,
            sphere_symm=isinstance(sphere, HemiSphere),
            max_angle=radians(max_angle),
            step_size=step_size,
            min_pts=minlen,
            max_pts=maxlen,
            rng_seed=tracking_kwargs["random_seed"],
            chunk_size=jit_chunk_size,
        ) as jit_tracker:
            if trx:
                res = jit_tracker.generate_trx(seeds, params_img)
            else:
                res = jit_tracker.generate_sft(seeds, params_img)

        if n_threads != 0:
            numba.set_num_threads(old_numba_n_threads)
        return res
    else:
        if directions == "det":
            tracker = deterministic_tracking
        elif directions == "pft":
            tracker = pft_tracking
        else:
            raise ValueError(f"Unrecognized direction '{directions}'.")

        logger.info("Note there will be a long initial delay as seeds are initialized")

        seeds = gen_seeds(
            seed_mask,
            seed_threshold,
            n_seeds,
            thresholds_as_percentages,
            random_seeds,
            rng_seed,
            params_img.affine,
        )

        start_time = time()
        tracker = tqdm(
            tracker(
                seeds,
                stopping_criterion,
                params_img.affine,
                max_angle=max_angle,
                sphere=sphere,
                basis_type=basis_type,
                legacy=legacy,
                step_size=step_size,
                min_len=minlen,
                max_len=maxlen,
                return_all=False,
                nbr_threads=int(n_threads),
                **tracking_kwargs,
            ),
            total=len(seeds),
            desc="Tracking, note that the total is an overestimate...",
        )
        logger.info((f"Seed initialization took {time() - start_time:.2f} seconds."))

        if trx:
            return LazyTractogram(lambda: tracker, affine_to_rasmm=params_img.affine)
        else:
            return StatefulTractogram(tracker, params_img, Space.RASMM)

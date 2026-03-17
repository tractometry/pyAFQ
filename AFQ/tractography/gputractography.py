import logging
from math import radians

import nibabel as nib
import numpy as np
from dipy.align import resample
from dipy.reconst import shm

from AFQ.tractography.utils import gen_seeds

logger = logging.getLogger("AFQ")


# Modified from https://github.com/dipy/GPUStreamlines/blob/master/run_dipy_gpu.py
def gpu_track(
    data,
    gtab,
    seed_path,
    pve_path,
    odf_model,
    sphere,
    directions,
    seed_threshold,
    thresholds_as_percentages,
    max_angle,
    step_size,
    minlen,
    maxlen,
    n_seeds,
    random_seeds,
    rng_seed,
    use_trx,
    ngpus,
    chunk_size,
    gpu_backend,
):
    """
    Perform GPU tractography on DWI data.

    Parameters
    ----------
    data : ndarray
        DWI data.
    gtab : GradientTable
        The gradient table.
    seed_path : str
        Float or binary mask describing the ROI within which we seed for
        tracking.
    pve_path : str
        Estimations of partial volumes of WM, GM, and CSF.
    odf_model : str, optional
        One of {"OPDT", "CSA"}
    seed_threshold : float
        The value of the seed_path above which tracking is seeded.
    thresholds_as_percentages : bool
        Interpret seed_threshold as percentages of the
        total non-nan voxels in the seed mask to include
        (between 0 and 100), instead of as a threshold on the
        values themselves.
    max_angle : float
        The maximum turning angle in each step.
    step_size : float
        The size of a step (in mm) of tractography.
    n_seeds : int
        The seeding density: if this is an int, it is is how many seeds in each
        voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
        array, these are the coordinates of the seeds. Unless random_seeds is
        set to True, in which case this is the total number of random seeds
        to generate within the mask. Default: 1
    minlen: int, optional
        The minimal length (mm) in a streamline
    maxlen: int, optional
        The maximum length (mm) in a streamline
    random_seeds : bool
        If True, n_seeds is total number of random seeds to generate.
        If False, n_seeds encodes the density of seeds to generate.
    rng_seed : int
        random seed used to generate random seeds if random_seeds is
        set to True. Default: None
    use_trx : bool
        Whether to use trx.
    ngpus : int
        Number of GPUs to use.
    chunk_size : int
        Chunk size for GPU tracking.
    gpu_backend : str, optional
        GPU backend to use for tractography.
        One of {"auto", "cuda", "metal", "webgpu"}.
    Returns
    -------
    """
    gpu_backend = gpu_backend.lower()
    if gpu_backend == "auto":
        from cuslines import (
            BootDirectionGetter,
            GPUTracker,
            ProbDirectionGetter,
            PttDirectionGetter,
        )
    elif gpu_backend == "cuda":
        from cuslines.cuda_python import (
            BootDirectionGetter,
            GPUTracker,
            ProbDirectionGetter,
            PttDirectionGetter,
        )
    elif gpu_backend == "metal":
        from cuslines.metal import (
            MetalBootDirectionGetter as BootDirectionGetter,
        )
        from cuslines.metal import (
            MetalGPUTracker as GPUTracker,
        )
        from cuslines.metal import (
            MetalProbDirectionGetter as ProbDirectionGetter,
        )
        from cuslines.metal import (
            MetalPttDirectionGetter as PttDirectionGetter,
        )
    elif gpu_backend == "webgpu":
        from cuslines.webgpu import (
            WebGPUBootDirectionGetter as BootDirectionGetter,
        )
        from cuslines.webgpu import (
            WebGPUProbDirectionGetter as ProbDirectionGetter,
        )
        from cuslines.webgpu import (
            WebGPUPttDirectionGetter as PttDirectionGetter,
        )
        from cuslines.webgpu import (
            WebGPUTracker as GPUTracker,
        )
    else:
        raise ValueError(
            "gpu_backend must be one of 'auto', 'cuda', "
            f"'metal', or 'webgpu', not {gpu_backend}"
        )

    seed_img = nib.load(seed_path)
    directions = directions.lower()

    minlen = int(minlen / step_size)
    maxlen = int(maxlen / step_size)

    R = seed_img.affine[0:3, 0:3]
    vox_dim = np.mean(np.diag(np.linalg.cholesky(R.T.dot(R))))
    step_size = step_size / vox_dim

    # Roughly handle ACT/CMC for now
    wm_threshold = 0.5

    pve_img = nib.load(pve_path)

    wm_img = resample(
        pve_img.get_fdata()[..., 2],
        seed_img.get_fdata(),
        moving_affine=pve_img.affine,
        static_affine=seed_img.affine,
    )
    wm_data = wm_img.get_fdata()

    seed_data = seed_img.get_fdata()

    if directions == "boot":
        if odf_model.lower() == "opdt":
            dg = BootDirectionGetter.from_dipy_opdt(gtab, sphere)
        elif odf_model.lower() == "csa":
            dg = BootDirectionGetter.from_dipy_csa(gtab, sphere)
        else:
            raise ValueError(f"odf_model must be 'opdt' or 'csa', not {odf_model}")
    else:
        # Convert SH coefficients to ODFs
        sym_order = (-3.0 + np.sqrt(1.0 + 8.0 * data.shape[3])) / 2.0
        if sym_order.is_integer():
            sh_order_max = sym_order
            full_basis = False
        full_order = np.sqrt(data.shape[3]) - 1.0
        if full_order.is_integer():
            sh_order_max = full_order
            full_basis = True

        theta = sphere.theta
        phi = sphere.phi

        sampling_matrix, _, _ = shm.real_sh_descoteaux(
            sh_order_max, theta, phi, full_basis=full_basis, legacy=False
        )
        model = shm.SphHarmModel(gtab)
        model.cache_set("sampling_matrix", sphere, sampling_matrix)
        model_fit = shm.SphHarmFit(model, data, None)
        data = model_fit.odf(sphere).clip(min=0)

        if directions == "ptt":
            # Set FOD to 0 outside mask for probing
            data[wm_data < wm_threshold, :] = 0
            dg = PttDirectionGetter()
        elif directions == "prob":
            dg = ProbDirectionGetter()
        else:
            raise ValueError(
                f"directions must be 'boot', 'ptt', or 'prob', not {directions}"
            )

    seeds = gen_seeds(
        seed_data,
        seed_threshold,
        n_seeds,
        thresholds_as_percentages,
        random_seeds,
        rng_seed,
        np.eye(4),
    )

    if rng_seed is None:
        rng_seed = np.random.randint(0, 2**31 - 1)

    with GPUTracker(
        dg,
        data,
        wm_data,
        wm_threshold,
        sphere.vertices,
        sphere.edges,
        max_angle=radians(max_angle),
        step_size=step_size,
        min_pts=minlen,
        max_pts=maxlen,
        ngpus=ngpus,
        rng_seed=rng_seed,
        chunk_size=chunk_size,
    ) as gpu_tracker:
        if use_trx:
            return gpu_tracker.generate_trx(seeds, seed_img)
        else:
            return gpu_tracker.generate_sft(seeds, seed_img)

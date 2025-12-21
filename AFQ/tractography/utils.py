import logging

import dipy.tracking.utils as dtu
import numpy as np

logger = logging.getLogger("AFQ")


def get_percentile_threshold(mask, threshold):
    zero_mask = mask == 0
    mask[zero_mask] = np.nan
    new_threshold = np.nanpercentile(mask, 100 - threshold)
    mask[zero_mask] = 0
    return new_threshold


def _is_int(val):
    if isinstance(val, np.ndarray):
        return False
    if isinstance(val, int):
        return True
    elif isinstance(val, float):
        if val.is_integer():
            return True
        else:
            raise ValueError(f"n_seeds is set to {val}, which is not an integer")


def gen_seeds(
    seed_mask,
    seed_threshold,
    n_seeds,
    thresholds_as_percentages,
    random_seeds,
    rng_seed,
    affine,
):
    """
    Generate seeds for tracking. For parameters, see
    :func:`AFQ.tractography.tractography.track`.
    """
    logger.info("Generating Seeds...")
    if _is_int(n_seeds):
        # If it's float type, cast to integer:
        if isinstance(n_seeds, float):
            n_seeds = int(n_seeds)
        if len(np.unique(seed_mask)) > 2:
            if thresholds_as_percentages:
                seed_threshold = get_percentile_threshold(seed_mask, seed_threshold)
            seed_mask = seed_mask > seed_threshold
        if random_seeds:
            seeds = dtu.random_seeds_from_mask(
                seed_mask,
                seeds_count=n_seeds,
                seed_count_per_voxel=False,
                affine=affine,
                random_seed=rng_seed,
            )
        else:
            seeds = dtu.seeds_from_mask(seed_mask, density=n_seeds, affine=affine)
    else:
        # If user provided an array, we'll use n_seeds as the seeds:
        seeds = n_seeds
    return seeds

import logging
import os
import os.path as op

import dipy.tracking.streamlinespeed as dps
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.streamline import select_random_set_of_streamlines

import AFQ.recognition.sparse_decisions as ars
import AFQ.recognition.utils as abu
from AFQ.api.bundle_dict import BundleDict
from AFQ.recognition.criteria import run_bundle_rec_plan
from AFQ.recognition.preprocess import get_preproc_plan
from AFQ.utils.path import write_json

logger = logging.getLogger("AFQ")


def recognize(
    tg,
    img,
    mapping,
    bundle_dict,
    reg_template,
    n_cpus,
    nb_points=False,
    nb_streamlines=False,
    clip_edges=False,
    rb_recognize_params=None,
    refine_reco=False,
    prob_threshold=0,
    dist_to_waypoint=None,
    rng=None,
    return_idx=False,
    filter_by_endpoints=True,
    dist_to_atlas=4,
    save_intermediates=None,
    cleaning_params=None,
):
    """
    Segment streamlines into bundles.

    Parameters
    ----------
    tg : StatefulTractogram, or TrxFile
        Tractogram to segment.
    img : str, nib.Nifti1Image
        Image for reference.
    mapping : MappingDefinition
        Mapping from subject to template.
    bundle_dict : dict or AFQ.api.BundleDict
        Dictionary of bundles to segment.
    reg_template : str, nib.Nifti1Image
        Template image for registration.
    n_cpus : int
        Number of CPUs to use for parallelization.
    nb_points : int, boolean
        Resample streamlines to nb_points number of points.
        If False, no resampling is done. Can only be done
        on a StatefulTractogram.
        Default: False
    nb_streamlines : int, boolean
        Subsample streamlines to nb_streamlines.
        Can only be done on a StatefulTractogram.
        If False, no subsampling is done.
        Default: False
    clip_edges : bool
        Whether to clip the streamlines to be only in between the ROIs.
        Default: False
    rb_recognize_params : dict
        RecoBundles parameters for the recognize function.
        Default: dict(model_clust_thr=1.25, reduction_thr=25, pruning_thr=12)
    refine_reco : bool
        Whether to refine the RecoBundles segmentation.
        Default: False
    prob_threshold : float.
        Using AFQ Algorithm.
        Initial cleaning of fiber groups is done using probability maps
        from [Hua2008]_. Here, we choose an average probability that
        needs to be exceeded for an individual streamline to be retained.
        Default: 0.
    dist_to_waypoint : float.
        The distance that a streamline node has to be from the waypoint
        ROI in order to be included or excluded.
        If set to None (default), will be calculated as the
        center-to-corner distance of the voxel in the diffusion data.
        If a bundle has inc_addtol or exc_addtol in its bundle_dict, that
        tolerance will be added to this distance.
        For example, if you wanted to increase tolerance for the right
        arcuate waypoint ROIs by 3 each, you could make the following
        modification to your bundle_dict:
        bundle_dict["Right Arcuate"]["inc_addtol"] = [3, 3]
        Additional tolerances can also be negative.
        Default: None.
    rng : RandomState or int
        If None, creates RandomState.
        If int, creates RandomState with seed rng.
        Used in RecoBundles Algorithm.
        Default: None.
    return_idx : bool
        Whether to return the indices in the original streamlines as part
        of the output of segmentation.
        Default: False.
    filter_by_endpoints: bool
        Whether to filter the bundles based on their endpoints.
        Default: True.
    dist_to_atlas : float
        If filter_by_endpoints is True, this is the required distance
        from the endpoints to the atlas ROIs.
        Default: 4
    save_intermediates : str, optional
        The full path to a folder into which intermediate products
        are saved. Default: None, means no saving of intermediates.
    cleaning_params : dict, optional
        Cleaning params to pass to seg.clean_bundle. This will
        override the default parameters of that method. However, this
        can be overridden by setting the cleaning parameters in the
        bundle_dict. Default: {}.

    References
    ----------
    .. [Hua2008] Hua K, Zhang J, Wakana S, Jiang H, Li X, et al. (2008)
        Tract probability maps in stereotaxic spaces: analyses of white
        matter anatomy and tract-specific quantification. Neuroimage 39:
        336-347

    .. [Yeatman2012] Yeatman, Jason D., Robert F. Dougherty, Nathaniel J.
        Myall, Brian A. Wandell, and Heidi M. Feldman. 2012. "Tract Profiles
        of White Matter Properties: Automating Fiber-Tract Quantification"
        PloS One 7 (11): e49790.

    .. [Garyfallidis2018] Garyfallidis et al. Recognition of white matter
        bundles using local and global streamline-based registration and
        clustering, Neuroimage, 2017.
    """
    if cleaning_params is None:
        cleaning_params = {}
    if rb_recognize_params is None:
        rb_recognize_params = dict(
            model_clust_thr=1.25, reduction_thr=50, pruning_thr=12
        )
    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)

    if (save_intermediates is not None) and (not op.exists(save_intermediates)):
        os.makedirs(save_intermediates, exist_ok=True)

    logger.info("Preprocessing Streamlines")
    if not isinstance(bundle_dict, BundleDict):
        bundle_dict = BundleDict(bundle_dict)

    if isinstance(tg, StatefulTractogram):
        if nb_streamlines and len(tg) > nb_streamlines:
            tg = StatefulTractogram(
                select_random_set_of_streamlines(
                    tg.streamlines, nb_streamlines, rng=rng
                ),
                tg,
                tg.space,
            )

        if nb_points:
            tg = StatefulTractogram(
                dps.set_number_of_points(tg.streamlines, nb_points), tg, tg.space
            )

        tg.to_rasmm()

    n_streamlines = len(tg)
    recognized_bundles_dict = {}

    fiber_groups = {}
    meta = {}

    preproc_imap = get_preproc_plan(img, tg, dist_to_waypoint, dist_to_atlas)

    logger.info("Assigning Streamlines to Bundles")
    for bundle_name in bundle_dict.bundle_names:
        logger.info(f"Finding Streamlines for {bundle_name}")
        run_bundle_rec_plan(
            bundle_dict,
            tg.streamlines,
            mapping,
            img,
            reg_template,
            preproc_imap,
            bundle_name,
            recognized_bundles_dict,
            clip_edges=clip_edges,
            n_cpus=n_cpus,
            rb_recognize_params=rb_recognize_params,
            prob_threshold=prob_threshold,
            refine_reco=refine_reco,
            rng=rng,
            return_idx=return_idx,
            filter_by_endpoints=filter_by_endpoints,
            save_intermediates=save_intermediates,
            cleaning_params=cleaning_params,
        )

    if save_intermediates is not None:
        os.makedirs(save_intermediates, exist_ok=True)
        bc_path = op.join(save_intermediates, "sls_bundle_decisions.json")
        write_json(
            bc_path,
            {
                b_name: b_sls.selected_fiber_idxs.tolist()
                for b_name, b_sls in recognized_bundles_dict.items()
            },
        )

    sparse_dists = ars.compute_sparse_decisions(recognized_bundles_dict, n_streamlines)

    conflicts = ars.get_conflict_count(sparse_dists)
    if conflicts > 0:
        logger.info(
            (
                "Conflicts in bundle assignment detected. "
                f"{conflicts} conflicts detected in total out of "
                f"{n_streamlines} total streamlines. "
                "Defaulting to whichever bundle is closest to the include ROI,"
                "followed by whichever appears first "
                "in the bundle_dict."
            )
        )

        ars.remove_conflicts(sparse_dists, recognized_bundles_dict)

    # We do another round through, so that we can:
    # 1. Clip streamlines according to ROIs
    # 2. Re-orient streamlines
    logger.info("Re-orienting streamlines to consistent directions")
    for b_name, r_bd in recognized_bundles_dict.items():
        logger.info(f"Processing {b_name}")

        if len(r_bd.selected_fiber_idxs) == 0:
            # There's nothing here, set and move to the next bundle:
            if "bundlesection" in bundle_dict.get_b_info(b_name):
                for sb_name in bundle_dict.get_b_info(b_name)["bundlesection"]:
                    _return_empty(sb_name, return_idx, fiber_groups, img)
            else:
                _return_empty(b_name, return_idx, fiber_groups, img)
            continue

        b_def = r_bd.bundle_def
        if "bundlesection" in b_def:
            for sb_name, sb_include_cuts in b_def["bundlesection"].items():
                bundlesection_select_sl = abu.cut_sls_by_closest(
                    r_bd.get_selected_sls(),
                    r_bd.roi_closest,
                    sb_include_cuts,
                    in_place=False,
                )
                _add_bundle_to_fiber_group(
                    sb_name,
                    bundlesection_select_sl,
                    r_bd.selected_fiber_idxs,
                    r_bd.sls_flipped,
                    return_idx,
                    fiber_groups,
                    img,
                )
                _add_bundle_to_meta(sb_name, b_def, meta)
        else:
            _add_bundle_to_fiber_group(
                b_name,
                r_bd.get_selected_sls(cut=clip_edges),
                r_bd.selected_fiber_idxs,
                r_bd.sls_flipped,
                return_idx,
                fiber_groups,
                img,
            )
            _add_bundle_to_meta(b_name, b_def, meta)
    return fiber_groups, meta


# Helper functions for formatting the results
def _return_empty(bundle_name, return_idx, fiber_groups, img):
    """
    Helper function to return an empty dict under
    some conditions.
    """
    if return_idx:
        fiber_groups[bundle_name] = {}
        fiber_groups[bundle_name]["sl"] = StatefulTractogram([], img, Space.RASMM)
        fiber_groups[bundle_name]["idx"] = np.array([])
    else:
        fiber_groups[bundle_name] = StatefulTractogram([], img, Space.RASMM)


def _add_bundle_to_fiber_group(b_name, sl, idx, to_flip, return_idx, fiber_groups, img):
    """
    Helper function to add a bundle to a fiber group.
    """
    sl = abu.flip_sls(sl, to_flip, in_place=False)

    sl = StatefulTractogram(sl, img, Space.RASMM)

    if return_idx:
        fiber_groups[b_name] = {}
        fiber_groups[b_name]["sl"] = sl
        fiber_groups[b_name]["idx"] = idx
    else:
        fiber_groups[b_name] = sl


def _add_bundle_to_meta(bundle_name, b_def, meta):
    # remove keys that can never be serialized
    for key in ["include", "exclude", "prob_map", "start", "end", "curvature"]:
        b_def.pop(key, None)
    meta[bundle_name] = b_def

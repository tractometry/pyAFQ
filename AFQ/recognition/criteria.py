import logging
from time import time

import dipy.tracking.streamline as dts
import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
from dipy.segment.bundles import RecoBundles
from dipy.segment.clustering import QuickBundles
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
from trx.io import load as load_trx

import AFQ.recognition.cleaning as abc
import AFQ.recognition.curvature as abv
import AFQ.recognition.other_bundles as abo
import AFQ.recognition.roi as abr
import AFQ.recognition.utils as abu
from AFQ.api.bundle_dict import apply_to_roi_dict
from AFQ.recognition.clustering import subcluster_by_atlas
from AFQ.recognition.preprocess import PreprocPlan
from AFQ.recognition.utils import resample_tg
from AFQ.utils.streamlines import move_streamlines

# Criteria that are purely per-streamline and safe to run on a chunk
# without needing to see the rest of the tractogram. These run in the
# chunk-local phase.
criteria_order_chunk_local = [
    "length",
    "endpoint_dists",
    "cross_midline",
    "start",
    "end",
    "prob_map",
    "primary_axis",
    "include",
    "exclude",
    "curvature",
]

# RecoBundles needs the whole candidate pool for a bundle, so it runs
# in the global phase even though it's nominally a "pre-other-bundles"
# criterion.
criteria_order_pre_other_bundles = criteria_order_chunk_local + ["recobundles"]

criteria_order_post_other_bundles = ["orient_mahal", "isolation_forest", "qb_thresh"]


valid_noncriterion = [
    "space",
    "mahal",
    "inc_addtol",
    "exc_addtol",
    "exact_endpoints",
    "ORG_spectral_subbundles",
    "cluster_IDs",
    "startpoint_location",
    "endpoint_location",
    "primary_axis_core_only",
]


logger = logging.getLogger("AFQ")


def prob_map(b_sls, bundle_def, preproc_plan, prob_threshold, img, **kwargs):
    b_sls.initiate_selection("Prob. Map")
    fiber_probabilities = dts.values_from_volume(
        bundle_def["prob_map"].get_fdata(),
        preproc_plan.fgarray[b_sls.selected_fiber_idxs],
        img.affine,
    )
    fiber_probabilities = np.mean(fiber_probabilities, -1)
    b_sls.select(fiber_probabilities > prob_threshold, "Prob. Map")


def cross_midline(b_sls, bundle_def, preproc_plan, **kwargs):
    b_sls.initiate_selection("Cross Mid.")
    accepted = preproc_plan.crosses[b_sls.selected_fiber_idxs]
    if not bundle_def["cross_midline"]:
        accepted = np.invert(accepted)
    b_sls.select(accepted, "Cross Mid.")


def start(b_sls, bundle_def, preproc_plan, **kwargs):
    b_sls.initiate_selection("Startpoint")
    exact_endpoints = bundle_def.get("exact_endpoints", False)
    if exact_endpoints:
        tol = 0
    else:
        tol = kwargs["dist_to_atlas"]

    accept_idx = abr.clean_by_endpoints(
        preproc_plan.fgarray[b_sls.selected_fiber_idxs],
        bundle_def["start"],
        0,
        tol=tol,
        flip_sls=b_sls.sls_flipped,
    )
    if not b_sls.oriented_yet:
        accepted_idx_flipped = abr.clean_by_endpoints(
            preproc_plan.fgarray[b_sls.selected_fiber_idxs],
            bundle_def["start"],
            -1,
            tol=tol,
        )
        new_accept_idx = np.logical_or(accepted_idx_flipped, accept_idx)
        special_idx = np.logical_and(accept_idx, accepted_idx_flipped)
        special_idx_to_flip = abu.manual_orient_sls(
            preproc_plan.fgarray[b_sls.selected_fiber_idxs][special_idx]
        )
        accepted_idx_flipped[special_idx] = special_idx_to_flip
        b_sls.reorient(accepted_idx_flipped)
        accept_idx = new_accept_idx

    b_sls.select(accept_idx, "Startpoint")


def end(b_sls, bundle_def, preproc_plan, **kwargs):
    b_sls.initiate_selection("endpoint")
    exact_endpoints = bundle_def.get("exact_endpoints", False)
    if exact_endpoints:
        tol = 0
    else:
        tol = kwargs["dist_to_atlas"]

    accept_idx = abr.clean_by_endpoints(
        preproc_plan.fgarray[b_sls.selected_fiber_idxs],
        bundle_def["end"],
        -1,
        tol=tol,
        flip_sls=b_sls.sls_flipped,
    )
    if not b_sls.oriented_yet:
        accepted_idx_flipped = abr.clean_by_endpoints(
            preproc_plan.fgarray[b_sls.selected_fiber_idxs],
            bundle_def["end"],
            0,
            tol=tol,
        )
        new_accept_idx = np.logical_or(accepted_idx_flipped, accept_idx)
        special_idx = np.logical_and(accept_idx, accepted_idx_flipped)
        special_idx_to_flip = abu.manual_orient_sls(
            preproc_plan.fgarray[b_sls.selected_fiber_idxs][special_idx]
        )
        accepted_idx_flipped[special_idx] = special_idx_to_flip
        b_sls.reorient(accepted_idx_flipped)
        accept_idx = new_accept_idx
    b_sls.select(accept_idx, "endpoint")


def length(b_sls, bundle_def, preproc_plan, **kwargs):
    b_sls.initiate_selection("length")
    min_len = bundle_def["length"].get("min_len", 0)
    max_len = bundle_def["length"].get("max_len", np.inf)

    sl_lens = preproc_plan.lengths

    accept_idx = (sl_lens >= min_len) & (sl_lens <= max_len)
    b_sls.select(accept_idx, "length")


def endpoint_dists(b_sls, bundle_def, preproc_plan, **kwargs):
    b_sls.initiate_selection("endpoint_dists")
    min_dist = bundle_def["endpoint_dists"].get("min_dist", 0)
    max_dist = bundle_def["endpoint_dists"].get("max_dist", np.inf)

    sl_endpoint_dists = preproc_plan.endpoint_dists[b_sls.selected_fiber_idxs]

    accept_idx = (sl_endpoint_dists >= min_dist) & (sl_endpoint_dists <= max_dist)
    b_sls.select(accept_idx, "endpoint_dists")


def primary_axis(b_sls, bundle_def, **kwargs):
    b_sls.initiate_selection("orientation")
    accept_idx = abc.clean_by_orientation(
        b_sls.get_selected_sls(),
        bundle_def["primary_axis"],
        bundle_def.get("primary_axis_core_only", 0.6),
    )
    b_sls.select(accept_idx, "orientation")


def include(b_sls, bundle_def, **kwargs):
    accept_idx = b_sls.initiate_selection("include")
    flip_using_include = len(bundle_def["include"]) > 1 and not b_sls.oriented_yet

    if "inc_addtol" in bundle_def:
        include_roi_tols = []
        for inc_tol in bundle_def["inc_addtol"]:
            include_roi_tols.append((inc_tol / kwargs["vox_dim"] + kwargs["tol"]) ** 2)
    else:
        include_roi_tols = [kwargs["tol"] ** 2] * len(bundle_def["include"])

    inc_results = abr.check_sls_with_inclusion(
        b_sls.get_selected_sls(), bundle_def["include"], include_roi_tols
    )

    n_inc = len(bundle_def["include"])
    roi_closest = np.zeros((n_inc, len(b_sls)), dtype=np.int32)
    roi_dists = np.zeros((n_inc, len(b_sls)), dtype=np.float32)
    if flip_using_include:
        to_flip = np.ones_like(accept_idx, dtype=np.bool_)
    for sl_idx, inc_result in enumerate(inc_results):
        sl_accepted, sl_closest, sl_dists = inc_result

        if sl_accepted:
            roi_closest[:, sl_idx] = sl_closest
            roi_dists[:, sl_idx] = sl_dists
            if len(sl_closest) > 1:
                if (len(sl_closest) < 2) or abs(sl_closest[0] - sl_closest[-1]) > 1:
                    if flip_using_include:
                        to_flip[sl_idx] = sl_closest[0] > sl_closest[-1]
                        if to_flip[sl_idx]:
                            roi_closest[:, sl_idx] = np.flip(sl_closest)
                            roi_dists[:, sl_idx] = np.flip(sl_dists)
                    accept_idx[sl_idx] = 1
            else:
                accept_idx[sl_idx] = 1

    b_sls.roi_closest = roi_closest.T
    b_sls.roi_dists = roi_dists.T
    if flip_using_include:
        b_sls.reorient(to_flip)
    b_sls.select(accept_idx, "include")


def curvature(b_sls, bundle_def, mapping, img, save_intermediates, **kwargs):
    accept_idx = b_sls.initiate_selection("curvature")
    if "sft" in bundle_def["curvature"]:
        ref_sl = bundle_def["curvature"]["sft"]
    else:
        ref_sl = load_tractogram(
            bundle_def["curvature"]["path"], "same", bbox_valid_check=False
        )
    moved_ref_sl = move_streamlines(
        ref_sl, "subject", mapping, img, save_intermediates=save_intermediates
    )
    moved_ref_sl = moved_ref_sl.streamlines[0]
    moved_ref_curve = abv.sl_curve(moved_ref_sl, len(moved_ref_sl))
    ref_curve_threshold = np.radians(bundle_def["curvature"].get("thresh", 10))
    cut = bundle_def["curvature"].get("cut", True)
    for idx, sl in enumerate(b_sls.get_selected_sls(cut=cut, flip=True)):
        if len(sl) > 1:
            this_sl_curve = abv.sl_curve(sl, len(moved_ref_sl))
            dist = abv.sl_curve_dist(this_sl_curve, moved_ref_curve)
            if dist <= ref_curve_threshold:
                accept_idx[idx] = 1
    b_sls.select(accept_idx, "curvature", cut=cut)


def exclude(b_sls, bundle_def, **kwargs):
    accept_idx = b_sls.initiate_selection("exclude")
    if "exc_addtol" in bundle_def:
        exclude_roi_tols = []
        for exc_tol in bundle_def["exc_addtol"]:
            exclude_roi_tols.append((exc_tol / kwargs["vox_dim"] + kwargs["tol"]) ** 2)
    else:
        exclude_roi_tols = [kwargs["tol"] ** 2] * len(bundle_def["exclude"])
    for sl_idx, sl in enumerate(b_sls.get_selected_sls()):
        if abr.check_sl_with_exclusion(sl, bundle_def["exclude"], exclude_roi_tols):
            accept_idx[sl_idx] = 1
    b_sls.select(accept_idx, "exclude")


def recobundles(
    b_sls,
    mapping,
    bundle_def,
    reg_template,
    img,
    refine_reco,
    save_intermediates,
    rng,
    rb_recognize_params,
    **kwargs,
):
    b_sls.initiate_selection("Recobundles")
    moved_sl = move_streamlines(
        StatefulTractogram(b_sls.get_selected_sls(), img, Space.RASMM),
        "template",
        mapping,
        reg_template,
        to_space=Space.RASMM,
        save_intermediates=save_intermediates,
    ).streamlines
    moved_sl_resampled = abu.resample_tg(moved_sl, 100)
    rb = RecoBundles(moved_sl, verbose=True, rng=rng)
    _, rec_labels = rb.recognize(bundle_def["recobundles"]["sl"], **rb_recognize_params)
    if refine_reco:
        _, rec_labels = rb.refine(
            bundle_def["recobundles"]["sl"],
            moved_sl_resampled[rec_labels],
            **rb_recognize_params,
        )
    if not b_sls.oriented_yet and np.sum(rec_labels) > 0:
        standard_sl = next(iter(bundle_def["recobundles"]["centroid"]))
        oriented_idx = abu.orient_by_streamline(moved_sl[rec_labels], standard_sl)
        b_sls.reorient(rec_labels[oriented_idx])
    rec_labels = sorted(rec_labels)
    b_sls.select(rec_labels, "Recobundles")


def qb_thresh(b_sls, bundle_def, clip_edges, **kwargs):
    b_sls.initiate_selection("qb_thresh")
    cut = clip_edges or ("bundlesection" in bundle_def)
    qbx = QuickBundles(
        bundle_def["qb_thresh"],
        AveragePointwiseEuclideanMetric(ResampleFeature(nb_points=12)),
    )
    clusters = qbx.cluster(b_sls.get_selected_sls(cut=cut, flip=True))
    cleaned_idx = clusters[np.argmax(clusters.clusters_sizes())].indices
    b_sls.select(cleaned_idx, "qb_thresh", cut=cut)


def clean_by_other_bundle(
    b_sls, bundle_def, img, other_bundle_name, other_bundle_sls, **kwargs
):
    cleaned_idx = b_sls.initiate_selection(other_bundle_name)
    cleaned_idx = 1
    flipped_sls = b_sls.get_selected_sls(flip=True)

    if "overlap" in bundle_def[other_bundle_name]:
        cleaned_idx_overlap = abo.clean_by_overlap(
            flipped_sls,
            other_bundle_sls,
            bundle_def[other_bundle_name]["overlap"],
            img,
            remove=False,
            project=bundle_def[other_bundle_name].get("project", None),
        )
        cleaned_idx = np.logical_and(cleaned_idx, cleaned_idx_overlap)

    if "node_thresh" in bundle_def[other_bundle_name]:
        cleaned_idx_node_thresh = abo.clean_by_overlap(
            flipped_sls,
            other_bundle_sls,
            bundle_def[other_bundle_name]["node_thresh"],
            img,
            remove=True,
            project=bundle_def[other_bundle_name].get("project", None),
        )
        cleaned_idx = np.logical_and(cleaned_idx, cleaned_idx_node_thresh)

    if "core" in bundle_def[other_bundle_name]:
        consideration = bundle_def[other_bundle_name].get("consideration", 10.0)
        if isinstance(consideration, (int, float)):
            consideration = float(consideration)
            consideration = consideration / kwargs["vox_dim"]

        cleaned_idx_core = abo.clean_relative_to_other_core(
            bundle_def[other_bundle_name]["core"].lower(),
            np.array(abu.resample_tg(flipped_sls, 100)),
            np.array(abu.resample_tg(other_bundle_sls, 100)),
            consideration=consideration,
        )
        cleaned_idx = np.logical_and(cleaned_idx, cleaned_idx_core)

    b_sls.select(cleaned_idx, other_bundle_name)


def orient_mahal(b_sls, bundle_def, **kwargs):
    b_sls.initiate_selection("orient_mahal")
    accept_idx = abc.clean_by_orientation_mahalanobis(
        b_sls.get_selected_sls(), **bundle_def.get("orient_mahal", {})
    )
    b_sls.select(accept_idx, "orient_mahal")


def isolation_forest(b_sls, bundle_def, rng, **kwargs):
    b_sls.initiate_selection("isolation_forest")
    accept_idx = abc.clean_by_isolation_forest(
        b_sls.get_selected_sls(),
        distance_threshold=bundle_def["isolation_forest"].get("distance_threshold", 3),
        n_rounds=bundle_def["isolation_forest"].get("n_rounds", 5),
        random_state=rng,
    )
    b_sls.select(accept_idx, "isolation_forest")


def mahalanobis(b_sls, bundle_def, clip_edges, cleaning_params, **kwargs):
    b_sls.initiate_selection("Mahalanobis")
    clean_params = bundle_def.get("mahal", {})
    clean_params = {**cleaning_params, **clean_params}
    clean_params["return_idx"] = True
    cut = clip_edges or ("bundlesection" in bundle_def)
    _, cleaned_idx = abc.clean_bundle(
        b_sls.get_selected_sls(cut=cut, flip=True), **clean_params
    )
    b_sls.select(cleaned_idx, "Mahalanobis", cut=cut)


def _prepare_bundle_def(bundle_dict, bundle_name, mapping, img):
    """
    Warp ROIs and apply distance-transform conversion
    """
    tqdm.write(f"Preparing ROIs for {bundle_name}")
    start_time = time()
    bundle_def = dict(bundle_dict.get_b_info(bundle_name))
    bundle_def.update(
        bundle_dict.transform_rois(bundle_name, mapping, img, apply_to_recobundles=True)
    )

    def check_space(roi):
        if not np.allclose(img.affine, roi.affine):
            logger.warning(
                "Resampling set to False in case where affines "
                "do not match. This is likely due to subject space ROIs"
                " not being in the right space. This found for bundle "
                f"{bundle_name}"
            )

    apply_to_roi_dict(bundle_def, check_space, dry_run=True, apply_to_prob_map=True)

    apply_to_roi_dict(
        bundle_def,
        lambda roi_img: nib.Nifti1Image(
            distance_transform_edt(np.where(roi_img.get_fdata() == 0, 1, 0)),
            roi_img.affine,
        ),
        dry_run=False,
        apply_to_recobundles=False,
        apply_to_prob_map=False,
    )
    tqdm.write(f"Time to prep ROIs: {time() - start_time}s")
    return bundle_def


def _validate_criteria(bundle_def, bundle_name, bundle_dict, recognized_bundles_dict):
    for potential_criterion in bundle_def.keys():
        if (
            (potential_criterion not in criteria_order_post_other_bundles)
            and (potential_criterion not in criteria_order_pre_other_bundles)
            and (potential_criterion not in recognized_bundles_dict.keys())
            and (potential_criterion not in valid_noncriterion)
        ):
            if potential_criterion in bundle_dict.bundle_names:
                raise ValueError(
                    f"Bundle {potential_criterion} is being used as a criterion in "
                    f"the definition of bundle {bundle_name}, however this bundle "
                    "was not found. This could because of insufficient streamlines"
                )
            else:
                raise ValueError(
                    "Invalid criterion in bundle definition:\n"
                    f"{potential_criterion} in bundle {bundle_name}.\n"
                    "Valid criteria are:\n"
                    f"{criteria_order_pre_other_bundles}\n"
                    f"{criteria_order_post_other_bundles}\n"
                    f"{recognized_bundles_dict.keys()}\n"
                    f"{valid_noncriterion}\n"
                )


def _run_chunk_local(
    bundle_def,
    chunk_streamlines,
    bundle_name,
    img,
    preproc_plan,
    save_intermediates,
    vox_dim,
    tol,
    dist_to_atlas,
    **segmentation_params,
):
    b_sls = abu.SlsBeingRecognized(
        chunk_streamlines,
        save_intermediates,
        bundle_name,
        img,
        len(bundle_def.get("include", [])),
    )

    inputs = {
        "b_sls": b_sls,
        "preproc_plan": preproc_plan,
        "bundle_def": bundle_def,
        "img": img,
        "save_intermediates": save_intermediates,
        "vox_dim": vox_dim,
        "tol": tol,
        "dist_to_atlas": dist_to_atlas,
    }
    inputs.update(segmentation_params)

    for criterion in criteria_order_chunk_local:
        if b_sls and criterion in bundle_def:
            inputs[criterion] = globals()[criterion](**inputs)

    return b_sls


def _run_global_phase(
    bundle_def,
    bundle_name,
    b_sls,
    fgarray_for_candidates,
    candidate_global_idx,
    mapping,
    img,
    reg_template,
    preproc_scalars,
    recognized_bundles_dict,
    is_subbundle=False,
    **segmentation_params,
):
    if not b_sls:
        return

    inputs = {
        "b_sls": b_sls,
        "preproc_plan": preproc_scalars,
        "bundle_def": bundle_def,
        "mapping": mapping,
        "img": img,
        "reg_template": reg_template,
    }
    inputs.update(segmentation_params)

    if "recobundles" in bundle_def:
        recobundles(**inputs)

    if b_sls:
        for o_bundle_name in recognized_bundles_dict.keys():
            if o_bundle_name in bundle_def.keys():
                clean_by_other_bundle(
                    **inputs,
                    other_bundle_name=o_bundle_name,
                    other_bundle_sls=recognized_bundles_dict[
                        o_bundle_name
                    ].get_selected_sls(flip=True),
                )

    for criterion in criteria_order_post_other_bundles:
        if b_sls and criterion in bundle_def:
            inputs[criterion] = globals()[criterion](**inputs)

    if b_sls:
        if "mahal" in bundle_def or (
            "isolation_forest" not in bundle_def
            and "orient_mahal" not in bundle_def
            and "ORG_spectral_subbundles" not in bundle_def
        ):
            mahalanobis(**inputs)

    # Wrong-side-of-midline cleanup. fgarray_for_candidates is in
    # candidate-local order; b_sls.selected_fiber_idxs is in global
    # order. searchsorted translates between them.
    if (
        b_sls
        and not is_subbundle
        and "cross_midline" in bundle_def
        and not bundle_def["cross_midline"]
        and fgarray_for_candidates is not None
        and candidate_global_idx is not None
    ):
        pos = np.searchsorted(candidate_global_idx, b_sls.selected_fiber_idxs)
        b_sls.initiate_selection("Wrong side of mid.")
        avg_side = np.sign(np.mean(fgarray_for_candidates[pos, :, 0], axis=1))
        majority_side = np.sign(np.sum(avg_side))
        b_sls.select(avg_side == majority_side, "Wrong side of mid.")

    if b_sls and not b_sls.oriented_yet:
        raise ValueError(
            "pyAFQ was unable to consistently orient streamlines "
            f"in bundle {bundle_name} using the provided ROIs. "
            "This can be fixed by including at least 2 "
            "waypoint ROIs, or by using endpoint ROIs."
        )

    if not b_sls:
        return

    if "ORG_spectral_subbundles" in bundle_def:
        if is_subbundle:
            raise ValueError("Nested ORG_spectral_subbundles are not supported.")
        subdict = bundle_def["ORG_spectral_subbundles"]
        b_sls.initiate_selection(
            f"ORG spectral clustering, {len(subdict.bundle_names)} "
            "subbundles being recognized"
        )

        sub_sft = StatefulTractogram(
            b_sls.get_selected_sls(flip=True), img, Space.RASMM
        )
        cluster_labels = subcluster_by_atlas(
            sub_sft, mapping, img, subdict.all_cluster_IDs, n_points=40
        )

        for sub_b_name in subdict.bundle_names:
            c_ids = subdict._dict[sub_b_name]["cluster_IDs"]
            n_roi = len(subdict._dict[sub_b_name].get("include", []))
            cluster_b_sls = b_sls.copy(sub_b_name, n_roi)
            selected = np.zeros(len(b_sls), dtype=bool)
            for c_id in c_ids:
                selected = np.logical_or(selected, cluster_labels == c_id)
            cluster_b_sls.select(selected, f"Clusters {c_ids}")

            sub_bundle_def = _prepare_bundle_def(subdict, sub_b_name, mapping, img)
            _validate_criteria(
                sub_bundle_def, sub_b_name, subdict, recognized_bundles_dict
            )
            _run_global_phase(
                sub_bundle_def,
                sub_b_name,
                cluster_b_sls,
                None,
                None,
                mapping,
                img,
                reg_template,
                preproc_scalars,
                recognized_bundles_dict,
                is_subbundle=True,
                **segmentation_params,
            )
    else:
        b_sls.bundle_def = bundle_def
        recognized_bundles_dict[bundle_name] = b_sls


def recognize_bundles(
    tg,
    bundle_dict,
    mapping,
    img,
    reg_template,
    chunk_size,
    dist_to_waypoint,
    dist_to_atlas,
    save_intermediates,
    **segmentation_params,
):
    if isinstance(tg, str):
        tg_path = tg
        tg = load_trx(tg_path, img)
    else:
        tg_path = None

    n_streamlines = len(tg)
    recognized_bundles_dict = {}

    tqdm.write(
        f"Recognizing bundles over {n_streamlines} streamlines "
        f"in chunks of {chunk_size}"
    )

    tol, dist_to_atlas, vox_dim = abu.tolerance_mm_to_vox(
        img, dist_to_waypoint, dist_to_atlas
    )
    preproc_scalars = {
        "vox_dim": vox_dim,
        "tol": tol,
        "dist_to_atlas": dist_to_atlas,
    }

    bundle_defs = {}
    survivor_dicts = {}
    for bundle_name in bundle_dict.bundle_names:
        bd = _prepare_bundle_def(bundle_dict, bundle_name, mapping, img)
        bundle_defs[bundle_name] = bd
        survivor_dicts[bundle_name] = []

    total_chunks = (n_streamlines + chunk_size - 1) // chunk_size
    for chunk_start in tqdm(
        range(0, n_streamlines, chunk_size),
        total=total_chunks,
        desc="Batched Portion of Recognition",
    ):
        chunk_end = min(chunk_start + chunk_size, n_streamlines)
        tqdm.write(
            f"Processing chunk {chunk_start}:{chunk_end} of {n_streamlines} "
            f"({(chunk_end / n_streamlines) * 100:.2f}%)"
        )

        if tg_path is not None:
            tg = load_trx(tg_path, img)
        chunk_streamlines = tg.streamlines[chunk_start:chunk_end].copy()
        if tg_path is not None:
            tg.close()
            del tg

        chunk_preproc = PreprocPlan(chunk_streamlines)

        for bundle_name in bundle_dict.bundle_names:
            tqdm.write(f"Running chunk-local phase for bundle {bundle_name}")
            chunk_b_sls = _run_chunk_local(
                bundle_defs[bundle_name],
                chunk_streamlines,
                bundle_name,
                img,
                chunk_preproc,
                save_intermediates,
                mapping=mapping,
                reg_template=reg_template,
                vox_dim=vox_dim,
                tol=tol,
                dist_to_atlas=dist_to_atlas,
                **segmentation_params,
            )
            survivor_dicts[bundle_name].append(chunk_b_sls.export_selected(chunk_start))
            del chunk_b_sls
        del chunk_preproc, chunk_streamlines

    if tg_path is not None:
        tg = load_trx(tg_path, img)

    for bundle_name in bundle_dict.bundle_names:
        tqdm.write(f"Running global phase for bundle {bundle_name}")
        bundle_def = bundle_defs[bundle_name]

        merged = abu.SlsBeingRecognized.from_selected(
            survivor_dicts[bundle_name],
            tg.streamlines,
            save_intermediates,
            bundle_name,
            img,
            len(bundle_def.get("include", [])),
        )
        survivor_dicts[bundle_name] = None  # free per-chunk dicts

        if merged is None:
            tqdm.write(
                f"Bundle {bundle_name}: 0 candidates after chunk-local filtering"
            )
            continue

        _validate_criteria(
            bundle_def, bundle_name, bundle_dict, recognized_bundles_dict
        )

        tqdm.write(
            f"Bundle {bundle_name}: {len(merged)} candidates after "
            "chunk-local filtering"
        )

        need_fgarray = "cross_midline" in bundle_def and not bundle_def["cross_midline"]
        if need_fgarray:
            candidate_global_idx = np.array(merged.selected_fiber_idxs, dtype=np.int64)
            cand_streamlines = [tg.streamlines[int(i)] for i in candidate_global_idx]
            fgarray_for_candidates = np.asarray(
                resample_tg(cand_streamlines, 20), dtype=np.float32
            )
            del cand_streamlines
        else:
            candidate_global_idx = None
            fgarray_for_candidates = None

        _run_global_phase(
            bundle_def,
            bundle_name,
            merged,
            fgarray_for_candidates,
            candidate_global_idx,
            mapping,
            img,
            reg_template,
            preproc_scalars,
            recognized_bundles_dict,
            save_intermediates=save_intermediates,
            **segmentation_params,
        )

    return recognized_bundles_dict

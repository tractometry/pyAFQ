import logging
from time import time

import numpy as np
import nibabel as nib
import ray

from scipy.ndimage import distance_transform_edt

import dipy.tracking.streamline as dts
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.segment.featurespeed import ResampleFeature
from dipy.io.streamline import load_tractogram
from dipy.segment.bundles import RecoBundles
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from AFQ.api.bundle_dict import apply_to_roi_dict
import AFQ.recognition.utils as abu
import AFQ.recognition.cleaning as abc
import AFQ.recognition.curvature as abv
import AFQ.recognition.roi as abr
import AFQ.recognition.other_bundles as abo
from AFQ.utils.stats import chunk_indices


criteria_order_pre_other_bundles = [
    "prob_map", "cross_midline", "start", "end",
    "length", "primary_axis", "include", "exclude",
    "curvature", "recobundles"]


criteria_order_post_other_bundles = [
    "orient_mahal", "isolation_forest", "qb_thresh"]


valid_noncriterion = [
    "space", "mahal", "primary_axis_percentage",
    "inc_addtol", "exc_addtol"]


logger = logging.getLogger('AFQ')


def prob_map(b_sls, bundle_def, preproc_imap, prob_threshold, **kwargs):
    b_sls.initiate_selection("Prob. Map")
    # using entire fgarray here only because it is the first step
    fiber_probabilities = dts.values_from_volume(
        bundle_def["prob_map"].get_fdata(),
        preproc_imap["fgarray"], np.eye(4))
    fiber_probabilities = np.mean(fiber_probabilities, -1)
    b_sls.select(
        fiber_probabilities > prob_threshold,
        "Prob. Map")


def cross_midline(b_sls, bundle_def, preproc_imap, **kwargs):
    b_sls.initiate_selection("Cross Mid.")
    accepted = preproc_imap["crosses"][b_sls.selected_fiber_idxs]
    if not bundle_def["cross_midline"]:
        accepted = np.invert(accepted)
    b_sls.select(accepted, "Cross Mid.")


def start(b_sls, bundle_def, preproc_imap, **kwargs):
    accept_idx = b_sls.initiate_selection("Startpoint")
    abr.clean_by_endpoints(
        b_sls.get_selected_sls(),
        bundle_def["start"],
        0,
        tol=preproc_imap["dist_to_atlas"],
        flip_sls=b_sls.sls_flipped,
        accepted_idxs=accept_idx)
    if not b_sls.oriented_yet:
        accepted_idx_flipped = abr.clean_by_endpoints(
            b_sls.get_selected_sls(),
            bundle_def["start"],
            -1,
            tol=preproc_imap["dist_to_atlas"])
        b_sls.reorient(accepted_idx_flipped)
        accept_idx = np.logical_xor(
            accepted_idx_flipped, accept_idx)
    b_sls.select(accept_idx, "Startpoint")


def end(b_sls, bundle_def, preproc_imap, **kwargs):
    accept_idx = b_sls.initiate_selection("endpoint")
    abr.clean_by_endpoints(
        b_sls.get_selected_sls(),
        bundle_def["end"],
        -1,
        tol=preproc_imap["dist_to_atlas"],
        flip_sls=b_sls.sls_flipped,
        accepted_idxs=accept_idx)
    if not b_sls.oriented_yet:
        accepted_idx_flipped = abr.clean_by_endpoints(
            b_sls.get_selected_sls(),
            bundle_def["end"],
            0,
            tol=preproc_imap["dist_to_atlas"])
        b_sls.reorient(accepted_idx_flipped)
        accept_idx = np.logical_xor(
            accepted_idx_flipped, accept_idx)
    b_sls.select(accept_idx, "endpoint")


def length(b_sls, bundle_def, preproc_imap, **kwargs):
    accept_idx = b_sls.initiate_selection("length")
    min_len = bundle_def["length"].get(
        "min_len", 0) / preproc_imap["vox_dim"]
    max_len = bundle_def["length"].get(
        "max_len", np.inf) / preproc_imap["vox_dim"]
    for idx, sl in enumerate(b_sls.get_selected_sls()):
        sl_len = np.sum(
            np.linalg.norm(np.diff(sl, axis=0), axis=1))
        if sl_len >= min_len and sl_len <= max_len:
            accept_idx[idx] = 1
    b_sls.select(accept_idx, "length")


def primary_axis(b_sls, bundle_def, img, **kwargs):
    b_sls.initiate_selection("orientation")
    accept_idx = abc.clean_by_orientation(
        b_sls.get_selected_sls(),
        bundle_def["primary_axis"],
        img.affine,
        bundle_def.get(
            "primary_axis_percentage", None))
    b_sls.select(accept_idx, "orientation")


def include(b_sls, bundle_def, preproc_imap, max_includes,
            n_cpus, **kwargs):
    accept_idx = b_sls.initiate_selection("include")
    flip_using_include = len(bundle_def["include"]) > 1\
        and not b_sls.oriented_yet

    if f'inc_addtol' in bundle_def:
        include_roi_tols = []
        for inc_tol in bundle_def["inc_addtol"]:
            include_roi_tols.append((
                inc_tol / preproc_imap["vox_dim"] + preproc_imap["tol"])**2)
    else:
        include_roi_tols = [preproc_imap["tol"]**2] * len(
            bundle_def["include"])

    # with parallel segmentation, the first for loop will
    # only collect streamlines and does not need tqdm
    if n_cpus > 1:
        inc_results = np.zeros(len(b_sls), dtype=tuple)

        inc_rois_id = ray.put(bundle_def["include"])
        inc_roi_tols_id = ray.put(include_roi_tols)

        _check_inc_parallel = ray.remote(
            num_cpus=n_cpus)(abr.check_sls_with_inclusion)

        sls_chunks = list(chunk_indices(np.arange(len(b_sls)), n_cpus))
        futures = [
            _check_inc_parallel.remote(
                b_sls.get_selected_sls()[sls_chunk],
                inc_rois_id,
                inc_roi_tols_id)
            for sls_chunk in sls_chunks
        ]

        for ii, future in enumerate(futures):
            inc_results[sls_chunks[ii]] = ray.get(future)
    else:
        inc_results = abr.check_sls_with_inclusion(
            b_sls.get_selected_sls(),
            bundle_def["include"],
            include_roi_tols)

    roi_closest = -np.ones(
        (max_includes, len(b_sls)),
        dtype=np.int32)
    if flip_using_include:
        to_flip = np.ones_like(accept_idx, dtype=np.bool_)
    for sl_idx, inc_result in enumerate(inc_results):
        sl_accepted, sl_closest = inc_result

        if sl_accepted:
            if len(sl_closest) > 1:
                roi_closest[:len(sl_closest), sl_idx] = sl_closest
                # Only accept SLs that, when cut, are meaningful
                if (len(sl_closest) < 2) or abs(
                        sl_closest[0] - sl_closest[-1]) > 1:
                    # Flip sl if it is close to second ROI
                    # before its close to the first ROI
                    if flip_using_include:
                        to_flip[sl_idx] =\
                            sl_closest[0] > sl_closest[-1]
                        if to_flip[sl_idx]:
                            roi_closest[:len(sl_closest), sl_idx] =\
                                np.flip(sl_closest)
                    accept_idx[sl_idx] = 1
            else:
                accept_idx[sl_idx] = 1

    b_sls.roi_closest = roi_closest.T
    if flip_using_include:
        b_sls.reorient(to_flip)
    b_sls.select(accept_idx, "include")


def curvature(b_sls, bundle_def, mapping, img, save_intermediates, **kwargs):
    '''
    Filters streamlines by how well they match
    a curve in orientation and shape but not scale
    '''
    accept_idx = b_sls.initiate_selection("curvature")
    if "sft" in bundle_def["curvature"]:
        ref_sl = bundle_def["curvature"]["sft"]
    else:
        ref_sl = load_tractogram(
            bundle_def["curvature"]["path"], "same",
            bbox_valid_check=False)
    moved_ref_sl = abu.move_streamlines(
        ref_sl, "subject", mapping, img,
        save_intermediates=save_intermediates)
    moved_ref_sl.to_vox()
    moved_ref_sl = moved_ref_sl.streamlines[0]
    moved_ref_curve = abv.sl_curve(
        moved_ref_sl,
        len(moved_ref_sl))
    ref_curve_threshold = np.radians(bundle_def["curvature"].get(
        "thresh", 10))
    cut = bundle_def["curvature"].get("cut", True)
    for idx, sl in enumerate(b_sls.get_selected_sls(
            cut=cut, flip=True)):
        if len(sl) > 1:
            this_sl_curve = abv.sl_curve(sl, len(moved_ref_sl))
            dist = abv.sl_curve_dist(this_sl_curve, moved_ref_curve)
            if dist <= ref_curve_threshold:
                accept_idx[idx] = 1
    b_sls.select(accept_idx, "curvature", cut=cut)


def exclude(b_sls, bundle_def, preproc_imap, **kwargs):
    accept_idx = b_sls.initiate_selection("exclude")
    if f'exc_addtol' in bundle_def:
        exclude_roi_tols = []
        for exc_tol in bundle_def["exc_addtol"]:
            exclude_roi_tols.append((
                exc_tol / preproc_imap["vox_dim"] + preproc_imap["tol"])**2)
    else:
        exclude_roi_tols = [
            preproc_imap["tol"]**2] * len(bundle_def["exclude"])
    for sl_idx, sl in enumerate(b_sls.get_selected_sls()):
        if abr.check_sl_with_exclusion(
                sl, bundle_def["exclude"], exclude_roi_tols):
            accept_idx[sl_idx] = 1
    b_sls.select(accept_idx, "exclude")


def recobundles(b_sls, mapping, bundle_def, reg_template, img, refine_reco,
                save_intermediates, rng, rb_recognize_params, **kwargs):
    b_sls.initiate_selection("Recobundles")
    moved_sl = abu.move_streamlines(
        StatefulTractogram(b_sls.get_selected_sls(), img, Space.VOX),
        "template", mapping, reg_template,
        save_intermediates=save_intermediates).streamlines
    moved_sl_resampled = abu.resample_tg(moved_sl, 100)
    rb = RecoBundles(moved_sl, verbose=True, rng=rng)
    _, rec_labels = rb.recognize(
        bundle_def['recobundles']['sl'],
        **rb_recognize_params)
    if refine_reco:
        _, rec_labels = rb.refine(
            bundle_def['recobundles']['sl'], moved_sl_resampled[rec_labels],
            **rb_recognize_params)
    if not b_sls.oriented_yet and np.sum(rec_labels) > 0:
        standard_sl = next(iter(bundle_def['recobundles']['centroid']))
        oriented_idx = abu.orient_by_streamline(
            moved_sl[rec_labels],
            standard_sl)
        b_sls.reorient(rec_labels[oriented_idx])
    b_sls.select(rec_labels, "Recobundles")


def qb_thresh(b_sls, bundle_def, preproc_imap, clip_edges, **kwargs):
    b_sls.initiate_selection("qb_thresh")
    cut = clip_edges or ("bundlesection" in bundle_def)
    qbx = QuickBundles(
        bundle_def["qb_thresh"] / preproc_imap["vox_dim"],
        AveragePointwiseEuclideanMetric(
            ResampleFeature(nb_points=12)))
    clusters = qbx.cluster(b_sls.get_selected_sls(
        cut=cut, flip=True))
    cleaned_idx = clusters[np.argmax(
        clusters.clusters_sizes())].indices
    b_sls.select(cleaned_idx, "qb_thresh", cut=cut)


def clean_by_other_bundle(b_sls, bundle_def,
                          img,
                          preproc_imap,
                          other_bundle_name,
                          other_bundle_sls, **kwargs):
    cleaned_idx = b_sls.initiate_selection(other_bundle_name)
    cleaned_idx = 1

    if 'overlap' in bundle_def[other_bundle_name]:
        cleaned_idx_overlap = abo.clean_by_overlap(
            b_sls.get_selected_sls(),
            other_bundle_sls,
            bundle_def[other_bundle_name]["overlap"],
            img, False)
        cleaned_idx = np.logical_and(cleaned_idx, cleaned_idx_overlap)

    if 'node_thresh' in bundle_def[other_bundle_name]:
        cleaned_idx_node_thresh = abo.clean_by_overlap(
            b_sls.get_selected_sls(),
            other_bundle_sls,
            bundle_def[other_bundle_name]["node_thresh"],
            img, True)
        cleaned_idx = np.logical_and(cleaned_idx, cleaned_idx_node_thresh)

    if 'core' in bundle_def[other_bundle_name]:
        cleaned_idx_core = abo.clean_relative_to_other_core(
            bundle_def[other_bundle_name]['core'].lower(),
            preproc_imap["fgarray"][b_sls.selected_fiber_idxs],
            np.array(abu.resample_tg(other_bundle_sls, 20)),
            img.affine, False)
        cleaned_idx = np.logical_and(cleaned_idx, cleaned_idx_core)

    if 'entire_core' in bundle_def[other_bundle_name]:
        cleaned_idx_core = abo.clean_relative_to_other_core(
            bundle_def[other_bundle_name]['entire_core'].lower(),
            preproc_imap["fgarray"][b_sls.selected_fiber_idxs],
            np.array(abu.resample_tg(other_bundle_sls, 20)),
            img.affine, True)
        cleaned_idx = np.logical_and(cleaned_idx, cleaned_idx_core)

    b_sls.select(cleaned_idx, other_bundle_name)


def orient_mahal(b_sls, bundle_def, **kwargs):
    b_sls.initiate_selection("orient_mahal")
    accept_idx = abc.clean_by_orientation_mahalanobis(
        b_sls.get_selected_sls(),
        **bundle_def.get("orient_mahal", {}))
    b_sls.select(accept_idx, "orient_mahal")


def isolation_forest(b_sls, bundle_def, n_cpus, rng, **kwargs):
    b_sls.initiate_selection("isolation_forest")
    accept_idx = abc.clean_by_isolation_forest(
        b_sls.get_selected_sls(),
        percent_outlier_thresh=bundle_def["isolation_forest"].get(
            "percent_outlier_thresh", 25),
        n_jobs=n_cpus, random_state=rng)
    b_sls.select(accept_idx, "isolation_forest")


def mahalanobis(b_sls, bundle_def, clip_edges, cleaning_params, **kwargs):
    b_sls.initiate_selection("Mahalanobis")
    clean_params = bundle_def.get("mahal", {})
    clean_params = {
        **cleaning_params,
        **clean_params}
    clean_params["return_idx"] = True
    cut = clip_edges or ("bundlesection" in bundle_def)
    _, cleaned_idx = abc.clean_bundle(
        b_sls.get_selected_sls(cut=cut, flip=True),
        **clean_params)
    b_sls.select(cleaned_idx, "Mahalanobis", cut=cut)


def run_bundle_rec_plan(
        bundle_dict, tg, mapping, img, reg_template, preproc_imap,
        bundle_name, bundle_idx, bundle_to_flip, bundle_roi_closest,
        bundle_decisions,
        **segmentation_params):
    # Warp ROIs
    logger.info(f"Preparing ROIs for {bundle_name}")
    start_time = time()
    bundle_def = dict(bundle_dict.get_b_info(bundle_name))
    bundle_def.update(bundle_dict.transform_rois(
        bundle_name,
        mapping,
        img.affine,
        apply_to_recobundles=True))

    def check_space(roi):
        if not np.allclose(img.affine, roi.affine):
            logger.warning(
                "Resampling set to False in case where affines "
                "do not match. This is likely due to subject space ROIs"
                " not being in the right space. This found for bundle "
                f"{bundle_name}")

    apply_to_roi_dict(
        bundle_def,
        check_space,
        dry_run=True,
        apply_to_prob_map=True)

    apply_to_roi_dict(
        bundle_def,
        lambda roi_img: nib.Nifti1Image(
            distance_transform_edt(
                np.where(roi_img.get_fdata() == 0, 1, 0)),
            roi_img.affine),
        dry_run=False,
        apply_to_recobundles=False,
        apply_to_prob_map=False)
    logger.info(f"Time to prep ROIs: {time()-start_time}s")

    b_sls = abu.SlsBeingRecognized(
        tg.streamlines, logger,
        segmentation_params["save_intermediates"],
        bundle_name,
        img, len(bundle_def.get("include", [])))

    inputs = {}
    inputs["b_sls"] = b_sls
    inputs["preproc_imap"] = preproc_imap
    inputs["bundle_def"] = bundle_def
    inputs["max_includes"] = bundle_dict.max_includes
    inputs["mapping"] = mapping
    inputs["img"] = img
    inputs["reg_template"] = reg_template
    for key, value in segmentation_params.items():
        inputs[key] = value

    for potential_criterion in bundle_def.keys():
        if (potential_criterion not in criteria_order_post_other_bundles) and\
            (potential_criterion not in criteria_order_pre_other_bundles) and\
                (potential_criterion not in bundle_dict.bundle_names) and\
                (potential_criterion not in valid_noncriterion):
            raise ValueError((
                "Invalid criterion in bundle definition:\n"
                f"{potential_criterion} in bundle {bundle_name}.\n"
                "Valid criteria are:\n"
                f"{criteria_order_pre_other_bundles}\n"
                f"{criteria_order_post_other_bundles}\n"
                f"{bundle_dict.bundle_names}\n"
                f"{valid_noncriterion}\n"))

    for criterion in criteria_order_pre_other_bundles:
        if b_sls and criterion in bundle_def:
            inputs[criterion] = globals()[criterion](**inputs)
    if b_sls:
        for ii, bundle_name in enumerate(bundle_dict.bundle_names):
            if bundle_name in bundle_def.keys():
                idx = np.where(bundle_decisions[:, ii])[0]
                clean_by_other_bundle(
                    **inputs,
                    other_bundle_name=bundle_name,
                    other_bundle_sls=tg.streamlines[idx])
    for criterion in criteria_order_post_other_bundles:
        if b_sls and criterion in bundle_def:
            inputs[criterion] = globals()[criterion](**inputs)
    if b_sls:
        if "mahal" in bundle_def or (
            "isolation_forest" not in bundle_def
                and "orient_mahal" not in bundle_def):
            mahalanobis(**inputs)

    if b_sls and not b_sls.oriented_yet:
        raise ValueError(
            "pyAFQ was unable to consistently orient streamlines "
            f"in bundle {bundle_name} using the provided ROIs. "
            "This can be fixed by including at least 2 "
            "waypoint ROIs, or by using "
            "endpoint ROIs.")

    if b_sls:
        bundle_to_flip[
            b_sls.selected_fiber_idxs,
            bundle_idx] = b_sls.sls_flipped.copy()
        bundle_decisions[
            b_sls.selected_fiber_idxs,
            bundle_idx] = 1
        if hasattr(b_sls, "roi_closest"):
            bundle_roi_closest[
                b_sls.selected_fiber_idxs,
                bundle_idx,
                :
            ] = b_sls.roi_closest.copy()

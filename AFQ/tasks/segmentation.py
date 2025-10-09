import nibabel as nib
import os
import os.path as op
from time import time
import numpy as np
import pandas as pd
import logging

import immlib

from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import get_fname, with_name, str_to_desc
from AFQ.recognition.recognize import recognize
from AFQ.utils.path import drop_extension, write_json
import AFQ.utils.streamlines as aus
from AFQ.tasks.utils import get_default_args
import AFQ.utils.volume as auv
from AFQ._fixes import gaussian_weights
import AFQ.recognition.utils as abu

try:
    from trx.io import load as load_trx
    from trx.io import save as save_trx
    from trx.trx_file_memmap import TrxFile
    has_trx = True
except ModuleNotFoundError:
    has_trx = False

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.stats.analysis import afq_profile
from dipy.tracking.streamline import set_number_of_points, values_from_volume
from nibabel.affines import voxel_sizes
from nibabel.orientations import aff2axcodes
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.align import resample
from scipy.spatial import cKDTree


import gzip
import shutil
import os.path as op
from tempfile import mkdtemp

logger = logging.getLogger('AFQ')


@immlib.calc("bundles")
@as_file('_desc-bundles_tractography')
def segment(data_imap, mapping_imap,
            tractography_imap, segmentation_params):
    """
    full path to a trk/trx file containing containing
    segmented streamlines, labeled by bundle

    Parameters
    ----------
    segmentation_params : dict, optional
        The parameters for segmentation.
        Defaults to using the default behavior of the seg.Segmentation object.
    """
    bundle_dict = data_imap["bundle_dict"]
    reg_template = data_imap["reg_template"]
    streamlines = tractography_imap["streamlines"]
    if streamlines.endswith(".trk") or\
        streamlines.endswith(".tck") or\
            streamlines.endswith(".vtk"):
        tg = load_tractogram(
            streamlines, data_imap["dwi"], Space.VOX,
            bbox_valid_check=False)
        is_trx = False
    elif streamlines.endswith(".trx"):
        is_trx = True
        trx = load_trx(streamlines, data_imap["dwi"])

        # Prepare StatefulTractogram
        affine = np.array(trx.header["VOXEL_TO_RASMM"], dtype=np.float32)
        dimensions = np.array(trx.header["DIMENSIONS"], dtype=np.uint16)
        vox_sizes = np.array(voxel_sizes(affine), dtype=np.float32)
        vox_order = "".join(aff2axcodes(affine))
        space_attributes = (affine, dimensions, vox_sizes, vox_order)

        # Avoid deep copy triggered by to_sft
        tg = StatefulTractogram(
            trx.streamlines,
            space_attributes,
            Space.RASMM)
        del trx
    elif streamlines.endswith(".tck.gz"):
        # uncompress tck.gz to a temporary tck:
        temp_tck = op.join(mkdtemp(), op.split(
            streamlines.replace(".gz", ""))[1])
        logger.info(f"Temporary tck file created at: {temp_tck}")
        with gzip.open(streamlines, 'rb') as f_in:
            with open(temp_tck, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        # initialize stateful tractogram from tck file:
        tg = load_tractogram(
            temp_tck, data_imap["dwi"], Space.VOX,
            bbox_valid_check=False)
        is_trx = False
    if len(tg.streamlines) == 0:
        raise ValueError(f"There are no streamlines in {streamlines}."
                         " This is likely due to errors in defining the "
                         " tractography parameters or the"
                         " seed/stop masks.")

    if not is_trx:
        indices_to_remove, _ = tg.remove_invalid_streamlines()
        if len(indices_to_remove) > 0:
            logger.warning((
                f"{len(indices_to_remove)} invalid "
                "streamlines removed"))

    start_time = time()
    bundles, bundle_meta = recognize(
        tg,
        data_imap["dwi"],
        mapping_imap["mapping"],
        bundle_dict,
        reg_template,
        data_imap["n_cpus"],
        **segmentation_params)

    seg_sft = aus.SegmentedSFT(bundles, Space.VOX)

    if len(seg_sft.sft) < 1:
        raise ValueError("Fatal: No bundles recognized.")

    if is_trx:
        seg_sft.sft.dtype_dict = {'positions': np.float16,
                                  'offsets': np.uint32}
        tgram = TrxFile.from_sft(seg_sft.sft)
        tgram.groups = seg_sft.bundle_idxs
        meta = {}

    else:
        tgram, meta = seg_sft.get_sft_and_sidecar()

    seg_params_out = {}
    for arg_name, value in segmentation_params.items():
        if isinstance(value, (int, float, bool, str)):
            seg_params_out[arg_name] = value
        elif isinstance(value, (list, tuple)):
            seg_params_out[arg_name] = [str(v) for v in value]
        elif isinstance(value, dict):
            for k, v in value.items():
                seg_params_out[k] = str(v)
        else:
            seg_params_out[arg_name] = str(value)

    meta["source"] = streamlines
    meta["Recognition Parameters"] = seg_params_out
    meta["Bundle Parameters"] = bundle_meta
    meta["Timing"] = time() - start_time
    return tgram, meta


@immlib.calc("indiv_bundles")
def export_bundles(base_fname, output_dir,
                   bundles,
                   tracking_params):
    """
    dictionary of paths, where each path is
    a full path to a trk file containing the streamlines of a given bundle.
    """
    is_trx = tracking_params.get("trx", False)
    if is_trx:
        extension = ".trx"
    else:
        extension = ".trk"

    base_fname = op.join(output_dir, op.split(base_fname)[1])
    seg_sft = aus.SegmentedSFT.fromfile(bundles)
    for bundle in seg_sft.bundle_names:
        fname = get_fname(
            base_fname,
            f'_desc-{str_to_desc(bundle)}'
            f'_tractography{extension}',
            subfolder="bundles")
        if op.exists(fname):
            logger.info(
                f"Bundle {bundle} already exists at {fname}. "
                "Skipping export.")
        else:
            bundle_sft = seg_sft.get_bundle(bundle)
            if len(bundle_sft) > 0:
                logger.info(f"Saving {fname}")
                if is_trx:
                    seg_sft.sft.dtype_dict = {
                        'positions': np.float16,
                        'offsets': np.uint32}
                    trxfile = TrxFile.from_sft(bundle_sft)
                    save_trx(trxfile, fname)
                else:
                    save_tractogram(
                        bundle_sft, fname,
                        bbox_valid_check=False)
            else:
                logger.info(f"No bundle to save for {bundle}")
            meta = dict(
                source=bundles,
                params=seg_sft.get_bundle_param_info(bundle))
            meta_fname = drop_extension(fname) + '.json'
            write_json(meta_fname, meta)
    return op.dirname(fname)


@immlib.calc("sl_counts")
@as_file('_desc-slCount_tractography.csv',
         subfolder="stats")
def export_sl_counts(bundles):
    """
    full path to a JSON file containing streamline counts
    """
    sl_counts = []
    seg_sft = aus.SegmentedSFT.fromfile(bundles)

    for bundle in seg_sft.bundle_names:
        sl_counts.append(len(
            seg_sft.get_bundle(bundle).streamlines))
    sl_counts.append(len(seg_sft.sft.streamlines))

    counts_df = pd.DataFrame(
        data=dict(
            n_streamlines=sl_counts),
        index=seg_sft.bundle_names + ["Total Recognized"])
    return counts_df, dict(source=bundles)


@immlib.calc("median_bundle_lengths")
@as_file(
    '_desc-medianBundleLengths_tractography.csv',
    subfolder="stats")
def export_bundle_lengths(bundles):
    """
    full path to a JSON file containing median bundle lengths
    """
    med_len_counts = []
    seg_sft = aus.SegmentedSFT.fromfile(bundles)

    for bundle in seg_sft.bundle_names:
        these_lengths = seg_sft.get_bundle(
            bundle)._tractogram._streamlines._lengths
        if len(these_lengths) > 0:
            med_len_counts.append(np.median(
                these_lengths))
        else:
            med_len_counts.append(0)
    med_len_counts.append(np.median(
        seg_sft.sft._tractogram._streamlines._lengths))

    counts_df = pd.DataFrame(
        data=dict(
            median_len=med_len_counts),
        index=seg_sft.bundle_names + ["Total Recognized"])
    return counts_df, dict(source=bundles)


@immlib.calc("density_maps")
@as_file('_desc-density_tractography.nii.gz')
def export_density_maps(bundles, data_imap):
    """
    full path to 4d nifti file containing streamline counts per voxel
    per bundle, where the 4th dimension encodes the bundle
    """
    seg_sft = aus.SegmentedSFT.fromfile(
        bundles)
    entire_density_map = np.zeros((
        *data_imap["data"].shape[:3],
        len(seg_sft.bundle_names)))
    for ii, bundle_name in enumerate(seg_sft.bundle_names):
        bundle_sl = seg_sft.get_bundle(bundle_name)
        bundle_density = auv.density_map(bundle_sl).get_fdata()
        entire_density_map[..., ii] = bundle_density

    return nib.Nifti1Image(
        entire_density_map, data_imap["dwi_affine"]), dict(
            source=bundles, bundles=list(seg_sft.bundle_names))


@immlib.calc("endpoint_maps")
@as_file('_desc-endpoints_tractography.nii.gz')
def export_endpoint_maps(bundles, data_imap, endpoint_threshold=3):
    """
    full path to a NIfTI file containing endpoint maps for each bundle

    Parameters
    ----------
    endpoint_threshold : float, optional
        The threshold for the endpoint maps.
        If None, no endpoint maps are exported as distance to endpoints maps,
        which the user can then threshold as needed.
        Default: 3
    """
    seg_sft = aus.SegmentedSFT.fromfile(bundles)
    entire_endpoint_map = np.zeros((
        *data_imap["data"].shape[:3],
        len(seg_sft.bundle_names)))

    b0_img = nib.load(data_imap["b0"])
    pve_img = nib.load(data_imap["t1w_pve"])
    pve_data = pve_img.get_fdata()
    gm = resample(pve_data[..., 1], b0_img.get_fdata(),
                  pve_img.affine, b0_img.affine).get_fdata()

    R = b0_img.affine[0:3, 0:3]
    vox_to_mm = np.mean(np.diag(np.linalg.cholesky(R.T.dot(R))))

    for ii, bundle_name in enumerate(seg_sft.bundle_names):
        bundle_sl = seg_sft.get_bundle(bundle_name)
        if len(bundle_sl.streamlines) == 0:
            continue

        bundle_sl.to_vox()

        endpoints = np.vstack([s[0] for s in bundle_sl.streamlines]
                              + [s[-1] for s in bundle_sl.streamlines])

        shape = b0_img.get_fdata().shape
        xv, yv, zv = np.meshgrid(np.arange(shape[0]),
                                 np.arange(shape[1]),
                                 np.arange(shape[2]), indexing='ij')
        grid_points = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])

        kdtree = cKDTree(endpoints)
        distances, _ = kdtree.query(grid_points)
        tractogram_distance = distances.reshape(shape)

        entire_endpoint_map[..., ii] = tractogram_distance * (
            gm > 0.5).astype(np.float32) * vox_to_mm

    if endpoint_threshold is not None:
        entire_endpoint_map = np.logical_and(
            entire_endpoint_map < endpoint_threshold,
            entire_endpoint_map != 0.0).astype(np.float32)

    return nib.Nifti1Image(
        entire_endpoint_map, data_imap["dwi_affine"]), dict(
            source=bundles, bundles=list(seg_sft.bundle_names))


@immlib.calc("profiles")
@as_file('_desc-profiles_tractography.csv')
def tract_profiles(bundles,
                   scalar_dict, data_imap,
                   profile_weights="gauss",
                   n_points_profile=100):
    """
    full path to a CSV file containing tract profiles

    Parameters
    ----------
    profile_weights : str, 1D array, 2D array, or callable, optional
        How to weight each streamline (1D) or each node (2D)
        when calculating the tract-profiles. If callable, this is a
        function that calculates weights. If None, no weighting will
        be applied. If "gauss", gaussian weights will be used.
        If "median", the median of values at each node will be used
        instead of a mean or weighted mean.
        Default: "gauss"
    n_points_profile : int, optional
        Number of points to resample each streamline to before
        calculating the tract-profiles.
        Default: 100
    """
    if not (profile_weights is None
            or isinstance(profile_weights, str)
            or callable(profile_weights)
            or hasattr(profile_weights, "__len__")):
        raise TypeError(
            "profile_weights must be string, None, callable, or"
            + "a 1D or 2D array")
    if isinstance(profile_weights, str):
        profile_weights = profile_weights.lower()
    if isinstance(profile_weights, str) and\
            profile_weights != "gauss" and profile_weights != "median":
        raise TypeError(
            "if profile_weights is a string,"
            + " it must be 'gauss' or 'median'")

    bundle_names = []
    node_numbers = []
    profiles = np.empty((len(scalar_dict), 0)).tolist()
    this_profile = np.zeros((len(scalar_dict), n_points_profile))
    reference = nib.load(scalar_dict[list(scalar_dict.keys())[0]])
    seg_sft = aus.SegmentedSFT.fromfile(
        bundles,
        reference=reference)

    seg_sft.sft.to_rasmm()
    for bundle_name in seg_sft.bundle_names:
        this_sl = seg_sft.get_bundle(bundle_name).streamlines
        if len(this_sl) == 0:
            continue
        if profile_weights == "gauss":
            # calculate only once per bundle
            bundle_profile_weights = gaussian_weights(
                this_sl,
                n_points=n_points_profile)
        for ii, (scalar, scalar_file) in enumerate(scalar_dict.items()):
            if isinstance(scalar_file, str):
                scalar_file = nib.load(scalar_file)
            scalar_data = scalar_file.get_fdata()
            if isinstance(profile_weights, str):
                if profile_weights == "gauss":
                    this_prof_weights = np.asarray(bundle_profile_weights)
                elif profile_weights == "median":
                    # weights bundle to only return the mean
                    def _median_weight(bundle):
                        fgarray = set_number_of_points(
                            bundle, n_points_profile)
                        values = np.array(
                            values_from_volume(
                                scalar_data,
                                fgarray,
                                data_imap["dwi_affine"]))
                        weights = np.zeros(values.shape)
                        for ii, jj in enumerate(
                            np.argsort(values, axis=0)[
                                len(values) // 2, :]):
                            weights[jj, ii] = 1
                        return weights
                    this_prof_weights = _median_weight
            else:
                this_prof_weights = np.asarray(profile_weights)
            if isinstance(this_prof_weights, np.ndarray) and \
                    np.any(np.isnan(this_prof_weights)):  # fit failed
                logger.warning((
                    f"Even weighting used for "
                    f"bundle {bundle_name}, scalar {scalar} "
                    f"in profiling due inability to estimate weights. "
                    "This is often caused by low streamline count or "
                    "low variance in the scalar data."))
                this_prof_weights = np.ones_like(this_prof_weights)
            this_profile[ii] = afq_profile(
                scalar_data,
                this_sl,
                data_imap["dwi_affine"],
                weights=this_prof_weights,
                n_points=n_points_profile)
            profiles[ii].extend(list(this_profile[ii]))
        nodes = list(np.arange(this_profile[0].shape[0]))
        bundle_names.extend([bundle_name] * len(nodes))
        node_numbers.extend(nodes)

    profile_dict = dict()
    profile_dict["tractID"] = bundle_names
    profile_dict["nodeID"] = node_numbers
    for ii, scalar in enumerate(scalar_dict.keys()):
        profile_dict[scalar] = profiles[ii]

    profile_dframe = pd.DataFrame(profile_dict)
    meta = dict(source=bundles,
                parameters=get_default_args(afq_profile),
                scalars=list(scalar_dict.keys()),
                bundles=list(seg_sft.bundle_names))

    return profile_dframe, meta


@immlib.calc("scalar_dict")
def get_scalar_dict(data_imap, mapping_imap, scalars=["dti_fa", "dti_md"]):
    """
    dicionary mapping scalar names
    to their respective file paths

    Parameters
    ----------
    scalars : list of strings and/or scalar definitions, optional
        List of scalars to use.
        Can be any of: "dti_fa", "dti_md", "dki_fa", "dki_md", "dki_awf",
        "dki_mk". Can also be a scalar from AFQ.definitions.image.
        Defaults for single shell data to ["dti_fa", "dti_md"],
        and for multi-shell data to ["dki_fa", "dki_md"].
        Default: ['dti_fa', 'dti_md']
    """
    # Note: some scalars preprocessing done in plans, before this step
    scalar_dict = {}
    for scalar in scalars:
        if isinstance(scalar, str):
            sc = scalar.lower()
            scalar_dict[sc] = data_imap[f"{sc}"]
        elif f"{scalar.get_name()}" in mapping_imap:
            scalar_dict[scalar.get_name()] = mapping_imap[
                f"{scalar.get_name()}"]
    return {"scalar_dict": scalar_dict}


def get_segmentation_plan(kwargs):
    if "segmentation_params" in kwargs\
            and not isinstance(kwargs["segmentation_params"], dict):
        raise TypeError(
            "segmentation_params a dict")
    if "cleaning_params" in kwargs:
        raise ValueError(
            "cleaning_params should be passed inside of"
            "segmentation_params")
    segmentation_tasks = with_name([
        get_scalar_dict,
        export_sl_counts,
        export_bundle_lengths,
        export_bundles,
        export_density_maps,
        export_endpoint_maps,
        segment,
        tract_profiles])

    default_seg_params = get_default_args(recognize)
    if "segmentation_params" in kwargs:
        for k in kwargs["segmentation_params"]:
            default_seg_params[k] = kwargs["segmentation_params"][k]

    kwargs["segmentation_params"] = default_seg_params
    return immlib.plan(**segmentation_tasks)

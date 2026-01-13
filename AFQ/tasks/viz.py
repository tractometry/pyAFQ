import logging
import os.path as op
import re
from time import time

import immlib
import nibabel as nib
import numpy as np
import pandas as pd
from dipy.align import resample
from plotly.subplots import make_subplots

import AFQ.utils.streamlines as aus
from AFQ.tasks.utils import get_fname, get_tp, str_to_desc, with_name
from AFQ.utils.path import drop_extension, write_json
from AFQ.viz.utils import Viz

logger = logging.getLogger("AFQ")


def _viz_prepare_vol(vol, xform, mapping, scalar_dict, ref):
    if vol in scalar_dict.keys():
        vol = scalar_dict[vol]

    if isinstance(vol, str):
        vol = nib.load(vol)

    vol = resample(vol, ref)

    vol = vol.get_fdata()
    if xform:
        vol = mapping.transform_inverse(vol)
    vol[np.isnan(vol)] = 0
    return vol


@immlib.calc("all_bundles_figure")
def viz_bundles(
    base_fname,
    viz_backend,
    structural_imap,
    data_imap,
    tissue_imap,
    mapping_imap,
    segmentation_imap,
    best_scalar,
    sbv_lims_bundles=None,
    volume_opacity_bundles=0.5,
    n_points_bundles=40,
):
    """
    figure for the visualization of the recognized
    bundles in the subject's brain.

    Parameters
    ----------
    sbv_lims_bundles : ndarray
        Of the form (lower bound, upper bound). Shading based on
        shade_by_volume will only differentiate values within these bounds.
        If lower bound is None, will default to 0.
        If upper bound is None, will default to the maximum value in
        shade_by_volume.
        Default: [None, None]
    volume_opacity_bundles : float, optional
        Opacity of volume slices.
        Default: 0.3
    n_points_bundles : int or None
        n_points to resample streamlines to before plotting. If None, no
        resampling is done.
        Default: 40

    Returns
    -------
    List of Figure, String or just the Figure:
    If file can be generated, returns a tuple including the figure and the
    path to the file.
    Otherwise, returns the figure.
    """
    if sbv_lims_bundles is None:
        sbv_lims_bundles = [None, None]
    mapping = mapping_imap["mapping"]
    scalar_dict = segmentation_imap["scalar_dict"]
    profiles_file = segmentation_imap["profiles"]
    t1_img = nib.load(structural_imap["t1_masked"])
    shade_by_volume = get_tp(best_scalar, structural_imap, data_imap, tissue_imap)
    shade_by_volume = _viz_prepare_vol(
        shade_by_volume, False, mapping, scalar_dict, t1_img
    )
    volume = _viz_prepare_vol(t1_img, False, mapping, scalar_dict, t1_img)

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = t1_img.affine[i, i] < 0

    if "plotly" in viz_backend.backend:
        figure = make_subplots(
            rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]]
        )
    else:
        figure = None

    figure = viz_backend.visualize_volume(
        volume,
        opacity=volume_opacity_bundles,
        flip_axes=flip_axes,
        interact=False,
        inline=False,
        figure=figure,
    )

    figure = viz_backend.visualize_bundles(
        segmentation_imap["bundles"],
        img=t1_img,
        shade_by_volume=shade_by_volume,
        sbv_lims=sbv_lims_bundles,
        include_profiles=(pd.read_csv(profiles_file), best_scalar),
        n_points=n_points_bundles,
        flip_axes=flip_axes,
        interact=False,
        inline=False,
        figure=figure,
    )

    fname = None
    if "no_gif" not in viz_backend.backend:
        fname = get_fname(base_fname, ".gif", "..")

        try:
            viz_backend.create_gif(figure, fname)
        except PermissionError as e:
            logger.warning(f"Failed to write GIF file: {fname} \n{e}")
    if "plotly" in viz_backend.backend:
        fname = get_fname(base_fname, ".html", "..")

        try:
            figure.write_html(fname)
        except PermissionError as e:
            logger.warning(f"Failed to write HTML file: {fname}\n{e}")
    if fname is None:
        return figure
    else:
        return [figure, fname]


@immlib.calc("indiv_bundles_figures")
def viz_indivBundle(
    base_fname,
    output_dir,
    viz_backend,
    structural_imap,
    data_imap,
    tissue_imap,
    mapping_imap,
    segmentation_imap,
    best_scalar,
    sbv_lims_indiv=None,
    volume_opacity_indiv=0.5,
    n_points_indiv=40,
):
    """
    list of full paths to html or gif files
    containing visualizations of individual bundles

    Parameters
    ----------
    sbv_lims_indiv : ndarray
        Of the form (lower bound, upper bound). Shading based on
        shade_by_volume will only differentiate values within these bounds.
        If lower bound is None, will default to 0.
        If upper bound is None, will default to the maximum value in
        shade_by_volume.
        Default: [None, None]
    volume_opacity_indiv : float, optional
        Opacity of volume slices.
        Default: 0.3
    n_points_indiv : int or None
        n_points to resample streamlines to before plotting. If None, no
        resampling is done.
        Default: 40
    """
    if sbv_lims_indiv is None:
        sbv_lims_indiv = [None, None]
    mapping = mapping_imap["mapping"]
    bundle_dict = data_imap["bundle_dict"]
    scalar_dict = segmentation_imap["scalar_dict"]
    volume_img = nib.load(structural_imap["t1_masked"])
    shade_by_volume = get_tp(best_scalar, structural_imap, data_imap, tissue_imap)
    profiles = pd.read_csv(segmentation_imap["profiles"])

    start_time = time()
    volume = _viz_prepare_vol(volume_img, False, mapping, scalar_dict, volume_img)
    shade_by_volume = _viz_prepare_vol(
        shade_by_volume, False, mapping, scalar_dict, volume_img
    )

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = volume_img.affine[i, i] < 0

    bundles = aus.SegmentedSFT.fromfile(segmentation_imap["bundles"])

    # This dictionary contains a mapping to which ROIs
    # should be used from the bundle dict, based on the
    # name from the segmented SFT file. Currently,
    # This is only different when using bundle sections.
    segmented_bname_to_roi_bname = {}
    for b_name, b_info in bundle_dict.items():
        if "bundlesection" in b_info:
            for sb_name in b_info["bundlesection"]:
                segmented_bname_to_roi_bname[sb_name] = b_name
        else:
            segmented_bname_to_roi_bname[b_name] = b_name

    figures = {}
    failed_write = False
    for bundle_name in bundles.bundle_names:
        logger.info(f"Generating {bundle_name} visualization...")
        roi_bname = segmented_bname_to_roi_bname[bundle_name]

        figure = viz_backend.visualize_volume(
            volume,
            opacity=volume_opacity_indiv,
            flip_axes=flip_axes,
            interact=False,
            inline=False,
        )
        if len(bundles.get_bundle(bundle_name)) > 0:
            figure = viz_backend.visualize_bundles(
                bundles,
                img=volume_img,
                shade_by_volume=shade_by_volume,
                sbv_lims=sbv_lims_indiv,
                bundle=bundle_name,
                n_points=n_points_indiv,
                flip_axes=flip_axes,
                interact=False,
                inline=False,
                figure=figure,
            )
        else:
            logger.info("No streamlines found to visualize for " + bundle_name)

        for roi_fname in mapping_imap["rois"][roi_bname]:
            name = roi_fname.split("desc-")[1].split("_")[0]
            if "probseg" in roi_fname:
                name = f"{name}Probseg"
            roi_img = nib.load(roi_fname)
            roi_img = resample(roi_img, volume_img)
            figure = viz_backend.visualize_roi(
                roi_img,
                name=name,
                flip_axes=flip_axes,
                inline=False,
                interact=False,
                figure=figure,
            )

        base_fname = op.join(output_dir, op.split(base_fname)[1])
        figures[bundle_name] = figure
        if "no_gif" not in viz_backend.backend:
            fname = get_fname(
                base_fname,
                f"_desc-{str_to_desc(bundle_name)}_tractography.gif",
                "viz_bundles",
            )

            try:
                viz_backend.create_gif(figure, fname)
            except PermissionError as e:
                if not failed_write:
                    logger.warning(
                        (f"Failed to write individual visualization files \n{e}")
                    )
                    failed_write = True
        if "plotly" in viz_backend.backend:
            fname = get_fname(
                base_fname,
                f"_desc-{str_to_desc(bundle_name)}_tractography.html",
                "viz_bundles",
            )

            try:
                figure.write_html(fname)
            except PermissionError as e:
                if not failed_write:
                    logger.warning(
                        (f"Failed to write individual visualization files \n{e}")
                    )
                    failed_write = True

            # also do the core visualizations when using the plotly backend
            indiv_profile = profiles[profiles.tractID == bundle_name][
                best_scalar
            ].to_numpy()
            if len(indiv_profile) > 1:
                fname = get_fname(
                    base_fname,
                    f"_desc-{str_to_desc(bundle_name)}Core_tractography.html",
                    "viz_core_bundles",
                )
                core_fig = make_subplots(
                    rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]]
                )
                core_fig = viz_backend.visualize_volume(
                    volume,
                    opacity=volume_opacity_indiv,
                    flip_axes=flip_axes,
                    figure=core_fig,
                    interact=False,
                    inline=False,
                )
                core_fig = viz_backend.visualize_bundles(
                    segmentation_imap["bundles"],
                    img=volume_img,
                    shade_by_volume=shade_by_volume,
                    sbv_lims=sbv_lims_indiv,
                    bundle=bundle_name,
                    colors={bundle_name: [0.5, 0.5, 0.5]},
                    n_points=n_points_indiv,
                    flip_axes=flip_axes,
                    interact=False,
                    inline=False,
                    figure=core_fig,
                )
                core_fig = viz_backend.single_bundle_viz(
                    indiv_profile,
                    segmentation_imap["bundles"],
                    bundle_name,
                    best_scalar,
                    img=volume_img,
                    flip_axes=flip_axes,
                    figure=core_fig,
                    include_profile=True,
                )
                try:
                    core_fig.write_html(fname)
                except PermissionError as e:
                    if not failed_write:
                        logger.warning(
                            (f"Failed to write individual visualization files \n{e}")
                        )
                        failed_write = True
    if not failed_write:
        meta_fname = drop_extension(fname) + ".json"
        meta = dict(Timing=time() - start_time)
        write_json(meta_fname, meta)
    return {"indiv_bundles_figures": figures}


@immlib.calc("tract_profile_plots")
def plot_tract_profiles(base_fname, output_dir, scalars, segmentation_imap):
    """
    list of full paths to png files,
    where files contain plots of the tract profiles
    """
    from AFQ.viz.plot import visualize_tract_profiles

    start_time = time()
    fnames = []
    base_fname = op.join(output_dir, op.split(base_fname)[1])
    for scalar in scalars:
        this_scalar = scalar if isinstance(scalar, str) else scalar.get_name()
        fname = get_fname(
            base_fname,
            f"_param-{str_to_desc(this_scalar)}_desc-alltracts_tractography",
            "tract_profile_plots",
        )

        if not op.exists(fname):
            visualize_tract_profiles(
                segmentation_imap["profiles"],
                scalar=this_scalar,
                file_name=fname,
                n_boot=100,
            )
            fnames.append(fname)

            meta_fname = drop_extension(fname) + ".json"
            meta = dict(Timing=time() - start_time)
            write_json(meta_fname, meta)
    return fnames


@immlib.calc("viz_backend")
def init_viz_backend(viz_backend_spec="plotly_no_gif", virtual_frame_buffer=False):
    """
    An instance of the `AFQ.viz.utils.viz_backend` class.

    Parameters
    ----------
    virtual_frame_buffer : bool, optional
        Whether to use a virtual frame buffer. This is  if
        generating GIFs in a headless environment.
        Default: False
    viz_backend_spec : str, optional
        Which visualization backend to use.
        See Visualization Backends page in documentation for details
        https://tractometry.org/pyAFQ/reference/viz_backend.html
        One of {"fury", "plotly", "plotly_no_gif"}.
        Default: "plotly_no_gif"
    """
    if not isinstance(virtual_frame_buffer, bool):
        raise TypeError("virtual_frame_buffer must be a bool")
    if "fury" not in viz_backend_spec and "plotly" not in viz_backend_spec:
        raise TypeError("viz_backend_spec must contain either 'fury' or 'plotly'")

    if virtual_frame_buffer:
        from xvfbwrapper import Xvfb

        vdisplay = Xvfb(width=1280, height=1280)
        vdisplay.start()

    return Viz(backend=viz_backend_spec.lower())


@immlib.calc("citations")
def citations(base_fname, citations):
    """
    Export Bibtex citation file for methods used by pyAFQ.
    """
    bib_fname = get_fname(base_fname, "_citations.bib")

    if op.exists(bib_fname):
        with open(bib_fname, "r") as f:
            content = f.read()
            existing_keys = set(re.findall(r"@\w+\{([^,]+),", content))
    else:
        existing_keys = set()

    with open(bib_fname, "a") as f:
        for entry in citations:
            match = re.search(r"@\w+\{([^,]+),", entry)
            if match:
                key = match.group(1)
                if key not in existing_keys:
                    f.write("\n" + entry + "\n")
                    existing_keys.add(key)


def get_viz_plan(kwargs):
    viz_tasks = with_name(
        [plot_tract_profiles, viz_bundles, viz_indivBundle, init_viz_backend, citations]
    )
    return immlib.plan(**viz_tasks)

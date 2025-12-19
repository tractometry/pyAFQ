import logging
import os.path as op

import immlib
import nibabel as nib
import numpy as np
from dipy.align import resample
from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import load_tractogram

import AFQ.data.fetch as afd
from AFQ.definitions.image import ImageDefinition
from AFQ.definitions.mapping import SynMap
from AFQ.definitions.utils import Definition
from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import get_fname, get_tp, str_to_desc, with_name
from AFQ.utils.path import drop_extension, space_from_fname, write_json

logger = logging.getLogger('AFQ')


@immlib.calc("b0_warped")
def export_registered_b0(base_fname, data_imap, mapping):
    """
    full path to a nifti file containing
    b0 transformed to template space
    """
    warped_b0_fname = get_fname(
        base_fname,
        f'_space-{data_imap["tmpl_name"]}_b0ref.nii.gz')
    if not op.exists(warped_b0_fname):
        mean_b0 = nib.load(data_imap["b0"]).get_fdata()
        warped_b0 = mapping.transform(mean_b0)
        warped_b0 = nib.Nifti1Image(warped_b0,
                                    data_imap["reg_template"].affine)
        logger.info(f"Saving {warped_b0_fname}")
        nib.save(warped_b0, warped_b0_fname)
        meta = dict(
            b0InSubject=data_imap["b0"],
            dependent="dwi")
        meta_fname = get_fname(
            base_fname,
            f'_space-{data_imap["tmpl_name"]}_b0ref.json')
        write_json(meta_fname, meta)

    return warped_b0_fname


@immlib.calc("template_xform")
def template_xform(base_fname, dwi_data_file, data_imap, mapping):
    """
    full path to a nifti file containing
    registration template transformed to subject space
    """
    subject_space = space_from_fname(dwi_data_file)
    template_xform_fname = get_fname(
        base_fname,
        f'_space-{subject_space}_desc-template_anat.nii.gz')
    if not op.exists(template_xform_fname):
        template_xform = mapping.transform_inverse(
            data_imap["reg_template"].get_fdata())
        template_xform = nib.Nifti1Image(
            template_xform, data_imap["dwi_affine"])
        logger.info(f"Saving {template_xform_fname}")
        nib.save(template_xform, template_xform_fname)
        meta = dict()
        meta_fname = get_fname(
            base_fname,
            f'_space-{subject_space}_desc-template_anat.json')
        write_json(meta_fname, meta)

    return template_xform_fname


@immlib.calc("rois")
def export_rois(base_fname, output_dir, dwi_data_file, data_imap, mapping):
    """
    dictionary of full paths to Nifti1Image files of ROIs
    transformed to subject space
    """
    bundle_dict = data_imap["bundle_dict"]
    roi_files = {}
    base_roi_fname = op.join(output_dir, op.split(base_fname)[1])
    to_space = space_from_fname(dwi_data_file)
    for bundle_name in bundle_dict:
        roi_files[bundle_name] = []
        for roi, roi_fname in zip(*bundle_dict.transform_rois(
                bundle_name, mapping, data_imap["dwi_affine"],
                base_fname=base_roi_fname,
                to_space=to_space)):
            roi_files[bundle_name].append(roi_fname)
            if not op.exists(roi_fname):
                logger.info(f"Saving {roi_fname}")
                nib.save(
                    nib.Nifti1Image(
                        roi.get_fdata().astype(np.float32),
                        roi.affine), roi_fname)
                meta = {
                    "Bundle Definition": bundle_dict.get_b_info(bundle_name)}
                meta_fname = f'{drop_extension(roi_fname)}.json'
                write_json(meta_fname, meta)
    return {'rois': roi_files}


@immlib.calc("mapping")
def mapping(base_fname, dwi_data_file, reg_subject, data_imap,
            mapping_definition=None):
    """
    mapping from subject to template space.

    Parameters
    ----------
    mapping_definition : instance of `AFQ.definitions.mapping`, optional
        This defines how to either create a mapping from
        each subject space to template space or load a mapping from
        another software. If creating a map, will register reg_subject and
        reg_template.
        If None, use SynMap()
        Default: None
    """
    reg_template = data_imap["reg_template"]
    tmpl_name = data_imap["tmpl_name"]

    if mapping_definition is None:
        mapping_definition = SynMap()

    if isinstance(reg_subject, str):
        reg_subject = nib.load(reg_subject)

    if not isinstance(mapping_definition, Definition):
        raise TypeError(
            "mapping must be a mapping defined"
            + " in `AFQ.definitions.mapping`")
    return mapping_definition.get_for_subses(
        base_fname, data_imap["dwi"], dwi_data_file,
        reg_subject, reg_template, tmpl_name)


@immlib.calc("mapping")
def sls_mapping(base_fname, dwi_data_file, reg_subject, data_imap,
                tractography_imap, mapping_definition=None):
    """
    mapping from subject to template space.

    Parameters
    ----------
    mapping_definition : instance of `AFQ.definitions.mapping`, optional
        This defines how to either create a mapping from
        each subject space to template space or load a mapping from
        another software. If creating a map, will register reg_subject and
        reg_template.
        If None, use SynMap()
        Default: None
    """
    reg_template = data_imap["reg_template"]
    tmpl_name = data_imap["tmpl_name"]
    if mapping_definition is None:
        mapping_definition = SynMap()
    if not isinstance(mapping_definition, Definition):
        raise TypeError(
            "mapping must be a mapping defined"
            + " in `AFQ.definitions.mapping`")
    streamlines_file = tractography_imap["streamlines"]
    tg = load_tractogram(
        streamlines_file, reg_subject,
        Space.VOX, bbox_valid_check=False)
    tg.to_rasmm()

    atlas_fname = op.join(
        afd.afq_home,
        'hcp_atlas_16_bundles',
        'Atlas_in_MNI_Space_16_bundles',
        'whole_brain',
        'whole_brain_MNI.trk')
    if not op.exists(atlas_fname):
        afd.fetch_hcp_atlas_16_bundles()
    hcp_atlas = load_tractogram(
        atlas_fname,
        'same', bbox_valid_check=False)
    return mapping_definition.get_for_subses(
        base_fname, data_imap["dwi"],
        dwi_data_file,
        reg_subject, reg_template,
        tmpl_name,
        subject_sls=tg.streamlines,
        template_sls=hcp_atlas.streamlines)


@immlib.calc("reg_subject")
def get_reg_subject(structural_imap, data_imap, tissue_imap,
                    reg_subject_spec="t1w"):
    """
    Nifti1Image which represents this subject
    when registering the subject to the template

    Parameters
    ----------
    reg_subject_spec : str, instance of `AFQ.definitions.ImageDefinition`, optional  # noqa
        The source image data to be registered.
        Can either be a Nifti1Image, an ImageFile, or str.
        if "b0", "dti_fa_subject", "subject_sls", "t1w", or "power_map,"
        image data will be loaded automatically.
        If "subject_sls" is used, slr registration will be used
        and reg_template should be "hcp_atlas".
        Default: "t1w"
    """
    if not isinstance(reg_subject_spec, str)\
            and not isinstance(reg_subject_spec, nib.Nifti1Image):
        # Note the ImageDefinition case is handled in get_mapping_plan
        raise TypeError(
            "reg_subject must be a str, ImageDefinition, or Nifti1Image")

    filename_dict = {
        "b0": "b0",
        "power_map": "csd_pmap",
        "dti_fa_subject": "dti_fa",
        "subject_sls": "b0",
        "t1w": "t1_masked"
    }
    bm = nib.load(data_imap["brain_mask"])

    if reg_subject_spec in filename_dict:
        reg_subject_spec = get_tp(
            filename_dict[reg_subject_spec],
            structural_imap,
            data_imap,
            tissue_imap)
    if isinstance(reg_subject_spec, str):
        img = nib.load(reg_subject_spec)

    if not np.allclose(img.affine, bm.affine) or not np.allclose(
            img.get_fdata().shape, bm.get_fdata().shape):
        img = resample(img, bm)

    bm = bm.get_fdata().astype(bool)
    masked_data = img.get_fdata()
    masked_data[~bm] = 0
    img = nib.Nifti1Image(masked_data, img.affine)
    return img


def get_mapping_plan(kwargs, use_sls=False):
    mapping_tasks = with_name([
        export_registered_b0, template_xform, export_rois, mapping,
        get_reg_subject])

    # add custom scalars
    for scalar in kwargs["scalars"]:
        if isinstance(scalar, Definition):
            mapping_tasks[f"{scalar.get_name()}_res"] =\
                immlib.calc(f"{scalar.get_name()}")(
                    as_file((
                        f'_desc-{str_to_desc(scalar.get_name())}'
                        '_dwimap.nii.gz'), subfolder="models")(
                            scalar.get_image_getter("mapping")))

    if use_sls:
        mapping_tasks["mapping_res"] = sls_mapping

    reg_ss = kwargs.get("reg_subject_spec", None)
    if isinstance(reg_ss, ImageDefinition):
        mapping_tasks["get_reg_subject_res"] = immlib.calc("reg_subject")(
            as_file((
                f'_desc-{str_to_desc(reg_ss.get_name())}'
                '_dwiref.nii.gz'))(reg_ss.get_image_getter("mapping")))

    return immlib.plan(**mapping_tasks)

import numpy as np
import logging

import nibabel as nib

from dipy.align import resample

from AFQ.definitions.utils import Definition, find_file, name_from_path
from AFQ.tasks.utils import get_tp


__all__ = [
    "ImageFile", "FullImage", "RoiImage", "LabelledImageFile",
    "ThresholdedImageFile", "ScalarImage", "ThresholdedScalarImage",
    "TemplateImage", "PVEImage", "PVEImages"]


logger = logging.getLogger('AFQ')


def _resample_image(image_data, ref_data, image_affine, ref_affine):
    '''
    Helper function
    Resamples image to dwi if necessary
    '''
    if len(ref_data.shape) > 3:  # DWI data
        ref_data = ref_data[..., 0]

    def _resample_slice(slice_data):
        return resample(
            slice_data.astype(float),
            ref_data,
            moving_affine=image_affine,
            static_affine=ref_affine).get_fdata().astype(image_type)

    image_type = image_data.dtype
    if ((ref_data is not None)
        and (ref_affine is not None)
            and ((ref_data.shape[:3] != image_data.shape[:3]) or (
                not np.allclose(ref_affine, image_affine)))):
        if len(image_data.shape) < 4:
            return _resample_slice(image_data), True
        else:
            return np.stack([_resample_slice(
                image_data[..., ii]) for ii in range(
                    image_data.shape[-1])], axis=-1), True
    else:
        return image_data, False


class ImageDefinition(Definition):
    '''
    All Image Definitions should inherit this.
    '''

    def get_name(self):
        raise NotImplementedError("Please implement a get_name method")

    def get_image_getter(self, task_name):
        raise NotImplementedError(
            "Please implement a get_image_getter method")

    # This function is set up to be overriden by other images
    def apply_conditions(self, image_data_orig, image_file):
        return image_data_orig, dict(source=image_file)


class CombineImageMixin(object):
    """
    Helper Class
    Useful for making an image by combining different conditions
    """

    def __init__(self, combine):
        self.combine = combine.lower()

    def reset_image_draft(self, shape):
        if self.combine == "or":
            self.image_draft = np.zeros(shape, dtype=bool)
        elif self.combine == "and":
            self.image_draft = np.ones(shape, dtype=bool)
        else:
            self.combine_illdefined()

    def __mul__(self, other_image):
        if self.combine == "or":
            return np.logical_or(self.image_draft, other_image)
        elif self.combine == "and":
            return np.logical_and(self.image_draft, other_image)
        else:
            self.combine_illdefined()

    def combine_illdefined(self):
        raise TypeError((
            f"combine should be either 'or' or 'and',"
            f" you set combine to {self.combine}"))


class ThresholdMixin(CombineImageMixin):
    def __init__(self, combine, lower_bound, upper_bound, as_percentage):
        CombineImageMixin.__init__(self, combine)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.as_percentage = as_percentage

    def apply_conditions(self, image_data_orig, image_file):
        # Apply thresholds
        self.reset_image_draft(image_data_orig.shape)
        if self.upper_bound is not None:
            if self.as_percentage:
                upper_bound = np.nanpercentile(
                    image_data_orig,
                    self.upper_bound)
            else:
                upper_bound = self.upper_bound
            self.image_draft = self * (image_data_orig < upper_bound)
        if self.lower_bound is not None:
            if self.as_percentage:
                lower_bound = np.nanpercentile(
                    image_data_orig,
                    100 - self.lower_bound)
            else:
                lower_bound = self.lower_bound
            self.image_draft = self * (image_data_orig > lower_bound)

        meta = dict(source=image_file,
                    upper_bound=self.upper_bound,
                    lower_bound=self.lower_bound,
                    combined_with=self.combine)
        return self.image_draft, meta


class ImageFile(ImageDefinition):
    """
    Define an image based on a file.
    Does not apply any labels or thresholds;
    Generates image with floating point data.
    Useful for seed images, where threshold can be applied
    after interpolation (see example).

    Parameters
    ----------
    path : str, optional
        path to file to get image from. Use this or suffix.
        Default: None
    suffix : str, optional
        suffix to pass to bids_layout.get() to identify the file.
        Default: None
    filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}
    resample : bool, optional
        Whether to resample the image to the DWI data.
        Default: True

    Examples
    --------
    seed_image = ImageFile(
        suffix="WM",
        filters={"scope":"dmriprep"})
    api.GroupAFQ(tracking_params={"seed_image": seed_image,
                                "seed_threshold": 0.1})
    """

    def __init__(self, path=None, suffix=None, filters={}, resample=True):
        if path is None and suffix is None:
            raise ValueError((
                "One of `path` or `suffix` must set to "
                "a value other than None."))

        if path is not None:
            self._from_path = True
            self.fname = path
        else:
            self._from_path = False
            self.suffix = suffix
            self.filters = filters
            self.fnames = {}
        self.resample = resample

    def find_path(self, bids_layout, from_path,
                  subject, session, required=True):
        if self._from_path:
            return

        nearest_image = find_file(
            bids_layout, from_path, self.filters, self.suffix, session,
            subject, required=required)

        if nearest_image is None:
            return False

        self.fnames[from_path] = nearest_image

    def get_path_data_affine(self, dwi_path):
        if self._from_path:
            image_file = self.fname
        else:
            image_file = self.fnames[dwi_path]
        image_img = nib.load(image_file)
        return image_file, image_img.get_fdata(), image_img.affine

    def get_name(self):
        return name_from_path(self.fname) if self._from_path else self.suffix

    def get_image_getter(self, task_name):
        def _image_getter_helper(resample_to, ref_file):
            # Load data
            image_file, image_data_orig, image_affine = \
                self.get_path_data_affine(ref_file)

            # Apply any conditions on the data
            image_data, meta = self.apply_conditions(
                image_data_orig, image_file)

            if self.resample:
                if isinstance(resample_to, str):
                    resample_to_img = nib.load(resample_to)
                    meta["resampled"] = resample_to
                else:
                    resample_to_img = resample_to
                    meta["resampled"] = True
                image_data, _ = _resample_image(
                    image_data,
                    resample_to_img.get_fdata(),
                    image_affine,
                    resample_to_img.affine)
                image_affine = resample_to_img.affine
            else:
                meta["resampled"] = False

            return nib.Nifti1Image(
                image_data.astype(np.float32),
                image_affine), meta
        
        # In these tasks, use T1 as ref
        if task_name == "structural" or task_name == "tissue":
            def image_getter(t1_file):
                return _image_getter_helper(t1_file, t1_file)
        elif task_name == "data": # Otherwise, use DWI
            def image_getter(dwi, dwi_data_file):
                return _image_getter_helper(dwi, dwi_data_file)
        else:
            def image_getter(data_imap, dwi_data_file):
                return _image_getter_helper(data_imap["dwi"], dwi_data_file)
        return image_getter


class FullImage(ImageDefinition):
    """
    Define an image which covers a full volume.

    Examples
    --------
    brain_image_definition = FullImage()
    """

    def __init__(self):
        pass

    def get_name(self):
        return "entire_volume"

    def get_image_getter(self, task_name):
        def _image_getter_helper(dwi):
            return nib.Nifti1Image(
                np.ones(dwi.get_fdata()[..., 0].shape, dtype=np.float32),
                dwi.affine), dict(source="Entire Volume")
        if task_name == "data":
            def image_getter(dwi):
                return _image_getter_helper(dwi)
        else:
            def image_getter(data_imap):
                return _image_getter_helper(data_imap["dwi"])
        return image_getter


class RoiImage(ImageDefinition):
    """
    Define an image which is all include ROIs or'd together.

    Parameters
    ----------
    use_waypoints : bool
        Whether to use the include ROIs to generate the image.
    use_presegment : bool
        Whether to use presegment bundle dict from segmentation params
        to get ROIs.
    use_endpoints : bool
        Whether to use the endpoints ("start" and "end") to generate
        the image.
    tissue_property : str or None
        Tissue property from `scalars` to multiply the ROI image with.
        Can be useful to limit seed mask to the core white matter.
        Note: this must be a built-in tissue property.
        Default: None
    tissue_property_n_voxel : int or None
        Threshold `tissue_property` to a boolean mask with
        tissue_property_n_voxel number of voxels set to True.
        Default: None
    tissue_property_threshold : int or None
        Threshold to threshold `tissue_property` if a boolean mask is
        desired. This threshold is interpreted as a percentile.
        Overrides tissue_property_n_voxel.
        Default: None
    Examples
    --------
    seed_image = RoiImage()
    api.GroupAFQ(tracking_params={"seed_image": seed_image})
    """

    def __init__(self,
                 use_waypoints=True,
                 use_presegment=False,
                 use_endpoints=False,
                 only_wmgmi=False,
                 tissue_property=None,
                 tissue_property_n_voxel=None,
                 tissue_property_threshold=None):
        self.use_waypoints = use_waypoints
        self.use_presegment = use_presegment
        self.use_endpoints = use_endpoints
        self.only_wmgmi = only_wmgmi
        self.tissue_property = tissue_property
        self.tissue_property_n_voxel = tissue_property_n_voxel
        self.tissue_property_threshold = tissue_property_threshold
        if not np.logical_or(self.use_waypoints, np.logical_or(
                self.use_endpoints, self.use_presegment)):
            raise ValueError((
                "One of use_waypoints, use_presegment, "
                "use_endpoints, must be True"))

    def get_name(self):
        return "roi"

    def get_image_getter(self, task_name):
        def _image_getter_helper(mapping_imap,
                                 data_imap,
                                 structural_imap,
                                 tissue_imap,
                                 segmentation_params):
            image_data = None
            bundle_dict = data_imap["bundle_dict"]
            if self.use_presegment:
                bundle_dict = \
                    segmentation_params["presegment_bundle_dict"]
            else:
                bundle_dict = bundle_dict

            for bundle_name in bundle_dict:
                bundle_entry = bundle_dict.transform_rois(
                    bundle_name,
                    mapping_imap["mapping"],
                    data_imap["dwi_affine"])
                rois = []
                if self.use_endpoints:
                    rois.extend(
                        [bundle_entry[end_type] for end_type in
                            ["start", "end"] if end_type in bundle_entry])
                if self.use_waypoints:
                    rois.extend(bundle_entry.get('include', []))
                for roi in rois:
                    warped_roi = roi.get_fdata()
                    if image_data is None:
                        image_data = np.zeros(warped_roi.shape)
                    image_data = np.logical_or(
                        image_data,
                        warped_roi.astype(bool))
            if self.tissue_property is not None:
                tp = nib.load(get_tp(
                    self.tissue_property,
                    structural_imap,
                    data_imap,
                    tissue_imap)).get_fdata()
                image_data = image_data.astype(np.float32) * tp
                if self.tissue_property_threshold is not None:
                    zero_mask = image_data == 0
                    image_data[zero_mask] = np.nan
                    tp_thresh = np.nanpercentile(
                        image_data,
                        100 - self.tissue_property_threshold)
                    image_data[zero_mask] = 0
                    image_data = image_data > tp_thresh
                elif self.tissue_property_n_voxel is not None:
                    tp_thresh = np.sort(image_data.flatten())[
                        -1 - self.tissue_property_n_voxel]
                    image_data = image_data > tp_thresh
            if image_data is None:
                raise ValueError((
                    "BundleDict does not have enough ROIs to generate "
                    f"an ROI Image: {bundle_dict._dict}"))

            if self.only_wmgmi:
                wmgmi = nib.load(
                    tissue_imap["wm_gm_interface"]).get_fdata()
                if not np.allclose(wmgmi.shape, image_data.shape):
                    logger.error("WM/GM Interface shape: %s", wmgmi.shape)
                    logger.error("ROI image shape: %s", image_data.shape)
                    raise ValueError((
                        "wm_gm_interface and ROI image do not have the "
                        "same shape, cannot apply wm_gm_interface."
                        "If ROI image shape is different from DWI shape, "
                        "consider if you need to map your ROIs to DWI space. "
                        "If only resampling is required, "
                        "set resample_subject_to "
                        "to True in your BundleDict instantiation."))

                image_data = np.logical_and(
                    image_data, wmgmi)
                if np.sum(image_data) == 0:
                    raise ValueError((
                        "BundleDict does not have enough ROIs to generate "
                        "an ROI Image with WM/GM interface applied."))

            return nib.Nifti1Image(
                image_data.astype(np.float32),
                data_imap["dwi_affine"]), dict(source="ROIs")

        if task_name == "data"\
            or task_name == "structural" or\
                task_name == "tissue" or\
                    task_name == "mapping":
            raise ValueError((
                "RoiImage cannot be used in this context, as they"
                "require later derivatives to be calculated"))
        return _image_getter_helper


class LabelledImageFile(ImageFile, CombineImageMixin):
    """
    Define an image based on labels in a file.

    Parameters
    ----------
    path : str, optional
        path to file to get image from. Use this or suffix.
        Default: None
    suffix : str, optional
        suffix to pass to bids_layout.get() to identify the file.
        Default: None
    filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}
    inclusive_labels : list of ints, optional
        The labels from the file to include from the boolean image.
        If None, no inclusive labels are applied.
    exclusive_labels : list of ints, optional
        The labels from the file to exclude from the boolean image.
        If None, no exclusive labels are applied.
        Default: None.
    combine : str, optional
        How to combine the boolean images generated by inclusive_labels
        and exclusive_labels. If "and", they will be and'd together.
        If "or", they will be or'd.
        Note: in this class, you will most likely want to either set
        inclusive_labels or exclusive_labels, not both,
        so combine will not matter.
        Default: "or"

    Examples
    --------
    brain_image_definition = LabelledImageFile(
        suffix="aseg",
        filters={"scope": "dmriprep"},
        exclusive_labels=[0])
    api.GroupAFQ(brain_image_definition=brain_image_definition)
    """

    def __init__(self, path=None, suffix=None, filters={},
                 inclusive_labels=None,
                 exclusive_labels=None, combine="or"):
        ImageFile.__init__(self, path, suffix, filters)
        CombineImageMixin.__init__(self, combine)
        self.inclusive_labels = inclusive_labels
        self.exclusive_labels = exclusive_labels

    def apply_conditions(self, image_data_orig, image_file):
        # For different sets of labels, extract all the voxels that
        # have any / all of these values:
        self.reset_image_draft(image_data_orig.shape)
        if self.inclusive_labels is not None:
            for label in self.inclusive_labels:
                self.image_draft = self * (image_data_orig == label)
        if self.exclusive_labels is not None:
            for label in self.exclusive_labels:
                self.image_draft = self * (image_data_orig != label)

        meta = dict(source=image_file,
                    inclusive_labels=self.inclusive_labels,
                    exclusive_lavels=self.exclusive_labels,
                    combined_with=self.combine)
        return self.image_draft, meta


class ThresholdedImageFile(ThresholdMixin, ImageFile):
    """
    Define an image based on thresholding a file.
    Note that this should not be used to directly make a seed image.
    In those cases, consider thresholding after
    interpolation, as in the example for ImageFile.

    Parameters
    ----------
    path : str, optional
        path to file to get image from. Use this or suffix.
        Default: None
    suffix : str, optional
        suffix to pass to bids_layout.get() to identify the file.
        Default: None
    filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}
    lower_bound : float, optional
        Lower bound to generate boolean image from data in the file.
        If None, no lower bound is applied.
        Default: None.
    upper_bound : float, optional
        Upper bound to generate boolean image from data in the file.
        If None, no upper bound is applied.
        Default: None.
    as_percentage : bool, optional
        Interpret lower_bound and upper_bound as percentages of the
        total non-nan voxels in the image to include (between 0 and 100),
        instead of as a threshold on the values themselves.
        Default: False
    combine : str, optional
        How to combine the boolean images generated by lower_bound
        and upper_bound. If "and", they will be and'd together.
        If "or", they will be or'd.
        Default: "and"

    Examples
    --------
    brain_image_definition = ThresholdedImageFile(
        suffix="BM",
        filters={"scope":"dmriprep"},
        lower_bound=0.1)
    api.GroupAFQ(brain_image_definition=brain_image_definition)
    """

    def __init__(self, path=None, suffix=None, filters={}, lower_bound=None,
                 upper_bound=None, as_percentage=False, combine="and"):
        ImageFile.__init__(self, path, suffix, filters)
        ThresholdMixin.__init__(self, combine, lower_bound,
                                upper_bound, as_percentage)


class ScalarImage(ImageDefinition):
    """
    Define an image based on a scalar.
    Does not apply any labels or thresholds;
    Generates image with floating point data.
    Useful for seed images, where threshold can be applied
    after interpolation (see example).

    Parameters
    ----------
    scalar : str
        Scalar to threshold.
        Can be one of "dti_fa", "dti_md", "dki_fa", "dki_md".

    Examples
    --------
    seed_image = ScalarImage(
        "dti_fa")
    api.GroupAFQ(tracking_params={
        "seed_image": seed_image,
        "seed_threshold": 0.2})
    """

    def __init__(self, scalar):
        self.scalar = scalar

    def get_name(self):
        return self.scalar

    def get_image_getter(self, task_name):
        if task_name in ["structural", "data"]:
            raise ValueError((
                "ThresholdedScalarImage cannot be used in this context, as it"
                " requires later derivatives to be calculated"))

        def _helper(structural_imap, data_imap, tissue_imap):
            scalar_path = get_tp(
                self.scalar,
                structural_imap,
                data_imap,
                tissue_imap,
            )

            img = nib.load(scalar_path)
            img_data = img.get_fdata()

            thresh_data, meta = self.apply_conditions(
                img_data, scalar_path
            )

            return nib.Nifti1Image(
                thresh_data.astype(np.float32),
                img.affine
            ), meta

        if task_name == "tissue":
            def image_getter(structural_imap, data_imap):
                return _helper(structural_imap, data_imap, None)
        else:
            def image_getter(data_imap, structural_imap, tissue_imap):
                return _helper(structural_imap, data_imap, tissue_imap)

        return image_getter


class ThresholdedScalarImage(ThresholdMixin, ScalarImage):
    """
    Define an image based on thresholding a scalar image.
    Note that this should not be used to directly make a seed image.
    In those cases, consider thresholding after
    interpolation, as in the example for ScalarImage.

    Parameters
    ----------
    scalar : str
        Scalar to threshold.
        Can be one of "dti_fa", "dti_md", "dki_fa", "dki_md".
    lower_bound : float, optional
        Lower bound to generate boolean image from data in the file.
        If None, no lower bound is applied.
        Default: None.
    upper_bound : float, optional
        Upper bound to generate boolean image from data in the file.
        If None, no upper bound is applied.
        Default: None.
    as_percentage : bool, optional
        Interpret lower_bound and upper_bound as percentages of the
        total non-nan voxels in the image to include (between 0 and 100),
        instead of as a threshold on the values themselves.
        Default: False
    combine : str, optional
        How to combine the boolean images generated by lower_bound
        and upper_bound. If "and", they will be and'd together.
        If "or", they will be or'd.
        Default: "and"

    Examples
    --------
    seed_image = ThresholdedScalarImage(
        "dti_fa",
        lower_bound=0.2)
    api.GroupAFQ(tracking_params={"seed_image": seed_image})
    """

    def __init__(self, scalar, lower_bound=None, upper_bound=None,
                 as_percentage=False, combine="and"):
        CombineImageMixin.__init__(self, combine)
        ThresholdMixin.__init__(self, combine, lower_bound,
                                upper_bound, as_percentage)
        self.scalar = scalar


class PVEImage(ImageDefinition):
    """
    Define an CSF/GM/WM PVE image from a single file.

    Parameters
    ----------
    pve_order : str
        Order of PVEs in file. Should be a list of three strings,
        each of "csf", "gm", or "wm", indicating which volume
        in the file corresponds to which tissue type.
    path : str, optional
        path to file to get image from. Use this or suffix.
        Default: None
    suffix : str, optional
        suffix to pass to bids_layout.get() to identify the file.
        Default: None
    filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the file.
        Default: {}
    resample : bool, optional
        Whether to resample the image to the DWI data.
        Default: True
    Examples
    --------
    pve = PVEImage(
        pve_order=["csf", "gm", "wm"],
        suffix="pve")
    api.GroupAFQ(..., pve=pve)
    """

    def __init__(self,
                 pve_order=["csf", "gm", "wm"],
                 path=None,
                 suffix=None,
                 filters={},
                 resample=True):
        self.pve_order = pve_order
        super().__init__(
            path=path,
            suffix=suffix,
            filters=filters,
            resample=resample
        )
    
    def get_image_getter(self, task_name):
        def image_getter(t1_file):
            # Load data
            image_file, image_data_orig, image_affine = \
                self.get_path_data_affine(t1_file)

            # Apply any conditions on the data
            image_data, meta = self.apply_conditions(
                image_data_orig, image_file)

            if self.resample:
                resample_to_img = nib.load(t1_file)
                meta["resampled"] = t1_file
                image_data, _ = _resample_image(
                    image_data,
                    resample_to_img.get_fdata(),
                    image_affine,
                    resample_to_img.affine)
                image_affine = resample_to_img.affine
            else:
                meta["resampled"] = False

            pve_data = []
            for tissue_type in ["csf", "gm", "wm"]:
                pve_index = self.pve_order.index(tissue_type)
                pve_data.append(image_data[..., pve_index])
            return nib.Nifti1Image(
                np.asarray(pve_data).astype(np.float32),
                image_affine), meta

        return image_getter


class PVEImages(ImageDefinition):
    """
    Define a CSF/GM/WM PVE image from three separate files.

    Parameters
    ----------
    CSF_probseg : ImageFile
        Corticospinal fluid segmentation file.
    GM_probseg : ImageFile
        Gray matter segmentation file.
    WM_probseg : ImageFile
        White matter segmentation file.

    Examples
    --------
    pve = PVEImages(
        afm.ImageFile(suffix="CSFprobseg"),
        afm.ImageFile(suffix="GMprobseg"),
        afm.ImageFile(suffix="WMprobseg"))
    api.GroupAFQ(..., pve=pve)
    """

    def __init__(self, CSF_probseg, GM_probseg, WM_probseg):
        self.probsegs = (CSF_probseg, GM_probseg, WM_probseg)

    def find_path(self, bids_layout, from_path,
                  subject, session, required=True):
        if required == False:
            raise ValueError(
                "PVEImage cannot be used in this context")
        for probseg in self.probsegs:
            if hasattr(probseg, 'find_path'):
                probseg.find_path(
                    bids_layout, from_path, subject, session,
                    required=required)

    def get_name(self):
        return "pves"

    def get_image_getter(self, task_name):
        self.probseg_funcs = [
            probseg.get_image_getter(task_name) for probseg in self.probsegs]
        def _image_getter_helper(pve_csf, pve_gm, pve_wm, dwi_affine):
            pve_data = np.stack(
                [nib.load(p).get_fdata() for p in (pve_csf, pve_gm, pve_wm)],
                axis=-1)
            return nib.Nifti1Image(
                np.asarray(pve_data).astype(np.float32),
                dwi_affine), {
                    "CSF PVE": pve_csf,
                    "GM PVE": pve_gm,
                    "WM PVE": pve_wm}
        return _image_getter_helper


class TemplateImage(ImageDefinition):
    """
    Define a scalar based on a template.
    This template will be transformed into subject space before use.

    Parameters
    ----------
    path : str
        path to the template.

    Examples
    --------
    my_scalar = TemplateImage(
        "path/to/my_scalar_in_MNI.nii.gz")
    api.GroupAFQ(scalars=["dti_fa", "dti_md", my_scalar])
    """

    def __init__(self, path):
        self.path = path

    def get_name(self):
        return name_from_path(self.path)

    def get_image_getter(self, task_name):
        def _image_getter_helper(mapping, reg_template, reg_subject):
            img = nib.load(self.path)
            img_data = resample(
                img.get_fdata(),
                reg_template,
                moving_affine=img.affine,
                static_affine=reg_template.affine).get_fdata()

            scalar_data = mapping.transform_inverse(
                img_data, interpolation='nearest')
            return nib.Nifti1Image(
                scalar_data.astype(np.float32),
                reg_subject.affine), dict(source=self.path)

        if task_name == "data":
            raise ValueError((
                "TemplateImage cannot be used in this context, as they"
                "require later derivatives to be calculated"))
        elif task_name == "mapping":
            def image_getter(mapping, reg_subject, data_imap):
                return _image_getter_helper(
                    mapping, data_imap["reg_template"],
                    reg_subject)
        else:
            def image_getter(mapping_imap, data_imap):
                return _image_getter_helper(
                    mapping_imap["mapping"], data_imap["reg_template"],
                    mapping_imap["reg_subject"])
        return image_getter

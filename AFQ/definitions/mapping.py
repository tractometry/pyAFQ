import nibabel as nib
import numpy as np
import logging
from time import time
import os.path as op

from AFQ.definitions.utils import Definition, find_file
from dipy.align import syn_registration, affine_registration
import AFQ.registration as reg
from AFQ.utils.path import write_json
from AFQ.tasks.utils import get_fname

from dipy.align.imaffine import AffineMap

try:
    from fsl.data.image import Image
    from fsl.transform.fnirt import readFnirt
    from fsl.transform.nonlinear import applyDeformation
    has_fslpy = True
except ModuleNotFoundError:
    has_fslpy = False

__all__ = ["FnirtMap", "SynMap", "SlrMap", "AffMap", "IdentityMap"]


logger = logging.getLogger('AFQ')


# For map defintions, get_for_subses should return only the mapping
# Where the mapping has transform and transform_inverse functions
# which each accept data, **kwargs


class FnirtMap(Definition):
    """
    Use an existing FNIRT map. Expects a warp file
    and an image file for each subject / session; image file
    is used as src space for warp.

    Parameters
    ----------
    warp_path : str, optional
        path to file to get warp from. Use this or warp_suffix.
        Default: None
    space_path : str, optional
        path to file to get warp from. Use this or space_suffix.
        Default: None
    warp_suffix : str, optional
        suffix to pass to bids_layout.get() to identify the warp file.
        Default: None
    space_suffix : str, optional
        suffix to pass to bids_layout.get() to identify the space file.
        Default: None
    warp_filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the warp file.
        Default: {}
    space_filters : str, optional
        Additional filters to pass to bids_layout.get() to identify
        the space file.
        Default: {}

    Notes
    -----
    If you have an existing mapping calculated using Fnirt,
    you can pass bids filters to :class:`AFQ.definitions.mapping.FnirtMap`
    and pyAFQ will find and use that mapping.

    Examples
    --------
    fnirt_map = FnirtMap(
        warp_suffix="warp",
        space_suffix="MNI",
        warp_filters={"scope": "TBSS"},
        space_filters={"scope": "TBSS"})
    api.GroupAFQ(mapping=fnirt_map)
    """

    def __init__(self, warp_path=None, space_path=None,
                 warp_suffix=None, space_suffix=None,
                 warp_filters={}, space_filters={}):
        if not has_fslpy:
            raise ImportError(
                "Please install fslpy if you want to use FnirtMap")
        if warp_path is None and warp_suffix is None:
            raise ValueError((
                "One of `warp_path` or `warp_suffix` should be set "
                "to a value other than None."))
        if space_path is None and space_suffix is None:
            raise ValueError(
                "One of space_path or space_suffix must not be None.")
        if warp_path is not None and space_path is None\
                or space_path is not None and warp_path is None:
            raise ValueError((
                "If passing a value for `warp_path`, "
                "you must also pass a value for `space_path`"))
        if warp_path is not None:
            self._from_path = True
            self.fnames = (warp_path, space_path)
        else:
            self._from_path = False
            self.warp_suffix = warp_suffix
            self.warp_filters = warp_filters
            self.space_suffix = space_suffix
            self.space_filters = space_filters
            self.fnames = {}

    def find_path(self, bids_layout, from_path,
                  subject, session, required=True):
        if self._from_path:
            return
        if session not in self.fnames:
            self.fnames[session] = {}

        nearest_warp = find_file(
            bids_layout, from_path, self.warp_filters, self.warp_suffix,
            session, subject, required=required)

        nearest_space = find_file(
            bids_layout, from_path, self.space_filters, self.space_suffix,
            session, subject, required=required)

        self.fnames[from_path] = (nearest_warp, nearest_space)

    def get_for_subses(self, base_fname, dwi, dwi_data_file, reg_subject,
                       reg_template):
        if self._from_path:
            nearest_warp, nearest_space = self.fnames
        else:
            nearest_warp, nearest_space = self.fnames[dwi_data_file]

        our_templ = reg_template
        subj = Image(dwi)
        their_templ = Image(nearest_space)
        warp = readFnirt(nearest_warp, their_templ, subj)

        return ConformedFnirtMapping(warp, our_templ.affine)


class ConformedFnirtMapping():
    """
        ConformedFnirtMapping which matches the generic mapping API.
    """

    def __init__(self, warp, ref_affine):
        self.ref_affine = ref_affine
        self.warp = warp

    def transform_inverse(self, data, **kwargs):
        data_img = Image(nib.Nifti1Image(
            data.astype(np.float32), self.ref_affine))
        return np.asarray(applyDeformation(data_img, self.warp).data)

    def transform_inverse_pts(self, pts):
        # This should only be used for curvature analysis,
        # Because I think the results still need to be shifted
        pts = nib.affines.apply_affine(
            self.warp.src.getAffine('voxel', 'world'), pts)
        pts = nib.affines.apply_affine(
            np.linalg.inv(self.ref_affine), pts)
        pts = self.warp.transform(pts, 'fsl', "world")
        return pts

    def transform(self, data, **kwargs):
        raise NotImplementedError(
            "Fnirt based mappings can currently"
            + " only transform from template to subject space")


class IdentityMap(Definition):
    """
    Does not perform any transformations from MNI to subject where
    pyAFQ normally would.

    Examples
    --------
    my_example_mapping = IdentityMap()
    api.GroupAFQ(mapping=my_example_mapping)
    """

    def __init__(self):
        pass

    def get_for_subses(self, base_fname, dwi, dwi_data_file, reg_subject,
                       reg_template):
        return ConformedAffineMapping(
            np.identity(4),
            domain_grid_shape=reg.reduce_shape(
                reg_subject.shape),
            domain_grid2world=reg_subject.affine,
            codomain_grid_shape=reg.reduce_shape(
                reg_template.shape),
            codomain_grid2world=reg_template.affine)


class GeneratedMapMixin(object):
    """
    Helper Class
    Useful for maps that are generated by pyAFQ
    """

    def get_fnames(self, extension, base_fname):
        mapping_file = get_fname(
            base_fname,
            '_desc-mapping_from-DWI_to-MNI_xform')
        meta_fname = f'{mapping_file}.json'
        mapping_file = mapping_file + extension
        return mapping_file, meta_fname

    def prealign(self, base_fname, reg_subject, reg_template, save=True):
        prealign_file_desc = "_desc-prealign_from-DWI_to-MNI_xform"
        prealign_file = get_fname(
            base_fname, f'{prealign_file_desc}.npy')
        if not op.exists(prealign_file):
            start_time = time()
            _, aff = affine_registration(
                reg_subject,
                reg_template,
                **self.affine_kwargs)
            meta = dict(
                type="rigid",
                dependent="dwi",
                timing=time() - start_time)
            if not save:
                return aff
            logger.info(f"Saving {prealign_file}")
            np.save(prealign_file, aff)
            meta_fname = get_fname(
                base_fname, f'{prealign_file_desc}.json')
            write_json(meta_fname, meta)
        return prealign_file if save else np.load(prealign_file)

    def get_for_subses(self, base_fname, dwi, dwi_data_file, reg_subject,
                       reg_template, subject_sls=None, template_sls=None):
        mapping_file, meta_fname = self.get_fnames(
            self.extension, base_fname)

        if self.use_prealign:
            reg_prealign = np.load(self.prealign(
                base_fname, reg_subject, reg_template))
        else:
            reg_prealign = None
        if not op.exists(mapping_file):
            start_time = time()
            mapping = self.gen_mapping(
                base_fname, reg_subject, reg_template,
                subject_sls, template_sls,
                reg_prealign)
            total_time = time() - start_time

            logger.info(f"Saving {mapping_file}")
            reg.write_mapping(mapping, mapping_file)
            meta = dict(
                type="displacementfield",
                timing=total_time)
            if subject_sls is None:
                meta["dependent"] = "dwi"
            else:
                meta["dependent"] = "trk"
            if isinstance(reg_subject, str):
                meta["reg_subject"] = reg_subject
            if isinstance(reg_template, str):
                meta["reg_template"] = reg_template
            write_json(meta_fname, meta)
        reg_prealign_inv = np.linalg.inv(reg_prealign) if self.use_prealign\
            else None
        mapping = reg.read_mapping(
            mapping_file,
            dwi,
            reg_template,
            prealign=reg_prealign_inv)
        return mapping


class SynMap(GeneratedMapMixin, Definition):
    """
    Calculate a Syn registration for each subject/session
    using reg_subject and reg_template.

    Parameters
    ----------
    use_prealign : bool
        Whether to perform a linear pre-registration.
        Default: True
    affine_kwargs : dictionary, optional
        Parameters to pass to affine_registration
        in dipy.align, which does the linear pre-alignment.
        Only used if use_prealign is True.
        Default: {}
    syn_kwargs : dictionary, optional
        Parameters to pass to syn_registration
        in dipy.align, which does the SyN alignment.
        Default: {}

    Notes
    -----
    The default mapping class is to
    use Symmetric Diffeomorphic Image Registration (SyN).
    This is done with an optional linear pre-alignment by default.
    The parameters of the pre-alginment can be specified when
    initializing the SynMap.

    Examples
    --------
    api.GroupAFQ(mapping=SynMap())
    """

    def __init__(self, use_prealign=True, affine_kwargs={}, syn_kwargs={}):
        self.use_prealign = use_prealign
        self.affine_kwargs = affine_kwargs
        self.syn_kwargs = syn_kwargs
        self.extension = ".nii.gz"

    def gen_mapping(self, base_fname, reg_subject, reg_template,
                    subject_sls, template_sls,
                    reg_prealign):
        _, mapping = syn_registration(
            reg_subject.get_fdata(),
            reg_template.get_fdata(),
            moving_affine=reg_subject.affine,
            static_affine=reg_template.affine,
            prealign=reg_prealign,
            **self.syn_kwargs)
        if self.use_prealign:
            mapping.codomain_world2grid = np.linalg.inv(reg_prealign)
        return mapping


class SlrMap(GeneratedMapMixin, Definition):
    """
    Calculate a SLR registration for each subject/session
    using reg_subject and reg_template.

    Parameters
    ----------
    slr_kwargs : dictionary, optional
        Parameters to pass to whole_brain_slr
        in dipy, which does the SLR alignment.
        Default: {}

    Notes
    -----
    Use this class to tell pyAFQ to use
    Streamline-based Linear Registration (SLR) 
    for registration. Note that the reg_template and reg_subject
    parameters passed to :class:`AFQ.api.group.GroupAFQ` should
    be streamlines when using this registration.

    Examples
    --------
    api.GroupAFQ(mapping=SlrMap())
    """

    def __init__(self, slr_kwargs={}):
        self.slr_kwargs = {}
        self.use_prealign = False
        self.extension = ".npy"

    def gen_mapping(self, base_fname, reg_template, reg_subject,
                    subject_sls, template_sls, reg_prealign):
        return reg.slr_registration(
            subject_sls, template_sls,
            moving_affine=reg_subject.affine,
            moving_shape=reg_subject.shape,
            static_affine=reg_template.affine,
            static_shape=reg_template.shape,
            **self.slr_kwargs)


class AffMap(GeneratedMapMixin, Definition):
    """
    Calculate an affine registration for each subject/session
    using reg_subject and reg_template.

    Parameters
    ----------
    affine_kwargs : dictionary, optional
        Parameters to pass to affine_registration
        in dipy.align, which does the linear pre-alignment.
        Default: {}

    Notes
    -----
    This will only perform a linear alignment for registration.

    Examples
    --------
    api.GroupAFQ(mapping=AffMap())
    """

    def __init__(self, affine_kwargs={}):
        self.use_prealign = False
        self.affine_kwargs = affine_kwargs
        self.extension = ".npy"

    def gen_mapping(self, base_fname, reg_subject, reg_template,
                    subject_sls, template_sls,
                    reg_prealign):
        return ConformedAffineMapping(
            np.linalg.inv(self.prealign(
                base_fname, reg_subject, reg_template, save=False)),
            domain_grid_shape=reg.reduce_shape(
                reg_subject.shape),
            domain_grid2world=reg_subject.affine,
            codomain_grid_shape=reg.reduce_shape(
                reg_template.shape),
            codomain_grid2world=reg_template.affine)


class ConformedAffineMapping(AffineMap):
    """
    Modifies AffineMap API to match DiffeomorphicMap API.
    Important for SLR maps API to be indistinguishable from SYN maps API.
    """

    def transform(self, *args, interpolation='linear', **kwargs):
        kwargs['interp'] = interpolation
        return super().transform_inverse(*args, **kwargs)

    def transform_inverse(self, *args, interpolation='linear', **kwargs):
        kwargs['interp'] = interpolation
        return super().transform(*args, **kwargs)

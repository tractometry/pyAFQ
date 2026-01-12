import logging

import immlib
import nibabel as nib
import numpy as np
from dipy.align import resample
from dipy.core.gradients import unique_bvals_tolerance
from dipy.data import get_sphere
from dipy.reconst.mcsd import (
    mask_for_response_msmt,
    multi_shell_fiber_response,
    response_from_mask_msmt,
)

from AFQ.definitions.image import PVEImage, PVEImages
from AFQ.models.asym_filtering import (
    compute_asymmetry_index,
    compute_nufid_asym,
    compute_odd_power_map,
    unified_filtering,
)
from AFQ.models.msmt import MultiShellDeconvModel
from AFQ.models.QBallTP import anisotropic_power
from AFQ.models.wmgm_interface import fit_wm_gm_interface
from AFQ.nn.brainchop import pve_from_subcortex
from AFQ.nn.multiaxial import extract_pve
from AFQ.nn.synthseg import pve_from_synthseg
from AFQ.tasks.decorators import as_file, as_img
from AFQ.tasks.utils import with_name

logger = logging.getLogger("AFQ")


@immlib.calc("wm_gm_interface")
@as_file(suffix="_desc-wmgmi_mask.nii.gz")
def wm_gm_interface(pve_internal, data_imap):
    """
    full path to a nifti file containing the white
    matter/gray matter interface
    """
    PVE_img = nib.load(pve_internal)
    b0_img = nib.load(data_imap["b0"])

    wmgmi_img = fit_wm_gm_interface(PVE_img, b0_img)

    return wmgmi_img, dict(FromPVE=pve_internal)


@immlib.calc("pve_internal")
@as_file(suffix="_desc-pve_probseg.nii.gz")
def pve_internal(structural_imap, pve="synthseg"):
    """
    WM+GM+CSF segmentation

    Parameters
    ----------
    pve : str or PVEImage, optional
        Method to use for PVE estimation.
        Can be a string defining a built-in method from neural networks,
        or a Definition object to import the PVE.
        Importing a PVE from software like Freesurfer or FSL FAST is
        recommended if they are available.
        The built-in methods are "synthseg" or "multiaxial+brainchop".
        "synthseg" uses SynthSeg2 [1] to get the PVE.
        "multiaxial+brainchop" uses MultiAxial [2] and BrainChop [3]
        segmentations to get the PVE. Note this requires downloading
        the pre-trained multi-axial model which is licensed with
        Creative Commons Attribution-NonCommercial-ShareAlike 4.0
        International.
        Default: "synthseg"
    """
    if isinstance(pve, str):
        if pve == "synthseg":
            logger.warning(
                (
                    "Using SynthSeg2 for PVE estimation. "
                    "This may use considerable memory resources. "
                )
            )

            synthseg_seg = nib.load(structural_imap["synthseg_model"])
            PVE = pve_from_synthseg(synthseg_seg.get_fdata())

            return nib.Nifti1Image(PVE, synthseg_seg.affine), dict(
                SynthsegParcellation=structural_imap["synthseg_model"],
                labels=["csf", "gm", "wm"],
            )
        elif pve == "multiaxial+brainchop":
            logger.warning(
                (
                    "Using MultiAxial+BrainChop for PVE estimation. "
                    "MultiAxial requires downloading "
                    "the pre-trained multi-axial model which is licensed with "
                    "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 "
                    "International."
                )
            )
            t1_subcortex_img = nib.load(structural_imap["t1_subcortex"])
            mx_model_img = nib.load(structural_imap["mx_model"])

            PVE_brainchop = pve_from_subcortex(t1_subcortex_img.get_fdata()).astype(
                np.float32
            )
            PVE_multiaxial = extract_pve(mx_model_img).get_fdata().astype(np.float32)

            # Use predictions from both to get final estimates
            PVE = (PVE_brainchop + PVE_multiaxial) / 2

            return nib.Nifti1Image(PVE, t1_subcortex_img.affine), dict(
                SubCortexParcellation=structural_imap["t1_subcortex"],
                MultiAxialSegmentation=structural_imap["mx_model"],
                labels=["csf", "gm", "wm"],
            )

    raise ValueError(
        "pve must be a PVEImage, PVEImages, 'synthseg', or 'multiaxial+brainchop'"
    )


@immlib.calc("msmtcsd_params")
@as_file(suffix="_model-msmtcsd_param-fod_dwimap.nii.gz", subfolder="models")
@as_img
def msmt_params(data_imap, pve_internal, msmt_sh_order=8, msmt_fa_thr=0.7):
    """
    full path to a nifti file containing
    parameters for the MSMT CSD fit

    Parameters
    ----------
    msmt_sh_order : int, optional.
        Spherical harmonic order to use for the MSMT CSD fit.
        Default: 8
    msmt_fa_thr : float, optional.
        The threshold on the FA used to calculate the multi shell auto
        response. Can be useful to reduce for baby subjects.
        Default: 0.7

    References
    ----------
    .. [1] B. Jeurissen, J.-D. Tournier, T. Dhollander, A. Connelly,
            and J. Sijbers. Multi-tissue constrained spherical
            deconvolution for improved analysis of multi-shell diffusion
            MRI data. NeuroImage, 103 (2014), pp. 411–426
    """
    mask = nib.load(data_imap["brain_mask"]).get_fdata()

    pve_img = nib.load(pve_internal)
    pve_data = pve_img.get_fdata()
    csf = resample(
        pve_data[..., 0],
        data_imap["data"][..., 0],
        pve_img.affine,
        data_imap["dwi_affine"],
    ).get_fdata()
    gm = resample(
        pve_data[..., 1],
        data_imap["data"][..., 0],
        pve_img.affine,
        data_imap["dwi_affine"],
    ).get_fdata()
    wm = resample(
        pve_data[..., 2],
        data_imap["data"][..., 0],
        pve_img.affine,
        data_imap["dwi_affine"],
    ).get_fdata()

    mask_wm, mask_gm, mask_csf = mask_for_response_msmt(
        data_imap["gtab"],
        data_imap["data"],
        roi_radii=10,
        wm_fa_thr=msmt_fa_thr,
        gm_fa_thr=0.3,
        csf_fa_thr=0.15,
        gm_md_thr=0.001,
        csf_md_thr=0.0032,
    )
    mask_wm *= wm > 0.5
    mask_gm *= gm > 0.5
    mask_csf *= csf > 0.5
    response_wm, response_gm, response_csf = response_from_mask_msmt(
        data_imap["gtab"], data_imap["data"], mask_wm, mask_gm, mask_csf
    )
    ubvals = unique_bvals_tolerance(data_imap["gtab"].bvals)
    response_mcsd = multi_shell_fiber_response(
        msmt_sh_order, ubvals, response_wm, response_gm, response_csf
    )

    mcsd_model = MultiShellDeconvModel(data_imap["gtab"], response_mcsd)
    logger.info("Fitting Multi-Shell CSD model...")
    mcsd_fit = mcsd_model.fit(data_imap["data"], mask, n_cpus=data_imap["n_cpus"])

    meta = dict(
        SphericalHarmonicDegree=msmt_sh_order, SphericalHarmonicBasis="DESCOTEAUX"
    )
    return mcsd_fit.shm_coeff, meta


@immlib.calc("msmt_apm")
@as_file(suffix="_model-msmtcsd_param-apm_dwimap.nii.gz", subfolder="models")
@as_img
def msmt_apm(msmtcsd_params):
    """
    full path to a nifti file containing
    the anisotropic power map
    """
    sh_coeff = nib.load(msmtcsd_params).get_fdata()
    pmap = anisotropic_power(sh_coeff)
    return pmap, dict(MSMTCSDParamsFile=msmtcsd_params)


@immlib.calc("msmt_aodf_params")
@as_file(suffix="_model-msmtcsd_param-aodf_dwimap.nii.gz", subfolder="models")
@as_img
def msmt_aodf(msmtcsd_params, data_imap):
    """
    full path to a nifti file containing
    MSMT CSD ODFs filtered by unified filtering [1]

    References
    ----------
    [1] Poirier and Descoteaux, 2024, "A Unified Filtering Method for
        Estimating Asymmetric Orientation Distribution Functions",
        Neuroimage, https://doi.org/10.1016/j.neuroimage.2024.120516
    """
    sh_coeff = nib.load(msmtcsd_params).get_fdata()

    logger.info("Applying unified filtering to generate asymmetric MSMT CSD ODFs...")
    aodf = unified_filtering(
        sh_coeff,
        get_sphere(name="repulsion724"),
        n_threads=data_imap["n_threads"],
        low_mem=data_imap["low_mem"],
    )

    return aodf, dict(MSMTCSDParamsFile=msmtcsd_params, Sphere="repulsion724")


@immlib.calc("msmt_aodf_asi")
@as_file(suffix="_model-msmtcsd_param-asi_dwimap.nii.gz", subfolder="models")
@as_img
def msmt_aodf_asi(msmt_aodf_params, data_imap):
    """
    full path to a nifti file containing
    the MSMT CSD Asymmetric Index (ASI) [1]

    References
    ----------
    [1] S. Cetin Karayumak, E. Özarslan, and G. Unal,
        "Asymmetric Orientation Distribution Functions (AODFs)
        revealing intravoxel geometry in diffusion MRI"
        Magnetic Resonance Imaging, vol. 49, pp. 145-158, Jun. 2018,
        doi: https://doi.org/10.1016/j.mri.2018.03.006.
    """

    aodf = nib.load(msmt_aodf_params).get_fdata()
    brain_mask = nib.load(data_imap["brain_mask"]).get_fdata().astype(bool)
    asi = compute_asymmetry_index(aodf, brain_mask)

    return asi, dict(MSMTCSDParamsFile=msmt_aodf_params)


@immlib.calc("msmt_aodf_opm")
@as_file(suffix="_model-msmtcsd_param-opm_dwimap.nii.gz", subfolder="models")
@as_img
def msmt_aodf_opm(msmt_aodf_params, data_imap):
    """
    full path to a nifti file containing
    the MSMT CSD odd-power map [1]

    References
    ----------
    [1] C. Poirier, E. St-Onge, and M. Descoteaux,
        "Investigating the Occurrence of Asymmetric Patterns in
        White Matter Fiber Orientation Distribution Functions"
        [Abstract], In: Proc. Intl. Soc. Mag. Reson. Med. 29 (2021),
        2021 May 15-20, Vancouver, BC, Abstract number 0865.
    """

    aodf = nib.load(msmt_aodf_params).get_fdata()
    brain_mask = nib.load(data_imap["brain_mask"]).get_fdata().astype(bool)
    opm = compute_odd_power_map(aodf, brain_mask)

    return opm, dict(MSMTCSDParamsFile=msmt_aodf_params)


@immlib.calc("msmt_aodf_nufid")
@as_file(suffix="_model-msmtcsd_param-nufid_dwimap.nii.gz", subfolder="models")
@as_img
def msmt_aodf_nufid(msmt_aodf_params, data_imap, pve_internal):
    """
    full path to a nifti file containing
    the MSMT CSD Number of fiber directions (nufid) map [1]

    References
    ----------
    [1] C. Poirier and M. Descoteaux,
        "Filtering Methods for Asymmetric ODFs:
        Where and How Asymmetry Occurs in the White Matter."
        bioRxiv. 2022 Jan 1; 2022.12.18.520881.
        doi: https://doi.org/10.1101/2022.12.18.520881
    """
    pve_img = nib.load(pve_internal)
    pve_data = pve_img.get_fdata()

    aodf_img = nib.load(msmt_aodf_params)
    aodf = aodf_img.get_fdata()

    csf = resample(
        pve_data[..., 0], aodf[..., 0], pve_img.affine, aodf_img.affine
    ).get_fdata()

    # Only sphere we use for AODF currently
    sphere = get_sphere(name="repulsion724")

    brain_mask = nib.load(data_imap["brain_mask"]).get_fdata().astype(bool)

    logger.info("Number of fiber directions (nufid) map from AODF...")
    nufid = compute_nufid_asym(aodf, sphere, csf, brain_mask)

    return nufid, dict(MSMTCSDParamsFile=msmt_aodf_params, PVE=pve_internal)


@immlib.calc("csd_aodf_nufid")
@as_file(suffix="_model-csd_param-nufid_dwimap.nii.gz", subfolder="models")
@as_img
def csd_aodf_nufid(data_imap, pve_internal):
    """
    full path to a nifti file containing
    the CSD Number of fiber directions (nufid) map [1]

    References
    ----------
    [1] C. Poirier and M. Descoteaux,
        "Filtering Methods for Asymmetric ODFs:
        Where and How Asymmetry Occurs in the White Matter."
        bioRxiv. 2022 Jan 1; 2022.12.18.520881.
        doi: https://doi.org/10.1101/2022.12.18.520881
    """
    pve_img = nib.load(pve_internal)
    pve_data = pve_img.get_fdata()

    aodf_img = nib.load(data_imap["csd_aodf_params"])
    aodf = aodf_img.get_fdata()

    csf = resample(
        pve_data[..., 0], aodf[..., 0], pve_img.affine, aodf_img.affine
    ).get_fdata()

    # Only sphere we use for AODF currently
    sphere = get_sphere(name="repulsion724")

    brain_mask = nib.load(data_imap["brain_mask"]).get_fdata().astype(bool)

    logger.info("Number of fiber directions (nufid) map from AODF...")
    nufid = compute_nufid_asym(aodf, sphere, csf, brain_mask)

    return nufid, dict(CSDParamsFile=data_imap["csd_aodf_params"], PVE=pve_internal)


def get_tissue_plan(kwargs):
    tissue_tasks = with_name(
        [
            pve_internal,
            wm_gm_interface,
            msmt_params,
            msmt_apm,
            msmt_aodf,
            msmt_aodf_asi,
            msmt_aodf_opm,
            msmt_aodf_nufid,
            csd_aodf_nufid,
        ]
    )

    pve = kwargs.get("pve", None)
    if isinstance(pve, PVEImages):
        probseg_func = pve.get_image_getter("tissue")
        tissue_tasks["csf_res"] = immlib.calc("pve_csf")(
            as_file("_desc-csf_probseg.nii.gz", subfolder="models")(
                pve.probseg_funcs[0]
            )
        )
        tissue_tasks["gm_res"] = immlib.calc("pve_gm")(
            as_file("_desc-gm_probseg.nii.gz", subfolder="models")(pve.probseg_funcs[1])
        )
        tissue_tasks["wm_res"] = immlib.calc("pve_wm")(
            as_file("_desc-wm_probseg.nii.gz", subfolder="models")(pve.probseg_funcs[2])
        )
        tissue_tasks["pve_internal_res"] = immlib.calc("pve_internal")(
            as_file("_desc-pve_probseg.nii.gz")(probseg_func)
        )
    elif isinstance(pve, PVEImage):
        tissue_tasks["pve_internal_res"] = immlib.calc("pve_internal")(
            as_file("_desc-pve_probseg.nii.gz")(pve.get_image_getter("tissue"))
        )
    else:
        logger.warning(
            "It is recommended to provide CSF/GM/WM "
            "segmentations using PVEImage or PVEImages "
            "in AFQ.definitions.image. Otherwise, "
            "SynthSeg2 will be used"
        )

    return immlib.plan(**tissue_tasks)

import immlib
import nibabel as nib
import logging

from AFQ.tasks.utils import with_name
from AFQ.tasks.decorators import as_file

from AFQ.nn.brainchop import run_brainchop
from AFQ.nn.multiaxial import run_multiaxial
from AFQ.nn.synthseg import run_synthseg


logger = logging.getLogger('AFQ')


@immlib.calc("synthseg_model")
@as_file(suffix='_model-synthseg2_probseg.nii.gz',
         subfolder="nn")
def synthseg_model(t1_masked):
    """
    full path to the synthseg2 model segmentations

    References
    ----------
    [1] Billot, Benjamin, et al. "Robust machine learning segmentation
        for large-scale analysis of heterogeneous clinical brain MRI
        datasets." Proceedings of the National Academy of Sciences 120.9
        (2023): e2216399120.
    [2] Billot, Benjamin, et al. "SynthSeg: Segmentation of brain MRI scans
        of any contrast and resolution without retraining." Medical image
        analysis 86 (2023): 102789.
    """
    t1_img = nib.load(t1_masked)
    predictions = run_synthseg(t1_img, "synthseg2")
    return predictions, dict(T1w=t1_masked)


@immlib.calc("mx_model")
@as_file(suffix='_model-multiaxial_probseg.nii.gz',
         subfolder="nn")
def mx_model(t1_masked):
    """
    full path to the multi-axial model for brain extraction
    outputs

    References
    ----------
    [1] Birnbaum, Andrew M., et al. "Full-head segmentation of MRI
        with abnormal brain anatomy: model and data release." Journal of
        Medical Imaging 12.5 (2025): 054001-054001.
    """
    t1_img = nib.load(t1_masked)
    predictions = run_multiaxial(t1_img)
    return predictions, dict(T1w=t1_masked)


@immlib.calc("t1w_brain_mask")
@as_file(suffix='_desc-T1w_mask.nii.gz')
def t1w_brain_mask(t1_file):
    """
    full path to a nifti file containing brain mask from T1w image

    References
    ----------
    [1] Masoud, M., Hu, F., & Plis, S. (2023). Brainchop: In-browser MRI
        volumetric segmentation and rendering. Journal of Open Source
        Software, 8(83), 5098.
        https://doi.org/10.21105/joss.05098
    """
    return run_brainchop(nib.load(t1_file), "mindgrab"), dict(
        T1w=t1_file,
        model="mindgrab")


@immlib.calc("t1_masked")
@as_file(suffix='_desc-masked_T1w.nii.gz')
def t1_masked(t1_file, t1w_brain_mask):
    """
    full path to a nifti file containing the T1w masked
    """
    t1_img = nib.load(t1_file)
    t1_data = t1_img.get_fdata()
    t1_mask = nib.load(t1w_brain_mask)
    t1_data[t1_mask.get_fdata() == 0] = 0
    t1_img_masked = nib.Nifti1Image(
        t1_data, t1_img.affine)
    return t1_img_masked, dict(
        T1w=t1_file,
        BrainMask=t1w_brain_mask)


@immlib.calc("t1_subcortex")
@as_file(suffix='_desc-subcortex_probseg.nii.gz', subfolder="nn")
def t1_subcortex(t1_masked):
    """
    full path to a nifti file containing segmentation of
    subcortical structures from T1w image using Brainchop

    References
    ----------
    [1] Masoud, M., Hu, F., & Plis, S. (2023). Brainchop: In-browser MRI
        volumetric segmentation and rendering. Journal of Open Source
        Software, 8(83), 5098.
        https://doi.org/10.21105/joss.05098
    """
    t1_img_masked = nib.load(t1_masked)

    subcortical_img = run_brainchop(
        t1_img_masked, "subcortical")

    meta = dict(
        T1w=t1_masked,
        model="subcortical",
        labels=[
            "Unknown", "Cerebral-White-Matter", "Cerebral-Cortex",
            "Lateral-Ventricle", "Inferior-Lateral-Ventricle",
            "Cerebellum-White-Matter", "Cerebellum-Cortex",
            "Thalamus", "Caudate", "Putamen", "Pallidum",
            "3rd-Ventricle", "4th-Ventricle", "Brain-Stem",
            "Hippocampus", "Amygdala", "Accumbens-area", "VentralDC"])

    return subcortical_img, meta


def get_structural_plan(kwargs):
    structural_tasks = with_name([
        mx_model, synthseg_model,
        t1w_brain_mask, t1_subcortex, t1_masked])
    
    return immlib.plan(**structural_tasks)

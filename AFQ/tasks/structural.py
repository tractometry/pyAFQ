import logging
import multiprocessing

import immlib
import nibabel as nib
from numba import get_num_threads

from AFQ.definitions.utils import Definition
from AFQ.nn.brainchop import run_brainchop
from AFQ.nn.multiaxial import run_multiaxial
from AFQ.nn.synthseg import run_synthseg
from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import check_onnxruntime, with_name

logger = logging.getLogger("AFQ")


@immlib.calc("n_cpus", "n_threads", "low_mem")
def configure_ncpus_nthreads(ray_n_cpus=None, numba_n_threads=None, low_memory=False):
    """
    Configure the number of CPUs to use for parallel processing with Ray,
    the number of threads to use for Numba,
    and whether to use low-memory versions of algorithms
    where available

    Parameters
    ----------
    ray_n_cpus : int, optional
        The number of CPUs to use for parallel processing with Ray.
        If None, uses the number of available CPUs minus one.
        Tractography, Recognition, and MSMT use Ray.
        Default: None
    numba_n_threads : int, optional
        The number of threads to use for Numba.
        If None, uses the number of available CPUs minus one,
        but with a maximum of 16.
        ASYM fit uses Numba.
        Default: None
    low_memory : bool, optional
        Whether to use low-memory versions of algorithms
        where available.
        Default: False
    """
    if ray_n_cpus is None:
        ray_n_cpus = max(multiprocessing.cpu_count() - 1, 1)
    if numba_n_threads is None:
        numba_n_threads = min(max(get_num_threads() - 1, 1), 16)

    return ray_n_cpus, numba_n_threads, low_memory


@immlib.calc("onnx_kwargs")
def onnx_kwargs(low_mem, onnx_execution_provider="CPUExecutionProvider"):
    """
    The execution provider to use for onnx models

    Parameters
    ----------
    onnx_execution_provider : str, optional
        The execution provider to use for onnx models.
        By default this is set to CPUExecutionProvider
        which should work on all systems. If you have a
        compatible GPU and the appropriate onnxruntime installed
        you can set this to "CUDAExecutionProvider" or
        "OpenVINOExecutionProvider" for potentially faster
        inference.
        Default: "CPUExecutionProvider"

    Returns
    -------
    str
        The ONNX execution provider to use for onnx models.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        # In this case, we can throw a more informative error
        # when the user tries to run a model
        # that requires onnxruntime
        return onnx_execution_provider
    if onnx_execution_provider not in ort.get_available_providers():
        logger.warning(
            f"{onnx_execution_provider} is not available. "
            f"Available providers are: {ort.get_available_providers()}. "
            "Falling back to CPUExecutionProvider."
        )
        onnx_execution_provider = "CPUExecutionProvider"
    options = ort.SessionOptions()
    if low_mem:
        options.add_session_config_entry("session.use_mem_arena", "0")
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    onnx_kwargs = {"providers": [onnx_execution_provider], "options": options}

    return {"onnx_kwargs": onnx_kwargs}


@immlib.calc("synthseg_model")
@as_file(suffix="_model-synthseg2_probseg.nii.gz", subfolder="nn")
def synthseg_model(t1_masked, citations, onnx_kwargs):
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
    citations.add("billot_synthseg_2023")
    citations.add("billot_robust_2023")
    ort = check_onnxruntime(
        "SynthSeg 2.0",
        "Or, provide your own segmentations using PVEImage or PVEImages.",
    )
    t1_img = nib.load(t1_masked)
    predictions = run_synthseg(ort, t1_img, "synthseg2", onnx_kwargs)
    return predictions, dict(T1w=t1_masked)


@immlib.calc("mx_model")
@as_file(suffix="_model-multiaxial_probseg.nii.gz", subfolder="nn")
def mx_model(t1_file, t1w_brain_mask, citations, onnx_kwargs):
    """
    full path to the multi-axial model for brain extraction
    outputs

    References
    ----------
    [1] Birnbaum, Andrew M., et al. "Full-head segmentation of MRI
        with abnormal brain anatomy: model and data release." Journal of
        Medical Imaging 12.5 (2025): 054001-054001.
    """
    citations.add("birnbaum2025full")
    ort = check_onnxruntime(
        "Multi-axial", "Or, provide your own segmentations using PVEImage or PVEImages."
    )
    t1_img = nib.load(t1_file)
    t1_mask = nib.load(t1w_brain_mask)
    predictions = run_multiaxial(ort, t1_img, onnx_kwargs)
    predictions = nib.Nifti1Image(
        predictions.get_fdata() * t1_mask.get_fdata(), t1_img.affine
    )
    return predictions, dict(T1w=t1_file, mask=t1w_brain_mask)


@immlib.calc("t1w_brain_mask")
@as_file(suffix="_desc-T1w_mask.nii.gz")
def t1w_brain_mask(t1_file, citations, onnx_kwargs, brain_mask_definition=None):
    """
    full path to a nifti file containing brain mask from T1w image

    Parameters
    ----------
    brain_mask_definition : instance from `AFQ.definitions.image`, optional
        This will be used to create
        the brain mask, which gets applied before registration to a
        template.
        If you want no brain mask to be applied, use FullImage.
        If None, use Brainchop Mindgrab model.
        Default: None

    References
    ----------
    [1] Masoud, M., Hu, F., & Plis, S. (2023). Brainchop: In-browser MRI
        volumetric segmentation and rendering. Journal of Open Source
        Software, 8(83), 5098.
        https://doi.org/10.21105/joss.05098
    """
    # Note that any case where brain_mask_definition is not None
    # is handled in get_data_plan
    # This is just the default

    citations.add("fani2025mindgrab")

    ort = check_onnxruntime(
        "Mindgrab", "Or, provide your own brain mask using brain_mask_definition."
    )
    return run_brainchop(ort, nib.load(t1_file), "mindgrab", onnx_kwargs), dict(
        T1w=t1_file, model="mindgrab"
    )


@immlib.calc("t1_masked")
@as_file(suffix="_desc-masked_T1w.nii.gz")
def t1_masked(t1_file, t1w_brain_mask):
    """
    full path to a nifti file containing the T1w masked
    """
    t1_img = nib.load(t1_file)
    t1_data = t1_img.get_fdata()
    t1_mask = nib.load(t1w_brain_mask)
    t1_data[t1_mask.get_fdata() == 0] = 0
    t1_img_masked = nib.Nifti1Image(t1_data, t1_img.affine)
    return t1_img_masked, dict(T1w=t1_file, BrainMask=t1w_brain_mask)


@immlib.calc("t1_subcortex")
@as_file(suffix="_desc-subcortex_probseg.nii.gz", subfolder="nn")
def t1_subcortex(t1_masked, citations, onnx_kwargs):
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
    ort = check_onnxruntime(
        "Brainchop Subcortical",
        "Or, provide your own segmentations using PVEImage or PVEImages.",
    )

    citations.add("masoud2023brainchop")

    t1_img_masked = nib.load(t1_masked)

    subcortical_img = run_brainchop(ort, t1_img_masked, "subcortical", onnx_kwargs)

    meta = dict(
        T1w=t1_masked,
        model="subcortical",
        labels=[
            "Unknown",
            "Cerebral-White-Matter",
            "Cerebral-Cortex",
            "Lateral-Ventricle",
            "Inferior-Lateral-Ventricle",
            "Cerebellum-White-Matter",
            "Cerebellum-Cortex",
            "Thalamus",
            "Caudate",
            "Putamen",
            "Pallidum",
            "3rd-Ventricle",
            "4th-Ventricle",
            "Brain-Stem",
            "Hippocampus",
            "Amygdala",
            "Accumbens-area",
            "VentralDC",
        ],
    )

    return subcortical_img, meta


def get_structural_plan(kwargs):
    structural_tasks = with_name(
        [
            mx_model,
            synthseg_model,
            t1w_brain_mask,
            t1_subcortex,
            t1_masked,
            onnx_kwargs,
            configure_ncpus_nthreads,
        ]
    )

    bm_def = kwargs.get("brain_mask_definition", None)
    if bm_def is not None:
        if not isinstance(bm_def, Definition):
            raise TypeError("brain_mask_definition must be a Definition")
        structural_tasks["t1w_brain_mask_res"] = immlib.calc("t1w_brain_mask")(
            as_file(suffix=("_desc-T1w_mask.nii.gz"))(
                bm_def.get_image_getter("structural")
            )
        )

    return immlib.plan(**structural_tasks)

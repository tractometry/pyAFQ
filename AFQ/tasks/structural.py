import logging

import immlib
import nibabel as nib

from AFQ.definitions.utils import Definition
from AFQ.nn.brainchop import run_brainchop
from AFQ.nn.multiaxial import run_multiaxial
from AFQ.nn.synthseg import run_synthseg
from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import check_onnxruntime, with_name

logger = logging.getLogger("AFQ")


@immlib.calc("synthseg_model")
@as_file(suffix="_model-synthseg2_probseg.nii.gz", subfolder="nn")
def synthseg_model(t1_masked, citations):
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
    citations.update(
        {
            """
@article{billot2023robust,
  title={Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets},
  author={Billot, Benjamin and Magdamo, Colin and Cheng, You and Arnold, Steven E and Das, Sudeshna and Iglesias, Juan Eugenio},
  journal={Proceedings of the National Academy of Sciences},
  volume={120},
  number={9},
  pages={e2216399120},
  year={2023},
  publisher={National Academy of Sciences}
}""",  # noqa: E501
            """
@article{billot2023synthseg,
  title={SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining},
  author={Billot, Benjamin and Greve, Douglas N and Puonti, Oula and Thielscher, Axel and Van Leemput, Koen and Fischl, Bruce and Dalca, Adrian V and Iglesias, Juan Eugenio and others},
  journal={Medical image analysis},
  volume={86},
  pages={102789},
  year={2023},
  publisher={Elsevier}
}""",  # noqa: E501
        }
    )
    ort = check_onnxruntime(
        "SynthSeg 2.0",
        "Or, provide your own segmentations using PVEImage or PVEImages.",
    )
    t1_img = nib.load(t1_masked)
    predictions = run_synthseg(ort, t1_img, "synthseg2")
    return predictions, dict(T1w=t1_masked)


@immlib.calc("mx_model")
@as_file(suffix="_model-multiaxial_probseg.nii.gz", subfolder="nn")
def mx_model(t1_file, t1w_brain_mask, citations):
    """
    full path to the multi-axial model for brain extraction
    outputs

    References
    ----------
    [1] Birnbaum, Andrew M., et al. "Full-head segmentation of MRI
        with abnormal brain anatomy: model and data release." Journal of
        Medical Imaging 12.5 (2025): 054001-054001.
    """
    citations.add("""
@article{birnbaum2025full,
  title={Full-head segmentation of MRI with abnormal brain anatomy: model and data release},
  author={Birnbaum, Andrew M and Buchwald, Adam and Turkeltaub, Peter and Jacks, Adam and Carr, George and Kannan, Shreya and Huang, Yu and Datta, Abhisheck and Parra, Lucas C and Hirsch, Lukas A},
  journal={Journal of Medical Imaging},
  volume={12},
  number={5},
  pages={054001--054001},
  year={2025},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}""")  # noqa: E501
    ort = check_onnxruntime(
        "Multi-axial", "Or, provide your own segmentations using PVEImage or PVEImages."
    )
    t1_img = nib.load(t1_file)
    t1_mask = nib.load(t1w_brain_mask)
    predictions = run_multiaxial(ort, t1_img)
    predictions = nib.Nifti1Image(
        predictions.get_fdata() * t1_mask.get_fdata(), t1_img.affine
    )
    return predictions, dict(T1w=t1_file, mask=t1w_brain_mask)


@immlib.calc("t1w_brain_mask")
@as_file(suffix="_desc-T1w_mask.nii.gz")
def t1w_brain_mask(t1_file, citations, brain_mask_definition=None):
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

    citations.add("""
@article{fani2025mindgrab,
  title={MindGrab for BrainChop: Fast and Accurate Skull Stripping for Command Line and Browser},
  author={Fani, Armina and Doan, Mike and Le, Isabelle and Fedorov, Alex and Hoffmann, Malte and Rorden, Chris and Plis, Sergey},
  journal={arXiv preprint arXiv:2506.11860},
  year={2025}
}""")  # noqa: E501

    ort = check_onnxruntime(
        "Mindgrab", "Or, provide your own brain mask using brain_mask_definition."
    )
    return run_brainchop(ort, nib.load(t1_file), "mindgrab"), dict(
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
def t1_subcortex(t1_masked, citations):
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

    citations.add("""
@article{masoud2023brainchop,
  title={Brainchop: In-browser MRI volumetric segmentation and rendering},
  author={Masoud, Mohamed and Hu, Farfalla and Plis, Sergey},
  journal={Journal of Open Source Software},
  volume={8},
  number={83},
  pages={5098},
  year={2023}
}""")

    t1_img_masked = nib.load(t1_masked)

    subcortical_img = run_brainchop(ort, t1_img_masked, "subcortical")

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
        [mx_model, synthseg_model, t1w_brain_mask, t1_subcortex, t1_masked]
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

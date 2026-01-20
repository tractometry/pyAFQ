import os.path as op
import tempfile

import nibabel as nib
import numpy.testing as npt
import pytest

try:
    import onnxruntime as ort

    has_onnx = True
except ImportError:
    has_onnx = False


import AFQ.data.fetch as afd
from AFQ.nn.brainchop import run_brainchop
from AFQ.nn.multiaxial import run_multiaxial


@pytest.mark.skipif(not has_onnx, reason="onnxruntime is not installed")
def test_run_brainchop():
    tmpdir = tempfile.mkdtemp()
    afd.organize_stanford_data(path=tmpdir)

    t1_path = op.join(
        tmpdir,
        (
            "stanford_hardi/derivatives/freesurfer/"
            "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
        ),
    )
    chopped_brain = run_brainchop(ort, nib.load(t1_path), "mindgrab")

    npt.assert_(chopped_brain.get_fdata().sum() > 200000)


@pytest.mark.skipif(not has_onnx, reason="onnxruntime is not installed")
def test_run_multiaxial():
    tmpdir = tempfile.mkdtemp()
    afd.organize_stanford_data(path=tmpdir)

    t1_path = op.join(
        tmpdir,
        (
            "stanford_hardi/derivatives/freesurfer/"
            "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
        ),
    )
    chopped_brain = run_multiaxial(ort, nib.load(t1_path))

    npt.assert_(chopped_brain.get_fdata().sum() > 200000)

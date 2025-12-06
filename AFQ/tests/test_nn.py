import nibabel as nib
import os.path as op
import tempfile
import numpy.testing as npt

from AFQ.nn.brainchop import run_brainchop
from AFQ.nn.multiaxial import run_multiaxial
import AFQ.data.fetch as afd

def test_run_brainchop():
    tmpdir = tempfile.mkdtemp()
    afd.organize_stanford_data(path=tmpdir)

    t1_path = op.join(
        tmpdir,
        (
            "stanford_hardi/derivatives/freesurfer/"
            "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"))
    chopped_brain = run_brainchop(nib.load(t1_path), "mindgrab")

    npt.assert_(chopped_brain.get_fdata().sum() > 200000)

def test_run_multiaxial():
    tmpdir = tempfile.mkdtemp()
    afd.organize_stanford_data(path=tmpdir)

    t1_path = op.join(
        tmpdir,
        (
            "stanford_hardi/derivatives/freesurfer/"
            "sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"))
    chopped_brain = run_multiaxial(nib.load(t1_path))

    npt.assert_(chopped_brain.get_fdata().sum() > 200000)

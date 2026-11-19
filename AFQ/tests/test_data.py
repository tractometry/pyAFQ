import nibabel as nib
import numpy as np
import numpy.testing as npt

import AFQ.data.fetch as afd


def test_read_roi():
    aff1 = np.eye(4)
    template = nib.Nifti1Image(np.ones((10, 10, 10)), aff1)
    aff2 = aff1[:]
    aff2[0, 0] = -1
    roi = nib.Nifti1Image(np.zeros((10, 10, 10)), aff2)
    img = afd.read_resample_roi(roi, resample_to=template)
    npt.assert_equal(img.affine, template.affine)


def test_brain_mask_binary(tmp_path):
    from AFQ.tasks.data import brain_mask

    # T1w and DWI at different resolutions, so resampling interpolates
    ijk = np.indices((60, 60, 60))
    t1_mask = nib.Nifti1Image(
        (((ijk - 30) ** 2).sum(0) < 20**2).astype(np.uint8), np.eye(4)
    )
    t1_mask_file = str(tmp_path / "t1_mask.nii.gz")
    nib.save(t1_mask, t1_mask_file)
    b0_file = str(tmp_path / "b0.nii.gz")
    nib.save(
        nib.Nifti1Image(np.ones((33, 33, 33)), np.diag([1.8, 1.8, 1.8, 1])), b0_file
    )

    mask_file = brain_mask(
        {"t1w_brain_mask": t1_mask_file},
        b0_file,
        base_fname=str(tmp_path / "sub"),
        output_dir=str(tmp_path),
        tracking_params={},
    )
    mask = nib.load(mask_file).get_fdata()
    npt.assert_equal(np.unique(mask), [0, 1])

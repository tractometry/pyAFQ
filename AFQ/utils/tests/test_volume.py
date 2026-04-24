import nibabel as nib
import numpy as np
import numpy.testing as npt
from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import StatefulTractogram

import AFQ.data.fetch as afd
import AFQ.utils.volume as afv


def test_density_map():
    file_dict = afd.read_stanford_hardi_tractography()

    # subsample even more
    subsampled_tractography = file_dict["tractography_subsampled"][441:444]
    sft = StatefulTractogram(subsampled_tractography, file_dict["dwi"], Space.RASMM)
    density_map = afv.density_map(sft)
    npt.assert_equal(int(np.sum(density_map.get_fdata())), 36)

    density_map = afv.density_map(sft, normalize=True)
    npt.assert_equal(density_map.get_fdata().max(), 1)


def test_dice_coeff():
    affine = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    img1 = nib.Nifti1Image(np.asarray([[0.8, 0.9, 0], [0, 0, 0], [0, 0, 0]]), affine)
    img2 = nib.Nifti1Image(np.asarray([[0.5, 0, 0], [0.6, 0, 0], [0, 0, 0]]), affine)
    npt.assert_equal(afv.dice_coeff(img1, img2), (0.5 + 0.8) / (0.5 + 0.6 + 0.8 + 0.9))

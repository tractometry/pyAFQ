import nibabel as nib
import numpy as np
import numpy.testing as npt
import scipy.ndimage as ndim
from dipy.align.imwarp import DiffeomorphicMap
from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import StatefulTractogram

import AFQ.data.fetch as afd
import AFQ.utils.volume as afv

SHAPE = (40, 40, 40)
CENTER = 20
RADIUS = 8


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


def _corrugation_mapping(amplitude=4.5, frequency=4):
    displacement = np.zeros(SHAPE + (3,), dtype=np.float32)
    xx = np.arange(SHAPE[0], dtype=np.float32)
    displacement[..., 2] = (amplitude * np.sin(2 * np.pi * frequency * xx / SHAPE[0]))[
        :, None, None
    ]

    mapping = DiffeomorphicMap(3, SHAPE, domain_shape=SHAPE, codomain_shape=SHAPE)
    mapping.allocate()
    mapping.forward = np.ascontiguousarray(displacement, dtype=np.float32)
    return mapping


def _n_components(mask):
    return ndim.label(mask, structure=np.ones((3, 3, 3)))[1]


def test_transform_roi_corrugation_mapping():
    mapping = _corrugation_mapping()

    xx, yy, zz = np.meshgrid(*[np.arange(s) for s in SHAPE], indexing="ij")
    in_circle = ((xx - CENTER) ** 2 + (yy - CENTER) ** 2) <= RADIUS**2

    # A solid sphere
    sphere = (
        in_circle
        & (((xx - CENTER) ** 2 + (yy - CENTER) ** 2 + (zz - CENTER) ** 2) <= RADIUS**2)
    ).astype(np.uint8)
    warped_sphere, sphere_tol = afv.transform_roi(
        nib.Nifti1Image(sphere.astype(np.float32), np.eye(4)),
        mapping,
        True,
        return_tol=True,
    )

    npt.assert_equal(sphere_tol, 0)
    npt.assert_equal(warped_sphere.dtype, np.uint8)
    npt.assert_equal(_n_components(warped_sphere), 1)
    npt.assert_array_less(
        abs(int(warped_sphere.sum()) - int(sphere.sum())) / sphere.sum(), 0.1
    )

    # A one-voxel-thick disk
    disk = (in_circle & (zz == CENTER)).astype(np.uint8)
    warped_disk, disk_tol = afv.transform_roi(disk, mapping, True, return_tol=True)

    npt.assert_equal(disk_tol, 1.1)
    npt.assert_equal(_n_components(warped_disk), 1)
    npt.assert_array_less(
        abs(int(warped_disk.sum()) - int(disk.sum())) / disk.sum(), 2.0
    )


def test_transform_roi_real_mapping():
    # this mapping is too light to trigger non-zero tolerance
    mapping = afd.read_stanford_hardi_tractography()["mapping"]
    shape = tuple(mapping.codomain_shape)
    xx, yy, zz = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    cx, cy, cz = (s // 2 for s in shape)

    sphere = (((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= 20**2).astype(
        np.uint8
    )
    warped, tol = afv.transform_roi(sphere, mapping, True, return_tol=True)

    npt.assert_equal(warped.shape, tuple(mapping.domain_shape))
    npt.assert_equal(tol, 0)
    npt.assert_equal(_n_components(warped), 1)
    npt.assert_array_less(0, warped.sum())

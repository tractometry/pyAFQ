import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from AFQ._fixes import get_simplified_transform
from AFQ.definitions.mapping import FireANTsMap
from AFQ.registration import read_fireants_mapping, read_syn_mapping


def _points(shape):
    return np.moveaxis(np.indices(shape, dtype=float), 0, -1)


def _ramp(points):
    return points[..., 0] + 2 * points[..., 1] + 3 * points[..., 2]


def _coords(disp, template):
    # encode a template-grid displacement (mm) as FireANTs coordinates
    shape = np.array(template.shape)
    vox = _points(template.shape) + disp @ np.linalg.inv(template.affine[:3, :3]).T
    return (2 * vox / (shape - 1) - 1).transpose(2, 1, 0, 3)[None]


@pytest.fixture
def grids():
    # different shapes, spacings and origins; oblique, flipped axes
    angle = np.pi / 6
    template_affine = np.array(
        [
            [-2 * np.cos(angle), -3 * np.sin(angle), 0, 23],
            [-2 * np.sin(angle), 3 * np.cos(angle), 0, -19],
            [0, 0, 4, 11],
            [0, 0, 0, 1],
        ]
    )
    relative = np.diag([0.7, 0.8, 0.9, 1])
    relative[:3, 3] = 4
    subject = nib.Nifti1Image(_ramp(_points((15, 17, 19))), template_affine @ relative)
    template = nib.Nifti1Image(_ramp(_points((25, 29, 33))), template_affine)
    return subject, template


@pytest.mark.parametrize("same_grid", [False, True])
def test_read_fireants_mapping_translation(grids, same_grid):
    subject, template = grids
    if same_grid:
        subject = template
    shift = np.array([0.4, -0.3, 0.2])  # template voxels
    disp = np.broadcast_to(template.affine[:3, :3] @ shift, template.shape + (3,))
    mapping = get_simplified_transform(
        read_fireants_mapping(_coords(disp, template), subject, template)
    )

    subject_points = _points(subject.shape)[3:-3, 3:-3, 3:-3].reshape(-1, 3)
    expected = (
        nib.affines.apply_affine(
            np.linalg.inv(template.affine) @ subject.affine, subject_points
        )
        - shift
    )
    npt.assert_allclose(mapping.transform_points(subject_points), expected, atol=1e-4)
    npt.assert_allclose(
        mapping.transform_points_inverse(expected), subject_points, atol=1e-4
    )
    warped = mapping.transform(template.get_fdata())
    npt.assert_allclose(warped[3:-3, 3:-3, 3:-3].ravel(), _ramp(expected), atol=1e-4)


def test_read_fireants_mapping_nonlinear_prealign(grids, tmp_path):
    subject, template = grids

    # two successive shears, with an explicit inverse that is not -disp
    def forward(points):
        out = points.copy()
        out[..., 0] += 1.6 * np.sin(points[..., 1] / 5)
        out[..., 1] += 1.2 * np.sin(out[..., 0] / 5)
        return out

    def inverse(points):
        out = points.copy()
        out[..., 1] -= 1.2 * np.sin(points[..., 0] / 5)
        out[..., 0] -= 1.6 * np.sin(out[..., 1] / 5)
        return out

    voxel_prealign = np.array(
        [[1.05, 0.1, 0, -0.7], [0, 0.95, 0, 0.9], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    prealign = template.affine @ voxel_prealign @ np.linalg.inv(template.affine)
    template_points = _points(template.shape)
    disp = (forward(template_points) - template_points) @ template.affine[:3, :3].T
    mapping = get_simplified_transform(
        read_fireants_mapping(_coords(disp, template), subject, template, prealign)
    )

    subject_points = _points(subject.shape)[3:-3, 3:-3, 3:-3].reshape(-1, 3)
    expected = inverse(
        nib.affines.apply_affine(
            np.linalg.inv(template.affine) @ np.linalg.inv(prealign) @ subject.affine,
            subject_points,
        )
    )
    subject_to_template = np.linalg.inv(template.affine) @ subject.affine
    for actual, desired in (
        (mapping.transform_points(subject_points), expected),
        (
            nib.affines.apply_affine(
                subject_to_template, mapping.transform_points_inverse(expected)
            ),
            nib.affines.apply_affine(subject_to_template, subject_points),
        ),
    ):
        error = np.linalg.norm(actual - desired, axis=-1)
        assert np.percentile(error, 95) <= 0.1
        assert error.max() <= 0.5

    # mapping survives SynMap's save / read_syn_mapping round trip
    fnames = []
    for name, field, img in (
        ("forward", mapping.forward, subject),
        ("backward", mapping.backward, template),
    ):
        fnames.append(str(tmp_path / f"{name}.nii.gz"))
        nib.save(nib.Nifti1Image(field, img.affine), fnames[-1])
    restored = read_syn_mapping(*fnames)
    npt.assert_allclose(
        restored.transform_points(subject_points),
        mapping.transform_points(subject_points),
        atol=1e-6,
    )
    npt.assert_allclose(
        restored.transform(template.get_fdata()),
        mapping.transform(template.get_fdata()),
        atol=1e-6,
    )


def test_fireants_map():
    pytest.importorskip("fireants")
    points = _points((32, 32, 32))

    def blob(center):
        return np.exp(-np.sum((points - center) ** 2, -1) / (2 * 4**2))

    template = nib.Nifti1Image(blob([16, 16, 16]).astype(np.float32), np.eye(4))
    subject = nib.Nifti1Image(blob([18, 16, 16]).astype(np.float32), np.eye(4))
    mapping = FireANTsMap(use_prealign=False, device="cpu").gen_mapping(
        subject, template, None
    )
    npt.assert_allclose(
        mapping.transform_points(np.array([[18.0, 16, 16]])), [[16, 16, 16]], atol=0.5
    )

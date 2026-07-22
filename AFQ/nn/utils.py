import nibabel as nib
import nibabel.processing as nbp
import numpy as np


def crop_to_nonzero(img):
    """Crops the image to the bounding box of non-zero voxels."""
    data = img.get_fdata()
    nonzero_coords = np.array(np.nonzero(data))

    start_coords = nonzero_coords.min(axis=1)
    end_coords = nonzero_coords.max(axis=1) + 1

    cropped_data = data[
        start_coords[0] : end_coords[0],
        start_coords[1] : end_coords[1],
        start_coords[2] : end_coords[2],
    ]

    new_affine = img.affine.copy()
    new_affine[:3, 3] = nib.affines.apply_affine(img.affine, start_coords)

    return nib.Nifti1Image(cropped_data, new_affine)


def prepare_t1_for_nn(t1_img, orientation="RAS"):
    t1_img_cropped = crop_to_nonzero(t1_img)

    t1_img_conformed = nbp.conform(
        t1_img_cropped,
        out_shape=(256, 256, 256),
        voxel_size=(1.0, 1.0, 1.0),
        orientation=orientation,
        order=1,
    )

    t1_data = t1_img_conformed.get_fdata()
    p02 = np.nanpercentile(t1_data, 2)
    p98 = np.nanpercentile(t1_data, 98)
    t1_data = np.clip(t1_data, p02, p98)
    t1_data = (t1_data - p02) / (p98 - p02)

    return t1_data, t1_img_conformed.affine


def resample_output(output, conformed_affine, t1_img):
    return nbp.resample_from_to(
        nib.Nifti1Image(output.astype(np.uint8), conformed_affine),
        t1_img,
        order=0,
    )


def merge_PVEs(PVE_a, PVE_b):
    """
    Merge two PVE (Partial Volume Estimation) images, PVE_a and PVE_b,
    into a single PVE image. The merging is done by taking the maximum
    WM estimate from both images, then the maximum GM estimate, and
    finally normalizing.
    """

    csf_a, gm_a, wm_a = PVE_a[..., 0], PVE_a[..., 1], PVE_a[..., 2]
    csf_b, gm_b, wm_b = PVE_b[..., 0], PVE_b[..., 1], PVE_b[..., 2]

    new_wm = np.maximum(wm_a, wm_b)

    # 2. Choose the source (a or b) with the larger GM estimate, per voxel
    use_a = gm_a >= gm_b
    chosen_csf = np.where(use_a, csf_a, csf_b)
    chosen_gm = np.where(use_a, gm_a, gm_b)

    # 3. Normalize chosen csf+gm to fill the remaining mass (1 - new_wm)
    remaining = 1.0 - new_wm
    chosen_sum = chosen_csf + chosen_gm

    # avoid divide-by-zero where chosen_sum is 0 (e.g. pure-WM voxel)
    scale = np.divide(
        remaining, chosen_sum, out=np.zeros_like(chosen_sum), where=chosen_sum > 0
    )

    new_csf = chosen_csf * scale
    new_gm = chosen_gm * scale

    PVE = np.stack([new_csf, new_gm, new_wm], axis=-1)

    return PVE

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


def prepare_t1_for_nn(t1_img):
    t1_img_cropped = crop_to_nonzero(t1_img)

    t1_img_conformed = nbp.conform(
        t1_img_cropped,
        out_shape=(256, 256, 256),
        voxel_size=(1.0, 1.0, 1.0),
        orientation="RAS",
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

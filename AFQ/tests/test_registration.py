import os.path as op

import dipy.data as dpd
import nibabel as nib
import nibabel.tmpdirs as nbtmp
import numpy as np
import numpy.testing as npt
from dipy.align.imaffine import AffineMap
from dipy.align.streamlinear import whole_brain_slr
from dipy.io.streamline import load_tractogram

import AFQ.data.fetch as afd
from AFQ.registration import read_affine_mapping, reduce_shape

MNI_T2 = afd.read_mni_template()
hardi_img, gtab = dpd.read_stanford_hardi()
MNI_T2_data = MNI_T2.get_fdata()
MNI_T2_affine = MNI_T2.affine
hardi_data = hardi_img.get_fdata()
hardi_affine = hardi_img.affine
b0 = hardi_data[..., gtab.b0s_mask]
mean_b0 = np.mean(b0, -1)

# We select some arbitrary chunk of data so this goes quicker:
subset_b0 = mean_b0[40:50, 40:50, 40:50]
subset_dwi_data = nib.Nifti1Image(hardi_data[40:50, 40:50, 40:50], hardi_affine)
subset_t2 = MNI_T2_data[40:60, 40:60, 40:60]
subset_b0_img = nib.Nifti1Image(subset_b0, hardi_affine)
subset_t2_img = nib.Nifti1Image(subset_t2, MNI_T2_affine)


def test_slr_registration():
    # have to import subject sls
    file_dict = afd.read_stanford_hardi_tractography()
    streamlines = file_dict["tractography_subsampled"]

    # have to import sls atlas
    afd.fetch_hcp_atlas_16_bundles()
    atlas_fname = op.join(
        afd.afq_home,
        "hcp_atlas_16_bundles",
        "Atlas_in_MNI_Space_16_bundles",
        "whole_brain",
        "whole_brain_MNI.trk",
    )
    hcp_atlas = load_tractogram(atlas_fname, "same", bbox_valid_check=False)

    with nbtmp.InTemporaryDirectory() as tmpdir:
        _, transform, _, _ = whole_brain_slr(
            streamlines,
            hcp_atlas.streamlines,
            x0="affine",
            verbose=False,
            progressive=False,
            greater_than=10,
            rm_small_clusters=1,
            rng=np.random.RandomState(seed=8),
        )

        mapping = AffineMap(
            transform,
            domain_grid_shape=reduce_shape(subset_b0_img.shape),
            domain_grid2world=subset_b0_img.affine,
            codomain_grid_shape=reduce_shape(subset_t2_img.shape),
            codomain_grid2world=subset_t2_img.affine,
        )

        warped_moving = mapping.transform_inverse(subset_b0)

        npt.assert_equal(warped_moving.shape, subset_t2.shape)
        mapping_fname = op.join(tmpdir, "mapping.npy")
        np.save(mapping_fname, transform)
        file_mapping = read_affine_mapping(mapping_fname, subset_b0_img, subset_t2_img)

        # Test that it has the same effect on the data:
        warped_from_file = file_mapping.transform_inverse(subset_b0)
        npt.assert_equal(warped_from_file, warped_moving)

        # Test that it is, attribute by attribute, identical:
        for k in mapping.__dict__:
            assert np.all(
                mapping.__getattribute__(k) == file_mapping.__getattribute__(k)
            )

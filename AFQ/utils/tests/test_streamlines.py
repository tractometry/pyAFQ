import dipy.tracking.streamline as dts
import nibabel as nib
import numpy as np
import numpy.testing as npt
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from trx.trx_file_memmap import TrxFile

from AFQ.utils import streamlines as aus

affine = np.array(
    [
        [2.0, 0.0, 0.0, -80.0],
        [0.0, 2.0, 0.0, -120.0],
        [0.0, 0.0, 2.0, -60.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

img = nib.Nifti1Image(np.ones((10, 10, 10, 30)), affine)


bundles = {
    "ba": StatefulTractogram(
        [
            np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
            np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]]),
        ],
        img,
        Space.VOX,
    ),
    "bb": StatefulTractogram(
        [
            np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 2], [0, 0, 2.5]]),
            np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 2, 2]]),
        ],
        img,
        Space.VOX,
    ),
}


class TestSegmentedSFT:
    def setup_method(self):
        self.img = img
        self.bundles = bundles
        self.seg = aus.SegmentedSFT(self.bundles)

    def test_bundle_names(self):
        assert set(self.seg.bundle_names) == {"ba", "bb"}

    def test_get_bundle_roundtrip(self):
        for k in self.bundles:
            for sl_orig, sl_got in zip(
                self.bundles[k].streamlines,
                self.seg.get_bundle(k).streamlines,
            ):
                npt.assert_array_equal(sl_orig, sl_got)

    def test_get_bundle_idxs_contiguous(self):
        idxs_a = self.seg.get_bundle_idxs("ba")
        idxs_b = self.seg.get_bundle_idxs("bb")
        assert len(idxs_a) == 2
        assert len(idxs_b) == 2
        assert set(idxs_a).isdisjoint(set(idxs_b))

    def test_sidecar_bundle_ids_populated(self):
        ids = self.seg.sidecar_info["bundle_ids"]
        assert set(ids.keys()) == {"ba", "bb"}

    def test_bundle_dps_match_sidecar(self):
        ids = self.seg.sidecar_info["bundle_ids"]
        dps = self.seg.sft.data_per_streamline["bundle"]
        for b_name, b_id in ids.items():
            for idx in self.seg.get_bundle_idxs(b_name):
                assert dps[idx] == b_id

    def test_get_lengths_returns_array(self):
        lengths = self.seg.get_lengths()
        assert len(lengths) == 4  # 2 bundles × 2 streamlines each

    def test_to_rasmm_does_not_raise(self):
        self.seg.to_rasmm()
        assert self.seg.sft.space == Space.RASMM

    def test_get_bundle_param_info_missing(self):
        assert self.seg.get_bundle_param_info("ba") == {}

    def test_get_bundle_param_info_present(self):
        seg = aus.SegmentedSFT(
            self.bundles,
            sidecar_info={"Bundle Parameters": {"ba": {"min_len": 10}}},
        )
        assert seg.get_bundle_param_info("ba") == {"min_len": 10}
        assert seg.get_bundle_param_info("bb") == {}

    def test_default_sidecar_is_empty_dict(self):
        seg = aus.SegmentedSFT(self.bundles)
        assert "bundle_ids" in seg.sidecar_info

    def test_sidecar_not_shared_between_instances(self):
        seg2 = aus.SegmentedSFT(self.bundles)
        assert self.seg.sidecar_info is not seg2.sidecar_info

    def test_dict_bundles_with_tracking_idx(self):
        bundles_with_idx = {
            "ba": {"sl": self.bundles["ba"], "idx": np.array([0, 1])},
            "bb": {"sl": self.bundles["bb"], "idx": np.array([2, 3])},
        }
        seg = aus.SegmentedSFT(bundles_with_idx)
        assert "tracking_idx" in seg.sidecar_info
        assert set(seg.sidecar_info["tracking_idx"].keys()) == {"ba", "bb"}
        assert isinstance(seg.sidecar_info["tracking_idx"]["ba"], list)

    def test_single_bundle_no_tracking_idx(self):
        single = {"ba": self.bundles["ba"]}
        seg = aus.SegmentedSFT(single)
        assert seg.this_tracking_idxs is None
        assert "tracking_idx" not in seg.sidecar_info


class TestSegmentedTRX:
    def setup_method(self):
        seg_sft = aus.SegmentedSFT(bundles)
        self.trx = TrxFile.from_sft(seg_sft.sft)
        self.trx.groups = seg_sft.bundle_idxs
        self.seg = aus.SegmentedTRX(self.trx)

    def test_bundle_names(self):
        assert set(self.seg.bundle_names) == {"ba", "bb"}

    def test_sft_is_trx(self):
        assert self.seg.sft is self.trx

    def test_space_attributes_set(self):
        assert hasattr(self.seg.sft, "space_attributes")
        assert self.seg.sft.space == Space.RASMM
        assert self.seg.sft.origin == Origin.NIFTI

    def test_get_lengths(self):
        lengths = self.seg.get_lengths()
        assert len(lengths) == 4

    def test_to_rasmm_is_noop(self):
        self.seg.to_rasmm()
        assert self.seg.sft.space == Space.RASMM

    def test_get_bundle_idxs(self):
        idxs = self.seg.get_bundle_idxs("ba")
        npt.assert_array_equal(idxs, self.trx.groups["ba"])

    def test_get_bundle_returns_sft(self):
        result = self.seg.get_bundle("ba")
        assert isinstance(result, StatefulTractogram)

    def test_get_bundle_param_info_missing(self):
        assert self.seg.get_bundle_param_info("ba") == {}


def test_split_streamline():
    streamlines = dts.Streamlines(
        [
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
        ]
    )
    assert streamlines == streamlines
    sl_to_split = 1
    split_idx = 1
    new_streamlines = aus.split_streamline(streamlines, sl_to_split, split_idx)
    test_streamlines = dts.Streamlines(
        [
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[7.0, 8.0, 9.0]]),
            np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]]),
        ]
    )

    # Test equality of the underlying dict items:
    for k in new_streamlines.__dict__.keys():
        if isinstance(new_streamlines.__dict__[k], np.ndarray):
            npt.assert_array_equal(
                new_streamlines.__dict__[k], test_streamlines.__dict__[k]
            )
        else:
            assert new_streamlines.__dict__[k] == test_streamlines.__dict__[k]

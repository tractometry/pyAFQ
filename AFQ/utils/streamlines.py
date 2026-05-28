import os.path as op

import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.utils import Origin
from nibabel.affines import voxel_sizes
from nibabel.orientations import aff2axcodes

try:
    from trx.io import load as load_trx

    has_trx = True
except ModuleNotFoundError:
    has_trx = False

from AFQ.definitions.mapping import ConformedFnirtMapping
from AFQ.utils.path import drop_extension, read_json


class SegmentedSFT:
    def __init__(self, bundles, sidecar_info=None, is_trx=False):
        if sidecar_info is None:
            sidecar_info = {}

        self.sidecar_info = sidecar_info
        self.is_trx = is_trx

        if is_trx:
            # Loading from TRX, we do not calculate additional information,
            # To avoid loading unnecessary streamlines into memory
            self.bundle_names = list(bundles.groups.keys())
            self.sft = bundles

            # mimic SFT attributes for compatibility
            affine = np.array(self.sft.header["VOXEL_TO_RASMM"], dtype=np.float32)
            dimensions = np.array(self.sft.header["DIMENSIONS"], dtype=np.uint16)
            vox_sizes = np.array(voxel_sizes(affine), dtype=np.float32)
            vox_order = "".join(aff2axcodes(affine))
            space_attributes = (affine, dimensions, vox_sizes, vox_order)
            self.sft.space_attributes = space_attributes
            self.sft.space = Space.RASMM
            self.sft.origin = Origin.NIFTI
            self.sft.dtype_dict = self.sft.get_dtype_dict()
        else:
            self.bundle_names = []
            sls = []
            idxs = {}
            this_tracking_idxs = {}
            idx_count = 0
            for b_name in bundles:
                if isinstance(bundles[b_name], dict):
                    this_sft = bundles[b_name]["sl"]
                    this_tracking_idxs[b_name] = bundles[b_name]["idx"]
                else:
                    this_sft = bundles[b_name]
                this_sls = list(this_sft.streamlines)
                sls.extend(this_sls)
                new_idx_count = idx_count + len(this_sls)
                idxs[b_name] = np.arange(idx_count, new_idx_count, dtype=np.uint32)
                idx_count = new_idx_count
                self.bundle_names.append(b_name)

            self._bundle_idxs = idxs
            if len(this_tracking_idxs) > 1:
                self.this_tracking_idxs = this_tracking_idxs
            else:
                self.this_tracking_idxs = None

            self.sidecar_info["bundle_ids"] = {}
            dps = np.zeros(len(sls))
            for ii, bundle_name in enumerate(self.bundle_names):
                self.sidecar_info["bundle_ids"][f"{bundle_name}"] = ii + 1
                dps[self._bundle_idxs[bundle_name]] = ii + 1
            self.sft = StatefulTractogram.from_sft(
                sls, this_sft, data_per_streamline={"bundle": dps}
            )
            if self.this_tracking_idxs is not None:
                for kk, _vv in self.this_tracking_idxs.items():
                    self.this_tracking_idxs[kk] = (
                        self.this_tracking_idxs[kk].astype(int).tolist()
                    )
                self.sidecar_info["tracking_idx"] = self.this_tracking_idxs

    def get_lengths(self):
        if self.is_trx:
            return self.sft.streamlines._lengths
        else:
            return self.sft._tractogram._streamlines._lengths

    def to_rasmm(self):
        if self.is_trx:
            pass  # always in RASMM
        else:
            self.sft.to_space(Space.RASMM)

    def bundle_idxs(self, b_name):
        if self.is_trx:
            return self.sft.groups[b_name]
        else:
            return self._bundle_idxs[b_name]

    def get_bundle(self, b_name):
        if self.is_trx:
            idx = self.sft.groups[b_name]
            return StatefulTractogram(self.sft.streamlines[idx], self.sft, Space.RASMM)
        else:
            return self.sft[self._bundle_idxs[b_name]]

    def get_bundle_param_info(self, b_name):
        return self.sidecar_info.get("Bundle Parameters", {}).get(b_name, {})

    @classmethod
    def fromfile(cls, trk_or_trx_file, reference="same", sidecar_file=None):
        if sidecar_file is None:
            # assume json sidecar has the same name as trk_file,
            # but with json suffix
            sidecar_file = f"{drop_extension(trk_or_trx_file)}.json"
            if not op.exists(sidecar_file):
                raise ValueError(
                    (
                        "JSON sidecars are required for trk files. "
                        f"JSON sidecar not found for: {sidecar_file}"
                    )
                )
        sidecar_info = read_json(sidecar_file)
        if trk_or_trx_file.endswith(".trx"):
            bundles = load_trx(trk_or_trx_file, reference)
            is_trx = True
        else:
            sft = load_tractogram(trk_or_trx_file, reference, to_space=Space.RASMM)

            if reference == "same":
                reference = sft
            bundles = {}
            if "bundle_ids" in sidecar_info:
                for b_name, b_id in sidecar_info["bundle_ids"].items():
                    idx = np.where(sft.data_per_streamline["bundle"] == b_id)[0]
                    bundles[b_name] = StatefulTractogram(
                        sft.streamlines[idx], reference, Space.RASMM
                    )
            else:
                bundles["whole_brain"] = sft
            is_trx = False

        return cls(bundles, sidecar_info, is_trx=is_trx)


def split_streamline(streamlines, sl_to_split, split_idx):
    """
    Given a Streamlines object, split one of the underlying streamlines

    Parameters
    ----------
    streamlines : a Streamlines class instance
        The group of streamlines, one of which is being split.
    sl_to_split : int
        The index of the streamline that is being split
    split_idx : int
        Where is the streamline being split
    """
    this_sl = streamlines[sl_to_split]

    streamlines._lengths = np.concatenate(
        [
            streamlines._lengths[:sl_to_split],
            np.array([split_idx]),
            np.array([this_sl.shape[0] - split_idx]),
            streamlines._lengths[sl_to_split + 1 :],
        ]
    )

    streamlines._offsets = np.concatenate(
        [np.array([0]), np.cumsum(streamlines._lengths[:-1])]
    )

    return streamlines


def move_streamlines(tg, to, mapping, img, to_space=None, save_intermediates=None):
    """Move streamlines to or from template space.

    to : str
        Either "template" or "subject". This determines
        whether we will use the forward or backwards displacement field.
    mapping : DIPY or pyAFQ mapping
        Mapping to use to move streamlines.
    img : Nifti1Image
        Image defining reference for where the streamlines move to.
    to_space : Space or None
        If not None, space to move streamlines to after moving them to the
        template or subject space. If None, streamlines will be moved back to
        their original space.
        Default: None.
    save_intermediates : str or None
        If not None, path to save intermediate tractogram after moving to template
        or subject space.
        Default: None.
    """
    tg_og_space = tg.space
    if isinstance(mapping, ConformedFnirtMapping):
        if to != "subject":
            raise ValueError(
                "Attempted to transform streamlines to template using "
                "unsupported mapping. "
                "Use something other than Fnirt."
            )
        tg.to_vox()
        moved_sl = []
        for sl in tg.streamlines:
            moved_sl.append(mapping.transform_pts(sl))
        moved_sft = StatefulTractogram(moved_sl, img, Space.RASMM)
    else:
        tg.to_vox()
        if to == "template":
            moved_sl = mapping.transform_points(tg.streamlines)
        else:
            moved_sl = mapping.transform_points_inverse(tg.streamlines)
        moved_sft = StatefulTractogram(moved_sl, img, Space.VOX)

    if save_intermediates is not None:
        save_tractogram(
            moved_sft,
            op.join(save_intermediates, f"sls_in_{to}.trk"),
            bbox_valid_check=False,
        )
    if to_space is None:
        moved_sft.to_space(tg_og_space)
    else:
        moved_sft.to_space(to_space)
    return moved_sft

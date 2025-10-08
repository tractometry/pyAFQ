import gc
from dipy.io.streamline import load_tractogram
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import os.path as op


try:
    from trx.io import load as load_trx
    has_trx = True
except ModuleNotFoundError:
    has_trx = False

from AFQ.utils.path import drop_extension, read_json


class SegmentedSFT():
    def __init__(self, bundles, space, sidecar_info={}):
        reference = None
        self.bundle_names = []
        self.sidecar_info = sidecar_info
        sls = []
        idxs = {}
        this_tracking_idxs = {}
        idx_count = 0
        for b_name in bundles:
            if isinstance(bundles[b_name], dict):
                this_sls = bundles[b_name]['sl']
                this_tracking_idxs[b_name] = bundles[b_name]['idx']
            else:
                this_sls = bundles[b_name]
            if reference is None:
                reference = this_sls
            this_sls = list(this_sls.streamlines)
            sls.extend(this_sls)
            new_idx_count = idx_count + len(this_sls)
            idxs[b_name] = np.arange(idx_count, new_idx_count, dtype=np.uint32)
            idx_count = new_idx_count
            self.bundle_names.append(b_name)

        self.sft = StatefulTractogram(sls, reference, space)
        self.bundle_idxs = idxs
        if len(this_tracking_idxs) > 1:
            self.this_tracking_idxs = this_tracking_idxs
        else:
            self.this_tracking_idxs = None

    def get_sft_and_sidecar(self):
        sidecar_info = {}
        sidecar_info["bundle_ids"] = {}
        dps = np.zeros(len(self.sft.streamlines))
        for ii, bundle_name in enumerate(self.bundle_names):
            sidecar_info["bundle_ids"][f"{bundle_name}"] = ii + 1
            dps[self.bundle_idxs[bundle_name]] = ii + 1
        dps = {"bundle": dps}
        self.sft.data_per_streamline = dps
        if self.this_tracking_idxs is not None:
            for kk, vv in self.this_tracking_idxs.items():
                for ii in range(len(self.this_tracking_idxs[kk])):
                    this = int(self.this_tracking_idxs[kk][ii])
                    self.this_tracking_idxs[kk][ii] = this
            sidecar_info["tracking_idx"] = self.this_tracking_idxs
        return self.sft, sidecar_info

    def get_bundle(self, b_name):
        return self.sft[self.bundle_idxs[b_name]]

    def get_bundle_param_info(self, b_name):
        return self.sidecar_info.get(
            "Bundle Parameters", {}).get(b_name, {})

    @classmethod
    def fromfile(cls, trk_or_trx_file, reference="same", sidecar_file=None):
        if sidecar_file is None:
            # assume json sidecar has the same name as trk_file,
            # but with json suffix
            sidecar_file = f'{drop_extension(trk_or_trx_file)}.json'
            if not op.exists(sidecar_file):
                raise ValueError((
                    "JSON sidecars are required for trk files. "
                    f"JSON sidecar not found for: {sidecar_file}"))
        sidecar_info = read_json(sidecar_file)
        if trk_or_trx_file.endswith(".trx"):
            trx = load_trx(trk_or_trx_file, reference)
            trx.streamlines._data = trx.streamlines._data.astype(np.float32)
            sft = trx.to_sft()
            if reference == "same":
                reference = sft
            bundles = {}
            for bundle in trx.groups:
                idx = trx.groups[bundle]
                bundles[bundle] = StatefulTractogram(
                    sft.streamlines[idx], reference, Space.RASMM)
        else:
            sft = load_tractogram(trk_or_trx_file, reference,
                                  to_space=Space.RASMM)

            if reference == "same":
                reference = sft
            bundles = {}
            if "bundle_ids" in sidecar_info:
                for b_name, b_id in sidecar_info["bundle_ids"].items():
                    idx = np.where(
                        sft.data_per_streamline['bundle'] == b_id)[0]
                    bundles[b_name] = StatefulTractogram(
                        sft.streamlines[idx], reference, Space.RASMM)
            else:
                bundles["whole_brain"] = sft

        return cls(bundles, Space.RASMM, sidecar_info)


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

    streamlines._lengths = np.concatenate([
        streamlines._lengths[:sl_to_split],
        np.array([split_idx]),
        np.array([this_sl.shape[0] - split_idx]),
        streamlines._lengths[sl_to_split + 1:]])

    streamlines._offsets = np.concatenate([
        np.array([0]),
        np.cumsum(streamlines._lengths[:-1])])

    return streamlines

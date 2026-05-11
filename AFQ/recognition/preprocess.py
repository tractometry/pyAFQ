from functools import cached_property

import numpy as np

import AFQ.recognition.utils as abu


class PreprocPlan:
    def __init__(self, tg):
        self.tg = tg

    @cached_property
    def fgarray(self):
        return np.asarray(abu.resample_tg(self.tg, 20), dtype=np.float32)

    @cached_property
    def crosses(self):
        return np.logical_and(
            np.any(self.fgarray[:, :, 0] > 0, axis=1),
            np.any(self.fgarray[:, :, 0] < 0, axis=1),
        )

    @cached_property
    def lengths(self):
        segments = np.diff(self.fgarray, axis=1)
        return np.sum(np.sqrt(np.sum(segments**2, axis=2)), axis=1)

    @cached_property
    def endpoint_dists(self):
        return np.linalg.norm(self.fgarray[:, 0, :] - self.fgarray[:, -1, :], axis=1)

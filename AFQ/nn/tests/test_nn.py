import numpy as np

from AFQ.nn.utils import merge_PVEs


def test_wm_is_elementwise_maximum():
    a = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    b = np.array([[0.1, 0.2, 0.7], [0.4, 0.3, 0.3]])
    np.testing.assert_allclose(merge_PVEs(a, b)[..., 2], [0.7, 0.8])


def test_larger_gm_source_wins():
    a = np.array([[0.6, 0.2, 0.2]])
    b = np.array([[0.2, 0.5, 0.3]])
    np.testing.assert_allclose(merge_PVEs(a, b), b)


def test_tie_prefers_a_and_renormalizes():
    a = np.array([[0.6, 0.2, 0.2]])
    b = np.array([[0.4, 0.2, 0.4]])
    np.testing.assert_allclose(merge_PVEs(a, b), [[0.45, 0.15, 0.4]])

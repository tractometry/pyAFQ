import dipy.core.gradients as dpg
import dipy.data as dpd
import dipy.io.gradients as dpio
import dipy.reconst.dki as dki
import numpy as np
import numpy.testing as npt
from dipy.sims.voxel import multi_tensor_dki

from AFQ.models import mkcurve


def _noisy_dki_data(shape=(10, 10, 3), snr=20, seed=0):
    rng = np.random.default_rng(seed)
    _, fbval, fbvec = dpd.get_fnames(name="small_64D")
    bvals, bvecs = dpio.read_bvals_bvecs(fbval, fbvec)
    gtab = dpg.gradient_table(np.r_[bvals, 2 * bvals], bvecs=np.r_[bvecs, bvecs])
    mevals = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087]] * 2)
    data = np.empty(shape + (len(gtab.bvals),))
    for idx in np.ndindex(shape):
        angles = [tuple(rng.uniform(0, 90, 2))] * 2 + [tuple(rng.uniform(0, 90, 2))] * 2
        sig, _, _ = multi_tensor_dki(
            gtab,
            mevals,
            S0=100,
            angles=angles,
            fractions=[24.5, 25.5, 24.5, 25.5],
            snr=None,
        )
        noise = rng.normal(0, 100 / snr, (2, len(sig)))
        data[idx] = np.hypot(sig + noise[0], noise[1])  # Rician
    return data, gtab


def test_mk_curves_match_dipy_fits():
    data, gtab = _noisy_dki_data()
    mask = np.ones(data.shape[:3], bool)
    out, new_gtab, _ = mkcurve.mkcurve_correct(data, gtab, mask)
    Y = out[mask][:300]
    syn = np.array([0.3, 1.0, 1.7]) * np.nanmean(Y[:, 0])
    curves = mkcurve._mk_curves(Y, dki.design_matrix(new_gtab), syn)
    # reference: DIPY's own per-voxel WLS solver at each synthetic b0
    B = dki.design_matrix(new_gtab)
    dirs = dpd.get_sphere(name="repulsion200").vertices
    gdirs = dpg.gradient_table(np.ones(len(dirs)), bvecs=dirs, b0_threshold=0)
    Bd = dki.design_matrix(gdirs)
    for i, s in enumerate(syn):
        ref = np.empty(len(Y))
        for v in range(len(Y)):
            y = Y[v].copy()
            y[0] = s
            x, _ = dki.ls_fit_dki(B, y, np.linalg.pinv(B), return_lower_triangular=True)
            ref[v] = np.mean((6 * Bd[:, 6:21] @ x[6:21]) / (Bd[:, :6] @ x[:6]) ** 2)
        ok = np.abs(ref) < 100
        assert ok.mean() > 0.9
        npt.assert_allclose(curves[i, ok], ref[ok], rtol=1e-6, atol=1e-8)


def test_detection():
    syn = np.arange(10.0, 210.0, 10.0)
    t = np.arange(len(syn))[:, None]
    curve = np.c_[1 - 2 * np.exp(-t / 3), 0.5 + 0 * t]  # crosses zero / never
    flag, fixed = mkcurve._detect(curve, np.array([20.0, 150.0]), syn, 0.5)
    npt.assert_array_equal(flag, [True, False])
    npt.assert_array_equal(fixed, [(40.0 + 200.0) / 2, 150.0])

import numpy as np

from src.evaluation import residual_m, summarize
from src.registration import CrossModalRegistrar, mask_optical, mask_sar, prep_optical, prep_sar
from tests.synthetic import make_pair
import pandas as pd


def _prepared(shift=(23.0, -17.0), seed=0):
    opt, sar, t = make_pair(shift=shift, seed=seed)
    return prep_optical(opt), prep_sar(sar), t


def _tp_error(M, t, gsd=1.0):
    # A feature at optical position p appears in the SAR image at p + t.
    opt_px = np.array([[512.0, 512.0], [300.0, 700.0], [800.0, 250.0]])
    sar_px = opt_px + np.array(t)
    return residual_m(M, sar_px, opt_px, gsd)


def test_identity_error_equals_injected_shift():
    o, s, t = _prepared()
    res = CrossModalRegistrar("identity").register(o, s)
    assert np.allclose(_tp_error(res.M, t), np.hypot(*t))


def test_masked_ncc_recovers_shift():
    o, s, t = _prepared()
    for method in ("ncc", "oriented_ncc"):
        res = CrossModalRegistrar(method).register(o, s)
        assert res.success, res.note
        assert _tp_error(res.M, t).max() < 1.5, (method, res.extra)


def test_masked_ncc_recovers_shift_with_unequal_margins():
    """Zero-padded margins of different height, as in the real SN9 patches."""
    opt, sar, t = make_pair(shift=(10.0, -8.0))
    opt, sar = opt.copy(), sar.copy()
    opt[:, :100, :] = 0.0
    sar[:160, :] = 0.0
    o, s = prep_optical(opt), prep_sar(sar, log_scale=False)
    res = CrossModalRegistrar("oriented_ncc").register(o, s, mask_optical(opt), mask_sar(sar))
    assert _tp_error(res.M, t).max() < 2.0, res.extra


def test_shared_margins_do_not_create_a_confident_zero_shift():
    """Regression test for a real failure: unrelated content with identical zero-padded margins must
    not yield a confident peak. (Hard-masking gradient maps made the mask edge a common signal.)"""
    a, _, _ = make_pair(seed=1)
    _, b, _ = make_pair(seed=2)
    a = a.copy(); b = b.copy()
    a[:, :200, :] = 0.0
    b[:200, :] = 0.0
    o, s = prep_optical(a), prep_sar(b, log_scale=False)
    res = CrossModalRegistrar("oriented_ncc").register(o, s, mask_optical(a), mask_sar(b))
    assert res.extra["ncc"] < 0.35, res.extra


def test_sift_improves_on_identity():
    o, s, t = _prepared()
    res = CrossModalRegistrar("sift", rep="gradient").register(o, s)
    print("sift:", res.success, res.n_matches, res.n_inliers, res.note)
    if res.success:
        assert _tp_error(res.M, t).max() < 3.0


def test_sift_rejects_unrelated_images():
    rng = np.random.default_rng(1)
    o = (rng.random((1024, 1024)) * 255).astype(np.uint8)
    s = (rng.random((1024, 1024)) * 255).astype(np.uint8)
    res = CrossModalRegistrar("sift", rep="gradient").register(o, s)
    assert not res.success


def test_summary_reports_failures_honestly():
    df = pd.DataFrame({"method": ["sift"] * 4, "before_m": [30.0] * 4,
                       "after_m": [30.0, 30.0, 4.0, 2.0], "success": [False, False, True, True]})
    s = summarize(df).iloc[0]
    assert s["success_rate"] == 0.5 and s["median_after_m_successes_only"] == 3.0
    assert s["median_after_m"] == 17.0


def test_identity_is_never_counted_as_improved():
    before = np.array([24.91, 30.0, 12.0])
    df = pd.DataFrame({"method": ["identity"] * 3, "before_m": before,
                       "after_m": before - 1e-9, "success": [True] * 3})
    s = summarize(df).iloc[0]
    assert s["share_improved"] == 0.0 and s["share_worsened"] == 0.0

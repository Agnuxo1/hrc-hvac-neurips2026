"""Implementation checks for the reservoir predictors (synthetic signals, no simulator).

These verify the code, not the controller: closed-loop results come only from the
official simulator (see experiments/LOG.md).
"""
import numpy as np
import pytest

from hrc_hvac.reservoir import (BatchedReservoir, LinearAR, Persistence, ReservoirPredictor,
                                SharedRLS, build_weights, make_predictor)


def test_weights_deterministic_and_scaled():
    W_in1, W1 = build_weights(80, 5, 0.9, seed=3)
    W_in2, W2 = build_weights(80, 5, 0.9, seed=3)
    assert np.array_equal(W_in1, W_in2) and np.array_equal(W1, W2)
    assert abs(np.max(np.abs(np.linalg.eigvals(W1))) - 0.9) < 1e-9
    _, W3 = build_weights(80, 5, 0.9, seed=4)
    assert not np.array_equal(W1, W3)


def test_zones_share_weights_but_keep_separate_state():
    r = BatchedReservoir(3, 4, size=50, seed=1)
    U = np.zeros((3, 4), dtype=np.float32)
    U[1] = 1.0
    X = r.step(U)
    assert np.allclose(X[0], 0) and np.allclose(X[2], 0) and np.abs(X[1]).sum() > 0


def test_rls_recovers_linear_map():
    rng = np.random.default_rng(0)
    w = rng.normal(size=6)
    rls = SharedRLS(6, lam=1.0, delta=100.0)
    for _ in range(200):
        F = rng.normal(size=(4, 6))
        rls.update(F, F @ w)
    assert np.allclose(rls.w, w, atol=1e-4)  # finite delta = small ridge bias


def test_persistence_is_zero_drift():
    p = Persistence(2, 3, horizon=4)
    y = np.array([20.0, 21.0])
    assert np.array_equal(p.step(np.zeros((2, 3)), y), y)


@pytest.mark.parametrize("kind", ["hrc", "ar"])
def test_predictors_learn_a_sine_better_than_persistence(kind):
    Z, D, h = 2, 3, 12
    p = make_predictor(kind, Z, D, horizon=h, size=60, delta=1.0, lam=0.9995, warmup=10, err_window=200)
    t = np.arange(3000)
    y = 22 + np.sin(2 * np.pi * t / 288)[:, None] * np.array([1.0, 2.0])
    for k in range(len(t)):
        u = np.stack([np.full(Z, np.sin(2 * np.pi * k / 288)), np.full(Z, np.cos(2 * np.pi * k / 288)),
                      (y[k] - 22) / 2], axis=1)
        p.step(u, y[k])
    rmse, prmse = p.rmse()
    assert p.ready and rmse < 0.5 * prmse


def test_invalid_step_only_advances_clock():
    p = ReservoirPredictor(2, 3, size=20)
    assert p.step(None, np.zeros(2), valid=False) is None
    assert p.t == 1 and p.n_err == 0


def test_ar_feature_width():
    p = LinearAR(2, 3, lags=2)
    assert p.n_feat == 9

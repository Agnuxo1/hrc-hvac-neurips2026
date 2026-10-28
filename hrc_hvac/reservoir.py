"""Batched holographic reservoir with a shared online RLS readout.

Attribution and license
-----------------------
The weight construction and the leaky state update are derived from
``holographic_reservoir/model.py`` (``HolographicReservoirModel``) and
``holographic_reservoir/backends.py`` (``SimulatedASICBackend``) of
https://github.com/Agnuxo1/Holographic-Reservoir by Francisco Angulo de Lafuente,
Apache License 2.0. This file is a derived work: it reproduces the weight
construction (same random-number sequence, same SHA-256 entropy of the simulated
backend, same spectral-radius scaling) and the leaky ``tanh`` state equation, and
adds batching over building zones with shared weights, an online recursive
least-squares (RLS) readout with forgetting, and ablation predictors.

Contents
--------
* ``BatchedReservoir``: state ``X`` of shape ``Z x N`` (one row per zone), shared
  weights, ``x <- (1-a) x + a tanh(W_in u + W x)``.
* ``SharedRLS``: recursive least squares with forgetting factor, one weight vector
  shared by all zones, block (Woodbury) update with several samples per step.
* ``ReservoirPredictor``, ``Persistence``, ``LinearAR``: same interface
  ``step(u, y) -> y_hat(t+h)``. They learn with a delayed target: the features seen
  ``h`` steps ago are kept in a ring buffer and, when ``y(t)`` arrives, the readout is
  updated with ``(features(t-h), y(t) - y(t-h))``. The predicted quantity is the
  increment ``y(t+h) - y(t)``.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np


# --------------------------------------------------------------------------
# Weight construction (same logic as the original repository)
# --------------------------------------------------------------------------


def _sim_entropy(random_seed: int, size: int) -> np.ndarray:
    """Entropy of ``SimulatedASICBackend(random_seed).mine(b"init", max(1, size//8))``."""
    salt = hashlib.sha256(b"holographic-reservoir-sim-" + str(int(random_seed)).encode()).digest()
    frames = [
        hashlib.sha256(salt + b"init" + i.to_bytes(8, "big")).digest()
        for i in range(max(1, size // 8))
    ]
    ent = np.concatenate([np.frombuffer(f, dtype=np.uint8) for f in frames]).astype(np.float64)
    return (ent[:size] / 127.5 - 1.0) * 0.05


@lru_cache(maxsize=16)
def _base_weights(size: int, input_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """(W_in, unscaled W, spectral radius) with the original RNG sequence."""
    rng = np.random.default_rng(seed)
    W_in = rng.uniform(-0.5, 0.5, size=(size, input_dim))
    W = rng.standard_normal((size, size))
    mask = rng.random((size, size)) < 0.1
    W = W * mask
    np.fill_diagonal(W, np.diag(W) + _sim_entropy(seed, size))
    radius = float(np.max(np.abs(np.linalg.eigvals(W))))
    W_in.setflags(write=False)
    W.setflags(write=False)
    return W_in, W, radius


def build_weights(
    size: int, input_dim: int, spectral_radius: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """``(W_in, W)`` identical to ``HolographicReservoirModel`` with the simulated backend."""
    W_in, W0, radius = _base_weights(int(size), int(input_dim), int(seed))
    W = W0 * (spectral_radius / radius) if radius > 0 else W0.copy()
    return W_in.copy(), W


class BatchedReservoir:
    """Leaky reservoir for a batch of Z zones with shared weights."""

    def __init__(
        self,
        Z: int,
        input_dim: int,
        size: int = 150,
        spectral_radius: float = 0.9,
        leak: float = 0.3,
        seed: int = 0,
        input_scale: float = 1.0,
        dtype=np.float32,
    ) -> None:
        self.Z, self.D, self.N = int(Z), int(input_dim), int(size)
        self.dtype = np.dtype(dtype)
        self.leak = float(leak)
        W_in, W = build_weights(self.N, self.D, spectral_radius, seed)
        self.W = W
        self.W_in = W_in
        # Weights are built in float64 (identical to the original); the dynamics run
        # in float32 for speed (the RLS readout stays in float64).
        self._Wt = np.ascontiguousarray(W.T, dtype=self.dtype)
        self._Wint = np.ascontiguousarray(W_in.T * float(input_scale), dtype=self.dtype)
        self.X = np.zeros((self.Z, self.N), dtype=self.dtype)
        self._pre = np.zeros((self.Z, self.N), dtype=self.dtype)

    def reset(self) -> None:
        self.X[:] = 0.0

    def step(self, U: np.ndarray) -> np.ndarray:
        """Advance one step with input ``U`` (Z x D); return the state (Z x N)."""
        pre = self._pre
        np.matmul(U.astype(self.dtype, copy=False), self._Wint, out=pre)
        pre += self.X @ self._Wt
        np.tanh(pre, out=pre)
        a = self.leak
        self.X *= 1.0 - a
        pre *= a
        self.X += pre
        return self.X


# --------------------------------------------------------------------------
# Shared RLS readout with forgetting
# --------------------------------------------------------------------------


class SharedRLS:
    """RLS with forgetting factor ``lam``; initial ``P = delta * I``; one readout for all zones."""

    def __init__(self, n: int, lam: float = 0.999, delta: float = 100.0, p_max: float = 1e8) -> None:
        self.n = int(n)
        self.lam = float(lam)
        self.delta = float(delta)
        self.p_max = float(p_max)
        self.reset()

    def reset(self) -> None:
        self.w = np.zeros(self.n)
        self.P = np.eye(self.n) * self.delta
        self.n_updates = 0

    def predict(self, F: np.ndarray) -> np.ndarray:
        return F @ self.w

    def update(self, F: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Update with M samples taken at the same instant. Returns the a-priori error."""
        F = np.atleast_2d(F)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        err = y - F @ self.w
        PFt = self.P @ F.T  # n x M
        G = F @ PFt
        G.flat[:: G.shape[0] + 1] += self.lam
        Kt = np.linalg.solve(G, PFt.T)  # M x n (G symmetric => K^T = G^-1 (P F^T)^T)
        self.w = self.w + Kt.T @ err
        self.P = (self.P - PFt @ Kt) / self.lam
        self.n_updates += 1
        if self.n_updates % 8 == 0:  # symmetrise and bound P every 8 updates
            self.P = 0.5 * (self.P + self.P.T)
            m = float(np.max(np.diag(self.P)))
            if not np.isfinite(m):
                raise FloatingPointError("non-finite P in RLS")
            if m > self.p_max:  # guard against P blow-up under poor excitation
                self.P *= self.p_max / m
        elif not np.isfinite(self.w[0]):
            raise FloatingPointError("non-finite RLS weights")
        return err


# --------------------------------------------------------------------------
# Delayed-target predictors (common interface)
# --------------------------------------------------------------------------


class _PredictorBase:
    """Predicts ``y(t+h)`` per zone. ``step(u, y)`` with ``u`` (Z x D) and ``y`` (Z,)."""

    uses_readout = True
    name = "base"

    def __init__(
        self,
        Z: int,
        input_dim: int,
        horizon: int = 12,
        lam: float = 0.999,
        delta: float = 100.0,
        rls_zones: int = 16,
        err_window: int = 288,
        warmup: int = 48,
        max_delta: float = 15.0,
    ) -> None:
        self.Z, self.D, self.h = int(Z), int(input_dim), int(horizon)
        self.lam, self.delta = float(lam), float(delta)
        self.rls_zones = max(1, min(int(rls_zones), self.Z))
        self.err_window = max(1.0, float(err_window))
        self.warmup = int(warmup)
        self.max_delta = float(max_delta)
        self.n_feat = self._n_features()
        self.rls = SharedRLS(self.n_feat, lam, delta) if self.uses_readout else None
        self._ring = self.h + 1
        self.reset()

    # -- to override ------------------------------------------------------
    def _n_features(self) -> int:
        return 0

    def _features(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _reset_extra(self) -> None:
        pass

    # -- state ------------------------------------------------------------
    def reset(self) -> None:
        self.t = 0
        R, Z = self._ring, self.Z
        self._F = np.zeros((R, Z, self.n_feat)) if self.uses_readout else None
        self._Y = np.zeros((R, Z))
        self._Pr = np.full((R, Z), np.nan)
        self._valid = np.zeros(R, dtype=bool)
        self._rot = 0
        self.err_ema = np.inf  # running MAE of the h-step prediction (same unit as y)
        self.persist_err_ema = np.inf  # running MAE of persistence, as a reference
        self.n_err = 0
        self.sq_err = 0.0  # accumulated squared errors (for RMSE reporting)
        self.sq_err_persist = 0.0
        if self.rls is not None:
            self.rls.reset()
        self._reset_extra()

    @property
    def ready(self) -> bool:
        return self.n_err >= self.warmup and np.isfinite(self.err_ema)

    def step(self, u: np.ndarray | None, y: np.ndarray, valid: bool = True) -> np.ndarray | None:
        """Advance one step. With ``valid=False`` only the clock advances."""
        t, h, R = self.t, self.h, self._ring
        slot = t % R
        self.t = t + 1
        if not valid:
            self._valid[slot] = False
            self._Pr[slot] = np.nan
            return None
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        f = self._features(u) if self.uses_readout else None
        if f is not None and not np.all(np.isfinite(f)):
            raise FloatingPointError("non-finite features")
        if f is not None:
            self._F[slot] = f
        self._Y[slot] = y
        self._valid[slot] = True

        # target arrival: y(t) against the prediction made at t-h
        if t >= h:
            o = (t - h) % R
            if self._valid[o]:
                pred_old = self._Pr[o]
                if np.all(np.isfinite(pred_old)):
                    err = y - pred_old
                    perr = y - self._Y[o]
                    mae = float(np.mean(np.abs(err)))
                    pmae = float(np.mean(np.abs(perr)))
                    self.n_err += 1
                    self.sq_err += float(np.mean(err * err))
                    self.sq_err_persist += float(np.mean(perr * perr))
                    a = max(1.0 / self.n_err, 1.0 / self.err_window)
                    if self.n_err == 1:
                        self.err_ema, self.persist_err_ema = mae, pmae
                    else:
                        self.err_ema += a * (mae - self.err_ema)
                        self.persist_err_ema += a * (pmae - self.persist_err_ema)
                if self.uses_readout:
                    rows = self._rows()
                    self.rls.update(self._F[o][rows], (y - self._Y[o])[rows])

        # h-step prediction
        if self.uses_readout:
            delta = np.clip(f @ self.rls.w, -self.max_delta, self.max_delta)
            yhat = y + delta
        else:
            yhat = y.copy()
        if not np.all(np.isfinite(yhat)):
            raise FloatingPointError("non-finite prediction")
        self._Pr[slot] = yhat
        return yhat

    def rmse(self) -> tuple[float, float]:
        """(RMSE of this predictor, RMSE of persistence) over all scored steps."""
        if self.n_err == 0:
            return float("nan"), float("nan")
        return (self.sq_err / self.n_err) ** 0.5, (self.sq_err_persist / self.n_err) ** 0.5

    def _rows(self) -> np.ndarray:
        M = self.rls_zones
        if M >= self.Z:
            return np.arange(self.Z)
        rows = (self._rot + np.arange(M)) % self.Z
        self._rot = (self._rot + M) % self.Z
        return rows


class ReservoirPredictor(_PredictorBase):
    """Shared holographic reservoir + RLS readout. Readout features: ``[x, u]``."""

    name = "hrc"

    def __init__(
        self,
        Z: int,
        input_dim: int,
        size: int = 150,
        spectral_radius: float = 0.9,
        leak: float = 0.3,
        seed: int = 0,
        input_scale: float = 1.0,
        **kw,
    ) -> None:
        self.size = int(size)
        self._D = int(input_dim)
        self.res = BatchedReservoir(
            Z, input_dim, size=self.size, spectral_radius=spectral_radius, leak=leak,
            seed=seed, input_scale=input_scale,
        )
        self._fbuf = np.zeros((int(Z), self.size + int(input_dim)))
        super().__init__(Z, input_dim, **kw)

    def _n_features(self) -> int:
        return self.size + self._D

    def _features(self, u: np.ndarray) -> np.ndarray:
        X = self.res.step(u)
        f = self._fbuf
        f[:, : self.size] = X
        f[:, self.size:] = u
        return f

    def _reset_extra(self) -> None:
        self.res.reset()


class Persistence(_PredictorBase):
    """Ablation: ``y(t+h) = y(t)``. Learns nothing (zero predicted drift)."""

    name = "persistence"
    uses_readout = False

    def _features(self, u):  # pragma: no cover - unused
        return None


class LinearAR(_PredictorBase):
    """Ablation: RLS on the raw inputs ``[u(t), u(t-1), ..., u(t-lags)]``, no reservoir."""

    name = "ar"

    def __init__(self, Z: int, input_dim: int, lags: int = 2, **kw) -> None:
        self.lags = int(lags)
        self._D = int(input_dim)
        super().__init__(Z, input_dim, **kw)

    def _n_features(self) -> int:
        return self._D * (self.lags + 1)

    def _reset_extra(self) -> None:
        self._hist: list[np.ndarray] = []

    def _features(self, u: np.ndarray) -> np.ndarray:
        if not self._hist:
            self._hist = [u.copy() for _ in range(self.lags + 1)]
        else:
            self._hist.insert(0, u.copy())
            self._hist.pop()
        return np.concatenate(self._hist, axis=1)


def make_predictor(kind: str, Z: int, input_dim: int, **kw):
    """``kind`` in {"hrc", "persistence", "ar"}; ``kw`` is split between reservoir and base."""
    kind = kind.lower()
    base_keys = {"horizon", "lam", "delta", "rls_zones", "err_window", "warmup", "max_delta"}
    res_keys = {"size", "spectral_radius", "leak", "seed", "input_scale"}
    base = {k: v for k, v in kw.items() if k in base_keys}
    if kind == "hrc":
        res = {k: v for k, v in kw.items() if k in res_keys}
        return ReservoirPredictor(Z, input_dim, **res, **base)
    if kind == "persistence":
        return Persistence(Z, input_dim, **base)
    if kind == "ar":
        return LinearAR(Z, input_dim, lags=kw.get("lags", 2), **base)
    raise ValueError(f"unknown predictor: {kind!r} (hrc|persistence|ar)")

"""HRC-PI: holographic-reservoir predictive feedforward on top of per-zone PI control.

Idea
----
Every zone of the building runs a PI loop on ``e = T_zone - T_target`` during work
hours (``zone_pi.ZTBestPolicy``). A PI loop only reacts to the error that has already
happened. HRC-PI adds an anticipatory term: a holographic reservoir, shared by all
zones and trained online (RLS readout, no offline training data, no building
identity), predicts each zone's temperature one hour ahead. The predicted drift
``d = T_hat(t+h) - T(t)`` enters the PI error as feedforward,

    e'(t) = T(t) - T_target(t) + kappa * g(t) * d(t),

where ``g(t)`` is a safety gate: it is 1 only when the predictor is warmed up and its
running one-hour MAE is below ``GATE_RATIO`` times the MAE of persistence (i.e. the
reservoir must currently beat "no change"). With ``kappa = 0`` the actions are
bit-identical to the PI controller (checked in ``tests/``).

Everything the predictor sees comes from live observations, the policy's own last
command and building metadata: zone temperature, its one-step change, outside air
temperature, time of day and week, work-hour flag, the zone's last PI output,
damper and reheat commands, occupancy and the serving air handler's run state.
"""
from __future__ import annotations

import numpy as np

from .reservoir import make_predictor
from .zone_pi import ZTBestPolicy, _outside_kelvin

N_FEATURES = 13


class HRCZonePIPolicy(ZTBestPolicy):
    """``ZTBestPolicy`` + reservoir feedforward on the per-zone PI error."""

    KEY = "hrc_pi"
    KAPPA = 0.5            # feedforward gain on the predicted 1 h drift (K -> K)
    PREDICTOR = "hrc"      # "hrc" | "ar" | "persistence" (ablations)
    SEED = 0               # reservoir weight seed
    HORIZON = 12           # 12 x 5 min = 1 h
    GATE_RATIO = 1.0       # predictor MAE must be below persistence MAE x ratio
    MAX_DRIFT_K = 2.0      # clip of the predicted drift used as feedforward
    RES_SIZE = 150
    RES_LEAK = 0.3
    RES_RADIUS = 0.9
    RES_INPUT_SCALE = 1.0
    RLS_LAMBDA = 0.999
    RLS_DELTA = 100.0      # initial P = delta * I (small delta = strong ridge prior towards zero drift)
    RLS_ZONES = 16
    TRACE = False          # keep (features, T) per step for offline predictor studies

    def __init__(self, building_meta_data, action_spec):
        super().__init__(building_meta_data, action_spec)
        rows, seen = [], set()
        for a, ctrl in enumerate(self.ahu_controllers):
            for vav_id, obs_idx, action_indices in ctrl["zones"]:
                if vav_id in seen:
                    continue
                seen.add(vav_id)
                rows.append((vav_id, obs_idx, action_indices or {}, a))
        self._hrc_rows = rows
        self._hrc_index = {v: i for i, (v, _, _, _) in enumerate(rows)}
        self._hrc_obs = np.array([o for _, o, _, _ in rows], dtype=int)
        Z = len(rows)
        self._hrc_damper = [ai.get("supply_air_damper_percentage_command") for _, _, ai, _ in rows]
        self._hrc_reheat = [ai.get("reheat_valve_percentage_command") for _, _, ai, _ in rows]
        self._hrc_ahu = np.array([a for _, _, _, a in rows], dtype=int)
        self._hrc_u = np.zeros((Z, N_FEATURES))
        self._hrc_drift = np.zeros(Z)
        self._hrc_T_prev = None
        self._hrc_last_cont = None
        self._hrc_last_disc = None
        self._hrc_gate = False
        self.trace_u, self.trace_T = [], []
        self.hrc_stats = {"steps": 0, "gate_steps": 0, "ff_abs_sum": 0.0, "ff_n": 0, "fallbacks": 0}
        self._hrc_build_predictor()

    # -- predictor --------------------------------------------------------
    def _hrc_build_predictor(self):
        self._hrc_pred = make_predictor(
            self.PREDICTOR, len(self._hrc_rows), N_FEATURES,
            horizon=self.HORIZON, lam=self.RLS_LAMBDA, delta=self.RLS_DELTA, rls_zones=self.RLS_ZONES,
            size=self.RES_SIZE, leak=self.RES_LEAK, spectral_radius=self.RES_RADIUS,
            input_scale=self.RES_INPUT_SCALE, seed=self.SEED,
        )

    def _hrc_features(self, obs, T):
        u = self._hrc_u
        u[:, 0] = (T - 295.15) / 3.0
        u[:, 1] = 0.0 if self._hrc_T_prev is None else np.clip((T - self._hrc_T_prev) * 5.0, -5, 5)
        oat = _outside_kelvin(self, obs[self._oat_any]) if self._oat_any is not None else 283.15
        u[:, 2] = (oat - 283.15) / 15.0
        u[:, 3] = obs[self.hod_sin_index]
        u[:, 4] = obs[self.hod_cos_index]
        u[:, 5] = obs[self.dow_sin_index]
        u[:, 6] = obs[self.dow_cos_index]
        u[:, 7] = 1.0 if self._pi_active else 0.0
        u[:, 8] = [self._zone_u.get(v, 0.0) for v, _, _, _ in self._hrc_rows]
        cont, disc = self._hrc_last_cont, self._hrc_last_disc
        if cont is not None:
            u[:, 9] = [cont[i] if i is not None else 0.0 for i in self._hrc_damper]
            u[:, 10] = [cont[i] if i is not None else 0.0 for i in self._hrc_reheat]
            runs = np.array([disc[c["run_discrete_idx"]] for c in self.ahu_controllers], dtype=float)
            u[:, 12] = runs[self._hrc_ahu]
        occ = float(obs[self._occ_idx]) if self._occ_idx is not None else 0.0
        u[:, 11] = np.clip(occ, -3.0, 3.0)
        return u

    def _hrc_update(self, obs):
        T = np.array([self._denormalize_zone_temp(obs[j]) for j in self._hrc_obs], dtype=float)
        if not np.all(np.isfinite(T)):
            self._hrc_gate = False
            self._hrc_drift[:] = 0.0
            return
        try:
            u = self._hrc_features(obs, T)
            if self.TRACE:
                self.trace_u.append(u.astype(np.float32))
                self.trace_T.append(T.astype(np.float32))
            yhat = self._hrc_pred.step(u, T)
            p = self._hrc_pred
            self._hrc_gate = bool(p.ready and p.err_ema < self.GATE_RATIO * p.persist_err_ema)
            self._hrc_drift = np.clip(yhat - T, -self.MAX_DRIFT_K, self.MAX_DRIFT_K)
        except (FloatingPointError, np.linalg.LinAlgError):
            # Safe fallback: plain PI, fresh predictor.
            self.hrc_stats["fallbacks"] += 1
            self._hrc_build_predictor()
            self._hrc_gate = False
            self._hrc_drift[:] = 0.0
        self._hrc_T_prev = T
        self.hrc_stats["steps"] += 1
        self.hrc_stats["gate_steps"] += int(self._hrc_gate)

    # -- control hooks ----------------------------------------------------
    def _zone_target(self, obs, lo, hi):
        target = super()._zone_target(obs, lo, hi)
        if self.KAPPA and self._hrc_gate:
            i = self._hrc_index.get(getattr(self, "_cur_vav", None))
            if i is not None:
                ff = self.KAPPA * float(self._hrc_drift[i])
                self.hrc_stats["ff_abs_sum"] += abs(ff)
                self.hrc_stats["ff_n"] += 1
                return target - ff
        return target

    def _base_action(self, obs):
        self._hrc_update(obs)
        return super()._base_action(obs)

    def get_action(self, obs):
        action = super().get_action(obs)
        self._hrc_last_cont = np.array(action["continuous_action"], dtype=float)
        self._hrc_last_disc = np.array(action["discrete_action"], dtype=int)
        return action

    def predictor_rmse(self):
        """(predictor RMSE, persistence RMSE) of the 1 h zone-temperature forecast, in K."""
        return self._hrc_pred.rmse()


def make_policy_class(kappa=0.5, predictor="hrc", seed=0, **overrides):
    """Return a ``HRCZonePIPolicy`` subclass with the given settings (for runners)."""
    attrs = dict(KAPPA=float(kappa), PREDICTOR=str(predictor), SEED=int(seed))
    attrs.update({k.upper(): v for k, v in overrides.items()})
    return type(f"HRCZonePI_{predictor}_k{kappa}_s{seed}", (HRCZonePIPolicy,), attrs)

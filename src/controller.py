"""Bumpless Transfer + Gravity-Compensated Joint Controller (G + PID + Anti-Windup)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import mujoco

from src.ik_solver import quintic_blend


class ControlMode(Enum):
    """Controller authority state machine."""

    PASSIVE = auto()   # External / idle / zero-torque or hold
    ENGAGING = auto()  # Bumpless handoff ramp window
    ACTIVE = auto()    # Full closed-loop G + PID + AW tracking


@dataclass
class ControllerGains:
    """Per-DOF gains calibrated for Firefly Dual-Arm."""

    kp: np.ndarray
    kd: np.ndarray
    ki: np.ndarray
    tau_limit: np.ndarray
    i_clamp: np.ndarray
    anti_windup_kb: float = 2.0

    @classmethod
    def default_for_model(cls, model: mujoco.MjModel) -> "ControllerGains":
        """Build default gains matching J1-J4 (20 Nm), J5-J6 (5 Nm), and gripper (5 Nm)."""
        nv = model.nv
        kp = np.zeros(nv)
        kd = np.zeros(nv)
        ki = np.zeros(nv)
        tau_limit = np.zeros(nv)
        i_clamp = np.zeros(nv)

        for jid in range(model.njnt):
            j = model.joint(jid)
            idx_v = j.dofadr[0]
            jname = j.name
            # Check _grip_joint_ BEFORE _joint_ because _grip_joint_ contains _joint_!
            if "_grip_joint_" in jname:
                if jname.endswith("_grip_joint_1"):
                    kp[idx_v], kd[idx_v], ki[idx_v] = 28.0, 1.8, 12.0
                    tau_limit[idx_v], i_clamp[idx_v] = 5.0, 0.25
                else:
                    # Passive mimic follower joint (no direct actuator)
                    kp[idx_v], kd[idx_v], ki[idx_v] = 0.0, 0.0, 0.0
                    tau_limit[idx_v], i_clamp[idx_v] = 0.0, 0.0
            elif "_joint_" in jname:
                num = int(jname.rsplit("_", 1)[1])
                if num <= 2:
                    kp[idx_v], kd[idx_v], ki[idx_v] = 240.0, 18.0, 320.0
                    tau_limit[idx_v], i_clamp[idx_v] = 20.0, 0.12
                elif num <= 4:
                    kp[idx_v], kd[idx_v], ki[idx_v] = 180.0, 14.0, 260.0
                    tau_limit[idx_v], i_clamp[idx_v] = 20.0, 0.12
                else:
                    kp[idx_v], kd[idx_v], ki[idx_v] = 65.0, 4.0, 95.0
                    tau_limit[idx_v], i_clamp[idx_v] = 5.0, 0.12

        return cls(
            kp=kp,
            kd=kd,
            ki=ki,
            tau_limit=tau_limit,
            i_clamp=i_clamp,
        )


class BumplessGravityController:
    """Joint-space torque controller with exact model-based gravity compensation,
    back-calculation anti-windup PID, and C^2 bumpless control-authority transfer.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        gains: ControllerGains | None = None,
        engage_duration: float = 0.30,
    ) -> None:
        self.model = model
        # Dedicated dynamics twin (only uses q_meas, never sim internals -> real-robot ready)
        self._dyn_data = mujoco.MjData(model)
        self.gains = gains or ControllerGains.default_for_model(model)
        self.engage_duration = float(engage_duration)

        # Map model actuators (nu) to generalized velocity DOFs (nv)
        self.act_dofs = np.array(
            [int(model.joint(model.actuator_trnid[a, 0]).dofadr[0]) for a in range(model.nu)],
            dtype=int,
        )

        self.mode = ControlMode.PASSIVE
        self._engage_elapsed = 0.0
        self._e_int = np.zeros(model.nv)
        self._tau_handoff_start = np.zeros(model.nv)
        self._q_latch = np.zeros(model.nq)
        self._last_tau = np.zeros(model.nu)

    def compute_gravity_torque(self, q_meas: np.ndarray) -> np.ndarray:
        """Compute static gravity compensation torque g(q_meas) via RNE (qvel=0), shape (nv,)."""
        self._dyn_data.qpos[:] = q_meas
        self._dyn_data.qvel[:] = 0.0
        self._dyn_data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self._dyn_data)
        return self._dyn_data.qfrc_bias.copy()

    def compute_gravity_ctrl(self, q_meas: np.ndarray) -> np.ndarray:
        """Compute gravity feedforward actuator vector matching model.nu."""
        return self.compute_gravity_torque(q_meas)[self.act_dofs].copy()

    def engage(
        self,
        q_meas: np.ndarray,
        dq_meas: np.ndarray,
        tau_prev: np.ndarray | None = None,
    ) -> None:
        """Initiate bumpless transfer from PASSIVE/external control to ACTIVE.

        1. Latches current measured state so initial position error e = 0.
        2. Pre-loads integrator state so tau_cmd(0+) matches tau_prev (or g(q_meas)).
        3. Starts C^2 quintic authority ramp over engage_duration.
        """
        tau_g = self.compute_gravity_torque(q_meas)
        if tau_prev is None:
            tau_prev_nv = tau_g.copy()
        elif len(tau_prev) == self.model.nu:
            tau_prev_nv = tau_g.copy()
            tau_prev_nv[self.act_dofs] = tau_prev
        else:
            tau_prev_nv = tau_prev.copy()

        self._q_latch = q_meas.copy()
        self._tau_handoff_start = tau_prev_nv.copy()

        # Pre-load integrator so Ki * e_int(0+) + tau_g - Kd * dq_meas == tau_prev
        tau_residual = tau_prev_nv - tau_g + self.gains.kd * dq_meas
        safe_ki = np.where(self.gains.ki > 1e-6, self.gains.ki, 1.0)
        e_int_init = np.where(self.gains.ki > 1e-6, tau_residual / safe_ki, 0.0)
        self._e_int = np.clip(e_int_init, -self.gains.i_clamp, self.gains.i_clamp)

        self._engage_elapsed = 0.0
        self.mode = ControlMode.ENGAGING if self.engage_duration > 0.0 else ControlMode.ACTIVE

    def disengage(self) -> None:
        """Return to PASSIVE mode."""
        self.mode = ControlMode.PASSIVE
        self._e_int[:] = 0.0

    def step(
        self,
        dt: float,
        q_meas: np.ndarray,
        dq_meas: np.ndarray,
        q_ref: np.ndarray,
        dq_ref: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute actuator torques (shape model.nu) for one control tick."""
        if self.mode == ControlMode.PASSIVE:
            self._last_tau[:] = 0.0
            return self._last_tau.copy()

        if dq_ref is None:
            dq_ref = np.zeros(self.model.nv)

        # Smoothly blend reference from latched pose during ENGAGING window
        if self.mode == ControlMode.ENGAGING:
            self._engage_elapsed += dt
            u = min(1.0, self._engage_elapsed / max(self.engage_duration, 1e-6))
            alpha, _ = quintic_blend(u)
            q_target = (1.0 - alpha) * self._q_latch + alpha * q_ref
            dq_target = alpha * dq_ref
            if u >= 1.0:
                self.mode = ControlMode.ACTIVE
        else:
            alpha = 1.0
            q_target = q_ref
            dq_target = dq_ref

        pos_err = q_target - q_meas
        vel_err = dq_target - dq_meas

        tau_g = self.compute_gravity_torque(q_meas)

        # Unsaturated G + PID torque
        tau_unsat = (
            tau_g
            + self.gains.kp * pos_err
            + self.gains.kd * vel_err
            + self.gains.ki * self._e_int
        )

        # Blend from previous controller torque during early engage window
        if alpha < 1.0:
            tau_unsat = (1.0 - alpha) * self._tau_handoff_start + alpha * tau_unsat

        tau_sat = np.clip(tau_unsat, -self.gains.tau_limit, self.gains.tau_limit)

        # Back-calculation anti-windup on integrator
        sat_diff = tau_sat - tau_unsat
        self._e_int += (pos_err + self.gains.anti_windup_kb * sat_diff) * dt
        self._e_int = np.clip(self._e_int, -self.gains.i_clamp, self.gains.i_clamp)

        self._last_tau = tau_sat[self.act_dofs].copy()
        return self._last_tau.copy()

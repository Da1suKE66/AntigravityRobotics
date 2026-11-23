"""Collision-free trajectory & path planner for Firefly Dual-Arm."""

from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass
import numpy as np
import mujoco
import mink

from src.ik_solver import FireflyDualIK, quintic_blend


@dataclass
class PlannedTrajectory:
    """Time-parameterized joint & TCP trajectory with collision verification metadata."""

    mode: str
    dt: float
    t: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    left_tcp_xyz: np.ndarray
    right_tcp_xyz: np.ndarray
    min_dist_mm: float
    max_ncon: int
    collision_free: bool

    def sample(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate (q_ref, dq_ref) at query time."""
        if time_s <= self.t[0]:
            return self.q[0].copy(), self.dq[0].copy()
        if time_s >= self.t[-1]:
            return self.q[-1].copy(), np.zeros_like(self.dq[-1])
        idx = int(np.searchsorted(self.t, time_s, side="right")) - 1
        idx = int(np.clip(idx, 0, len(self.t) - 2))
        alpha = (time_s - self.t[idx]) / max(self.t[idx + 1] - self.t[idx], 1e-9)
        q_out = (1.0 - alpha) * self.q[idx] + alpha * self.q[idx + 1]
        dq_out = (1.0 - alpha) * self.dq[idx] + alpha * self.dq[idx + 1]
        return q_out, dq_out

    def save(self, out_dir: pathlib.Path, stem: str) -> tuple[pathlib.Path, pathlib.Path]:
        """Persist planned path to JSON (summary + waypoints) and CSV (full dense trajectory)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{stem}.json"
        csv_path = out_dir / f"{stem}.csv"

        stride = max(1, len(self.t) // 100)
        summary = {
            "name": stem,
            "mode": self.mode,
            "dt_s": float(self.dt),
            "duration_s": float(self.t[-1]),
            "num_samples": int(len(self.t)),
            "collision_free": bool(self.collision_free),
            "max_ncon": int(self.max_ncon),
            "min_dist_mm": float(self.min_dist_mm),
            "start_q_rad": self.q[0].tolist(),
            "goal_q_rad": self.q[-1].tolist(),
            "keyframes": [
                {
                    "t_s": float(self.t[i]),
                    "q_rad": np.round(self.q[i], 6).tolist(),
                    "left_tcp_xyz_m": np.round(self.left_tcp_xyz[i], 6).tolist(),
                    "right_tcp_xyz_m": np.round(self.right_tcp_xyz[i], 6).tolist(),
                }
                for i in sorted(set(list(range(0, len(self.t), stride)) + [len(self.t) - 1]))
            ],
        }
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            q_cols = [f"q_{i}" for i in range(self.q.shape[1])]
            dq_cols = [f"dq_{i}" for i in range(self.dq.shape[1])]
            writer.writerow(
                ["t_s", "l_x", "l_y", "l_z", "r_x", "r_y", "r_z", *q_cols, *dq_cols]
            )
            for i in range(len(self.t)):
                writer.writerow(
                    [
                        f"{self.t[i]:.4f}",
                        *[f"{v:.6f}" for v in self.left_tcp_xyz[i]],
                        *[f"{v:.6f}" for v in self.right_tcp_xyz[i]],
                        *[f"{v:.6f}" for v in self.q[i]],
                        *[f"{v:.6f}" for v in self.dq[i]],
                    ]
                )
        return json_path, csv_path


class FireflyPlanner:
    """Collision-aware joint & Cartesian trajectory planner on the exact MuJoCo scene."""

    def __init__(self, sim_model: mujoco.MjModel, ik: FireflyDualIK) -> None:
        self.sim_model = sim_model
        self.ik = ik
        self._col_data = mujoco.MjData(sim_model)

    def _eval_trajectory_collisions(
        self, q_traj: np.ndarray, check_stride: int = 5
    ) -> tuple[int, float]:
        max_ncon = 0
        min_dist_mm = 1e6
        n_qpos = min(q_traj.shape[1], self.sim_model.nq)
        indices = list(range(0, len(q_traj), check_stride))
        if len(q_traj) - 1 not in indices:
            indices.append(len(q_traj) - 1)
        for i in indices:
            self._col_data.qpos[:n_qpos] = q_traj[i, :n_qpos]
            mujoco.mj_forward(self.sim_model, self._col_data)
            ncon = int(self._col_data.ncon)
            if ncon > max_ncon:
                max_ncon = ncon
            for c in self._col_data.contact:
                min_dist_mm = min(min_dist_mm, float(c.dist * 1000.0))
        if max_ncon == 0:
            min_dist_mm = 0.0
        return max_ncon, min_dist_mm

    def _compute_tcp_trace(self, q_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        l_xyz = np.zeros((len(q_traj), 3), dtype=float)
        r_xyz = np.zeros((len(q_traj), 3), dtype=float)
        n_qpos = min(q_traj.shape[1], self.sim_model.nq)
        for i in range(len(q_traj)):
            self._col_data.qpos[:n_qpos] = q_traj[i, :n_qpos]
            mujoco.mj_kinematics(self.sim_model, self._col_data)
            l_xyz[i] = self._col_data.site("left_tcp_site").xpos
            r_xyz[i] = self._col_data.site("right_tcp_site").xpos
        return l_xyz, r_xyz

    def _quintic_segment(
        self, q_start: np.ndarray, q_goal: np.ndarray, duration: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        steps = max(2, int(round(duration / dt)) + 1)
        t = np.linspace(0.0, duration, steps)
        q = np.zeros((steps, len(q_start)), dtype=float)
        dq = np.zeros((steps, len(q_start)), dtype=float)
        delta = q_goal - q_start
        for i, ti in enumerate(t):
            s, ds = quintic_blend(ti / max(duration, 1e-6))
            q[i] = q_start + s * delta
            dq[i] = (ds / max(duration, 1e-6)) * delta
        return t, q, dq

    def plan_joint_p2p(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        duration: float = 3.5,
        dt: float = 0.002,
    ) -> PlannedTrajectory:
        """Plan collision-free joint-space P2P trajectory (synchronous, left_first, right_first, or via)."""
        l_dofs = self.ik.left_arm_dofs
        r_dofs = self.ik.right_arm_dofs

        # 1. Synchronous direct quintic interpolation
        t_sync, q_sync, dq_sync = self._quintic_segment(q_start, q_goal, duration, dt)
        ncon, min_d = self._eval_trajectory_collisions(q_sync)
        if ncon == 0:
            l_xyz, r_xyz = self._compute_tcp_trace(q_sync)
            return PlannedTrajectory("synchronous", dt, t_sync, q_sync, dq_sync, l_xyz, r_xyz, min_d, 0, True)

        # 2. Sequential left_first / right_first for crossing maneuvers (e.g. H05)
        half_dur = max(duration * 0.55, 2.0)
        for order, first_dofs in (("left_first", l_dofs), ("right_first", r_dofs)):
            q_mid = q_start.copy()
            q_mid[first_dofs] = q_goal[first_dofs]
            t1, q1, dq1 = self._quintic_segment(q_start, q_mid, half_dur, dt)
            t2, q2, dq2 = self._quintic_segment(q_mid, q_goal, half_dur, dt)
            t_seq = np.concatenate([t1, t1[-1] + t2[1:]])
            q_seq = np.vstack([q1, q2[1:]])
            dq_seq = np.vstack([dq1, dq2[1:]])
            ncon_s, min_ds = self._eval_trajectory_collisions(q_seq)
            if ncon_s == 0:
                l_xyz, r_xyz = self._compute_tcp_trace(q_seq)
                return PlannedTrajectory(order, dt, t_seq, q_seq, dq_seq, l_xyz, r_xyz, min_ds, 0, True)

        # 3. Retreat via home posture midpoint if direct/sequential paths graze obstacles
        if self.ik.model.nkey > 0:
            q_home = q_start.copy()
            home = self.ik.model.key("home").qpos
            q_home[l_dofs + r_dofs] = home[l_dofs + r_dofs]
            t1, q1, dq1 = self._quintic_segment(q_start, q_home, half_dur, dt)
            t2, q2, dq2 = self._quintic_segment(q_home, q_goal, half_dur, dt)
            t_via = np.concatenate([t1, t1[-1] + t2[1:]])
            q_via = np.vstack([q1, q2[1:]])
            dq_via = np.vstack([dq1, dq2[1:]])
            ncon_v, min_dv = self._eval_trajectory_collisions(q_via)
            if ncon_v == 0:
                l_xyz, r_xyz = self._compute_tcp_trace(q_via)
                return PlannedTrajectory("home_via", dt, t_via, q_via, dq_via, l_xyz, r_xyz, min_dv, 0, True)

        # Fallback: return synchronous path with collision metadata flagged
        l_xyz, r_xyz = self._compute_tcp_trace(q_sync)
        return PlannedTrajectory("synchronous_fallback", dt, t_sync, q_sync, dq_sync, l_xyz, r_xyz, min_d, ncon, False)

    def plan_pose_p2p(
        self,
        q_start: np.ndarray,
        left_target: mink.SE3 | None = None,
        right_target: mink.SE3 | None = None,
        duration: float = 3.5,
        dt: float = 0.002,
    ) -> tuple[PlannedTrajectory, dict[str, float | int | bool]]:
        """Solve global IK for target SE(3) pose(s) and plan a collision-free joint trajectory."""
        q_goal, ik_info = self.ik.solve_pose_global(
            q_start,
            left_target=left_target,
            right_target=right_target,
            collision_check_model=self.sim_model,
        )
        traj = self.plan_joint_p2p(q_start, q_goal, duration=duration, dt=dt)
        return traj, ik_info

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

        # Build per-DOF velocity limits from URDF specs
        self.vel_limits = np.full(ik.model.nv, 5.585053606381854, dtype=float)
        for jid in range(ik.model.njnt):
            j = ik.model.joint(jid)
            jname = j.name
            dof = int(j.dofadr[0])
            if "_grip_joint_" in jname:
                self.vel_limits[dof] = 1.5
            elif "_joint_" in jname:
                idx = int(jname.rsplit("_", 1)[1])
                self.vel_limits[dof] = 5.585053606381854 if idx <= 4 else 12.566370614359172

    def _validate_and_clamp_q(self, q: np.ndarray, label: str) -> np.ndarray:
        """Verify joint positions respect model limits, raising ValueError on violations."""
        q_out = q.copy()
        tol = 1e-6
        for jid in range(self.ik.model.njnt):
            if not self.ik.model.jnt_limited[jid]:
                continue
            adr = int(self.ik.model.jnt_qposadr[jid])
            lo, hi = self.ik.model.jnt_range[jid]
            val = float(q_out[adr])
            if val < lo - tol or val > hi + tol:
                jname = self.ik.model.joint(jid).name
                raise ValueError(
                    f"{label} joint '{jname}'={val:.6f} rad exceeds limit [{lo:.6f}, {hi:.6f}]"
                )
            q_out[adr] = float(np.clip(val, lo, hi))
        return q_out

    def _eval_trajectory_collisions(
        self,
        q_traj: np.ndarray,
        check_stride: int = 1,
        scene_qpos_tail: np.ndarray | None = None,
        allowed_body_pairs: set[tuple[str, str]] | None = None,
    ) -> tuple[int, float]:
        max_ncon = 0
        min_dist_mm = 1e6
        n_robot = self.ik.model.nq
        indices = list(range(0, len(q_traj), max(1, check_stride)))
        if len(q_traj) - 1 not in indices:
            indices.append(len(q_traj) - 1)
        for i in indices:
            self._col_data.qpos[:n_robot] = q_traj[i, :n_robot]
            if scene_qpos_tail is not None and self.sim_model.nq > n_robot:
                n_tail = min(len(scene_qpos_tail), self.sim_model.nq - n_robot)
                self._col_data.qpos[n_robot : n_robot + n_tail] = scene_qpos_tail[:n_tail]
            mujoco.mj_forward(self.sim_model, self._col_data)
            ncon = FireflyDualIK.count_robot_collisions(
                self.sim_model, self._col_data, allowed_body_pairs
            )
            if ncon > max_ncon:
                max_ncon = ncon
            if ncon > 0:
                for c in self._col_data.contact:
                    b1 = self.sim_model.body(self.sim_model.geom_bodyid[c.geom1]).name
                    b2 = self.sim_model.body(self.sim_model.geom_bodyid[c.geom2]).name
                    if not (b1.startswith(("left_", "right_")) or b2.startswith(("left_", "right_"))):
                        continue
                    if allowed_body_pairs is not None and tuple(sorted((b1, b2))) in allowed_body_pairs:
                        continue
                    min_dist_mm = min(min_dist_mm, float(c.dist * 1000.0))
        if max_ncon == 0:
            min_dist_mm = 0.0
        return max_ncon, min_dist_mm

    def _compute_tcp_trace(self, q_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        l_xyz = np.zeros((len(q_traj), 3), dtype=float)
        r_xyz = np.zeros((len(q_traj), 3), dtype=float)
        n_robot = self.ik.model.nq
        for i in range(len(q_traj)):
            self._col_data.qpos[:n_robot] = q_traj[i, :n_robot]
            mujoco.mj_kinematics(self.sim_model, self._col_data)
            l_xyz[i] = self._col_data.site("left_tcp_site").xpos
            r_xyz[i] = self._col_data.site("right_tcp_site").xpos
        return l_xyz, r_xyz

    def _quintic_segment(
        self, q_start: np.ndarray, q_goal: np.ndarray, duration: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        delta = q_goal - q_start
        # Quintic peak velocity is 1.875 * |delta| / T; stretch T to guarantee |dq| <= vel_limits
        n_v = min(len(delta), len(self.vel_limits))
        min_dur = float(np.max(1.875 * np.abs(delta[:n_v]) / self.vel_limits[:n_v]))
        actual_dur = max(float(duration), min_dur * 1.01, dt)

        steps = max(2, int(np.ceil(actual_dur / dt)) + 1)
        t = np.linspace(0.0, actual_dur, steps)
        q = np.zeros((steps, len(q_start)), dtype=float)
        dq = np.zeros((steps, len(q_start)), dtype=float)
        for i, ti in enumerate(t):
            s, ds = quintic_blend(ti / actual_dur)
            q[i] = q_start + s * delta
            dq[i] = (ds / actual_dur) * delta
        return t, q, dq

    def plan_joint_p2p(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
        duration: float = 3.5,
        dt: float = 0.002,
        scene_qpos_tail: np.ndarray | None = None,
        allowed_body_pairs: set[tuple[str, str]] | None = None,
    ) -> PlannedTrajectory:
        """Plan limit-respecting, collision-checked joint-space P2P trajectory."""
        q_start_c = self._validate_and_clamp_q(q_start, "start")
        q_goal_c = self._validate_and_clamp_q(q_goal, "goal")

        l_dofs = self.ik.left_arm_dofs
        r_dofs = self.ik.right_arm_dofs

        # 1. Synchronous direct quintic interpolation
        t_sync, q_sync, dq_sync = self._quintic_segment(q_start_c, q_goal_c, duration, dt)
        ncon, min_d = self._eval_trajectory_collisions(
            q_sync, check_stride=1, scene_qpos_tail=scene_qpos_tail, allowed_body_pairs=allowed_body_pairs
        )
        if ncon == 0:
            l_xyz, r_xyz = self._compute_tcp_trace(q_sync)
            return PlannedTrajectory("synchronous", dt, t_sync, q_sync, dq_sync, l_xyz, r_xyz, min_d, 0, True)

        # 2. Sequential left_first / right_first for crossing maneuvers (e.g. H05)
        half_dur = max(duration * 0.55, 2.0)
        for order, first_dofs in (("left_first", l_dofs), ("right_first", r_dofs)):
            q_mid = q_start_c.copy()
            q_mid[first_dofs] = q_goal_c[first_dofs]
            t1, q1, dq1 = self._quintic_segment(q_start_c, q_mid, half_dur, dt)
            t2, q2, dq2 = self._quintic_segment(q_mid, q_goal_c, half_dur, dt)
            t_seq = np.concatenate([t1, t1[-1] + t2[1:]])
            q_seq = np.vstack([q1, q2[1:]])
            dq_seq = np.vstack([dq1, dq2[1:]])
            ncon_s, min_ds = self._eval_trajectory_collisions(
                q_seq, check_stride=1, scene_qpos_tail=scene_qpos_tail, allowed_body_pairs=allowed_body_pairs
            )
            if ncon_s == 0:
                l_xyz, r_xyz = self._compute_tcp_trace(q_seq)
                return PlannedTrajectory(order, dt, t_seq, q_seq, dq_seq, l_xyz, r_xyz, min_ds, 0, True)

        # 3. Retreat via home posture midpoint if direct/sequential paths graze obstacles
        if self.ik.model.nkey > 0:
            q_home = q_start_c.copy()
            home = self.ik.model.key("home").qpos
            q_home[l_dofs + r_dofs] = home[l_dofs + r_dofs]
            t1, q1, dq1 = self._quintic_segment(q_start_c, q_home, half_dur, dt)
            t2, q2, dq2 = self._quintic_segment(q_home, q_goal_c, half_dur, dt)
            t_via = np.concatenate([t1, t1[-1] + t2[1:]])
            q_via = np.vstack([q1, q2[1:]])
            dq_via = np.vstack([dq1, dq2[1:]])
            ncon_v, min_dv = self._eval_trajectory_collisions(
                q_via, check_stride=1, scene_qpos_tail=scene_qpos_tail, allowed_body_pairs=allowed_body_pairs
            )
            if ncon_v == 0:
                l_xyz, r_xyz = self._compute_tcp_trace(q_via)
                return PlannedTrajectory("home_via", dt, t_via, q_via, dq_via, l_xyz, r_xyz, min_dv, 0, True)

        # Fallback: return synchronous path explicitly flagged collision_free=False
        l_xyz, r_xyz = self._compute_tcp_trace(q_sync)
        return PlannedTrajectory("synchronous_collision", dt, t_sync, q_sync, dq_sync, l_xyz, r_xyz, min_d, ncon, False)

    def plan_pose_p2p(
        self,
        q_start: np.ndarray,
        left_target: mink.SE3 | None = None,
        right_target: mink.SE3 | None = None,
        duration: float = 3.5,
        dt: float = 0.002,
        scene_qpos_tail: np.ndarray | None = None,
        allowed_body_pairs: set[tuple[str, str]] | None = None,
        limit_margin_rad: float = 6e-4,
    ) -> tuple[PlannedTrajectory, dict[str, float | int | bool]]:
        """Solve global IK for target SE(3) pose(s) and plan a collision-free joint trajectory."""
        q_goal, ik_info = self.ik.solve_pose_global(
            q_start,
            left_target=left_target,
            right_target=right_target,
            collision_check_model=self.sim_model,
            scene_qpos_tail=scene_qpos_tail,
            allowed_body_pairs=allowed_body_pairs,
        )
        # Apply soft joint-limit buffer for dynamic P2P execution so servo tracking never strikes hard stops
        if ik_info.get("converged", False) and limit_margin_rad > 0.0:
            arm_dofs = set(self.ik.left_arm_dofs + self.ik.right_arm_dofs)
            for jid in range(self.ik.model.njnt):
                if not self.ik.model.jnt_limited[jid]:
                    continue
                adr = int(self.ik.model.jnt_qposadr[jid])
                if adr not in arm_dofs:
                    continue
                lo, hi = self.ik.model.jnt_range[jid]
                if hi - lo > 2.0 * limit_margin_rad:
                    q_goal[adr] = float(np.clip(q_goal[adr], lo + limit_margin_rad, hi - limit_margin_rad))
        traj = self.plan_joint_p2p(
            q_start,
            q_goal,
            duration=duration,
            dt=dt,
            scene_qpos_tail=scene_qpos_tail,
            allowed_body_pairs=allowed_body_pairs,
        )
        return traj, ik_info

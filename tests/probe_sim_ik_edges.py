"""Exercise planner joint limits, post-contact accounting and rated-torque grasping."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mink
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import grasp_pipeline as grasp
from src.controller import ControlMode, ControllerGains
from src.ik_solver import FireflyDualIK
from src.planner import FireflyPlanner


class PlannerBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        robot = mujoco.MjModel.from_xml_path(str(ROOT / "model/firefly_dual.xml"))
        scene = mujoco.MjModel.from_xml_path(str(ROOT / "model/scene.xml"))
        self.data = mujoco.MjData(scene)
        self.scene = scene
        self.start = robot.key("home").qpos.copy()
        joint = robot.joint("left_joint_1")
        self.start[joint.qposadr[0]] = joint.range[1]
        self.data.qpos[:] = self.start
        mujoco.mj_forward(scene, self.data)
        self.assertEqual(self.data.ncon, 0)
        self.ik = FireflyDualIK(robot, site_suffix="tcp_site")
        self.left, self.right = self.ik.reset_to_measured(self.start)
        self.planner = FireflyPlanner(scene, self.ik)

    def test_reported_residual_matches_planned_endpoint(self) -> None:
        path, info = self.planner.plan_pose_p2p(self.start, left_target=self.left, duration=0.5)
        self.data.qpos[:] = path.q[-1]
        mujoco.mj_forward(self.scene, self.data)
        residual_mm = 1000 * np.linalg.norm(self.data.site("left_tcp_site").xpos - self.left.translation())
        self.assertAlmostEqual(float(info["pos_err_mm"]), residual_mm, delta=1e-4)

    def test_right_arm_command_preserves_left_arm(self) -> None:
        goal = mink.SE3.from_rotation_and_translation(
            self.right.rotation(), self.right.translation() + [0.005, 0.0, 0.0]
        )
        path, info = self.planner.plan_pose_p2p(self.start, right_target=goal, duration=0.5)
        self.assertTrue(info["converged"] and path.collision_free)
        left = self.ik.left_arm_dofs
        drift = float(np.max(np.abs(path.q[:, left] - self.start[left])))
        self.assertLessEqual(drift, 1e-8, f"Uncommanded left-arm displacement was {drift:.6f} rad")


class GraspBoundaryTests(unittest.TestCase):
    def test_safe_hold_counts_contacts_and_peak_torque(self) -> None:
        executor = grasp.GraspExecutor(record_video=False)
        self.addCleanup(executor.close)
        model, data = executor.sim_model, executor.data
        obj = model.joint("blue_cylinder_freejoint").qposadr[0]
        data.qpos[obj:obj + 3] = data.geom("left_grip_link_1_col_0").xpos.copy()
        mujoco.mj_forward(model, data)
        with self.assertRaisesRegex(RuntimeError, "forbidden contact"):
            executor._step_control(executor.q_cmd.copy(), np.zeros(executor.dyn_model.nv))
        self.assertEqual(executor.ctrl.mode, ControlMode.SAFE_HOLD)
        contacts = executor.runtime_forbidden_contacts
        peak = executor.peak_tau
        for _ in range(round(0.6 / executor.dt)):
            tau = executor.step_safe_hold()
            contacts += FireflyDualIK.count_robot_collisions(model, data, set())
            peak = max(peak, float(np.max(np.abs(tau))))
        self.assertTrue(
            executor.runtime_forbidden_contacts == contacts and abs(executor.peak_tau - peak) < 1e-6,
            f"Reported {executor.runtime_forbidden_contacts} contact points and {executor.peak_tau:.3f} N·m. "
            f"Observed {contacts} contact points and {peak:.3f} N·m.",
        )

    def test_grasp_with_dataset_rated_torque(self) -> None:
        manifest = json.loads((ROOT / "ik_regression_25/manifest.json").read_text())
        ratings = manifest["physics"]["rated_nm_per_arm"]
        defaults = ControllerGains.default_for_model

        def rated_gains(model: mujoco.MjModel) -> ControllerGains:
            gains = defaults(model)
            for side in ("left", "right"):
                for i, limit in enumerate(ratings, start=1):
                    gains.tau_limit[model.joint(f"{side}_joint_{i}").dofadr[0]] = limit
            return gains

        with TemporaryDirectory() as tmp:
            with (
                patch.object(ControllerGains, "default_for_model", staticmethod(rated_gains)),
                patch.object(grasp, "PATHS_DIR", Path(tmp) / "paths"),
                patch.object(grasp, "REPORTS_DIR", Path(tmp) / "reports"),
            ):
                try:
                    report = grasp.run_dual_arm_pick_and_place(record_video=False)
                except RuntimeError as error:
                    if "forbidden contact" not in str(error):
                        raise
                    raise self.failureException(str(error)) from None
        self.assertTrue(report["all_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Measure finger speeds in the default demo and observe the response to a moving object."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mink
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import grasp_pipeline as grasp


def check_default_finger_speeds() -> bool:
    urdf = ET.parse(ROOT / "assets/firefly_dual.urdf")
    joints = [f"{side}_grip_joint_1" for side in ("left", "right")]
    limits = {j: float(urdf.find(f"./joint[@name='{j}']/limit").attrib["velocity"]) for j in joints}
    peaks: dict[tuple[str, str], tuple[float, float]] = {}
    advance = grasp.GraspExecutor._step_control

    def observe(executor: grasp.GraspExecutor, q_ref: np.ndarray, dq_ref: np.ndarray) -> None:
        advance(executor, q_ref, dq_ref)
        for name in joints:
            dof = executor.sim_model.joint(name).dofadr[0]
            key = (executor._current_label, name)
            ref, measured = peaks.get(key, (0.0, 0.0))
            peaks[key] = max(ref, abs(dq_ref[dof])), max(measured, abs(executor.data.qvel[dof]))

    with TemporaryDirectory() as tmp:
        with (
            patch.object(grasp, "REPORTS_DIR", Path(tmp) / "reports"),
            patch.object(grasp, "PATHS_DIR", Path(tmp) / "paths"),
            patch.object(grasp.GraspExecutor, "_step_control", observe),
        ):
            report = grasp.run_dual_arm_pick_and_place(record_video=False)

    within_limits = True
    for (stage, joint), (ref, measured) in peaks.items():
        if max(ref, measured) > limits[joint] + 1e-6:
            print(f"{stage} {joint}: command {ref:.3f}, measured {measured:.3f}, limit {limits[joint]} rad/s")
            within_limits = False
    return report["all_passed"] and within_limits


def observe_unexpected_contact() -> None:
    executor = grasp.GraspExecutor(record_video=False)
    try:
        left, _ = executor.ik.reset_to_measured(executor.q_cmd)
        goal = mink.SE3.from_rotation_and_translation(left.rotation(), left.translation() + [0.015, 0.0, -0.012])
        advance = executor._step_control
        ticks = 0
        first = None

        def disturb(q_ref: np.ndarray, dq_ref: np.ndarray) -> None:
            nonlocal ticks, first
            ticks += 1
            if ticks == 100:
                # Introduce an unexpected object after the executor has accepted the path.
                joint = executor.sim_model.joint("blue_cylinder_freejoint")
                pos = joint.qposadr[0]
                vel = joint.dofadr[0]
                executor.data.qpos[pos:pos + 3] = executor.data.geom("left_grip_link_1_col_0").xpos.copy()
                executor.data.qvel[vel:vel + 6] = 0
                mujoco.mj_forward(executor.sim_model, executor.data)
            try:
                advance(q_ref, dq_ref)
            finally:
                if first is None and executor.runtime_forbidden_contacts:
                    first = ticks

        with patch.object(executor, "_step_control", disturb):
            try:
                result = executor._move_pose("probe:unexpected_contact", "left", goal, 0.8, [])
                print(f"Stage returned collision_free={result['collision_free']}")
            except RuntimeError as error:
                if first is None:
                    raise
                print(f"Stage aborted: {error}")
        if first is None:
            raise AssertionError("The moving-object probe did not observe the induced contact")
        print(f"First forbidden contact at step {first}, last step {ticks}, "
              f"accumulated contact points {executor.runtime_forbidden_contacts}")
    finally:
        executor.close()


if __name__ == "__main__":
    speeds_ok = check_default_finger_speeds()
    observe_unexpected_contact()
    sys.exit(0 if speeds_ok else 1)

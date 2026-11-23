"""Verification test for dual-arm closed-loop pick-and-place in MuJoCo."""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.grasp_pipeline import run_dual_arm_pick_and_place


def test_dual_arm_pick_and_place() -> None:
    """Verify both left (red_block) and right (blue_cylinder) complete pick, lift, transport, place, and settle."""
    report = run_dual_arm_pick_and_place(record_video=False)
    assert report["all_passed"], f"Pick-and-place failed: {report['tasks']}"
    for task in report["tasks"]:
        assert task["lifted_z_gain_mm"] > 80.0, f"Insufficient lift: {task}"
        assert task["place_xy_err_mm"] < 3.0, f"Placement XY error too high: {task}"
        assert abs(task["place_z_err_mm"]) < 2.0, f"Placement Z error too high: {task}"
        assert task["settled_tilt_deg"] < 2.0, f"Object tilted after release: {task}"
        assert task["settled_lin_speed_m_s"] < 0.01, f"Object did not settle: {task}"
    for stage in report["stages"]:
        if "collision_free" in stage:
            assert stage["collision_free"], f"Stage collision detected: {stage}"
    print("ALL GRASP PICK-AND-PLACE ASSERTIONS PASSED!")


if __name__ == "__main__":
    test_dual_arm_pick_and_place()

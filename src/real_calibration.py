"""Hardware connection endpoints and calibrated camera / dual-arm extrinsics (261117 v1)."""

from __future__ import annotations

import numpy as np

# Hexfellow Firefly Y6 dual-arm controller WebSocket endpoints (CAN0=8439 left, CAN1=9439 right)
CONTROLLER_IPV6_HOST = "fe80::cc44:e4ff:fecf:de48%enx00e04c680044"
LEFT_ARM_WS_URL = f"ws://[{CONTROLLER_IPV6_HOST}]:8439"
RIGHT_ARM_WS_URL = f"ws://[{CONTROLLER_IPV6_HOST}]:9439"

# RealSense serial numbers and factory/SDK intrinsics (640x480)
CAMERA_SERIALS: dict[str, str] = {
    "left_d405": "260422274786",
    "right_d405": "260322270508",
    "top_d435": "262522074656",
}

CAMERA_INTRINSICS: dict[str, dict[str, float]] = {
    "left_d405": {
        "fx": 394.354706,
        "fy": 393.819916,
        "cx": 322.264221,
        "cy": 232.016556,
    },
    "right_d405": {
        "fx": 393.783875,
        "fy": 393.302246,
        "cx": 316.915253,
        "cy": 236.197098,
    },
    "top_d435": {
        "fx": 603.242737,
        "fy": 602.020325,
        "cx": 325.794067,
        "cy": 245.838181,
    },
}

# Hand-eye and base-to-base extrinsics (261117 initial calibration)
# Convention: T_target_from_source maps p_source -> p_target = T[:3, :3] @ p_source + T[:3, 3]

# Left wrist D405 (260422274786) -> arm1_link_6 (RMS: 5.02 mm, 0.73 deg)
T_ARM1_LINK6_FROM_LEFT_D405 = np.array(
    [
        [0.876035777, 0.016659105, 0.481958288, -0.065810385],
        [-0.008001913, 0.999767704, -0.020012671, -0.002553714],
        [-0.482179725, 0.013675227, 0.875965582, 0.085922521],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Right wrist D405 (260322270508) -> arm2_link_6 (RMS: 4.58 mm, 0.44 deg)
T_ARM2_LINK6_FROM_RIGHT_D405 = np.array(
    [
        [0.895823773, -0.020139976, 0.443952869, -0.067047795],
        [0.013415529, 0.999742842, 0.018283129, 0.004679868],
        [-0.444206924, -0.010422599, 0.895863593, 0.090268855],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Top D435 (262522074656) -> arm1_base (left arm base) (RMS: 7.16 mm, 2.32 deg)
T_ARM1_BASE_FROM_TOP_D435 = np.array(
    [
        [0.008673580, 0.854846368, 0.518808690, 0.037659997],
        [0.999858770, -0.014882761, 0.007806576, -0.373326740],
        [0.014394728, 0.518667708, -0.854854725, 0.516178397],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Top D435 (262522074656) -> arm2_base (right arm base) (RMS: 6.89 mm, 1.88 deg)
T_ARM2_BASE_FROM_TOP_D435 = np.array(
    [
        [-0.001546894, 0.883075525, 0.469228329, 0.060747607],
        [0.999644648, -0.011121474, 0.024225831, 0.303952972],
        [0.026611749, 0.469099063, -0.882744518, 0.507825950],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Right arm base (arm2_base) -> Left arm base (arm1_base)
T_ARM1_BASE_FROM_ARM2_BASE = np.array(
    [
        [0.998320222, 0.011731917, -0.056737078, 0.002261042],
        [-0.011026211, 0.999858107, 0.012735290, -0.683034078],
        [0.056878437, -0.012088302, 0.998307927, 0.009430772],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Nominal left_workcell_to_arm origin in assets/firefly_dual.urdf (rpy = 0)
URDF_WORKCELL_TO_LEFT_BASE = np.array([0.0605, 0.35, 0.095], dtype=np.float64)


def transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform T to (N, 3) points."""
    return pts_xyz @ T[:3, :3].T + T[:3, 3]


def check_se3_orthogonality(T: np.ndarray) -> dict[str, float]:
    """Return rotation orthogonality error and determinant for sanity checking."""
    R = T[:3, :3]
    err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det = float(np.linalg.det(R))
    return {"ortho_err": err, "det": det}

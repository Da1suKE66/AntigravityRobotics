"""Truly passive, zero-write WebSocket telemetry reader for Hexfellow Firefly arms.

Safety guarantee:
- Connects via raw WebSocket and ONLY calls `recv()`.
- NEVER calls `send()` and NEVER instantiates `HexDeviceApi` / `Arm._periodic()`,
  making it impossible to transmit `clear_parking_stop`, `start()`, session claims,
  or any motor/control command.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable

import numpy as np
import websockets
from hex_device.generated import public_api_types_pb2, public_api_up_pb2

ENCODER_MIDPOINT = 65535.0 / 2.0

# Matches hex_device.Arm.__init__: only these legacy arms subtract 65535/2;
# Firefly Y6 and newer arms use raw_pos / ppr * 2*pi directly.
LEGACY_OFFSET_ROBOT_TYPES = {
    int(public_api_types_pb2.RobotType.RtArmSaberD6x),
    int(public_api_types_pb2.RobotType.RtArmSaberD7x),
    int(public_api_types_pb2.RobotType.RtArmArcherD6Y_P1),
    int(public_api_types_pb2.RobotType.RtArmArcherY6L_V1),
}


def decode_api_up_arm_telemetry(raw_bytes: bytes) -> dict[str, Any] | None:
    """Decode one binary APIUp protobuf frame into a read-only telemetry dict."""
    up = public_api_up_pb2.APIUp()
    up.ParseFromString(raw_bytes)
    if not up.HasField("arm_status"):
        return None

    arm_st = up.arm_status
    motors = list(arm_st.motor_status)
    if not motors:
        return None

    raw_pos = np.array([m.position for m in motors], dtype=np.float64)
    ppr = np.array([m.pulse_per_rotation for m in motors], dtype=np.float64)
    # Exact conversion matching hex_device.Arm.__init__ + MotorBase.convert_positions_to_rad
    if int(up.robot_type) in LEGACY_OFFSET_ROBOT_TYPES:
        pos_rad = (raw_pos - ENCODER_MIDPOINT) / ppr * (2.0 * np.pi)
    else:
        pos_rad = raw_pos / ppr * (2.0 * np.pi)
    vel = np.array([m.speed for m in motors], dtype=np.float64)
    tau = np.array([m.torque for m in motors], dtype=np.float64)
    temps = [float(m.motor_temperature) for m in motors]
    volts = [float(m.voltage) for m in motors]
    err_codes = [
        (list(m.error) if hasattr(m.error, "__iter__") else (int(m.error) if m.error else None))
        for m in motors
    ]
    warnings = [str(m.motor_warning) for m in motors]

    parking_stop = ""
    parking_category = None
    parking_stop_active = False
    if arm_st.HasField("parking_stop_detail"):
        ps = arm_st.parking_stop_detail
        # In proto3, PscEmergencyStopButton == 0 with empty reason serializes to b"",
        # so arm_st.HasField("parking_stop_detail") is the ONLY reliable presence signal!
        parking_stop_active = True
        parking_category = int(ps.category)
        cat_name = public_api_types_pb2.ParkingStopCategory.Name(ps.category)
        parking_stop = f"category={cat_name}({ps.category}) reason={ps.reason}"

    gripper = None
    for sec in up.secondary_device_status:
        if sec.HasField("hand_status"):
            hm = list(sec.hand_status.motor_status)
            if hm:
                h_raw = np.array([m.position for m in hm], dtype=np.float64)
                h_ppr = np.array([m.pulse_per_rotation for m in hm], dtype=np.float64)
                h_rad = (h_raw - ENCODER_MIDPOINT) / np.where(h_ppr > 0, h_ppr, 1.0) * (2.0 * np.pi)
                gripper = {
                    "device_id": int(sec.device_id),
                    "device_type": int(sec.device_type),
                    "positions_rad": [float(x) for x in h_rad.tolist()],
                    "torques_nm": [float(m.torque) for m in hm],
                }

    return {
        "robot_type": int(up.robot_type),
        "session_id": int(up.session_id),
        "session_holder": int(arm_st.session_holder),
        "api_control_initialized": bool(arm_st.api_control_initialized),
        "calibrated": bool(arm_st.calibrated),
        "motor_count": len(motors),
        "joint_names": [f"joint_{i + 1}" for i in range(len(motors))],
        "positions_rad": [float(x) for x in pos_rad.tolist()],
        "velocities_rad_s": [float(x) for x in vel.tolist()],
        "torques_nm": [float(x) for x in tau.tolist()],
        "temperatures_c": temps,
        "voltages_v": volts,
        "error_codes": err_codes,
        "warnings": warnings,
        "parking_stop_active": parking_stop_active,
        "parking_stop": parking_stop,
        "parking_category": parking_category,
        "gripper": gripper,
    }


async def _read_arm_passive_async(
    ws_url: str,
    timeout_s: float = 5.0,
    ws_connect: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    connector = ws_connect or websockets.connect
    deadline = time.monotonic() + timeout_s
    async with connector(ws_url, open_timeout=timeout_s, proxy=None) as ws:
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError(f"No APIUp arm_status frame received from {ws_url} within {timeout_s}s")
            msg = await asyncio.wait_for(ws.recv(), timeout=rem)
            if isinstance(msg, (bytes, bytearray)):
                decoded = decode_api_up_arm_telemetry(bytes(msg))
                if decoded is not None:
                    return decoded


def read_arm_passive_readonly(
    name: str,
    ws_url: str,
    timeout_s: float = 5.0,
    ws_connect: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Read one arm telemetry snapshot with strictly zero uplink WebSocket sends."""
    result: dict[str, Any] = {
        "name": name,
        "ws_url": ws_url,
        "transport": "passive_zero_write_ws",
        "connected": False,
        "arm": None,
        "gripper": None,
        "error": None,
    }
    try:
        decoded = asyncio.run(
            _read_arm_passive_async(ws_url=ws_url, timeout_s=timeout_s, ws_connect=ws_connect)
        )
        result["connected"] = True
        result["gripper"] = decoded.pop("gripper", None)
        result["arm"] = decoded
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result

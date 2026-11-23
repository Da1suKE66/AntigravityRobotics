"""Build a normalized, physics-accurate MuJoCo MJCF model from firefly_dual.urdf using MjSpec."""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
import mujoco
import trimesh

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
URDF_PATH = PROJECT_ROOT / "assets" / "firefly_dual.urdf"
ASSETS_DIR = PROJECT_ROOT / "assets"
COL_MESH_DIR = ASSETS_DIR / "meshes" / "collision"
MODEL_DIR = PROJECT_ROOT / "model"
ASSET_XML_PATH = MODEL_DIR / "firefly_dual.xml"
SCENE_XML_PATH = MODEL_DIR / "scene.xml"

# Joint dynamics parameters calibrated for 20 Nm (J1-J4) and 5 Nm (J5-J6, gripper) actuators
ARM_DYNAMICS = {
    1: {"armature": 0.025, "damping": 0.6, "frictionloss": 0.15, "effort": 20.0},
    2: {"armature": 0.025, "damping": 0.6, "frictionloss": 0.15, "effort": 20.0},
    3: {"armature": 0.020, "damping": 0.5, "frictionloss": 0.12, "effort": 20.0},
    4: {"armature": 0.015, "damping": 0.4, "frictionloss": 0.10, "effort": 20.0},
    5: {"armature": 0.005, "damping": 0.15, "frictionloss": 0.05, "effort": 5.0},
    6: {"armature": 0.005, "damping": 0.15, "frictionloss": 0.05, "effort": 5.0},
}
GRIPPER_DYNAMICS = {"armature": 0.002, "damping": 0.08, "frictionloss": 0.03, "effort": 5.0}

# Multi-convex-hull decomposition specs: (source_stl_rel, cut_axis, cut_positions)
DECOMPOSITION_SPECS: dict[str, tuple[str, int, list[float]]] = {
    "link_3": ("meshes/arm/link_3.STL", 2, [0.055, 0.225, 0.255]),
    "link_4": ("meshes/arm/link_4.STL", 2, [0.030]),
    "link_5": ("meshes/arm/link_5.STL", 0, [0.025, 0.060]),
    "gr100_link_1": ("meshes/arm/gr100_link_1.STL", 2, [0.028, 0.055]),
    "gr100_link_2": ("meshes/arm/gr100_link_2.STL", 2, [0.028, 0.055]),
}


def _clip_points_by_slab(tm: trimesh.Trimesh, axis: int, lo: float, hi: float) -> "np.ndarray":
    """Extract mesh vertices inside [lo, hi] plus exact edge-plane intersections."""
    import numpy as np

    v = tm.vertices
    edges = tm.edges_unique
    pts = [v[(v[:, axis] >= lo) & (v[:, axis] <= hi)]]
    for cut in (lo, hi):
        if abs(cut) > 1e6:
            continue
        c0 = v[edges[:, 0], axis]
        c1 = v[edges[:, 1], axis]
        cross = ((c0 < cut) & (c1 > cut)) | ((c0 > cut) & (c1 < cut))
        if cross.any():
            ec = edges[cross]
            p0 = v[ec[:, 0]]
            p1 = v[ec[:, 1]]
            t = ((cut - p0[:, axis]) / (p1[:, axis] - p0[:, axis]))[:, None]
            pts.append(p0 + t * (p1 - p0))
    return np.vstack(pts)


def _hull_max_verts(pts: "np.ndarray", max_verts: int = 60) -> trimesh.Trimesh:
    """Compute a convex hull with at most max_verts vertices using farthest-point sampling."""
    import numpy as np

    h = trimesh.Trimesh(vertices=pts).convex_hull
    if len(h.vertices) <= max_verts:
        return h
    hp = h.vertices
    sel = [int(np.argmax(np.linalg.norm(hp - hp.mean(axis=0), axis=1)))]
    dists = np.full(len(hp), np.inf)
    for _ in range(max_verts - 1):
        d = np.linalg.norm(hp - hp[sel[-1]], axis=1)
        dists = np.minimum(dists, d)
        sel.append(int(np.argmax(dists)))
    return trimesh.Trimesh(vertices=hp[sel]).convex_hull


def build_collision_hull_meshes() -> dict[str, list[str]]:
    """Generate decomposed <=60-vertex convex hull .obj files in assets/meshes/collision/."""
    COL_MESH_DIR.mkdir(parents=True, exist_ok=True)
    generated: dict[str, list[str]] = {}
    for key, (rel_path, axis, cuts) in DECOMPOSITION_SPECS.items():
        tm = trimesh.load(str(ASSETS_DIR / rel_path), force="mesh")
        bounds = [-1e9] + cuts + [1e9]
        rels: list[str] = []
        idx = 0
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            pts = _clip_points_by_slab(tm, axis, lo, hi)
            if len(pts) < 4:
                continue
            hull = _hull_max_verts(pts, max_verts=60)
            out_rel = f"meshes/collision/{key}_col_{idx}.obj"
            hull.export(str(ASSETS_DIR / out_rel))
            rels.append(out_rel)
            idx += 1
        generated[key] = rels
    return generated


def _inject_mujoco_urdf_header(urdf_text: str) -> str:
    """Inject <mujoco><compiler .../></mujoco> so URDF visual meshes and frames are preserved."""
    compiler_tag = (
        '\n  <mujoco>\n'
        '    <compiler discardvisual="false" fusestatic="false" balanceinertia="true" strippath="false"/>\n'
        '  </mujoco>'
    )
    root = ET.fromstring(urdf_text)
    if root.find("mujoco") is not None:
        return urdf_text
    idx = urdf_text.find(">", urdf_text.find("<robot"))
    return urdf_text[: idx + 1] + compiler_tag + urdf_text[idx + 1 :]


def build_firefly_mjcf() -> tuple[pathlib.Path, pathlib.Path]:
    """Convert firefly_dual.urdf to normalized MJCF asset + simulation scene."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_urdf = URDF_PATH.read_text(encoding="utf-8")
    mod_urdf = _inject_mujoco_urdf_header(raw_urdf)

    # Write temporary URDF inside assets/ so relative mesh paths resolve cleanly
    tmp_urdf = URDF_PATH.parent / "_tmp_mujoco_import.urdf"
    try:
        tmp_urdf.write_text(mod_urdf, encoding="utf-8")
        spec = mujoco.MjSpec.from_file(str(tmp_urdf))
    finally:
        if tmp_urdf.exists():
            tmp_urdf.unlink()

    spec.modelname = "firefly_dual"
    spec.compiler.degree = False
    spec.compiler.meshdir = "../assets"

    # Solver & integrator defaults (implicitfast, 500 Hz)
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 0.002
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    # Convert oversized STLs (>180k faces) to .obj so MuJoCo's 200k STL face limit is never hit
    assets_dir = PROJECT_ROOT / "assets"
    for mesh in spec.meshes:
        if mesh.file and mesh.file.lower().endswith(".stl"):
            stl_path = assets_dir / mesh.file
            if stl_path.exists() and stl_path.stat().st_size > 8_000_000:
                tm = trimesh.load(str(stl_path), force="mesh")
                if len(tm.faces) > 180_000:
                    obj_rel = str(pathlib.Path(mesh.file).with_suffix(".obj"))
                    obj_path = assets_dir / obj_rel
                    if not obj_path.exists():
                        tm.export(str(obj_path))
                    mesh.file = obj_rel

    # Generate multi-hull collision meshes for concave arm links & gripper fingers
    col_mesh_map = build_collision_hull_meshes()
    for key, rels in col_mesh_map.items():
        for i, rel in enumerate(rels):
            spec.add_mesh(name=f"{key}_col_{i}", file=rel)

    # Organize visual (group 2, no collision, zero density/mass) vs collision (group 3)
    parent_map: list[tuple[str, str]] = []

    def _walk_bodies(body: mujoco.MjsBody, parent_name: str | None) -> None:
        if parent_name and parent_name != "world":
            parent_map.append((parent_name, body.name))
        col_idx = 0
        for geom in body.geoms:
            if geom.contype == 0 and geom.conaffinity == 0:
                geom.group = 2
                geom.density = 0.0
                geom.mass = 0.0
            else:
                if not geom.name:
                    geom.name = f"{body.name}_col_{col_idx}"
                col_idx += 1
                geom.group = 3
                geom.contype = 1
                geom.conaffinity = 1
                geom.rgba = [0.2, 0.65, 0.95, 0.40]
                geom.condim = 4
                geom.friction = [0.8, 0.02, 0.002]
                geom.solref = [0.008, 1.0]
        for child in body.bodies:
            _walk_bodies(child, body.name)

    _walk_bodies(spec.worldbody, None)

    # Replace single-hull STL colliders on concave arm links & fingers with decomposed multi-hulls
    for side in ("left", "right"):
        for b_suffix, key in [
            ("link_3", "link_3"),
            ("link_4", "link_4"),
            ("link_5", "link_5"),
            ("grip_link_1", "gr100_link_1"),
            ("grip_link_2", "gr100_link_2"),
        ]:
            body = spec.body(f"{side}_{b_suffix}")
            if body is None:
                continue
            for g in list(body.geoms):
                if g.group == 3:
                    spec.delete(g)
            is_grip = "grip" in b_suffix
            for i in range(len(col_mesh_map[key])):
                body.add_geom(
                    name=f"{side}_{b_suffix}_col_{i}",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=f"{key}_col_{i}",
                    group=3,
                    contype=1,
                    conaffinity=1,
                    condim=4,
                    friction=[1.2, 0.05, 0.005] if is_grip else [0.8, 0.02, 0.002],
                    solref=[0.004, 1.0] if is_grip else [0.008, 1.0],
                    rgba=[0.95, 0.55, 0.15, 0.50] if is_grip else [0.2, 0.65, 0.95, 0.40],
                    mass=0.0,
                    density=0.0,
                )

    # Add workcell, post, D435, and wrist D405 camera bracket collision primitives (group 3)
    workcell_boxes = [
        ("post_adapter_plate", "post_adapter_col", [0.0, 0.0, 0.005], [0.050, 0.050, 0.005]),
        ("post_socket", "post_socket_col", [0.0, 0.0, 0.0225], [0.022, 0.0382, 0.0225]),
        ("post_profile", "post_profile_col", [0.0, 0.0, 0.310], [0.015, 0.015, 0.310]),
        ("top_post_u_piece", "top_post_u_col", [0.0285, 0.015, 0.033], [0.020, 0.015, 0.0275]),
        ("top_camera_mount_plate", "top_cam_plate_col", [0.0, 0.0, -0.003], [0.0285, 0.015, 0.0125]),
        ("top_d435_visual_mesh_frame", "top_d435_col", [0.0, 0.0, -0.0125], [0.045, 0.0125, 0.0125]),
        ("e3_housing_cad_frame", "e3_housing_col", [0.0845, -0.0375, 0.0], [0.0945, 0.0475, 0.411]),
    ]
    for bname, gname, pos, size in workcell_boxes:
        b = spec.body(bname)
        if b is not None:
            b.add_geom(
                name=gname,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=pos,
                size=size,
                group=3,
                contype=1,
                conaffinity=1,
                rgba=[0.25, 0.55, 0.90, 0.45],
                mass=0.0,
                density=0.0,
            )

    for side in ("left", "right"):
        b_a = spec.body(f"{side}_A_fixed")
        if b_a is not None:
            b_a.add_geom(
                name=f"{side}_A_fixed_col",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.020, 0.030, 0.029],
                size=[0.020, 0.030, 0.029],
                group=3,
                contype=1,
                conaffinity=1,
                rgba=[0.25, 0.55, 0.90, 0.45],
                mass=0.0,
                density=0.0,
            )
        b_b = spec.body(f"{side}_B_camera_plate")
        if b_b is not None:
            b_b.add_geom(
                name=f"{side}_B_plate_col",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.020, 0.0115, 0.020],
                size=[0.020, 0.0115, 0.020],
                group=3,
                contype=1,
                conaffinity=1,
                rgba=[0.25, 0.55, 0.90, 0.45],
                mass=0.0,
                density=0.0,
            )
        b_cam = spec.body(f"{side}_wrist_d405_link")
        if b_cam is not None:
            b_cam.add_geom(
                name=f"{side}_d405_col",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=[0.0038, -0.009, -0.0115],
                size=[0.021, 0.021, 0.0115],
                group=3,
                contype=1,
                conaffinity=1,
                rgba=[0.25, 0.55, 0.90, 0.45],
                mass=0.0,
                density=0.0,
            )

    # Disable self-collision between adjacent bodies
    for b1, b2 in parent_map:
        spec.add_exclude(bodyname1=b1, bodyname2=b2)
    # Exclude gripper fingers from each other/wrist and welded camera brackets from local arm links
    for side in ("left", "right"):
        spec.add_exclude(bodyname1=f"{side}_grip_link_1", bodyname2=f"{side}_grip_link_2")
        spec.add_exclude(bodyname1=f"{side}_link_5", bodyname2=f"{side}_grip_link_1")
        spec.add_exclude(bodyname1=f"{side}_link_5", bodyname2=f"{side}_grip_link_2")
        spec.add_exclude(bodyname1=f"{side}_link_5", bodyname2=f"{side}_A_fixed")
        spec.add_exclude(bodyname1=f"{side}_grip_link_1", bodyname2=f"{side}_A_fixed")
        spec.add_exclude(bodyname1=f"{side}_grip_link_2", bodyname2=f"{side}_A_fixed")
        spec.add_exclude(bodyname1=f"{side}_link_6", bodyname2=f"{side}_B_camera_plate")
        spec.add_exclude(bodyname1=f"{side}_A_fixed", bodyname2=f"{side}_wrist_d405_link")
        spec.add_exclude(bodyname1="e3_housing_cad_frame", bodyname2=f"{side}_base_link")
        spec.add_exclude(bodyname1="e3_housing_cad_frame", bodyname2=f"{side}_link_1")

    # Configure joint dynamics and add torque-controlled motor actuators
    # Check _grip_joint_ BEFORE _joint_ because _grip_joint_ contains _joint_!
    joint_names_ordered: list[str] = []
    for joint in spec.joints:
        jname = joint.name
        joint_names_ordered.append(jname)
        if "_grip_joint_" in jname:
            dyn = GRIPPER_DYNAMICS
            is_mimic_follower = jname.endswith("_grip_joint_2")
        elif "_joint_" in jname:
            idx = int(jname.rsplit("_", 1)[1])
            dyn = ARM_DYNAMICS[idx]
            is_mimic_follower = False
        else:
            continue

        joint.armature = dyn["armature"]
        joint.damping = [dyn["damping"], 0.0, 0.0]
        joint.frictionloss = dyn["frictionloss"]
        joint.solref_limit = [0.004, 1.0]

        # URDF specifies grip_joint_2 as a passive mimic of grip_joint_1 (effort=0)
        if not is_mimic_follower:
            spec.add_actuator(
                name=f"{jname}_motor",
                target=jname,
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                ctrllimited=True,
                ctrlrange=[-dyn["effort"], dyn["effort"]],
                gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )

    # Tighten equality constraint solref on mimic gripper fingers for crisp symmetric closure
    for eq in spec.equalities:
        eq.solref = [0.004, 1.0]

    # Add end-effector sites (group 4): gripper pinch center (0.125 m) and TCP (0.170 m)
    for side in ("left", "right"):
        link6 = spec.body(f"{side}_link_6")
        if link6 is not None:
            link6.add_site(
                name=f"{side}_ee_site",
                pos=[0.0, 0.0, 0.125],
                quat=[1.0, 0.0, 0.0, 0.0],
                size=[0.012, 0.012, 0.012],
                rgba=[0.1, 0.95, 0.3, 0.6],
                group=4,
            )
            link6.add_site(
                name=f"{side}_tcp_site",
                pos=[0.0, 0.0, 0.170],
                quat=[1.0, 0.0, 0.0, 0.0],
                size=[0.010, 0.010, 0.010],
                rgba=[1.0, 0.4, 0.1, 0.7],
                group=4,
            )

    # Add optical cameras aligned with OpenCV optical frames (+Z forward, -Y up -> MuJoCo -Z forward, +Y up)
    cam_specs = [
        ("top_d435_cam", "top_d435_color_optical_frame_nominal", 58.0),
        ("left_wrist_d405_cam", "left_wrist_d405_color_optical_frame", 58.0),
        ("right_wrist_d405_cam", "right_wrist_d405_color_optical_frame", 58.0),
    ]
    for cam_name, body_name, fovy in cam_specs:
        b = spec.body(body_name)
        if b is not None:
            b.add_camera(
                name=cam_name,
                pos=[0.0, 0.0, 0.0],
                quat=[0.0, 1.0, 0.0, 0.0],  # 180 deg around X: +Z -> -Z, +Y -> -Y
                fovy=fovy,
            )

    # Add 'home' keyframe: safe bent-forward ready pose
    # Joint order: left_1..6, left_grip_1..2, right_1..6, right_grip_1..2
    home_map = {
        "left_joint_1": 0.0,
        "left_joint_2": 0.35,
        "left_joint_3": 1.35,
        "left_joint_4": -0.65,
        "left_joint_5": 0.0,
        "left_joint_6": 0.0,
        "left_grip_joint_1": 0.25,
        "left_grip_joint_2": 0.25,
        "right_joint_1": 0.0,
        "right_joint_2": 0.35,
        "right_joint_3": 1.35,
        "right_joint_4": -0.65,
        "right_joint_5": 0.0,
        "right_joint_6": 0.0,
        "right_grip_joint_1": 0.25,
        "right_grip_joint_2": 0.25,
    }
    home_qpos = [home_map.get(jn, 0.0) for jn in joint_names_ordered]
    spec.add_key(name="home", qpos=home_qpos)

    # Compile to validate & write clean MJCF XML
    model = spec.compile()
    xml_str = spec.to_xml()
    ASSET_XML_PATH.write_text(xml_str, encoding="utf-8")

    # Build scene.xml that includes firefly_dual.xml + floor, lighting, and a tabletop
    scene_xml = """<mujoco model="firefly_dual_scene">
  <include file="firefly_dual.xml"/>

  <statistic center="0.35 0 0.35" extent="1.2"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.05 0.08 0.12" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2"
             markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.15"/>
    <material name="table_mat" rgba="0.72 0.68 0.62 1" roughness="0.6" specular="0.1"/>
  </asset>

  <worldbody>
    <light pos="0.4 0 1.8" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" conaffinity="1"/>
    <body name="worktable" pos="0.42 0.0 0.04">
      <geom name="tabletop" type="box" size="0.32 0.55 0.04" material="table_mat" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
"""
    SCENE_XML_PATH.write_text(scene_xml, encoding="utf-8")
    print(f"Built {ASSET_XML_PATH} (nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}, ngeom={model.ngeom})")
    print(f"Built {SCENE_XML_PATH}")
    return ASSET_XML_PATH, SCENE_XML_PATH


if __name__ == "__main__":
    build_firefly_mjcf()

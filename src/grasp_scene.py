"""Build the Firefly Dual-Arm tabletop pick-and-place MuJoCo scene via MjSpec."""

from __future__ import annotations

import pathlib
import mujoco

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
BASE_SCENE_XML = MODEL_DIR / "scene.xml"
GRASP_SCENE_XML = MODEL_DIR / "grasp_scene.xml"


def build_grasp_scene() -> pathlib.Path:
    """Create model/grasp_scene.xml from model/scene.xml with graspable tabletop props and target zones."""
    spec = mujoco.MjSpec.from_file(str(BASE_SCENE_XML))
    spec.modelname = "firefly_dual_grasp_scene"
    # Keep discrete-friendly contact damping & 500 Hz implicitfast
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    # Add materials for props and placement pads
    spec.add_material(name="red_block_mat", rgba=[0.85, 0.18, 0.15, 1.0], roughness=0.45, specular=0.2)
    spec.add_material(name="blue_cyl_mat", rgba=[0.16, 0.45, 0.88, 1.0], roughness=0.40, specular=0.25)
    spec.add_material(name="zone_left_mat", rgba=[0.90, 0.45, 0.40, 0.85], roughness=0.8)
    spec.add_material(name="zone_right_mat", rgba=[0.40, 0.65, 0.92, 0.85], roughness=0.8)

    # Target placement pads (thin 2 mm visual+collision pads welded to tabletop at z=0.080..0.082)
    # Left target zone: [0.44, +0.07, 0.081]; Right target zone: [0.44, -0.07, 0.081]
    for zname, pos, mat in [
        ("place_zone_left", [0.41, 0.07, 0.081], "zone_left_mat"),
        ("place_zone_right", [0.41, -0.07, 0.081], "zone_right_mat"),
    ]:
        zb = spec.worldbody.add_body(name=zname, pos=pos)
        zb.add_geom(
            name=f"{zname}_vis",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.045, 0.045, 0.001],
            material=mat,
            group=2,
            contype=0,
            conaffinity=0,
            mass=0.0,
            density=0.0,
        )
        zb.add_geom(
            name=f"{zname}_col",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.045, 0.045, 0.001],
            group=3,
            contype=1,
            conaffinity=1,
            condim=4,
            friction=[0.9, 0.02, 0.002],
            rgba=[0.2, 0.8, 0.4, 0.45],
            mass=0.0,
            density=0.0,
        )
        zb.add_site(
            name=f"{zname}_site",
            pos=[0.0, 0.0, 0.001],
            size=[0.008, 0.008, 0.008],
            rgba=[0.1, 0.9, 0.3, 0.5],
            group=4,
        )

    # Graspable Prop 1: red_block (44 x 44 x 44 mm, 85 g) at [0.40, +0.22, 0.102]
    b_block = spec.worldbody.add_body(name="red_block", pos=[0.40, 0.22, 0.102])
    b_block.add_freejoint(name="red_block_freejoint")
    b_block.add_geom(
        name="red_block_vis",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.022, 0.022, 0.022],
        material="red_block_mat",
        group=2,
        contype=0,
        conaffinity=0,
        mass=0.0,
        density=0.0,
    )
    b_block.add_geom(
        name="red_block_col",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.022, 0.022, 0.022],
        group=3,
        contype=1,
        conaffinity=1,
        condim=4,
        friction=[1.1, 0.05, 0.005],
        solref=[0.004, 1.0],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        rgba=[0.20, 0.75, 0.35, 0.50],
        mass=0.085,
    )
    b_block.add_site(
        name="red_block_grasp_site",
        pos=[0.0, 0.0, 0.0],
        size=[0.008, 0.008, 0.008],
        rgba=[0.95, 0.25, 0.25, 0.45],
        group=4,
    )

    # Graspable Prop 2: blue_cylinder (r=21 mm, h=60 mm, 95 g) at [0.40, -0.22, 0.110]
    b_cyl = spec.worldbody.add_body(name="blue_cylinder", pos=[0.40, -0.22, 0.110])
    b_cyl.add_freejoint(name="blue_cylinder_freejoint")
    b_cyl.add_geom(
        name="blue_cylinder_vis",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.021, 0.030, 0.0],
        material="blue_cyl_mat",
        group=2,
        contype=0,
        conaffinity=0,
        mass=0.0,
        density=0.0,
    )
    b_cyl.add_geom(
        name="blue_cylinder_col",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.021, 0.030, 0.0],
        group=3,
        contype=1,
        conaffinity=1,
        condim=4,
        friction=[1.1, 0.05, 0.005],
        solref=[0.004, 1.0],
        solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
        rgba=[0.20, 0.75, 0.35, 0.50],
        mass=0.095,
    )
    b_cyl.add_site(
        name="blue_cylinder_grasp_site",
        pos=[0.0, 0.0, 0.0],
        size=[0.008, 0.008, 0.008],
        rgba=[0.95, 0.25, 0.25, 0.45],
        group=4,
    )

    # Add an angled tabletop overview camera for visual verification
    spec.worldbody.add_camera(
        name="overview_cam",
        pos=[1.15, -0.65, 0.95],
        xyaxes=[0.50, 0.87, 0.0, -0.45, 0.26, 0.85],
        fovy=45.0,
    )

    # Update 'home' keyframe to include the two freejoint poses (7 qpos each = 14 extra qpos)
    robot_home = list(spec.keys[0].qpos) if spec.keys else [0.0] * 16
    # Open grippers slightly at home ready pose
    robot_home[6:8] = [0.05, 0.05]
    robot_home[14:16] = [0.05, 0.05]
    full_home = robot_home + [
        0.40, 0.22, 0.102, 1.0, 0.0, 0.0, 0.0,   # red_block
        0.40, -0.22, 0.110, 1.0, 0.0, 0.0, 0.0,  # blue_cylinder
    ]
    if spec.keys:
        spec.keys[0].qpos = full_home
    else:
        spec.add_key(name="home", qpos=full_home)

    model = spec.compile()
    GRASP_SCENE_XML.write_text(spec.to_xml(), encoding="utf-8")
    print(f"Built {GRASP_SCENE_XML} (nbody={model.nbody}, nq={model.nq}, nv={model.nv}, nu={model.nu})")
    return GRASP_SCENE_XML


if __name__ == "__main__":
    build_grasp_scene()

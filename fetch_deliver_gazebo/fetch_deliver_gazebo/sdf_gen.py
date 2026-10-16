"""SDF generation for worlds and spawnable objects (Gazebo Classic, SDF 1.6)."""

import math
from typing import Iterable

from .layouts import Layout

# Graspable objects: small enough for the OpenMANIPULATOR-X gripper.
# Colors use Gazebo's built-in material scripts: inline <ambient>/<diffuse>
# materials make Gazebo Classic crash (Ogre "resource already exists") when a
# model with the same name is deleted and spawned again while a camera renders.
# Targets are tall cylinders grasped from the side. The red cube is the hard
# distractor for a red-cylinder target: same color, told apart only by its shape.
OBJECT_TYPES = {
    'red_cylinder': {'shape': 'cylinder', 'size': 0.03, 'height': 0.10, 'material': 'Gazebo/Red'},
    'green_cylinder': {'shape': 'cylinder', 'size': 0.03, 'height': 0.10, 'material': 'Gazebo/Green'},
    'blue_cylinder': {'shape': 'cylinder', 'size': 0.03, 'height': 0.10, 'material': 'Gazebo/Blue'},
    'yellow_cylinder': {'shape': 'cylinder', 'size': 0.03, 'height': 0.10, 'material': 'Gazebo/Yellow'},
    'red_cube': {'shape': 'box', 'size': 0.04, 'height': 0.04, 'material': 'Gazebo/Red'},
}

# Every object stands on a small table, so the gripper never reaches down to the
# floor (a floor-level grasp presses the fingers into the ground and ODE rocks the
# robot). The table stays below the camera so its top, and the object's base, are
# in view, and below the LiDAR plane, which sees the object itself.
TABLE_HEIGHT = 0.07
TABLE_SIZE = 0.16


def _script_material(name: str) -> str:
    return (f'<material><script><uri>file://media/materials/scripts/gazebo.material</uri>'
            f'<name>{name}</name></script></material>')


def _material(rgb) -> str:
    r, g, b = rgb
    return (f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
            f'<specular>0.1 0.1 0.1 1</specular></material>')


def table_sdf(model_name: str) -> str:
    s, h = TABLE_SIZE, TABLE_HEIGHT
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <pose>0 0 {h / 2} 0 0 0</pose>
      <collision name="collision"><geometry><box><size>{s} {s} {h}</size></box></geometry></collision>
      <visual name="visual"><geometry><box><size>{s} {s} {h}</size></box></geometry>
        {_script_material('Gazebo/Wood')}</visual>
    </link>
  </model>
</sdf>"""


def object_sdf(model_name: str, object_type: str) -> str:
    spec = OBJECT_TYPES[object_type]
    s, h = spec['size'], spec['height']
    if spec['shape'] == 'box':
        geom = f'<box><size>{s} {s} {h}</size></box>'
    else:
        geom = f'<cylinder><radius>{s / 2}</radius><length>{h}</length></cylinder>'
    m = 0.02
    i = m * s * s / 6.0
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <link name="link">
      <inertial><mass>{m}</mass>
        <inertia><ixx>{i:.3e}</ixx><iyy>{i:.3e}</iyy><izz>{i:.3e}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <collision name="collision">
        <geometry>{geom}</geometry>
        <surface>
          <friction><ode><mu>1.5</mu><mu2>1.5</mu2></ode></friction>
          <contact><ode><kp>1e6</kp><kd>100</kd><min_depth>0.001</min_depth><max_vel>0.1</max_vel></ode></contact>
        </surface>
      </collision>
      <visual name="visual"><geometry>{geom}</geometry>{_script_material(spec['material'])}</visual>
    </link>
  </model>
</sdf>"""


def obstacle_sdf(model_name: str, size=(0.3, 0.3, 0.5)) -> str:
    """A kinematic box used as a dynamic obstacle (moved via set_entity_state)."""
    sx, sy, sz = size
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>false</static>
    <link name="link">
      <kinematic>true</kinematic>
      <gravity>false</gravity>
      <pose>0 0 {sz / 2} 0 0 0</pose>
      <collision name="collision"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry></collision>
      <visual name="visual"><geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        {_script_material('Gazebo/Orange')}</visual>
    </link>
  </model>
</sdf>"""


def _box_link(name, x, y, z, yaw, sx, sy, sz, material_script=None, rgb=None) -> str:
    if material_script:
        mat = (f'<material><script><uri>file://media/materials/scripts/gazebo.material</uri>'
               f'<name>{material_script}</name></script></material>')
    else:
        mat = _material(rgb or (0.8, 0.8, 0.8))
    return f"""
      <link name="{name}">
        <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.4f}</pose>
        <collision name="collision"><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>{mat}</visual>
      </link>"""


def walls_model(layout: Layout) -> str:
    links = []
    t, h = layout.wall_thickness, layout.wall_height
    for i, (x1, y1, x2, y2) in enumerate(layout.walls):
        length = math.hypot(x2 - x1, y2 - y1) + t
        links.append(_box_link(f'wall_{i}', (x1 + x2) / 2, (y1 + y2) / 2, h / 2,
                               math.atan2(y2 - y1, x2 - x1), length, t, h,
                               rgb=(0.93, 0.91, 0.86)))
    return f"""
    <model name="walls">
      <static>true</static>{''.join(links)}
    </model>"""


def furniture_models(layout: Layout) -> str:
    out = []
    for f in layout.furniture:
        link = _box_link('link', 0, 0, f.sz / 2, 0, f.sx, f.sy, f.sz, material_script=f.color)
        out.append(f"""
    <model name="{f.name}">
      <static>true</static>
      <pose>{f.x} {f.y} 0 0 0 {f.yaw}</pose>{link}
    </model>""")
    return ''.join(out)


def dropoff_models(layout: Layout) -> str:
    x, y, _ = layout.dropoff_pose
    mx, my, mz, myaw = layout.marker_pose
    return f"""
    <model name="dropoff_zone">
      <static>true</static>
      <pose>{x} {y} 0.0005 0 0 0</pose>
      <link name="link">
        <visual name="visual">
          <geometry><box><size>0.30 0.30 0.001</size></box></geometry>
          {_material((0.1, 0.8, 0.9))}
        </visual>
      </link>
    </model>
    <include>
      <name>dropoff_marker</name>
      <uri>model://aruco_marker_{layout.marker_id}</uri>
      <pose>{mx} {my} {mz} 0 0 {myaw}</pose>
    </include>"""


def plugins_block(robot_model: str = 'turtlebot3_manipulation_system') -> str:
    return f"""
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros><namespace>/gazebo</namespace></ros>
      <update_rate>20.0</update_rate>
    </plugin>
    <plugin name="honest_link_attacher" filename="libhonest_link_attacher.so">
      <ros><namespace>/fetch_deliver</namespace></ros>
      <robot_model>{robot_model}</robot_model>
      <base_link>base_footprint</base_link>
      <ee_link>link5</ee_link>
      <ee_offset>0.126 0 0</ee_offset>
      <left_finger_joint>gripper_left_joint</left_finger_joint>
      <right_finger_joint>gripper_right_joint</right_finger_joint>
      <!-- finger opening = finger_base_gap + q_left + q_right (link origins) -->
      <finger_base_gap>0.042</finger_base_gap>
      <max_closed_opening>0.034</max_closed_opening>
      <pose_tolerance>0.02</pose_tolerance>
      <angle_tolerance>0.35</angle_tolerance>
      <object_tolerance>0.045</object_tolerance>
    </plugin>"""


def world_sdf(layout: Layout) -> str:
    return f"""<?xml version="1.0"?>
<!-- Generated by fetch_deliver_gazebo/scripts/generate_worlds.py from config/layouts.yaml.
     Do not edit by hand. Layout: {layout.name} -->
<sdf version="1.6">
  <world name="fetch_deliver_{layout.name}">
    <physics type="ode">
      <real_time_update_rate>1000.0</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <ode>
        <solver><type>quick</type><iters>150</iters><precon_iters>0</precon_iters><sor>1.4</sor></solver>
        <constraints><cfm>0.00001</cfm><erp>0.2</erp>
          <contact_max_correcting_vel>2000</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer></constraints>
      </ode>
    </physics>
    <scene>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.7 0.75 0.8 1</background>
      <shadows>false</shadows>
    </scene>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 -9 11 0 0.95 1.5708</pose>
      </camera>
    </gui>
    <include><uri>model://sun</uri></include>
    <include><uri>model://ground_plane</uri></include>
{walls_model(layout)}
{furniture_models(layout)}
{dropoff_models(layout)}
{plugins_block()}
  </world>
</sdf>
"""


# White quiet zone around the black marker square in the texture, as a fraction of
# the marker side on each edge. The textured plane is enlarged by it so the black
# square itself is exactly ``size`` wide, which is what pose estimation assumes.
MARKER_QUIET_ZONE = 1.0 / 6.0


def marker_model_files(marker_id: int, size: float = 0.20):
    """Return (model.config, model.sdf, material script) for an ArUco marker plate.

    ``size`` is the side of the black marker square (the PnP model size)."""
    name = f'aruco_marker_{marker_id}'
    plane = size * (1.0 + 2.0 * MARKER_QUIET_ZONE)
    config = f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <description>ArUco DICT_4X4_50 marker id {marker_id} ({size} m) marking the drop-off.</description>
</model>
"""
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>0.01 {plane + 0.02:.4f} {plane + 0.02:.4f}</size></box></geometry>
      </collision>
      <visual name="backing">
        <pose>-0.002 0 0 0 0 0</pose>
        <geometry><box><size>0.006 {plane + 0.02:.4f} {plane + 0.02:.4f}</size></box></geometry>
        <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse></material>
      </visual>
      <visual name="marker">
        <pose>0.0015 0 0 0 1.5708 0</pose>
        <geometry><plane><normal>0 0 1</normal><size>{plane:.4f} {plane:.4f}</size></plane></geometry>
        <material>
          <script>
            <uri>model://{name}/materials/scripts</uri>
            <uri>model://{name}/materials/textures</uri>
            <name>FetchDeliver/ArucoMarker{marker_id}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    material = f"""material FetchDeliver/ArucoMarker{marker_id}
{{
  receive_shadows off
  technique
  {{
    pass
    {{
      lighting off
      texture_unit
      {{
        texture aruco_{marker_id}.png
        filtering none
      }}
    }}
  }}
}}
"""
    return config, sdf, material


def iter_marker_ids(layouts: Iterable[Layout]):
    return sorted({lay.marker_id for lay in layouts})

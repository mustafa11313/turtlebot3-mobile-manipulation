"""SDF generation for worlds and spawnable objects (Gazebo Classic, SDF 1.6)."""

import math
from typing import Iterable

from .layouts import Layout

# Graspable objects: small enough for the OpenMANIPULATOR-X gripper.
OBJECT_TYPES = {
    'red_cube': {'shape': 'box', 'size': 0.025, 'color': (0.85, 0.05, 0.05)},
    'green_cube': {'shape': 'box', 'size': 0.025, 'color': (0.05, 0.65, 0.10)},
    'blue_cube': {'shape': 'box', 'size': 0.025, 'color': (0.05, 0.15, 0.85)},
    'yellow_cube': {'shape': 'box', 'size': 0.025, 'color': (0.90, 0.80, 0.05)},
    'red_cylinder': {'shape': 'cylinder', 'size': 0.025, 'color': (0.85, 0.05, 0.05)},
}


def _material(rgb) -> str:
    r, g, b = rgb
    return (f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
            f'<specular>0.1 0.1 0.1 1</specular></material>')


def object_sdf(model_name: str, object_type: str) -> str:
    spec = OBJECT_TYPES[object_type]
    s = spec['size']
    if spec['shape'] == 'box':
        geom = f'<box><size>{s} {s} {s}</size></box>'
    else:
        geom = f'<cylinder><radius>{s / 2}</radius><length>{s}</length></cylinder>'
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
      <visual name="visual"><geometry>{geom}</geometry>{_material(spec['color'])}</visual>
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
        {_material((0.9, 0.45, 0.0))}</visual>
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
      <object_tolerance>0.025</object_tolerance>
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


def marker_model_files(marker_id: int, size: float = 0.20):
    """Return (model.config, model.sdf, material script) for an ArUco marker plate."""
    name = f'aruco_marker_{marker_id}'
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
      <collision name="collision"><geometry><box><size>0.01 {size + 0.04} {size + 0.04}</size></box></geometry></collision>
      <visual name="backing">
        <pose>-0.002 0 0 0 0 0</pose>
        <geometry><box><size>0.006 {size + 0.04} {size + 0.04}</size></box></geometry>
        <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse></material>
      </visual>
      <visual name="marker">
        <pose>0.0015 0 0 0 1.5708 0</pose>
        <geometry><plane><normal>0 0 1</normal><size>{size} {size}</size></plane></geometry>
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

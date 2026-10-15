"""MoveIt 2 move_group for the TurtleBot3 Waffle Pi + OpenMANIPULATOR-X.

Adapted from ROBOTIS turtlebot3_manipulation_moveit_config (Apache-2.0).
"""

import os

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(pkg, rel):
    with open(os.path.join(get_package_share_directory(pkg), rel)) as f:
        return yaml.safe_load(f)


def moveit_params(camera_pitch='0.30'):
    desc = xacro.process_file(
        os.path.join(get_package_share_directory('fetch_deliver_description'),
                     'urdf', 'fetch_deliver_robot.urdf.xacro'),
        mappings={'use_sim': 'true', 'camera_pitch': camera_pitch}).toxml()
    with open(os.path.join(get_package_share_directory('fetch_deliver_moveit_config'),
                           'config', 'fetch_deliver_robot.srdf')) as f:
        srdf = f.read()
    pkg = 'fetch_deliver_moveit_config'
    ompl = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': (
                'default_planner_request_adapters/AddTimeOptimalParameterization '
                'default_planner_request_adapters/FixWorkspaceBounds '
                'default_planner_request_adapters/FixStartStateBounds '
                'default_planner_request_adapters/FixStartStateCollision '
                'default_planner_request_adapters/FixStartStatePathConstraints'),
            'start_state_max_bounds_error': 0.1,
        }
    }
    ompl['move_group'].update(_load_yaml(pkg, 'config/ompl_planning.yaml'))
    return [
        {'robot_description': desc},
        {'robot_description_semantic': srdf},
        {'robot_description_kinematics': _load_yaml(pkg, 'config/kinematics.yaml')},
        {'robot_description_planning': _load_yaml(pkg, 'config/joint_limits.yaml')},
        ompl,
        {
            'moveit_manage_controllers': True,
            'trajectory_execution.allowed_execution_duration_scaling': 1.5,
            'trajectory_execution.allowed_goal_duration_margin': 1.0,
            'trajectory_execution.allowed_start_tolerance': 0.05,
        },
        {
            'moveit_simple_controller_manager': _load_yaml(pkg, 'config/moveit_controllers.yaml'),
            'moveit_controller_manager':
                'moveit_simple_controller_manager/MoveItSimpleControllerManager',
        },
        {
            'publish_planning_scene': True,
            'publish_geometry_updates': True,
            'publish_state_updates': True,
            'publish_transforms_updates': True,
            'publish_robot_description': True,
            'publish_robot_description_semantic': True,
        },
    ]


def generate_launch_description():
    params = moveit_params()
    move_group = Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=params + [{'use_sim_time': LaunchConfiguration('use_sim_time')}])
    rviz = Node(
        package='rviz2', executable='rviz2', output='log',
        condition=IfCondition(LaunchConfiguration('rviz')),
        arguments=['-d', os.path.join(get_package_share_directory('fetch_deliver_moveit_config'),
                                      'config', 'moveit.rviz')],
        parameters=params[:3] + [{'use_sim_time': LaunchConfiguration('use_sim_time')}])
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        move_group, rviz,
    ])

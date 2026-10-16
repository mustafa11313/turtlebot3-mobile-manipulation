"""Gazebo Classic simulation: world + robot + controllers + randomized scenario.

ros2 launch fetch_deliver_gazebo sim.launch.py layout:=apartment seed:=3 difficulty:=hard gui:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ROBOT_NAME = 'turtlebot3_manipulation_system'


def _setup(context):
    gz_share = get_package_share_directory('fetch_deliver_gazebo')
    desc_share = get_package_share_directory('fetch_deliver_description')
    gazebo_ros_share = get_package_share_directory('gazebo_ros')

    import sys
    sys.path.insert(0, os.path.dirname(gz_share))  # not needed when installed, harmless
    from fetch_deliver_gazebo.layouts import load_layout

    layout_name = LaunchConfiguration('layout').perform(context)
    layout = load_layout(layout_name)
    x, y, yaw = layout.robot_start
    world = os.path.join(gz_share, 'worlds', f'{layout_name}.world')

    robot_description = ParameterValue(Command([
        'xacro ', os.path.join(desc_share, 'urdf', 'fetch_deliver_robot.urdf.xacro'),
        ' use_sim:=true camera_pitch:=', LaunchConfiguration('camera_pitch')]), value_type=str)

    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world, 'verbose': 'false', 'init': 'true', 'factory': 'true'}.items())
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_share, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(LaunchConfiguration('gui')))

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher',
               parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
               output='screen')

    spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-topic', 'robot_description', '-entity', ROBOT_NAME,
                            '-x', str(x), '-y', str(y), '-z', '0.01', '-Y', str(yaw),
                            '-timeout', '120'])

    def spawner(name):
        return Node(package='controller_manager', executable='spawner', output='screen',
                    arguments=[name, '--controller-manager', '/controller_manager',
                               '--controller-manager-timeout', '120'])

    controllers = RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[
        spawner('joint_state_broadcaster'), spawner('arm_controller'),
        spawner('gripper_controller'), spawner('imu_broadcaster')]))

    scenario = TimerAction(period=8.0, actions=[Node(
        package='fetch_deliver_gazebo', executable='scenario_spawner.py', output='screen',
        condition=IfCondition(LaunchConfiguration('spawn_scenario')),
        parameters=[{'layout': layout_name,
                     'seed': int(LaunchConfiguration('seed').perform(context)),
                     'difficulty': LaunchConfiguration('difficulty'),
                     'target_type': LaunchConfiguration('target_type'),
                     'scenario_file': LaunchConfiguration('scenario_file'),
                     'output_file': LaunchConfiguration('scenario_output'),
                     'use_sim_time': True}])])

    return [gzserver, gzclient, rsp, spawn, controllers, scenario]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('layout', default_value='arena', description='arena | apartment | office'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('difficulty', default_value='easy',
                              description='easy | distractors | occluded | dynamic | hard'),
        DeclareLaunchArgument('target_type', default_value='red_cylinder'),
        DeclareLaunchArgument('scenario_file', default_value=''),
        DeclareLaunchArgument('scenario_output', default_value=''),
        DeclareLaunchArgument('spawn_scenario', default_value='true'),
        DeclareLaunchArgument('camera_pitch', default_value='0.30'),
        DeclareLaunchArgument('gui', default_value='false'),
        OpaqueFunction(function=_setup),
    ])

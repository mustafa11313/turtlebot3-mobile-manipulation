"""Full fetch & deliver mission in simulation.

ros2 launch fetch_deliver_bringup mission.launch.py layout:=apartment difficulty:=hard seed:=4 gui:=true

Starts Gazebo (with a randomized scenario), localization + Nav2, MoveIt 2,
the pick/place server, perception, the align server and the behaviour tree.
With slam:=true the robot maps the environment while searching (frontier search).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    bringup = get_package_share_directory('fetch_deliver_bringup')
    gz = get_package_share_directory('fetch_deliver_gazebo')
    moveit = get_package_share_directory('fetch_deliver_moveit_config')
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    layout = lc('layout')
    slam = lc('slam').lower() == 'true'
    map_yaml = lc('map_file') or os.path.join(bringup, 'maps', f'{layout}.yaml')
    strategy = lc('search_strategy') or ('frontier' if slam else 'waypoints')
    sim_time = {'use_sim_time': True}

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz, 'launch', 'sim.launch.py')),
        launch_arguments={'layout': layout, 'seed': lc('seed'), 'difficulty': lc('difficulty'),
                          'target_type': lc('target'), 'gui': lc('gui'),
                          'scenario_output': lc('scenario_output'),
                          'scenario_file': lc('scenario_file')}.items())
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'navigation.launch.py')),
        launch_arguments={'slam': str(slam).lower(), 'map_yaml': map_yaml}.items())
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit, 'launch', 'move_group.launch.py')))
    nodes = [
        Node(package='fetch_deliver_manipulation', executable='pick_place_server', output='screen',
             parameters=[sim_time]),
        Node(package='fetch_deliver_perception', executable='object_detector', output='screen',
             parameters=[sim_time, {'target_label': lc('target'), 'backend': lc('detector'),
                                    'yolo_weights': lc('yolo_weights')}]),
        Node(package='fetch_deliver_perception', executable='aruco_detector', output='screen',
             parameters=[sim_time]),
        Node(package='fetch_deliver_mission', executable='align_server', output='screen',
             parameters=[sim_time]),
    ]
    mission = Node(package='fetch_deliver_mission', executable='mission', output='screen',
                   condition=IfCondition(LaunchConfiguration('start_mission')),
                   parameters=[sim_time, {'layout': layout, 'target_label': lc('target'),
                                          'search_strategy': strategy,
                                          'shutdown_when_done': lc('shutdown_when_done') == 'true'}])
    rviz = Node(package='rviz2', executable='rviz2', output='log',
                condition=IfCondition(LaunchConfiguration('rviz')),
                arguments=['-d', os.path.join(bringup, 'rviz', 'fetch_deliver.rviz')],
                parameters=[sim_time])
    return [sim,
            TimerAction(period=12.0, actions=[nav, move_group] + nodes),
            TimerAction(period=25.0, actions=[mission]),
            rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('layout', default_value='apartment', description='apartment | office'),
        DeclareLaunchArgument('seed', default_value='0'),
        DeclareLaunchArgument('difficulty', default_value='easy',
                              description='easy | distractors | occluded | dynamic | hard'),
        DeclareLaunchArgument('target', default_value='red_cube'),
        DeclareLaunchArgument('slam', default_value='false',
                              description='map while searching instead of AMCL on a saved map'),
        DeclareLaunchArgument('map_file', default_value='', description='saved map yaml (default: maps/<layout>.yaml)'),
        DeclareLaunchArgument('search_strategy', default_value='',
                              description='waypoints | frontier (default: frontier with slam)'),
        DeclareLaunchArgument('detector', default_value='color', description='color | yolo'),
        DeclareLaunchArgument('yolo_weights', default_value=''),
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('start_mission', default_value='true'),
        DeclareLaunchArgument('shutdown_when_done', default_value='false'),
        DeclareLaunchArgument('scenario_output', default_value=''),
        DeclareLaunchArgument('scenario_file', default_value=''),
        OpaqueFunction(function=_setup),
    ])

"""Build a map of a layout autonomously: Gazebo + slam_toolbox + Nav2 + frontier explorer.

ros2 launch fetch_deliver_bringup mapping.launch.py layout:=apartment gui:=true
The finished map is saved to <map_dir>/<layout>.yaml (default: the current directory).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('fetch_deliver_bringup')
    gz = get_package_share_directory('fetch_deliver_gazebo')
    layout = LaunchConfiguration('layout')
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz, 'launch', 'sim.launch.py')),
        launch_arguments={'layout': layout, 'gui': LaunchConfiguration('gui'),
                          'spawn_scenario': 'false'}.items())
    nav = TimerAction(period=10.0, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'navigation.launch.py')),
        launch_arguments={'slam': 'true'}.items())])
    explorer = TimerAction(period=25.0, actions=[Node(
        package='fetch_deliver_mission', executable='frontier_explorer', output='screen',
        condition=IfCondition(LaunchConfiguration('explore')),
        parameters=[{'use_sim_time': True,
                     'map_path': PathJoinSubstitution([LaunchConfiguration('map_dir'), layout])}])])
    rviz = Node(package='rviz2', executable='rviz2', output='log',
                condition=IfCondition(LaunchConfiguration('rviz')),
                arguments=['-d', os.path.join(bringup, 'rviz', 'fetch_deliver.rviz')],
                parameters=[{'use_sim_time': True}])
    return LaunchDescription([
        DeclareLaunchArgument('layout', default_value='apartment'),
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('explore', default_value='true'),
        DeclareLaunchArgument('map_dir', default_value=os.getcwd()),
        sim, nav, explorer, rviz,
    ])

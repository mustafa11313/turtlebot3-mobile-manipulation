"""Localization (AMCL on a saved map, or slam_toolbox) + Nav2 navigation.

Argument names are prefixed (map_yaml, nav2_params_file) because launch
configurations are global in Humble: gazebo_ros already declares "params_file".
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    bringup = get_package_share_directory('fetch_deliver_bringup')
    nav2 = get_package_share_directory('nav2_bringup')
    params = LaunchConfiguration('nav2_params_file')
    slam = LaunchConfiguration('slam')

    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2, 'launch', 'localization_launch.py')),
        condition=UnlessCondition(slam),
        launch_arguments={'map': LaunchConfiguration('map_yaml'), 'use_sim_time': 'true',
                          'params_file': params, 'autostart': 'true',
                          'use_composition': 'False'}.items())
    slam_toolbox = Node(
        package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
        output='screen', condition=IfCondition(slam),
        parameters=[os.path.join(bringup, 'config', 'slam_params.yaml'), {'use_sim_time': True}])
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2, 'launch', 'navigation_launch.py')),
        launch_arguments={'use_sim_time': 'true', 'params_file': params, 'autostart': 'true',
                          'use_composition': 'False'}.items())
    return LaunchDescription([
        DeclareLaunchArgument('slam', default_value='false'),
        DeclareLaunchArgument('map_yaml', default_value=PathJoinSubstitution(
            [bringup, 'maps', 'apartment.yaml'])),
        DeclareLaunchArgument('nav2_params_file', default_value=os.path.join(bringup, 'config',
                                                                        'nav2_params.yaml')),
        localization, slam_toolbox, navigation,
    ])

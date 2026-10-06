"""Bring up the full sorting stack:
    1. FANUC ROS 2 driver (action + msg + srv servers)
    2. Gripper controller (EIP register writer, exposes /gripper/open|close)
    3. Sorter node (this package)

Override defaults with launch args, e.g.:
    ros2 launch conveyor_sorter sorter.launch.py \\
        robot_name:=ER4IA robot_ip:=192.168.0.10
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


GRIPPER_CONTROLLER_PATH = os.path.expanduser(
    "~/ros2_ws/src/fanuc-color-sorter/gripper_control/gripper_controller.py"
)


def generate_launch_description():
    robot_name = LaunchConfiguration("robot_name")
    robot_ip = LaunchConfiguration("robot_ip")
    config_file = LaunchConfiguration("config")

    default_cfg = os.path.join(
        get_package_share_directory("conveyor_sorter"), "config", "sorter.yaml"
    )

    return LaunchDescription([
        DeclareLaunchArgument("robot_name", default_value="ER4IA"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.0.10",  # set your FANUC controller IP
                              description="FANUC controller IP"),
        DeclareLaunchArgument("config", default_value=default_cfg,
                              description="Path to sorter.yaml"),

        LogInfo(msg=["Starting FANUC driver components..."]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("action_servers"),
                    "launch", "action_servers.launch.py",
                ])
            ]),
            launch_arguments={
                "robot_name": robot_name,
                "robot_ip": robot_ip,
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("msg_publishers"),
                    "launch", "message_publishers.launch.py",
                ])
            ]),
            launch_arguments={
                "robot_name": robot_name,
                "robot_ip": robot_ip,
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare("srv_services"),
                    "launch", "srv_services.launch.py",
                ])
            ]),
            launch_arguments={
                "robot_name": robot_name,
                "robot_ip": robot_ip,
            }.items(),
        ),

        LogInfo(msg=["Starting gripper controller..."]),
        ExecuteProcess(
            cmd=["python3", GRIPPER_CONTROLLER_PATH],
            output="screen",
            name="gripper_controller",
        ),

        LogInfo(msg=["Starting sorter node..."]),
        Node(
            package="conveyor_sorter",
            executable="sorter",
            name="conveyor_sorter",
            output="screen",
            arguments=["--config", config_file],
        ),
    ])

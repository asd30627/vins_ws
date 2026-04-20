from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('kaist_player')
    default_config_file = os.path.join(pkg_dir, 'config', 'urban28_pankyo.yaml')

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to YAML config file'
    )

    dataset_root_arg = DeclareLaunchArgument(
        'dataset_root',
        default_value='/mnt/sata4t/datasets/kaist_complex_urban/extracted/urban28-pankyo',
        description='KAIST sequence root folder'
    )

    return LaunchDescription([
        config_file_arg,
        dataset_root_arg,
        Node(
            package='kaist_player',
            executable='kaist_player_node',
            name='kaist_player_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                {
                    'dataset_root': LaunchConfiguration('dataset_root')
                }
            ]
        )
    ])
from setuptools import setup
from glob import glob
import os

package_name = 'kaist_player'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ivlab3',
    maintainer_email='ivlab3@example.com',
    description='ROS2 native KAIST stereo + IMU player for VINS-Fusion',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'kaist_player_node = kaist_player.kaist_player_node:main',
        ],
    },
)

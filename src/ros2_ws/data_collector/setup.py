from setuptools import setup

package_name = 'data_collector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ivlab3',
    maintainer_email='asd30627@gmail.com',
    description='ROS2 dataset collector for image, imu, odom, camera_info, and clock.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'collector_node = data_collector.collector_node:main',
        ],
    },
)
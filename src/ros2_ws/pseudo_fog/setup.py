from setuptools import setup

package_name = 'pseudo_fog'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ivlab3',
    maintainer_email='asd30627@example.com',
    description='Pseudo FOG simulator package.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pseudo_fog_node = pseudo_fog.pseudo_fog_node:main',
            'fog_compare_node = pseudo_fog.fog_compare_node:main',
            'fog_spec_check_node = pseudo_fog.fog_spec_check_node:main',
        ],
    },
)
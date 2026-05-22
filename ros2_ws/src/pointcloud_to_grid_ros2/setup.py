from setuptools import setup
import os
from glob import glob

package_name = 'pointcloud_to_grid'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jdb00039',
    maintainer_email='jalen.beeman@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointcloud_to_grid_node = pointcloud_to_grid.pointcloud_to_grid_node:main',
            'interpolated_grid_node = pointcloud_to_grid.interpolated_grid_node:main',
        ],
    },
)

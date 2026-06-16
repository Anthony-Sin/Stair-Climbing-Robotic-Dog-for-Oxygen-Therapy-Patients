from glob import glob
from setuptools import find_packages, setup


package_name = "sim_lidar_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Codex",
    maintainer_email="codex@example.com",
    description="Bridge the Isaac sim XT16 cloud + odom onto the real ROS2 topics; forward Nav2 cmd_vel back to Isaac.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "sim_bridge_node = sim_lidar_bridge.sim_bridge_node:main",
        ],
    },
)

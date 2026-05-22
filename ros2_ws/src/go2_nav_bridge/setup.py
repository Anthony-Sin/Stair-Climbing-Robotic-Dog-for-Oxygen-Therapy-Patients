from setuptools import find_packages, setup


package_name = "go2_nav_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Codex",
    maintainer_email="codex@example.com",
    description="ROS 2 bridge for Go2 odometry and sport-mode commands.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bridge_node = go2_nav_bridge.bridge_node:main",
        ],
    },
)

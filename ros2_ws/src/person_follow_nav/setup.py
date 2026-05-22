from glob import glob
from setuptools import find_packages, setup


package_name = "person_follow_nav"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Codex",
    maintainer_email="codex@example.com",
    description="UDP target ingest and local path generation for Nav2 person following.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "follow_controller_node = person_follow_nav.follow_controller_node:main",
            "realsense_camera_client = person_follow_nav.camera_client:main",
        ],
    },
)

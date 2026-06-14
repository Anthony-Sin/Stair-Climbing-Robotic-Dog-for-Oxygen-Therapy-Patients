import os
import urllib.request
import pathlib

def download_file(url, dest):
    dest_path = pathlib.Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} -> {dest} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print("Success.")
    except Exception as e:
        print(f"Failed to download: {e}")

def main():
    base_url = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/"
    dest_dir = pathlib.Path(__file__).parent.parent.parent / "sim" / "isaac" / "assets" / "go2_fixed"
    
    files = [
        "go2.usd",
        "configuration/go2_description_base.usd",
        "configuration/go2_description_physics.usd",
        "configuration/go2_description_sensor.usd",
        "configuration/go2_description_robot.usd"
    ]
    
    for f in files:
        url = base_url + f
        dest = dest_dir / f
        download_file(url, dest)

if __name__ == "__main__":
    main()

import argparse
import json
import math
import socket
import sys
import threading
import time
 
# ---------------------------------------------------------------------------
# Isaac Sim bootstrap -- must happen before any omni imports
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp
 
parser = argparse.ArgumentParser(description="Isaac Sim Go2 environment")
parser.add_argument("--headless", action="store_true", help="Run without GUI")
parser.add_argument("--cmd-port", type=int, default=55001,
                    help="UDP port for incoming velocity commands")
parser.add_argument("--frame-port", type=int, default=55002,
                    help="UDP port for outgoing camera frames")
parser.add_argument("--physics-hz", type=int, default=60,
                    help="Physics simulation rate in Hz")
parser.add_argument("--render-every", type=int, default=2,
                    help="Publish a camera frame every N physics steps")
parser.add_argument("--person-x", type=float, default=2.5,
                    help="Initial X position of the person target")
parser.add_argument("--person-y", type=float, default=0.0,
                    help="Initial Y position of the person target")
parser.add_argument("--person-move", action="store_true",
                    help="Make the person walk a simple patrol path")
parser.add_argument("--frame-host", type=str, default='0.0.0.0',
                    help="Destination IP for camera frame UDP (WSL2 IP if running vision in WSL)")
args = parser.parse_args()
 
simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})
 
# ---------------------------------------------------------------------------
# Omniverse / Isaac imports (after SimulationApp is created)
# ---------------------------------------------------------------------------
import carb
import numpy as np
import omni.isaac.core.utils.nucleus as nucleus_utils
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.objects import DynamicCapsule, GroundPlane
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera
from pxr import Gf, UsdGeom, UsdPhysics
from omni.isaac.core.prims import GeometryPrim
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GO2_USD_PATH   = "/World/Go2"
CAMERA_PRIM    = "/World/Go2/trunk/front_camera"
PERSON_PRIM    = "/World/Person"
NUCLEUS_GO2    = "/Isaac/Robots/Unitree/Go2/go2.usd"
LOCAL_GO2      = str((
    __import__("pathlib").Path(__file__).parent / "assets" / "go2.usd"
).resolve())
 
# Go2 base link name inside the articulation (adjust if your USD differs)
BASE_LINK_NAME = "trunk"
 
# ---------------------------------------------------------------------------
# Shared state between threads
# ---------------------------------------------------------------------------
_cmd_lock   = threading.Lock()
_cmd_vel    = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "ts": 0.0}
_running    = True
 
# ---------------------------------------------------------------------------
# UDP command receiver  (background thread)
# ---------------------------------------------------------------------------
def _cmd_receiver_thread(port: int) -> None:
    """Receive velocity commands from main.py --sim via UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    print(f"[isaac_env] CMD receiver listening on UDP 0.0.0.0:{port}")
    while _running:
        try:
            data, _ = sock.recvfrom(256)
            payload = json.loads(data.decode("utf-8"))
            with _cmd_lock:
                _cmd_vel["vx"] = float(payload.get("vx", 0.0))
                _cmd_vel["vy"] = float(payload.get("vy", 0.0))
                _cmd_vel["wz"] = float(payload.get("wz", 0.0))
                _cmd_vel["ts"] = time.monotonic()
        except socket.timeout:
            continue
        except Exception as exc:
            print(f"[isaac_env] CMD receive error: {exc}")
    sock.close()
 
 
# ---------------------------------------------------------------------------
# Build the Isaac Sim world
# ---------------------------------------------------------------------------
def build_world(physics_hz: int) -> World:
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / physics_hz,
        rendering_dt=1.0 / physics_hz,
    )
    world.scene.add_default_ground_plane()
    return world
 
 
def _resolve_go2_usd() -> str:
    """Return the best available USD path for Go2."""
    nucleus_server = nucleus_utils.get_assets_root_path()
    if nucleus_server:
        candidate = nucleus_server + NUCLEUS_GO2
        try:
            nucleus_utils.is_file(candidate)
            print(f"[isaac_env] Using Nucleus asset: {candidate}")
            return candidate
        except Exception:
            pass
 
    import pathlib
    local = pathlib.Path(LOCAL_GO2)
    if local.exists():
        print(f"[isaac_env] Using local asset: {LOCAL_GO2}")
        return LOCAL_GO2
 
    raise FileNotFoundError(
        "Go2 USD not found in Nucleus or isaac/assets/go2.usd.\n"
        "Run:  python isaac/go2_usd_setup.py"
    )
 
 
def load_go2(world: World) -> Articulation:
    usd_path = _resolve_go2_usd()
    add_reference_to_stage(usd_path=usd_path, prim_path=GO2_USD_PATH)
    go2 = world.scene.add(
        Articulation(
            prim_path=GO2_USD_PATH, 
            name="go2",
            position=np.array([0.0, 0.0, 0.4])  # <--- Spawns dog 40cm in the air
        )
    )
    return go2
 
 
def add_camera(resolution: tuple = (1280, 720)) -> Camera:
    """Mount a front-facing RGB-D camera on the Go2 trunk."""
    camera = Camera(
        prim_path=CAMERA_PRIM,
        name="front_camera",
        resolution=resolution,
        # 25 cm forward, 10 cm up from trunk origin, facing forward
        position=np.array([0.25, 0.0, 0.1]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
    )
    camera.initialize()
    # Gives metric depth (meters) aligned to color image
    camera.add_distance_to_image_plane_to_frame()
    return camera
 
def spawn_person(world, x: float = 2.5, y: float = 0.0):
    # Using the direct AWS link you found
    human_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People/Characters/female_adult_police_01_new/female_adult_police_01_new.usd"
    
    print(f"[isaac_env] Downloading and loading humanoid asset from: {human_usd}")
    
    # Load the USD straight from the web
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.rotations import euler_angles_to_quat # <-- New Import
    
    add_reference_to_stage(usd_path=human_usd, prim_path=PERSON_PRIM)
    
    # Convert standard degrees [X, Y, Z] to a quaternion.
    # 90 degrees on the Y-axis should hinge him straight up.
    # (If he ends up doing a Matrix backbend, just change 90 to -90)
    stand_up_quat = euler_angles_to_quat(np.array([0, 90, 0]), degrees=True)
    
    # Wrap it in a GeometryPrim so your patrol script can move it
    person = GeometryPrim(
        prim_path=PERSON_PRIM,
        name="person",
        position=np.array([x, y, 0.0]), 
        orientation=stand_up_quat
    )
    world.scene.add(person)
    return person
# ---------------------------------------------------------------------------
# Locomotion helpers
# ---------------------------------------------------------------------------
 
def _get_root_xform(go2: Articulation):
    """Return the UsdGeom.Xformable for the Go2 root prim."""
    return UsdGeom.Xformable(go2.prim)
 
 
def _extract_yaw(xform_matrix) -> float:
    """Extract yaw (Z-axis rotation) from a 4x4 USD transform matrix."""
    return math.atan2(float(xform_matrix[0][1]), float(xform_matrix[0][0]))
 
 
def apply_velocity_to_go2(go2: Articulation, vx: float, vy: float, wz: float,
                           dt: float) -> None:
    """
    Move Go2 by directly setting root rigid-body velocities.
 
    Strategy 1 (preferred): UsdPhysics RigidBodyAPI velocity.
    Strategy 2 (fallback):  Teleport root position + yaw each step.
    Both preserve the visual appearance for the person-following use-case
    without requiring a full locomotion controller.
    """
    try:
        rb_api = UsdPhysics.RigidBodyAPI(root_prim)
        vel_attr = rb_api.GetVelocityAttr()
        root_prim = go2.prim
        if vel_attr and vel_attr.IsValid():
            xform = UsdGeom.Xformable(root_prim)
            mat   = xform.ComputeLocalToWorldTransform(0)
            yaw   = _extract_yaw(mat)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            # Rotate body-frame velocities into world frame
            wx = cos_y * vx - sin_y * vy
            wy = sin_y * vx + cos_y * vy
            rb_api.GetVelocityAttr().Set(Gf.Vec3f(float(wx), float(wy), 0.0))
            rb_api.GetAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, float(wz)))
            return
    except Exception:
        pass
 
    # Fallback: kinematic teleport
    try:
        xformable = UsdGeom.Xformable(go2.prim)
        mat = xformable.ComputeLocalToWorldTransform(0)
        yaw = _extract_yaw(mat)
        new_yaw = yaw + wz * dt
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        dx = (cos_y * vx - sin_y * vy) * dt
        dy = (sin_y * vx + cos_y * vy) * dt
 
        # Read current translation
        tx = float(mat[3][0])
        ty = float(mat[3][1])
        tz = float(mat[3][2])
 
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(tx + dx, ty + dy, tz))
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(new_yaw)))
    except Exception as exc:
        pass  # silently ignore during early init frames
 
 
# ---------------------------------------------------------------------------
# Person patrol path
# ---------------------------------------------------------------------------
_person_patrol_t = 0.0
_PATROL_RADIUS   = 1.8   # metres
_PATROL_SPEED    = 0.25  # rad/s
 
 
def update_person_patrol(person, dt: float) -> None:
    global _person_patrol_t
    _person_patrol_t += _PATROL_SPEED * dt
    px = _PATROL_RADIUS * math.cos(_person_patrol_t)
    py = _PATROL_RADIUS * math.sin(_person_patrol_t)
    person.set_world_pose(
        position=np.array([px, py, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
 
# ---------------------------------------------------------------------------
# Frame publisher
# ---------------------------------------------------------------------------
class FramePublisher:
    """Encodes RGB + depth frames and sends over UDP to SimCameraCapture."""

    PUBLISH_W = 160
    PUBLISH_H = 120
    
    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
        self._dest = (host, port)
        self._seq  = 0
        print(f"[isaac_env] Frame publisher sending to UDP {host}:{port}")
 
    def send(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        import cv2, base64, zlib
        w, h = self.PUBLISH_W, self.PUBLISH_H
        small_rgb   = cv2.resize(rgb,   (w, h), interpolation=cv2.INTER_LINEAR)
        small_depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # JPEG-encode color (~5-15KB)
        ok, buf = cv2.imencode('.jpg', small_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
            
        rgb_b64   = base64.b64encode(buf.tobytes()).decode('ascii')
        
        # zlib-compress depth (~15-25KB) with compression level 9
        depth_b64 = base64.b64encode(
            zlib.compress(small_depth.astype(np.uint16).tobytes(), level=9)
        ).decode('ascii')

        meta = {
            "seq":   self._seq,
            "ts":    time.time(),
            "w":     w,
            "h":     h,
            "enc":   "jpg+zlib",          # version flag
            "rgb":   rgb_b64,
            "depth": depth_b64,
        }
        self._seq += 1
        payload = json.dumps(meta).encode("utf-8")
        if len(payload) > 65000:
            print(f"[isaac_env] Warning: frame {self._seq} still too large ({len(payload)} bytes)")
            return
        try:
            self._sock.sendto(payload, self._dest)
        except Exception as exc:
            print(f"[isaac_env] Frame send error: {exc}")
            
    def close(self) -> None:
        self._sock.close()
# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def main() -> None:
    global _running
 
    print("[isaac_env] Building world ...")
    world = build_world(args.physics_hz)
 
    print("[isaac_env] Loading Go2 ...")
    go2 = load_go2(world)
 
    print("[isaac_env] Adding camera ...")
    camera = add_camera()
 
    print("[isaac_env] Spawning person target ...")
    person = spawn_person(world, x=args.person_x, y=args.person_y)
 
    world.reset()
    print(f"[isaac_env] World ready. Physics @ {args.physics_hz} Hz, "
          f"frames every {args.render_every} steps.")
 
    # Start background thread for receiving velocity commands
    cmd_thread = threading.Thread(
        target=_cmd_receiver_thread,
        args=(args.cmd_port,),
        daemon=True,
    )
    cmd_thread.start()
 
    publisher  = FramePublisher(host=args.frame_host, port=args.frame_port)
    dt         = 1.0 / args.physics_hz
    step_count = 0
 
    # Stale command timeout: stop robot if no command received for this long
    CMD_TIMEOUT_SEC = 1.0
 
    try:
        while simulation_app.is_running():
            world.step(render=not args.headless)
            step_count += 1
 
            # Read latest velocity command (zero out if stale)
            with _cmd_lock:
                age = time.monotonic() - _cmd_vel["ts"]
                if age > CMD_TIMEOUT_SEC:
                    vx, vy, wz = 0.0, 0.0, 0.0
                else:
                    vx = _cmd_vel["vx"]
                    vy = _cmd_vel["vy"]
                    wz = _cmd_vel["wz"]
 
            apply_velocity_to_go2(go2, vx, vy, wz, dt)
 
            if args.person_move:
                update_person_patrol(person, dt)
 
            # Publish camera frame at reduced rate
            if step_count % args.render_every == 0:
                try:
                    rgb_data   = camera.get_rgb()
                    depth_data = camera.get_depth()
                    if rgb_data is not None and depth_data is not None:
                        # depth_data is in metres; convert to uint16 millimetres
                        depth_mm = (depth_data * 1000.0).clip(0, 65535).astype(np.uint16)
                        publisher.send(rgb_data, depth_mm)
                except Exception as exc:
                    print(f"[isaac_env] Camera capture error: {exc}")
 
    except KeyboardInterrupt:
        print("\n[isaac_env] KeyboardInterrupt - shutting down.")
    finally:
        _running = False
        publisher.close()
        simulation_app.close()
        print("[isaac_env] Done.")
 
 
if __name__ == "__main__":
    main()
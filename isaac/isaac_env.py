import argparse
import json
import logging
import math
import socket
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Isaac Sim bootstrap -- must happen before any omni imports
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sim_logging_utils import configure_sim_logger, log_event

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
parser.add_argument("--log-dir", type=str, default=str(REPO_ROOT / "log"),
                    help="Directory for per-run Isaac Sim JSONL logs")
parser.add_argument("--quiet-console-log", action="store_true",
                    help="Write JSONL logs only and suppress pretty console log lines")
parser.add_argument("--no-view-follow-camera", action="store_true",
                    help="Do not switch the Isaac viewport to the dynamic Go2 follow camera")
parser.add_argument("--no-hold-motion-until-command", dest="hold_motion_until_command",
                    action="store_false", default=True,
                    help="Let autonomous scene motion start before the Docker/controller command stream is seen")
parser.add_argument("--view-camera-distance", type=float, default=3.2,
                    help="Viewport follow camera distance behind the Go2 robot")
parser.add_argument("--view-camera-height", type=float, default=1.45,
                    help="Viewport follow camera height above the route")
parser.add_argument("--view-camera-side-offset", type=float, default=-0.85,
                    help="Viewport follow camera side offset relative to the Go2 heading")
args = parser.parse_args()

# Setup logger
LOGGER = configure_sim_logger(
    "isaac_env",
    log_dir=args.log_dir,
    reset=True,
    console=not args.quiet_console_log,
)

log_event(
    LOGGER,
    logging.INFO,
    "isaac_env_bootstrap",
    "Starting Isaac Sim Go2 environment",
    headless=bool(args.headless),
    cmd_port=int(args.cmd_port),
    frame_port=int(args.frame_port),
    frame_host=args.frame_host,
    physics_hz=int(args.physics_hz),
    render_every=int(args.render_every),
    log_path=getattr(LOGGER, "sim_log_path", ""),
)

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
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from omni.isaac.core.prims import GeometryPrim

from sim_go2_locomotion import Go2LocomotionState, apply_go2_velocity, hold_go2_stable
from sim_person_actor import spawn_sim_person

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GO2_USD_PATH   = "/World/Go2"
CAMERA_PRIM    = "/World/Go2/trunk/front_camera"
VIEW_CAMERA_PRIM = "/World/View/Go2FollowCamera"
PERSON_PRIM    = "/World/Person"
NUCLEUS_GO2    = "/Isaac/Robots/Unitree/Go2/go2.usd"
LOCAL_GO2      = str((
    __import__("pathlib").Path(__file__).parent / "assets" / "go2.usd"
).resolve())

# Go2 base link name inside the articulation (adjust if your USD differs)
BASE_LINK_NAME = "trunk"

_go2_locomotion_state = Go2LocomotionState(target_height_m=0.32)

# ---------------------------------------------------------------------------
# Shared state between threads
# ---------------------------------------------------------------------------
_cmd_lock   = threading.Lock()
_cmd_vel    = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "ts": 0.0,
    "count": 0,
    "active_count": 0,
    "last_nonzero_ts": 0.0,
}
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
    log_event(
        LOGGER,
        logging.INFO,
        "cmd_receiver_started",
        "Isaac velocity command receiver is listening",
        bind_host="0.0.0.0",
        port=int(port),
    )
    while _running:
        try:
            data, _ = sock.recvfrom(256)
            payload = json.loads(data.decode("utf-8"))
            vx = float(payload.get("vx", 0.0))
            vy = float(payload.get("vy", 0.0))
            wz = float(payload.get("wz", 0.0))
            is_nonzero_command = (abs(vx) > 0.01) or (abs(vy) > 0.01) or (abs(wz) > 0.01)
            with _cmd_lock:
                _cmd_vel["vx"] = vx
                _cmd_vel["vy"] = vy
                _cmd_vel["wz"] = wz
                _cmd_vel["ts"] = time.monotonic()
                _cmd_vel["count"] = int(_cmd_vel.get("count", 0)) + 1
                cmd_count = int(_cmd_vel["count"])
                if is_nonzero_command:
                    _cmd_vel["active_count"] = int(_cmd_vel.get("active_count", 0)) + 1
                    _cmd_vel["last_nonzero_ts"] = _cmd_vel["ts"]
                active_count = int(_cmd_vel.get("active_count", 0))
            if cmd_count == 1:
                log_event(
                    LOGGER,
                    logging.INFO,
                    "controller_packet_stream_started",
                    "First Docker/controller velocity packet received",
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    nonzero=bool(is_nonzero_command),
                )
            if is_nonzero_command and active_count == 1:
                log_event(
                    LOGGER,
                    logging.INFO,
                    "controller_command_stream_started",
                    "First nonzero Docker/controller velocity command received; scene motion is released",
                    vx=vx,
                    vy=vy,
                    wz=wz,
                )
            log_event(
                LOGGER,
                logging.DEBUG,
                "cmd_received",
                "Velocity command received by Isaac",
                vx=vx,
                vy=vy,
                wz=wz,
            )
        except socket.timeout:
            continue
        except Exception as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "cmd_receive_error",
                "Isaac command receiver failed to parse a packet",
                error=str(exc),
            )
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
            log_event(
                LOGGER,
                logging.INFO,
                "go2_asset_selected",
                "Using Go2 Nucleus asset",
                asset_path=candidate,
            )
            return candidate
        except Exception:
            pass

    import pathlib
    local = pathlib.Path(LOCAL_GO2)
    if local.exists():
        log_event(
            LOGGER,
            logging.INFO,
            "go2_asset_selected",
            "Using local Go2 asset",
            asset_path=LOCAL_GO2,
        )
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
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        attach_robot_o2_tank(stage, GO2_USD_PATH + "/trunk")
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "o2_tank_mount_failed", "Failed to retrieve USD stage for mounting tank", error=str(exc))
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

    # Configure physical camera sensor properties to match RealSense D435
    # 36mm horizontal sensor, 26mm focal length -> ~69.4 deg hFOV
    try:
        prim = camera.prim
        prim.GetAttribute("focalLength").Set(26.0)
        prim.GetAttribute("horizontalAperture").Set(36.0)
        prim.GetAttribute("verticalAperture").Set(20.25)
        log_event(LOGGER, logging.INFO, "camera_intrinsics_configured", "Set D435 camera intrinsics: 36mm aperture, 26mm focal length")
    except Exception as e:
        log_event(LOGGER, logging.WARNING, "camera_intrinsics_failed", f"Failed to set camera intrinsics on USD prim: {e}")

    return camera


def get_terrain_height(x: float, y: float) -> float:
    """Return the exact terrain height at coordinate (x, y) based on spawned geometry."""
    if not (-1.05 <= y <= 1.05):
        return 0.0
    # Stairs: 2.0 to 3.5m
    if 2.0 <= x < 3.5:
        step_idx = int((x - 2.0) / 0.3)
        return min(0.40, (step_idx + 1) * 0.08)
    # Landing: 3.5 to 5.0m
    elif 3.5 <= x < 5.0:
        return 0.40
    # Ramp: 5.0 to 7.0m
    elif 5.0 <= x < 7.0:
        return 0.40 * (1.0 - (x - 5.0) / 2.0)
    # Flat ground
    return 0.0


def _add_visual_box(world: World, prim_path: str, name: str, position, scale, color, orientation=None) -> bool:
    try:
        from omni.isaac.core.objects import VisualCuboid as CuboidClass
    except Exception:
        from omni.isaac.core.objects import FixedCuboid as CuboidClass

    try:
        kwargs = {
            "prim_path": prim_path,
            "name": name,
            "position": np.array(position, dtype=float),
            "scale": np.array(scale, dtype=float),
            "color": np.array(color, dtype=float),
        }
        if orientation is not None:
            kwargs["orientation"] = np.array(orientation, dtype=float)
        world.scene.add(CuboidClass(**kwargs))
        return True
    except Exception as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "scene_prop_spawn_failed",
            f"Failed to spawn scene prop {name}",
            prim_path=prim_path,
            error=str(exc),
        )
        return False


def _set_xform_ops(prim, translate=None, rotate_xyz=None) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if translate is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    if rotate_xyz is not None:
        xform.AddRotateXYZOp().Set(Gf.Vec3f(float(rotate_xyz[0]), float(rotate_xyz[1]), float(rotate_xyz[2])))


def setup_scene_lighting(stage) -> None:
    from pxr import UsdLux

    try:
        if not stage.GetPrimAtPath("/World/Lighting").IsValid():
            stage.DefinePrim("/World/Lighting", "Xform")

        dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/SoftBlueDome")
        dome.CreateIntensityAttr().Set(420.0)
        dome.CreateColorAttr().Set(Gf.Vec3f(0.72, 0.80, 1.0))

        key = UsdLux.DistantLight.Define(stage, "/World/Lighting/WarmKey")
        key.CreateIntensityAttr().Set(1350.0)
        key.CreateAngleAttr().Set(1.8)
        key.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.92, 0.78))
        _set_xform_ops(key.GetPrim(), rotate_xyz=(-48.0, 0.0, 32.0))

        fill = UsdLux.RectLight.Define(stage, "/World/Lighting/WindowFill")
        fill.CreateIntensityAttr().Set(650.0)
        fill.CreateWidthAttr().Set(5.0)
        fill.CreateHeightAttr().Set(3.0)
        fill.CreateColorAttr().Set(Gf.Vec3f(0.68, 0.82, 1.0))
        _set_xform_ops(fill.GetPrim(), translate=(3.6, -2.4, 2.2), rotate_xyz=(-38.0, 0.0, 20.0))

        for idx, x_pos in enumerate((1.2, 3.6, 6.0, 8.0)):
            panel = UsdLux.RectLight.Define(stage, f"/World/Lighting/CeilingPanel_{idx}")
            panel.CreateIntensityAttr().Set(420.0)
            panel.CreateWidthAttr().Set(1.6)
            panel.CreateHeightAttr().Set(0.45)
            panel.CreateColorAttr().Set(Gf.Vec3f(0.92, 0.96, 1.0))
            _set_xform_ops(panel.GetPrim(), translate=(x_pos, 0.0, 2.35), rotate_xyz=(0.0, 90.0, 0.0))

        log_event(
            LOGGER,
            logging.INFO,
            "scene_lighting_configured",
            "Configured soft dome, key, fill, and ceiling panel lighting",
            dome="/World/Lighting/SoftBlueDome",
            key="/World/Lighting/WarmKey",
            fill="/World/Lighting/WindowFill",
        )
    except Exception as exc:
        log_event(
            LOGGER,
            logging.WARNING,
            "scene_lighting_failed",
            "Failed to configure enhanced scene lighting",
            error=str(exc),
        )


def update_scene_lighting(stage, elapsed_sec: float) -> None:
    from pxr import UsdLux

    try:
        key_prim = stage.GetPrimAtPath("/World/Lighting/WarmKey")
        if key_prim.IsValid():
            key = UsdLux.DistantLight(key_prim)
            key.GetIntensityAttr().Set(1300.0 + 120.0 * math.sin(0.16 * elapsed_sec))

        fill_prim = stage.GetPrimAtPath("/World/Lighting/WindowFill")
        if fill_prim.IsValid():
            fill = UsdLux.RectLight(fill_prim)
            fill.GetIntensityAttr().Set(620.0 + 90.0 * math.sin(0.11 * elapsed_sec + 0.7))

        for idx in range(4):
            panel_prim = stage.GetPrimAtPath(f"/World/Lighting/CeilingPanel_{idx}")
            if panel_prim.IsValid():
                panel = UsdLux.RectLight(panel_prim)
                panel.GetIntensityAttr().Set(390.0 + 40.0 * math.sin(0.23 * elapsed_sec + idx))
    except Exception:
        pass


def spawn_scene_visual_details(world: World) -> None:
    prop_count = 0

    # Colored path bands make the route readable from the follow camera.
    for side_y, color, name in (
        (-0.78, (0.12, 0.28, 0.38), "right_route_band"),
        (0.78, (0.10, 0.34, 0.28), "left_route_band"),
    ):
        if _add_visual_box(
            world,
            f"/World/Environment/{name}",
            name,
            (4.15, side_y, 0.011),
            (8.4, 0.045, 0.012),
            color,
        ):
            prop_count += 1

    for idx, x_pos in enumerate(np.linspace(0.6, 8.2, 13)):
        z_pos = get_terrain_height(float(x_pos), 0.0) + 0.016
        if _add_visual_box(
            world,
            f"/World/Environment/centerline_{idx:02d}",
            f"centerline_{idx:02d}",
            (float(x_pos), 0.0, z_pos),
            (0.24, 0.026, 0.014),
            (1.0, 0.82, 0.25),
        ):
            prop_count += 1

    for idx in range(5):
        x_pos = 2.0 + idx * 0.3 + 0.02
        z_pos = (idx + 1) * 0.08 + 0.018
        if _add_visual_box(
            world,
            f"/World/Environment/stair_nosing_{idx}",
            f"stair_nosing_{idx}",
            (x_pos, 0.0, z_pos),
            (0.035, 1.68, 0.018),
            (1.0, 0.72, 0.18),
        ):
            prop_count += 1

    for side_y in (-0.92, 0.92):
        if _add_visual_box(
            world,
            f"/World/Environment/ramp_rail_{'right' if side_y < 0 else 'left'}",
            f"ramp_rail_{'right' if side_y < 0 else 'left'}",
            (6.0, side_y, 0.72),
            (2.3, 0.035, 0.04),
            (0.78, 0.86, 0.90),
        ):
            prop_count += 1
        for post_idx, x_pos in enumerate((5.15, 5.75, 6.35, 6.9)):
            z_pos = get_terrain_height(float(x_pos), side_y) + 0.38
            if _add_visual_box(
                world,
                f"/World/Environment/ramp_post_{'r' if side_y < 0 else 'l'}_{post_idx}",
                f"ramp_post_{'r' if side_y < 0 else 'l'}_{post_idx}",
                (x_pos, side_y, z_pos),
                (0.035, 0.035, 0.55),
                (0.64, 0.72, 0.78),
            ):
                prop_count += 1

    for marker_name, x_pos, y_pos, color in (
        ("start_pad", 0.75, 0.0, (0.22, 0.55, 0.35)),
        ("stair_warning_pad", 2.0, 0.0, (0.92, 0.56, 0.18)),
        ("doorway_focus_pad", 4.25, 0.0, (0.26, 0.46, 0.74)),
        ("finish_pad", 8.15, 0.0, (0.35, 0.45, 0.58)),
    ):
        if _add_visual_box(
            world,
            f"/World/Environment/{marker_name}",
            marker_name,
            (x_pos, y_pos, get_terrain_height(x_pos, y_pos) + 0.014),
            (0.42, 0.42, 0.014),
            color,
        ):
            prop_count += 1

    log_event(
        LOGGER,
        logging.INFO,
        "scene_visual_details_spawned",
        "Spawned route bands, stair nosing, rails, posts, and testing pads",
        prop_count=int(prop_count),
    )


def spawn_obstacles(world: World) -> None:
    """Spawn static obstacles including stairs, landing platform, ramp, walls, and doorway."""
    from omni.isaac.core.objects import FixedCuboid
    
    # 1. Spawn Stairs (5 steps: 2.0 to 3.5m along X, 2.0m wide along Y, step height 0.08m)
    for i in range(5):
        step_x = 2.0 + i * 0.3 + 0.15
        step_z = (i + 1) * 0.08 / 2.0
        try:
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Environment/step_{i}",
                    name=f"step_{i}",
                    position=np.array([step_x, 0.0, step_z]),
                    scale=np.array([0.3, 2.0, (i + 1) * 0.08]),
                    color=np.array([0.5, 0.5, 0.5])
                )
            )
        except Exception as exc:
            log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", f"Failed to spawn step_{i}", error=str(exc))

    # 2. Spawn Landing Platform (3.5 to 5.0m along X, 2.0m wide along Y, height 0.40m)
    try:
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/landing",
                name="landing",
                position=np.array([4.25, 0.0, 0.20]),
                scale=np.array([1.5, 2.0, 0.40]),
                color=np.array([0.4, 0.4, 0.4])
            )
        )
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", "Failed to spawn landing", error=str(exc))

    # 3. Spawn Ramp (5.0 to 7.0m along X, 2.0m wide, slope down from 0.40m to 0.0m)
    try:
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/ramp",
                name="ramp",
                position=np.array([6.0, 0.0, 0.20]),
                orientation=np.array([0.9951, 0.0, -0.0985, 0.0]),
                scale=np.array([2.04, 2.0, 0.05]),
                color=np.array([0.45, 0.45, 0.45])
            )
        )
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", "Failed to spawn ramp", error=str(exc))

    # 4. Spawn Corridor Walls (Left at Y=1.05, Right at Y=-1.05, Height=1.2m, X from 0.0 to 8.0)
    try:
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/wall_left",
                name="wall_left",
                position=np.array([4.0, 1.05, 0.60]),
                scale=np.array([8.0, 0.1, 1.20]),
                color=np.array([0.7, 0.6, 0.5])
            )
        )
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/wall_right",
                name="wall_right",
                position=np.array([4.0, -1.05, 0.60]),
                scale=np.array([8.0, 0.1, 1.20]),
                color=np.array([0.7, 0.6, 0.5])
            )
        )
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", "Failed to spawn corridor walls", error=str(exc))

    # 5. Spawn Doorway at landing (X=4.25, leaving a 0.8m opening in the center)
    try:
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/door_partition_left",
                name="door_partition_left",
                position=np.array([4.25, 0.70, 0.80]),
                scale=np.array([0.1, 0.6, 0.80]),
                color=np.array([0.7, 0.6, 0.5])
            )
        )
        world.scene.add(
            FixedCuboid(
                prim_path="/World/Environment/door_partition_right",
                name="door_partition_right",
                position=np.array([4.25, -0.70, 0.80]),
                scale=np.array([0.1, 0.6, 0.80]),
                color=np.array([0.7, 0.6, 0.5])
            )
        )
        log_event(LOGGER, logging.INFO, "environment_spawned", "Stair-climbing corridor and doorway environment successfully spawned")
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "obstacle_spawn_failed", "Failed to spawn door partitions", error=str(exc))


def apply_realsense_depth_noise(depth_mm: np.ndarray, noise_multiplier: float = 1.0) -> np.ndarray:
    """
    Simulate realistic Intel RealSense D435 depth noise on a depth map (in mm).
    
    Includes:
    - Quadratic depth-dependent Gaussian noise (spatial noise)
    - Silhouette edge dropouts (due to stereo baseline shadows)
    - Random sensor dropouts (zero-fill holes)
    """
    if depth_mm is None or depth_mm.size == 0:
        return depth_mm
        
    noisy_depth = depth_mm.astype(np.float32)
    
    # 1. Quadratic depth-dependent noise
    depth_m = noisy_depth / 1000.0
    alpha = 0.003 * noise_multiplier
    sigma = alpha * (depth_m ** 2) * 1000.0
    
    # Add Gaussian noise
    noise = np.random.normal(0.0, 1.0, size=noisy_depth.shape) * sigma
    noisy_depth += noise
    
    # 2. Silhouette edge dropouts (stereo shadows)
    try:
        import cv2
        grad_x = cv2.Sobel(depth_mm, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_mm, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # High gradients (edges) get zeroed out
        edge_threshold = 1500.0  # mm difference per pixel
        edge_mask = grad_mag > edge_threshold
        
        # Dilate edge mask to simulate physical shadow width
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        noisy_depth[edge_mask] = 0.0
    except ImportError:
        pass
        
    # 3. Random dropouts / sensor holes (more likely at distance)
    dropout_prob = 0.01 + 0.08 * np.square(np.clip(depth_m / 4.0, 0.0, 1.0))
    random_vals = np.random.random(size=noisy_depth.shape)
    dropout_mask = random_vals < dropout_prob
    noisy_depth[dropout_mask] = 0.0
    
    # 4. Range limits (min 0.1m, max 10.0m for D435 color depth)
    noisy_depth[noisy_depth < 100.0] = 0.0
    noisy_depth[noisy_depth > 10000.0] = 0.0
    
    return np.clip(noisy_depth, 0.0, 65535.0).astype(np.uint16)

def attach_robot_o2_tank(stage, trunk_prim_path: str):
    """
    Physically mount a mockup of the Rhythm Healthcare P2-E6 portable oxygen concentrator
    and custom holder on top of the Go2 robot trunk.
    
    Measurements:
    - Holder: Weight 0.3 lbs (0.136 kg).
    - Tank: Dimensions 9.1" (L) x 3.5" (W) x 7.2" (H) -> 0.231m x 0.089m x 0.183m.
            Weight 4.6 lbs (2.086 kg).
    - Distance: Adjusted on back rails, 3.3 inches (0.084m) away from LiDAR center.
                LiDAR center is approximately at X = 0.05. Rails extend backward,
                so we place it at X = -0.15 relative to trunk origin.
    """
    from pxr import UsdGeom, Gf, UsdPhysics, Sdf

    def _set_fixed_joint_bodies(joint, body0_path: str, body1_path: str) -> None:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(body1_path)])
    
    try:
        # Create holder prim
        holder_path = f"{trunk_prim_path}/o2_holder"
        holder_geom = UsdGeom.Cube.Define(stage, holder_path)
        holder_geom.CreateSizeAttr(1.0)
        holder_geom.AddTranslateOp().Set(Gf.Vec3d(-0.15, 0.0, 0.08))
        holder_geom.AddScaleOp().Set(Gf.Vec3d(0.231, 0.089, 0.01))
        holder_geom.CreateDisplayColorAttr([(0.2, 0.2, 0.2)])
        
        holder_mass = UsdPhysics.MassAPI.Apply(holder_geom.GetPrim())
        holder_mass.CreateMassAttr(0.136)
        UsdPhysics.CollisionAPI.Apply(holder_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(holder_geom.GetPrim())
        
        # Weld holder to trunk
        holder_joint_path = f"{trunk_prim_path}/o2_holder_joint"
        holder_joint = UsdPhysics.FixedJoint.Define(stage, holder_joint_path)
        _set_fixed_joint_bodies(holder_joint, trunk_prim_path, holder_path)
        holder_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.15, 0.0, 0.08))
        holder_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        holder_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        holder_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        
        # Create tank prim
        tank_path = f"{trunk_prim_path}/o2_tank"
        tank_geom = UsdGeom.Cube.Define(stage, tank_path)
        tank_geom.CreateSizeAttr(1.0)
        tank_geom.AddTranslateOp().Set(Gf.Vec3d(-0.15, 0.0, 0.17))
        tank_geom.AddScaleOp().Set(Gf.Vec3d(0.231, 0.089, 0.183))
        tank_geom.CreateDisplayColorAttr([(0.9, 0.9, 0.9)])
        
        tank_mass = UsdPhysics.MassAPI.Apply(tank_geom.GetPrim())
        tank_mass.CreateMassAttr(2.086)
        UsdPhysics.CollisionAPI.Apply(tank_geom.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(tank_geom.GetPrim())
        
        # Weld tank to trunk
        tank_joint_path = f"{trunk_prim_path}/o2_tank_joint"
        tank_joint = UsdPhysics.FixedJoint.Define(stage, tank_joint_path)
        _set_fixed_joint_bodies(tank_joint, trunk_prim_path, tank_path)
        tank_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(-0.15, 0.0, 0.17))
        tank_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        tank_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        tank_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        
        log_event(LOGGER, logging.INFO, "o2_tank_attached", "Physically attached P2-E6 oxygen concentrator (4.6 lbs) and printed holder (0.3 lbs) to Go2 trunk rails via FixedJoints")
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "o2_tank_attachment_failed", f"Failed to attach oxygen tank to USD Go2 model: {exc}")


class PatientLocomotionState:
    def __init__(self):
        self.x = 2.5
        self.y = 0.0
        self.direction = 1.0  # +1 for forward through waypoints, -1 for backward
        self.stop_timer = 0.0
        self.turn_timer = 0.0
        self.gait_time = 0.0
        self.has_paused_at_door = False
        self.o2_sat = 98.0  # Oxygen saturation %
        # 2D Waypoints list: (X, Y)
        self.waypoints = [
            (1.0, 0.0),    # Start point
            (2.2, 0.0),    # Before stairs
            (3.2, 0.2),    # On stairs (slightly off-center)
            (4.25, 0.0),   # At doorway
            (5.5, -0.4),   # On ramp (off-center left)
            (6.5, 0.4),    # Off ramp (off-center right)
            (8.0, 0.0)     # End point
        ]
        self.current_wp_idx = 1
        self.wp_direction = 1  # 1 for forward, -1 for backward


_patient_state = None
_last_gt_patient_pose = None
_last_gt_distractor_pose = None


def spawn_person(world, x: float = 2.5, y: float = 0.0):
    global _patient_state
    _patient_state = PatientLocomotionState()
    _patient_state.x = x
    _patient_state.y = y
    
    person = spawn_sim_person(world, x=x, y=y, logger=LOGGER)
    
    # Spawn patient's portable oxygen tank on a trolley (physically enabled kinematic cylinder + handle)
    from omni.isaac.core.objects import DynamicCylinder, DynamicCuboid
    try:
        tank = world.scene.add(
            DynamicCylinder(
                prim_path="/World/Environment/patient_o2_tank",
                name="patient_o2_tank",
                position=np.array([x + 0.45, y, 0.25]),
                radius=0.08,
                height=0.45,
                color=np.array([0.2, 0.7, 0.2])
            )
        )
        rb_tank = UsdPhysics.RigidBodyAPI.Apply(tank.prim)
        rb_tank.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(tank.prim)
        
        handle = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Environment/patient_o2_handle",
                name="patient_o2_handle",
                position=np.array([x + 0.40, y, 0.55]),
                scale=np.array([0.02, 0.15, 0.02]),
                color=np.array([0.6, 0.6, 0.6])
            )
        )
        rb_handle = UsdPhysics.RigidBodyAPI.Apply(handle.prim)
        rb_handle.CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(handle.prim)
        
        log_event(LOGGER, logging.INFO, "patient_o2_spawned", "Spawned patient portable oxygen tank trolley with physical kinematic colliders")
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "patient_o2_spawn_failed", "Failed to spawn patient O2 tank trolley with colliders", error=str(exc))
        
    return person


def spawn_distractor_person(world, x: float, y: float):
    """Spawn a secondary distractor pedestrian crossing the hallway to create ReID occlusion."""
    try:
        from omni.isaac.core.utils.stage import add_reference_to_stage
        import sim_person_actor
        
        assets_root = nucleus_utils.get_assets_root_path()
        distractor_usd = None
        if assets_root:
            # Male character to distinguish from female patient
            distractor_usd = f"{assets_root}/Isaac/People/Characters/male_adult_police_01/male_adult_police_01.usd"
            try:
                if not nucleus_utils.is_file(distractor_usd):
                    distractor_usd = None
            except Exception:
                distractor_usd = None
                
        if not distractor_usd:
            distractor_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/People/Characters/male_adult_police_01_new/male_adult_police_01_new.usd"
            
        prim_path = "/World/Characters/DistractorWalker"
        add_reference_to_stage(usd_path=distractor_usd, prim_path=prim_path)
        
        sim_person_actor._set_xform_pose(prim_path, np.array([x, y, 0.0], dtype=float), 0.0)
        log_event(LOGGER, logging.INFO, "distractor_spawned", f"Spawned distractor pedestrian for ReID occlusion testing: {prim_path}")
        return prim_path
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "distractor_spawn_failed", "Failed to spawn distractor pedestrian", error=str(exc))
        return None


_distractor_t = 0.0

def update_distractor(prim_path: str, dt: float) -> None:
    """Animate distractor crossing back and forth along Y axis at X = 3.0m."""
    global _distractor_t
    _distractor_t += 0.8 * dt
    dy = math.sin(_distractor_t)
    dx = 3.0
    
    # Compute yaw to face movement direction
    vy = math.cos(_distractor_t)
    yaw = math.pi/2.0 if vy >= 0.0 else -math.pi/2.0
    
    qw = math.cos(yaw * 0.5)
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw * 0.5)
    
    global _last_gt_distractor_pose
    try:
        from sim_person_actor import _set_xform_pose
        dz = get_terrain_height(dx, dy)
        _set_xform_pose(prim_path, np.array([dx, dy, dz], dtype=float), yaw)
        _last_gt_distractor_pose = (dx, dy, dz)
    except Exception:
        pass


def update_person_patrol(person, dt: float) -> None:
    global _patient_state, _last_gt_patient_pose
    if _patient_state is None:
        return
        
    state = _patient_state
    
    # 0. Clinical Exertion & Oxygen Saturation updates
    if state.stop_timer > 0.0 or state.turn_timer > 0.0:
        # Recover O2 saturation while resting
        state.o2_sat = min(98.0, state.o2_sat + 0.8 * dt)
    else:
        # Higher exertion climbing stairs/ramps drops O2 faster
        px = state.x
        if 2.0 <= px < 3.5:
            state.o2_sat -= 0.18 * dt
        elif 5.0 <= px < 7.0:
            state.o2_sat -= 0.12 * dt
        else:
            state.o2_sat -= 0.05 * dt
            
    # Check desaturation thresholds
    is_stumbling = False
    if state.o2_sat < 86.0 and state.stop_timer <= 0.0:
        # Critical desaturation -> trigger immediate rest stop to recover
        state.stop_timer = 5.0
        log_event(
            LOGGER,
            logging.WARNING,
            "patient_desaturation_pause",
            f"Patient oxygen saturation critically low ({state.o2_sat:.1f}%); pausing to rest and catch breath",
            o2_saturation=state.o2_sat
        )
        return
    elif state.o2_sat < 90.0:
        # Moderate desaturation -> stumbling/staggering gait
        is_stumbling = True
        
    # Get current target waypoint
    target_wp = state.waypoints[state.current_wp_idx]
    tx, ty = target_wp
    dx = tx - state.x
    dy = ty - state.y
    dist = math.sqrt(dx**2 + dy**2)
    
    # Determine default angle to target
    if dist > 1e-3:
        yaw = math.atan2(dy, dx)
    else:
        yaw = 0.0 if state.wp_direction > 0 else math.pi
        
    # 1. State machine timers
    if state.stop_timer > 0.0:
        state.stop_timer -= dt
        # Labored heavy breathing bobbing while standing
        state.gait_time += dt
        px = state.x
        py_pos = state.y
        bob_amp = 0.035 if is_stumbling else 0.015
        pz = get_terrain_height(px, py_pos) + bob_amp * math.sin(6.0 * state.gait_time)
        yaw = 0.0 if state.wp_direction > 0 else math.pi
    elif state.turn_timer > 0.0:
        state.turn_timer -= dt
        # Interpolate yaw during turn
        state.gait_time += dt
        px = state.x
        py_pos = state.y
        pz = get_terrain_height(px, py_pos)
        progress = 1.0 - (state.turn_timer / 2.0)
        if state.wp_direction > 0:
            yaw = math.pi - progress * math.pi
        else:
            yaw = progress * math.pi
    else:
        # 2. Determine patient speed based on terrain section
        px = state.x
        if 2.0 <= px < 3.5:
            speed = 0.18
            unsteady_amp = 0.15 if is_stumbling else 0.07  # massive sway when stumbling
        elif 5.0 <= px < 7.0:
            speed = 0.22
            unsteady_amp = 0.12 if is_stumbling else 0.06
        else:
            speed = 0.35
            unsteady_amp = 0.10 if is_stumbling else 0.04
            
        if is_stumbling:
            speed *= 0.5  # stagger slowly
            
        # 3. Doorway Rest Stop Logic: pause near the doorway (X = 4.25m) to catch breath
        if abs(px - 4.25) < 0.15 and not state.has_paused_at_door and state.stop_timer <= 0.0:
            state.stop_timer = 3.0
            state.has_paused_at_door = True
            log_event(LOGGER, logging.INFO, "patient_doorway_rest", "Patient paused near doorway to catch breath")
            return
            
        # Reset doorway pause flag when far from doorway
        if abs(px - 4.25) > 1.0:
            state.has_paused_at_door = False
            
        # 4. Locomotion step towards waypoint
        step_dist = speed * dt
        if dist <= step_dist:
            state.x = tx
            state.y = ty
            state.current_wp_idx += state.wp_direction
            if state.current_wp_idx >= len(state.waypoints):
                state.current_wp_idx = len(state.waypoints) - 2
                state.wp_direction = -1
                state.turn_timer = 2.0
                return
            elif state.current_wp_idx < 0:
                state.current_wp_idx = 1
                state.wp_direction = 1
                state.turn_timer = 2.0
                return
        else:
            state.x += (dx / dist) * step_dist
            state.y += (dy / dist) * step_dist
            
        px = state.x
        py_pos = state.y
        
        # 5. Patient unsteadiness and gait sway
        state.gait_time += dt
        sway_y = unsteady_amp * math.sin(4.5 * state.gait_time)
        if is_stumbling:
            # Add staggered staggering
            sway_y += 0.05 * math.sin(1.2 * state.gait_time) + np.random.uniform(-0.02, 0.02)
            
        py_pos += sway_y
        
        # Gait bobbing (up and down)
        bob_amp = 0.04 if is_stumbling else 0.02
        bob_freq = 12.0 if is_stumbling else 9.0
        bob_z = bob_amp * math.cos(bob_freq * state.gait_time)
        pz = get_terrain_height(px, py_pos) + bob_z
        
    # Convert yaw to quaternion
    qw = math.cos(yaw * 0.5)
    qx = 0.0
    qy = 0.0
    qz = math.sin(yaw * 0.5)
    
    person.set_world_pose(
        position=np.array([px, py_pos, pz]),
        orientation=np.array([qw, qx, qy, qz]),
    )
    
    # Store ground truth pose for evaluations
    _last_gt_patient_pose = (px, py_pos, pz)
    
    # 6. Move the physical O2 tank trolley in front of the patient
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        
        for path_suffix, offset_x, offset_z in [("patient_o2_tank", 0.45, 0.225), ("patient_o2_handle", 0.40, 0.55)]:
            prim_path = f"/World/Environment/{path_suffix}"
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                # Face direction of waypoint target
                ptx = px + math.cos(yaw) * offset_x
                pty = py_pos + math.sin(yaw) * offset_x
                ptz = get_terrain_height(ptx, pty) + offset_z
                _set_xform_ops(prim, translate=(ptx, pty, ptz))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Perception Distortion & Noise Helpers
# ---------------------------------------------------------------------------
_distortion_maps = {}

def apply_lens_distortion(image: np.ndarray, is_depth: bool = False) -> np.ndarray:
    """Apply radial and tangential lens distortion (fisheye-like) matching D435."""
    import cv2
    import numpy as np
    
    h, w = image.shape[:2]
    key = (w, h)
    global _distortion_maps
    if key not in _distortion_maps:
        # fx = w / (2 * tan(69.4/2)) = w / 1.385, fy = h / (2 * tan(42.5/2)) = h / 0.787
        fx = w / 1.385
        fy = h / 0.787
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float32)
        
        # Distortion coefficients: [k1, k2, p1, p2, k3]
        dist_coef = np.array([0.15, -0.05, 0.002, 0.002, 0.0], dtype=np.float32)
        
        map1, map2 = cv2.initUndistortRectifyMap(K, dist_coef, None, K, (w, h), cv2.CV_32FC1)
        _distortion_maps[key] = (map1, map2)
        
    map1, map2 = _distortion_maps[key]
    interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    return cv2.remap(image, map1, map2, interpolation)


def apply_rgb_perception_noise(rgb: np.ndarray, vx: float, vy: float, wz: float) -> np.ndarray:
    """Simulate camera motion blur, dynamic exposure fluctuation, and sensor pixel noise."""
    import cv2
    import numpy as np
    import math
    import time
    
    if rgb is None or rgb.size == 0:
        return rgb
        
    h, w = rgb.shape[:2]
    # Remove alpha channel if present (RGBA to BGR)
    if rgb.shape[2] == 4:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    else:
        rgb_bgr = rgb.copy()
        
    noisy_rgb = rgb_bgr.astype(np.float32)
    
    # 1. Motion blur based on robot velocity (linear & angular)
    vel_mag = math.sqrt(vx**2 + vy**2) + abs(wz)
    if vel_mag > 0.08:
        ksize = int(np.clip(vel_mag * 10.0, 3, 7))
        if ksize % 2 == 0:
            ksize += 1
            
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        if abs(wz) > vel_mag * 0.4:
            # Rotational motion causes horizontal blur
            row_idx = ksize // 2
            kernel[row_idx, :] = 1.0
        else:
            # Linear motion causes vertical/diagonal blur
            angle = math.atan2(vy, vx)
            pt1 = (0, int((ksize - 1) * (0.5 - 0.5 * math.sin(angle))))
            pt2 = (ksize - 1, int((ksize - 1) * (0.5 + 0.5 * math.sin(angle))))
            cv2.line(kernel, pt1, pt2, 1.0, 1)
            
        kernel /= np.sum(kernel)
        noisy_rgb = cv2.filter2D(noisy_rgb, -1, kernel)
        
    # 2. Dynamic exposure fluctuation and ambient lighting variation
    t = time.monotonic()
    exposure = 1.0 + 0.04 * math.sin(0.4 * t) + 0.015 * math.cos(3.5 * t)
    flicker = np.random.normal(0, 1.0)
    noisy_rgb = noisy_rgb * exposure + flicker
    
    # 3. Sensor pixel noise (Gaussian color noise)
    noise = np.random.normal(0.0, 3.0, size=noisy_rgb.shape)
    noisy_rgb += noise
    
    return np.clip(noisy_rgb, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame publisher
# ---------------------------------------------------------------------------
class FramePublisher:
    """Encodes RGB + depth frames and sends over UDP to SimCameraCapture."""

    MAX_UDP_PAYLOAD_BYTES = 65000
    PUBLISH_ATTEMPTS = (
        (320, 180, 160, 90, 70),
        (256, 144, 128, 72, 60),
        (224, 126, 112, 63, 50),
    )
    
    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 22)
        self._dest = (host, port)
        self._seq  = 0
        self._warning_times = {}
        self._suppressed_warnings = {}
        first_rgb_w, first_rgb_h, first_depth_w, first_depth_h, first_quality = self.PUBLISH_ATTEMPTS[0]
        log_event(
            LOGGER,
            logging.INFO,
            "frame_publisher_started",
            "Camera frame publisher is ready",
            host=host,
            port=int(port),
            rgb_width=int(first_rgb_w),
            rgb_height=int(first_rgb_h),
            depth_width=int(first_depth_w),
            depth_height=int(first_depth_h),
            jpeg_quality=int(first_quality),
        )

    def _warn_rate_limited(self, event: str, message: str, *, interval_sec: float = 5.0, **fields) -> None:
        now = time.monotonic()
        last = self._warning_times.get(event, 0.0)
        if now - last < interval_sec:
            self._suppressed_warnings[event] = self._suppressed_warnings.get(event, 0) + 1
            return
        suppressed = self._suppressed_warnings.pop(event, 0)
        if suppressed:
            fields["suppressed_count"] = int(suppressed)
        self._warning_times[event] = now
        log_event(LOGGER, logging.WARNING, event, message, **fields)

    def send(self, rgb: np.ndarray, depth: np.ndarray, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0,
             gt_patient: tuple = None, gt_distractor: tuple = None) -> None:
        import cv2, base64, zlib

        seq = self._seq
        payload = None
        payload_meta = {}
        for rgb_w, rgb_h, depth_w, depth_h, jpeg_quality in self.PUBLISH_ATTEMPTS:
            small_rgb = cv2.resize(rgb, (rgb_w, rgb_h), interpolation=cv2.INTER_LINEAR)
            small_depth = cv2.resize(depth, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)

            # 1. Apply radial/tangential lens distortion (D435 simulation)
            small_rgb = apply_lens_distortion(small_rgb, is_depth=False)
            small_depth = apply_lens_distortion(small_depth, is_depth=True)

            # 2. Apply camera sensor noise and motion blur to RGB
            small_rgb_bgr = apply_rgb_perception_noise(small_rgb, vx, vy, wz)

            ok, buf = cv2.imencode('.jpg', small_rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not ok:
                continue

            rgb_b64 = base64.b64encode(buf.tobytes()).decode('ascii')
            depth_b64 = base64.b64encode(
                zlib.compress(small_depth.astype(np.uint16).tobytes(), level=9)
            ).decode('ascii')

            meta = {
                "seq": seq,
                "ts": time.time(),
                "w": rgb_w,
                "h": rgb_h,
                "rgb_w": rgb_w,
                "rgb_h": rgb_h,
                "depth_w": depth_w,
                "depth_h": depth_h,
                "enc": "jpg+zlib",
                "rgb": rgb_b64,
                "depth": depth_b64,
                "gt_patient": gt_patient,
                "gt_distractor": gt_distractor,
            }
            candidate_payload = json.dumps(meta).encode("utf-8")
            payload_meta = {
                "rgb_width": int(rgb_w),
                "rgb_height": int(rgb_h),
                "depth_width": int(depth_w),
                "depth_height": int(depth_h),
                "jpeg_quality": int(jpeg_quality),
                "candidate_payload_bytes": int(len(candidate_payload)),
            }
            if len(candidate_payload) <= self.MAX_UDP_PAYLOAD_BYTES:
                payload = candidate_payload
                break

        self._seq += 1
        if payload is None:
            self._warn_rate_limited(
                "frame_payload_too_large",
                "Camera frame payload is too large for UDP packet",
                seq=int(seq),
                max_payload_bytes=int(self.MAX_UDP_PAYLOAD_BYTES),
                **payload_meta,
            )
            return
        try:
            self._sock.sendto(payload, self._dest)
            log_event(
                LOGGER,
                logging.DEBUG,
                "frame_sent",
                "Camera frame sent to SimCameraCapture",
                seq=int(seq),
                payload_bytes=int(len(payload)),
                **payload_meta,
            )
        except Exception as exc:
            self._warn_rate_limited(
                "frame_send_error",
                "Camera frame send failed",
                seq=int(seq),
                error=str(exc),
            )
            
    def close(self) -> None:
        self._sock.close()


def apply_velocity_to_go2(go2: Articulation, vx: float, vy: float, wz: float, dt: float) -> None:
    apply_go2_velocity(go2, vx, vy, wz, dt, state=_go2_locomotion_state, base_link_name=BASE_LINK_NAME, logger=LOGGER)


def create_and_bind_friction_material(stage, prim_paths: list, material_path: str = "/World/PhysicsMaterials/HighFrictionMaterial"):
    from pxr import UsdPhysics, Sdf
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        material_prim = stage.DefinePrim(material_path, "Material")
        phys_mat = UsdPhysics.MaterialAPI.Apply(material_prim)
        phys_mat.CreateDynamicFrictionAttr().Set(1.0)
        phys_mat.CreateStaticFrictionAttr().Set(1.2)
        phys_mat.CreateRestitutionAttr().Set(0.0)
        log_event(LOGGER, logging.INFO, "physics_material_created", f"Created physics material {material_path} with dynamic=1.0, static=1.2")
        
    for p_path in prim_paths:
        prim = stage.GetPrimAtPath(p_path)
        if prim.IsValid():
            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
            try:
                collision_api.GetPhysicsMaterialRel().SetTargets([Sdf.Path(material_path)])
            except Exception:
                try:
                    prim.CreateRelationship("physics:material").SetTargets([Sdf.Path(material_path)])
                except Exception:
                    pass
            log_event(LOGGER, logging.INFO, "physics_material_bound", f"Bound {material_path} to {p_path}")


def _lerp_vec3(current: Gf.Vec3d, target: Gf.Vec3d, alpha: float) -> Gf.Vec3d:
    return Gf.Vec3d(
        float(current[0]) + (float(target[0]) - float(current[0])) * alpha,
        float(current[1]) + (float(target[1]) - float(current[1])) * alpha,
        float(current[2]) + (float(target[2]) - float(current[2])) * alpha,
    )


class ViewFollowCameraRig:
    """Viewport-only chase camera. It does not affect the robot's perception camera."""

    def __init__(self, stage, *, distance_m: float, height_m: float, side_offset_m: float) -> None:
        self.stage = stage
        self.distance_m = float(distance_m)
        self.height_m = float(height_m)
        self.side_offset_m = float(side_offset_m)
        self.path = VIEW_CAMERA_PRIM
        self._eye = None
        self._target = None
        self._warned = False

        if not stage.GetPrimAtPath("/World/View").IsValid():
            stage.DefinePrim("/World/View", "Xform")

        camera = UsdGeom.Camera.Define(stage, self.path)
        camera.CreateFocalLengthAttr().Set(30.0)
        camera.CreateHorizontalApertureAttr().Set(24.0)
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.05, 250.0))
        camera.CreateFocusDistanceAttr().Set(max(1.0, self.distance_m))

        self._xform = UsdGeom.Xformable(camera.GetPrim())
        self._xform.ClearXformOpOrder()
        self._transform_op = self._xform.AddTransformOp()
        self._set_active_viewport_camera()

        log_event(
            LOGGER,
            logging.INFO,
            "view_follow_camera_created",
            "Dynamic viewport follow camera is tracking the Go2 robot",
            camera_path=self.path,
            distance_m=self.distance_m,
            height_m=self.height_m,
            side_offset_m=self.side_offset_m,
        )

    def _set_active_viewport_camera(self) -> None:
        try:
            import omni.kit.viewport.utility as viewport_utility

            viewport = viewport_utility.get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.path
                log_event(
                    LOGGER,
                    logging.INFO,
                    "view_follow_camera_active",
                    "Isaac viewport switched to the dynamic Go2 follow camera",
                    camera_path=self.path,
                )
        except Exception as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "view_follow_camera_viewport_failed",
                "Could not switch the active viewport to the follow camera",
                camera_path=self.path,
                error=str(exc),
            )

    def _robot_pose(self, go2: Articulation):
        candidate_paths = (
            f"{GO2_USD_PATH}/{BASE_LINK_NAME}",
            f"{GO2_USD_PATH}/base",
            GO2_USD_PATH,
        )
        for path in candidate_paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
                return matrix, yaw, path

        root_prim = getattr(go2, "prim", None)
        if root_prim is None:
            return None, 0.0, ""
        matrix = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
        return matrix, yaw, str(root_prim.GetPath())

    def update(self, go2: Articulation, dt: float) -> None:
        try:
            matrix, yaw, _ = self._robot_pose(go2)
            if matrix is None:
                return

            rx = float(matrix[3][0])
            ry = float(matrix[3][1])
            rz = float(matrix[3][2])
            terrain_z = get_terrain_height(rx, ry)
            target = Gf.Vec3d(rx, ry, max(rz, terrain_z + 0.42))

            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            behind_x = -cos_y * self.distance_m
            behind_y = -sin_y * self.distance_m
            side_x = -sin_y * self.side_offset_m
            side_y = cos_y * self.side_offset_m
            desired_eye = Gf.Vec3d(
                rx + behind_x + side_x,
                ry + behind_y + side_y,
                terrain_z + self.height_m,
            )

            alpha = 1.0 - math.exp(-max(0.001, dt) * 4.5)
            self._eye = desired_eye if self._eye is None else _lerp_vec3(self._eye, desired_eye, alpha)
            self._target = target if self._target is None else _lerp_vec3(self._target, target, alpha)

            view_matrix = Gf.Matrix4d(1.0)
            view_matrix.SetLookAt(self._eye, self._target, Gf.Vec3d(0.0, 0.0, 1.0))
            self._transform_op.Set(view_matrix.GetInverse())
        except Exception as exc:
            if not self._warned:
                self._warned = True
                log_event(
                    LOGGER,
                    logging.WARNING,
                    "view_follow_camera_update_failed",
                    "Dynamic viewport follow camera update failed",
                    error=str(exc),
                )


def ensure_person_animation_loaded(world: World, person, *, render: bool, attempts: int = 4) -> bool:
    if not hasattr(person, "ensure_animation_ready"):
        return False

    for attempt_idx in range(max(1, attempts)):
        try:
            person.ensure_animation_ready(world, force_retry=attempt_idx > 0)
            if getattr(person, "animation_ready", False):
                log_event(
                    LOGGER,
                    logging.INFO,
                    "person_animation_confirmed",
                    "Person animation is loaded and ready",
                    attempts=int(getattr(person, "animation_attempt_count", attempt_idx + 1)),
                )
                return True
        except TypeError:
            person.ensure_animation_ready(world)
            return bool(getattr(person, "animation_ready", False))
        except Exception as exc:
            log_event(
                LOGGER,
                logging.WARNING,
                "person_animation_retry_failed",
                "Person animation readiness attempt failed",
                attempt=int(attempt_idx + 1),
                error=str(exc),
            )

        if attempt_idx < attempts - 1:
            for _ in range(3):
                try:
                    world.step(render=render)
                except Exception:
                    break
            try:
                import omni.kit.app

                omni.kit.app.get_app().update()
            except Exception:
                pass

    log_event(
        LOGGER,
        logging.WARNING,
        "person_animation_not_ready",
        "Person animation did not become ready; keeping kinematic walking fallback",
        attempts=int(getattr(person, "animation_attempt_count", attempts)),
    )
    return False

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
def main() -> None:
    global _running

    log_event(LOGGER, logging.INFO, "world_build_start", "Building Isaac world")
    world = build_world(args.physics_hz)

    log_event(LOGGER, logging.INFO, "obstacles_spawn_start", "Spawning static obstacles")
    spawn_obstacles(world)
    spawn_scene_visual_details(world)

    stage = None
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        setup_scene_lighting(stage)
        step_paths = [f"/World/Environment/step_{i}" for i in range(5)]
        step_paths.append("/World/Environment/landing")
        step_paths.append("/World/Environment/ramp")
        step_paths.append("/World/defaultGroundPlane")
        create_and_bind_friction_material(stage, step_paths)
    except Exception as exc:
        log_event(LOGGER, logging.WARNING, "physics_material_failed", "Failed to create/bind friction material", error=str(exc))

    log_event(LOGGER, logging.INFO, "go2_load_start", "Loading Go2 robot")
    go2 = load_go2(world)

    view_camera = None
    if stage is not None and not args.headless and not args.no_view_follow_camera:
        view_camera = ViewFollowCameraRig(
            stage,
            distance_m=args.view_camera_distance,
            height_m=args.view_camera_height,
            side_offset_m=args.view_camera_side_offset,
        )

    log_event(LOGGER, logging.INFO, "camera_add_start", "Adding front camera")
    camera = add_camera()

    log_event(LOGGER, logging.INFO, "person_spawn_start", "Spawning person target")
    person = spawn_person(world, x=args.person_x, y=args.person_y)

    distractor_prim = None
    if args.person_move:
        # Spawn distractor walking Y-axis crossing at X = 3.0 to test ReID occlusion
        distractor_prim = spawn_distractor_person(world, x=3.0, y=-1.0)

    world.reset()
    animation_ready = ensure_person_animation_loaded(world, person, render=not args.headless)

    log_event(
        LOGGER,
        logging.INFO,
        "world_ready",
        "Isaac world is ready",
        physics_hz=int(args.physics_hz),
        render_every=int(args.render_every),
        person_move=bool(args.person_move),
        person_animation_ready=bool(animation_ready),
        view_follow_camera=bool(view_camera is not None),
        hold_motion_until_command=bool(args.hold_motion_until_command),
    )

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
    motion_wait_logged = False
    motion_start_logged = False

    try:
        while simulation_app.is_running():
            if stage is not None:
                update_scene_lighting(stage, time.monotonic())

            world.step(render=not args.headless)
            step_count += 1

            # Read latest velocity command (zero out if stale)
            with _cmd_lock:
                age = time.monotonic() - _cmd_vel["ts"]
                cmd_count = int(_cmd_vel.get("count", 0))
                active_count = int(_cmd_vel.get("active_count", 0))
                if age > CMD_TIMEOUT_SEC:
                    vx, vy, wz = 0.0, 0.0, 0.0
                    command_fresh = False
                else:
                    vx = _cmd_vel["vx"]
                    vy = _cmd_vel["vy"]
                    wz = _cmd_vel["wz"]
                    command_fresh = True
            controller_ready = active_count > 0
            nonzero_command_fresh = (
                command_fresh
                and ((abs(vx) > 0.01) or (abs(vy) > 0.01) or (abs(wz) > 0.01))
            )
            scene_motion_allowed = (not args.hold_motion_until_command) or controller_ready
            if args.hold_motion_until_command and not controller_ready and not motion_wait_logged:
                motion_wait_logged = True
                log_event(
                    LOGGER,
                    logging.INFO,
                    "scene_motion_waiting_for_controller",
                    "Holding autonomous scene motion until Docker/controller command stream starts",
                    cmd_port=int(args.cmd_port),
                )
            elif scene_motion_allowed and motion_wait_logged and not motion_start_logged:
                motion_start_logged = True
                log_event(
                    LOGGER,
                    logging.INFO,
                    "scene_motion_started",
                    "Autonomous scene motion released after a nonzero controller command was observed",
                    command_count=cmd_count,
                    active_command_count=active_count,
                )

            if controller_ready and nonzero_command_fresh:
                apply_velocity_to_go2(go2, vx, vy, wz, dt)
            else:
                hold_go2_stable(go2, _go2_locomotion_state, dt, logger=LOGGER)
            if view_camera is not None:
                view_camera.update(go2, dt)

            # 2. Camera vibration / bobbing based on gait state
            try:
                state = _go2_locomotion_state
                t_gait = state.gait_time
                if t_gait > 0.0:
                    omega = 2.0 * math.pi / state.gait_period
                    # Z-bobbing (1.5cm amplitude)
                    dz = 0.015 * math.sin(2.0 * omega * t_gait)
                    # Lateral sway (1.0cm amplitude)
                    dy = 0.010 * math.cos(omega * t_gait)
                    # Longitudinal shake (0.5cm)
                    dx = 0.005 * math.sin(omega * t_gait)
                    
                    # Rotational shake (pitch, yaw, roll in radians)
                    pitch_shake = 0.03 * math.sin(2.0 * omega * t_gait)
                    yaw_shake = 0.02 * math.cos(omega * t_gait)
                    roll_shake = 0.01 * math.sin(omega * t_gait)
                else:
                    t_idle = time.monotonic()
                    dx = 0.001 * math.sin(10.0 * t_idle)
                    dy = 0.001 * math.cos(10.0 * t_idle)
                    dz = 0.001 * math.sin(15.0 * t_idle)
                    pitch_shake = 0.002 * math.sin(8.0 * t_idle)
                    yaw_shake = 0.002 * math.cos(8.0 * t_idle)
                    roll_shake = 0.0
                    
                # Add random high-frequency jitter
                dx += np.random.uniform(-0.002, 0.002)
                dy += np.random.uniform(-0.002, 0.002)
                dz += np.random.uniform(-0.002, 0.002)
                
                # Convert Euler angles to quaternion (qw, qx, qy, qz)
                cr = math.cos(roll_shake * 0.5)
                sr = math.sin(roll_shake * 0.5)
                cp = math.cos(pitch_shake * 0.5)
                sp = math.sin(pitch_shake * 0.5)
                cy = math.cos(yaw_shake * 0.5)
                sy = math.sin(yaw_shake * 0.5)
                
                qw = cr * cp * cy + sr * sp * sy
                qx = sr * cp * cy - cr * sp * sy
                qy = cr * sp * cy + sr * cp * sy
                qz = cr * cp * sy - sr * sp * cy
                
                camera.set_local_pose(
                    translation=np.array([0.25 + dx, dy, 0.1 + dz]),
                    orientation=np.array([qw, qx, qy, qz])
                )
            except Exception:
                pass

            if args.person_move and scene_motion_allowed:
                update_person_patrol(person, dt)
                if distractor_prim:
                    update_distractor(distractor_prim, dt)

            # Publish camera frame at reduced rate
            if step_count % args.render_every == 0:
                try:
                    rgb_data   = camera.get_rgb()
                    depth_data = camera.get_depth()
                    if rgb_data is not None and depth_data is not None:
                        # depth_data is in metres; convert to uint16 millimetres
                        depth_mm = (depth_data * 1000.0).clip(0, 65535).astype(np.uint16)
                        # Simulate realistic RealSense D435 sensor depth noise
                        depth_mm = apply_realsense_depth_noise(depth_mm)
                        
                        # Get ground truth coordinates
                        gt_patient = _last_gt_patient_pose
                        gt_distractor = _last_gt_distractor_pose if args.person_move else None
                        
                        publisher.send(rgb_data, depth_mm, vx, vy, wz, gt_patient, gt_distractor)
                except Exception as exc:
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "camera_capture_error",
                        "Camera capture failed during render step",
                        error=str(exc),
                    )

    except KeyboardInterrupt:
        log_event(LOGGER, logging.INFO, "keyboard_interrupt", "KeyboardInterrupt - shutting down")
    finally:
        _running = False
        publisher.close()
        simulation_app.close()
        log_event(LOGGER, logging.INFO, "simulation_shutdown", "Simulation shutdown completed")


if __name__ == "__main__":
    main()

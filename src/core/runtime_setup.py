"""Controller startup/initialization helpers extracted from core.main.

Pure, self-contained top-level helpers that run BEFORE the 50 Hz control loop:
CLI log-component parsing, camera/robot-controller construction (real vs. sim),
and the btop-style startup banner. Moved here verbatim from core/main.py as a
structural split; core.main re-exports every name so `core.main.X` still resolves.
"""

from typing import Set


def _parse_enabled_log_components(raw_value: str) -> Set[str]:
    components = {part.strip() for part in raw_value.split(',') if part.strip()}
    if not components or "none" in components:
        return set()
    if "all" in components:
        return {"all"}
    return components


def _build_camera(args):
    """Return the correct camera capture object based on --sim flag."""
    if args.sim:
        from sim_camera_capture import SimCameraCapture
        print("[main] Sim mode: using SimCameraCapture")
        sim_frame_timeout_exit_sec = getattr(args, "sim_frame_timeout_exit_sec", 30.0)
        timeout_sec = max(10.0, sim_frame_timeout_exit_sec) if sim_frame_timeout_exit_sec > 0.0 else 30.0
        return SimCameraCapture(
            width=1280,
            height=720,
            frame_port=args.frame_port,
            rotate=args.rotate,
            verbose=args.debug,
            timeout_sec=timeout_sec,
            latency_ms=getattr(args, "sim_latency_ms", 0.0),
            latency_jitter_ms=getattr(args, "sim_latency_jitter_ms", 0.0),
        )
    from camera_capture import CameraCapture
    return CameraCapture(
        mode=args.camera_mode,
        width=1280,
        height=720,
        fps=30,
        rotate=args.rotate,
        verbose=args.debug,
    )


def _build_robot_controller(args):
    """Return the correct robot controller based on --sim flag."""
    if args.sim:
        from sim_robot_controller import SimRobotController
        print("[main] Sim mode: using SimRobotController")
        ctrl = SimRobotController(
            cmd_host=args.cmd_host,
            cmd_port=args.cmd_port,
        )
        ctrl.initialize()
        return ctrl

    if getattr(args, "ros2", False):
        # Native ROS 2 path for the real Go2 EDU: a pure publisher that hands the
        # follow command to the low-level control node (which runs the policy and
        # writes /lowcmd). No joints, no unitree_sdk2 in this process. Imported
        # lazily so host/sim runs never need rclpy.
        from real.control.real_robot_controller import RealRobotController
        print("[main] ROS2 mode: using RealRobotController (native rclpy transport)")
        ctrl = RealRobotController(args)
        if not ctrl.initialize():
            return None
        return ctrl

    from robot_controller import RobotController
    low_level = getattr(args, "low_level_locomotion", False)
    base_model = getattr(args, "parkour_base_jit", "src/sim/models/locomotion/parkour/base_jit.pt")
    vision_model = getattr(args, "parkour_vision_weight", "src/sim/models/locomotion/parkour/vision_weight.pt")
    ctrl = RobotController(
        network_interface=args.network_interface,
        low_level_locomotion=low_level,
        base_model_path=base_model,
        vision_model_path=vision_model
    )
    if not ctrl.initialize():
        return None
    return ctrl


# The former _apply_sim_stair_gap_control() ground-truth stair-gap assist was
# removed: it drove the forward command from gt_patient (a sim-only cheat the real
# robot lacks) and was already dead code (never called). Stair approach is
# sensor-only via _apply_stair_command_policy; gt_patient/gt_distractor are kept
# elsewhere only as logged evaluation references, never as control inputs.


def _print_startup_banner(args) -> None:
    """Print a btop-style startup banner + config panel (DESIGN.md aesthetic).

    Shared by BOTH the sim shim and the real ROS2 entrypoint (both route through
    here), so the same controller summary shows on either target. Color is only
    emitted to an interactive TTY; when stdout is piped (Docker run logs, the
    launcher dashboard) term_ui degrades to plain ASCII, so nothing pollutes the
    captured logs.
    """
    try:
        from core.telemetry import term_ui as tu
    except Exception:
        return

    theme = tu.Theme.detect()
    if getattr(args, "sim", False):
        mode = "sim · Isaac"
    elif getattr(args, "ros2", False):
        mode = "real · ROS2 Go2 EDU"
    else:
        mode = "real · unitree sdk2"

    follow = "off"
    if getattr(args, "follow", False):
        follow = (f"{getattr(args, 'follow_backend', 'pid')}  "
                  f"target {getattr(args, 'target_distance', 0.0):.2f} m  "
                  f"kp {getattr(args, 'kp', 0.0):g}")

    if getattr(args, "sim", False):
        io = (f"frame :{getattr(args, 'frame_port', '?')}  "
              f"cmd {getattr(args, 'cmd_host', '?')}:{getattr(args, 'cmd_port', '?')}")
    elif getattr(args, "ros2", False):
        io = "rclpy /lowstate -> /lowcmd (native ROS2)"
    else:
        io = f"iface {getattr(args, 'network_interface', '?')}"

    width = 62
    lines = tu.panel("go2 controller", [
        tu.kv("mode", mode, theme, 9, "secondary"),
        tu.kv("follow", follow, theme, 9, "success" if follow != "off" else "muted"),
    ], width, accent="cpu", theme=theme)
    lines += tu.panel("perception / io", [
        tu.kv("pose", getattr(args, "trt_engine", "?"), theme, 9, "primary", bold_value=False),
        tu.kv("stairs", getattr(args, "stairs_model", "?"), theme, 9, "primary", bold_value=False),
        tu.kv("io", io, theme, 9, "blue"),
    ], width, accent="net", theme=theme,
        footer="headless" if getattr(args, "headless", False) else None)
    print("\n".join(lines), flush=True)

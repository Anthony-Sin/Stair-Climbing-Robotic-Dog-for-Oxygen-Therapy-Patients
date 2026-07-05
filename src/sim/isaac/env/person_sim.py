"""isaac_env.py extraction (Phase 2 split): person_sim. Verbatim bodies; only env_state requalification added."""
import logging
import math
import numpy as np
try:
    import omni.isaac.core.utils.nucleus as nucleus_utils
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import DynamicCapsule, GroundPlane
    from omni.isaac.core.utils.prims import is_prim_path_valid
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.prims import GeometryPrim
except ModuleNotFoundError:
    import isaacsim.storage.native as nucleus_utils
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation as Articulation, SingleGeometryPrim as GeometryPrim
    from isaacsim.core.api.objects import DynamicCapsule, GroundPlane
    from isaacsim.core.utils.prims import is_prim_path_valid
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
from pxr import Usd, UsdGeom
from sim_logging_utils import log_event
from world.sim_go2_locomotion import get_active_stairs

from env import env_state

from .terrain_queries import _get_person_pose_z, get_terrain_height

# Standing pelvis (root-body) height above the floor for the dynamic patient. The
# open-loop gait can't balance a free articulation, so the root Z is held here while
# the legs/feet do real contact physics. Tunable: too high -> feet dangle (double
# float); too low -> feet penetrate. ~0.92 m suits the 1.70 m-scaled CMU humanoid.
# Standing pelvis height. Must be LOWER than the straight-leg reach to the floor
# (~0.78 m for the 1.70 m-scaled CMU legs) so the gait IK has slack to BEND the knee;
# at 0.92 m the leg reached the floor dead-straight and the IK clamped it (knee ~1deg).
PELVIS_STAND_HEIGHT_M = 0.80
_patient_skel_cache = None

class PatientLocomotionState:
    def __init__(self, start_x: float = 0.8, start_y: float = 0.0):
        self.x = float(start_x)
        self.y = float(start_y)
        self.direction = 1.0  # +1 for forward through waypoints, -1 for backward
        self.stop_timer = 0.0
        self.turn_timer = 0.0
        self.gait_time = 0.0
        # Gait clock in full L/R cycles (one cycle == two footfalls == 0.6 m of
        # travel at speed = cadence * 0.3 m). Drives the visual bob; advanced only
        # while the patient is moving.
        self.gait_phase = 0.0
        # Throttled-trajectory-log bookkeeping (verify the climb from the JSONL).
        self.dbg_accum = 0.0
        self.elapsed_time = 0.0
        self.stair_phase_started = False
        self.stair_phase_logged = False
        self.o2_sat = 98.0  # Oxygen saturation %
        self.ground_follow_delay_sec = 20.0
        self.at_destination = False
        _stairs = get_active_stairs()
        self.heading_yaw = 0.0
        self.last_pz = _get_person_pose_z(self.x, self.y, smooth=True)
        # VISUAL root height: tracks the DISCRETE tread top (get_terrain_height),
        # smoothed, so the rendered body sits ON each step and its feet can reach the
        # tread (the smooth nosing-line ramp rides above the treads, which left the
        # feet floating). Ground-truth Z stays on the smooth ramp (no GT regression).
        self.visual_pz = get_terrain_height(self.x, self.y)
        # 2D waypoints: default scene stays straight; final scene prepends a
        # turning hospital corridor route before rejoining the stair centreline.
        if env_state.args.final_scene:
            from final_scene import build_patient_route
            self.waypoints = build_patient_route(
                env_state._FINAL_SCENE_SPEC,
                _stairs,
                start_xy=(self.x, self.y),
            )
        else:
            self.waypoints = [(self.x, self.y)]
            turns = getattr(env_state.args, "person_approach_turns", 0)
            amp   = getattr(env_state.args, "person_approach_amplitude", 1.2)
            # Flat-approach alignment point: person must be on-axis before the stairs.
            _ALIGN_X = 1.4
            if turns > 0 and _ALIGN_X > self.x + 0.5:
                # Distribute N turns evenly across [start_x, _ALIGN_X], alternating ±amp.
                span = _ALIGN_X - self.x
                for i in range(turns):
                    wx = self.x + span * (i + 1) / (turns + 1)
                    wy = amp if (i % 2 == 0) else -amp
                    self.waypoints.append((wx, wy))
            for waypoint_x in (1.4, 2.0):
                if waypoint_x > self.x + 0.05:
                    self.waypoints.append((waypoint_x, 0.0))
            if self.waypoints[-1][0] < 2.0:
                self.waypoints.append((2.0, 0.0))
        self.stair_base_wp_idx = len(self.waypoints) - 1
        # One waypoint per tread (tread centre) plus a top-landing target,
        # generated from the active StairSpec so the patient path matches the
        # spawned stairs for every preset (see --stair-preset).
        self.waypoints.extend(
            (_stairs.start_x_m + (i + 0.5) * _stairs.step_depth_m, 0.0)
            for i in range(_stairs.step_count)
        )
        # Top-landing destination. Placed 2.5 m onto the (3.5 m-deep) landing so the patient
        # walks well out onto the landing and the dog must FOLLOW it across the flat -- a real
        # test that the perception-follow standoff + landing creep-brake hold ~1.0 m without
        # the dog creeping into the patient. The patient just walks its OWN route and stops
        # here; holding the standoff on the flat landing is the DOG's job (perception-follow +
        # the landing creep-brake in isaac_env). Still within the 3.5 m landing (back edge +3.5).
        self.waypoints.append((_stairs.end_x_m + 2.5, 0.0))  # top landing (pull-off target)
        self.current_wp_idx = min(1, len(self.waypoints) - 1)
        self.wp_direction = 1

def _read_final_scene_robot_pose(stage):
    candidate_paths = (
        f"{env_state.GO2_USD_PATH}/{env_state.BASE_LINK_NAME}",
        f"{env_state.GO2_USD_PATH}/base",
        env_state.GO2_USD_PATH,
    )
    for path in candidate_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            yaw = math.atan2(float(matrix[0][1]), float(matrix[0][0]))
            return (
                (float(matrix[3][0]), float(matrix[3][1]), float(matrix[3][2])),
                float(yaw),
                path,
            )
    raise RuntimeError("final_scene: Go2 base pose prim was not found for recording cameras")

def spawn_distractor_person(world, x: float, y: float):
    """Spawn a secondary distractor pedestrian crossing the hallway for occlusion testing."""
    try:
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
        except ModuleNotFoundError:
            from isaacsim.core.utils.stage import add_reference_to_stage
        import world.sim_person_actor as sim_person_actor
        
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
            distractor_usd = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/People/Characters/male_adult_police_01_new/male_adult_police_01_new.usd"
            
        prim_path = "/World/Characters/DistractorWalker"
        add_reference_to_stage(usd_path=distractor_usd, prim_path=prim_path)
        
        sim_person_actor._set_xform_pose(prim_path, np.array([x, y, 0.0], dtype=float), 0.0)
        log_event(env_state.LOGGER, logging.INFO, "distractor_spawned", f"Spawned distractor pedestrian for occlusion testing: {prim_path}")
        return prim_path
    except Exception as exc:
        log_event(env_state.LOGGER, logging.WARNING, "distractor_spawn_failed", "Failed to spawn distractor pedestrian", error=str(exc))
        return None

def _patient_stand_height(person) -> float:
    """Root-above-floor height for the kinematic patient: the per-character snap-to-ground
    offset measured at spawn (SimPersonTarget.root_to_sole_m) if available, else the
    default pelvis-root constant. Seats the feet on the floor for any rig."""
    rts = getattr(person, "root_to_sole_m", None) if person is not None else None
    return float(rts) if rts is not None else PELVIS_STAND_HEIGHT_M

def _patient_gait_body_z(person) -> float:
    """Hip-above-floor height fed to the gait's foot-planting IK (which reaches each foot
    down from the hip). The per-character measured hip height if available, else the
    default constant. Distinct from _patient_stand_height (the VISUAL root) because a
    rig's root may sit at the feet, not the hip."""
    hh = getattr(person, "hip_height_m", None) if person is not None else None
    return float(hh) if hh is not None else PELVIS_STAND_HEIGHT_M

def _patient_body_log(person, ground_under: float) -> dict:
    """World positions of the patient's body parts, for the run log -- so we can SEE
    whether the feet sit on the floor (feet_z ~= ground), the hip height, head, etc.,
    without a screenshot. Best-effort and exception-safe (returns {} on any failure).

    feet_z / head_z come from the rendered mesh bounding box (always available); the
    per-joint positions come from the live UsdSkel pose when the query is available.
    """
    global _patient_skel_cache
    out: dict = {"ground": round(float(ground_under), 3)}
    try:
        import omni.usd
        from pxr import UsdGeom, Usd
        stage = omni.usd.get_context().get_stage()
        vp = getattr(person, "visual_prim_path", "") or ""
        prim = stage.GetPrimAtPath(vp)
        if prim and prim.IsValid():
            bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            rng = bc.ComputeWorldBound(prim).ComputeAlignedRange()
            if not rng.IsEmpty():
                _feet = float(rng.GetMin()[2])
                out["feet_z"] = round(_feet, 3)
                out["head_z"] = round(float(rng.GetMax()[2]), 3)
                out["float_m"] = round(_feet - float(ground_under), 3)
            # Robot feet as an independent GROUND-TRUTH reference: the Go2 physically
            # stands on the floor, so its lowest point IS the real ground contact. If the
            # patient's feet_z sits above robot_feet_z (on flat), the patient is floating.
            robot_prim = stage.GetPrimAtPath(env_state.GO2_USD_PATH)
            if robot_prim and robot_prim.IsValid():
                rrng = bc.ComputeWorldBound(robot_prim).ComputeAlignedRange()
                if not rrng.IsEmpty():
                    out["robot_feet_z"] = round(float(rrng.GetMin()[2]), 3)
                    if "feet_z" in out:
                        out["feet_vs_robot_m"] = round(out["feet_z"] - float(rrng.GetMin()[2]), 3)
    except Exception:
        pass
    # Per-joint WORLD transforms via UsdSkel, so we can report each FOOT's true world
    # height AND its clearance over the tread directly beneath it -- the real "is the
    # foot on the step / floating?" metric. (The bbox feet_z above is only the body's
    # single lowest point, not per foot.) The SkelCache MUST be Populate()-d with the
    # SkelRoot before GetSkelQuery returns a valid query; the prior code skipped Populate,
    # so the query came back empty and NO per-foot data was ever logged. On any failure
    # we stash the reason in skel_status so a run surfaces WHY instead of dropping silently.
    skel_status = "ok"
    try:
        import omni.usd
        from pxr import UsdSkel, UsdGeom, Usd
        stage = omni.usd.get_context().get_stage()
        skel_root_path = getattr(person, "_skel_root_path", "") or ""
        root_prim = stage.GetPrimAtPath(skel_root_path) if skel_root_path else None
        if not (root_prim and root_prim.IsValid()):
            skel_status = "no_skel_root"
        else:
            if _patient_skel_cache is None:
                _patient_skel_cache = UsdSkel.Cache()
            try:
                _patient_skel_cache.Populate(UsdSkel.Root(root_prim), Usd.PrimDefaultPredicate)
            except Exception:
                try:
                    _patient_skel_cache.Populate(UsdSkel.Root(root_prim))  # older USD signature
                except Exception:
                    pass
            skel = None
            for p in Usd.PrimRange(root_prim):
                if p.IsA(UsdSkel.Skeleton):
                    skel = UsdSkel.Skeleton(p)
                    break
            if skel is None:
                skel_status = "no_skeleton"
            else:
                q = _patient_skel_cache.GetSkelQuery(skel)
                xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
                xforms = None
                if not q:
                    skel_status = "no_skel_query"
                else:
                    # ComputeJointWorldTransforms signature varies by USD version (the
                    # 2-arg (xfCache, atRest:bool) rejects a TimeCode -> ArgumentError);
                    # try the bool form first, then the plain 1-arg. Clear the status on
                    # success so a first-try failure that the fallback recovers from is
                    # NOT mislabeled (the per-foot data was valid but read 'compute_err').
                    _last_err = None
                    for _args in ((xfc,), (xfc, False)):
                        try:
                            xforms = q.ComputeJointWorldTransforms(*_args)
                            if xforms:
                                break
                        except Exception as _e:
                            _last_err = type(_e).__name__
                    if not xforms and _last_err:
                        skel_status = f"compute_err:{_last_err}"
                joints = skel.GetJointsAttr().Get()
                if xforms and joints:
                    ci = {}
                    for j, xf in zip(joints, xforms):
                        ci[str(j).rsplit("/", 1)[-1].lower()] = xf
                    wanted = {
                        "hip": ("hips", "pelvis", "root"),
                        "l_foot": ("l_ankle", "leftfoot", "foot_l", "l_foot"),
                        "r_foot": ("r_ankle", "rightfoot", "foot_r", "r_foot"),
                        "l_toe": ("l_ball", "lefttoebase", "l_toe", "ball_l"),
                        "r_toe": ("r_ball", "righttoebase", "r_toe", "ball_r"),
                        "head": ("head",),
                    }
                    for nm, aliases in wanted.items():
                        for a in aliases:
                            xf = ci.get(a)
                            if xf is not None:
                                t = xf.ExtractTranslation()
                                pos = [round(float(t[0]), 3), round(float(t[1]), 3), round(float(t[2]), 3)]
                                out[nm] = pos
                                # Per-foot height over the DISCRETE tread under THAT foot.
                                # The ANKLE joint sits a fixed amount above the sole (~0.29 m
                                # on this mannequin), so raw clearance never reads ~0 even
                                # when planted. Track a per-foot running MINIMUM (the planted
                                # / standing height) and report lift ABOVE that, so the log
                                # reads ~0 = planted, >0 = raised -- the true "foot off the
                                # step" signal, robust to the ankle-above-sole offset.
                                if nm in ("l_foot", "r_foot"):
                                    try:
                                        terr = float(get_terrain_height(float(t[0]), float(t[1])))
                                        clr = pos[2] - terr
                                        out[nm + "_clear"] = round(clr, 3)
                                        mins = getattr(person, "_foot_clear_min", None)
                                        if mins is None:
                                            mins = {}
                                            person._foot_clear_min = mins
                                        mins[nm] = min(mins.get(nm, clr), clr)
                                        out[nm + "_lift"] = round(clr - mins[nm], 3)
                                    except Exception:
                                        pass
                                break
                elif skel_status == "ok":
                    skel_status = "empty_xforms"
    except Exception as _e:
        skel_status = f"err:{type(_e).__name__}"
    out["skel_status"] = skel_status
    return out

def _patient_lowest_foot(person):
    """(lowest animated foot-joint world Z, (x, y)) of the patient, or None.

    Reads the LIVE UsdSkel pose, so -- unlike the bind-pose mesh bbox that ``float_m``
    uses -- it reflects the actual posed (bent-leg) feet. This is the metric that reveals
    foot hover and the value the foot-grounding shifts onto the tread.
    """
    global _patient_skel_cache
    try:
        import omni.usd
        from pxr import UsdSkel, UsdGeom, Usd
        stage = omni.usd.get_context().get_stage()
        srp = getattr(person, "_skel_root_path", "") or ""
        root = stage.GetPrimAtPath(srp) if srp else None
        if not (root and root.IsValid()):
            return None
        if _patient_skel_cache is None:
            _patient_skel_cache = UsdSkel.Cache()
        try:
            _patient_skel_cache.Populate(UsdSkel.Root(root), Usd.PrimDefaultPredicate)
        except Exception:
            try:
                _patient_skel_cache.Populate(UsdSkel.Root(root))
            except Exception:
                return None
        skel = None
        for p in Usd.PrimRange(root):
            if p.IsA(UsdSkel.Skeleton):
                skel = UsdSkel.Skeleton(p)
                break
        if skel is None:
            return None
        q = _patient_skel_cache.GetSkelQuery(skel)
        if not q:
            return None
        xfc = UsdGeom.XformCache(Usd.TimeCode.Default())
        xforms = None
        for _a in ((xfc,), (xfc, False)):
            try:
                xforms = q.ComputeJointWorldTransforms(*_a)
                if xforms:
                    break
            except Exception:
                pass
        if not xforms:
            return None
        joints = skel.GetJointsAttr().Get()
        feet = ("l_ball", "lefttoebase", "l_toe", "ball_l", "r_ball", "righttoebase",
                "r_toe", "ball_r", "l_ankle", "leftfoot", "foot_l", "l_foot",
                "r_ankle", "rightfoot", "foot_r", "r_foot")
        best = None
        for j, xf in zip(joints, xforms):
            leaf = str(j).rsplit("/", 1)[-1].lower()
            if leaf in feet:
                tr = xf.ExtractTranslation()
                z = float(tr[2])
                if best is None or z < best[0]:
                    best = (z, (float(tr[0]), float(tr[1])))
        return best
    except Exception:
        return None

def _patient_upright_quat(yaw_rad: float) -> "np.ndarray":
    """Root orientation quaternion [w,x,y,z] = Rz(yaw)*Rx(90deg).

    The MJCF build rotates the humanoid +90deg about X to stand it upright; reproduce
    that and add a world-Z yaw to face the walking direction. Matches the readback
    convention used elsewhere (2*atan2(qz, qw) == yaw_rad).
    """
    a = 0.7071067811865476
    c = math.cos(yaw_rad / 2.0)
    s = math.sin(yaw_rad / 2.0)
    return np.array([c * a, c * a, s * a, s * a])

def ensure_person_animation_loaded(world: World, person, *, render: bool, attempts: int = 4) -> bool:
    if not hasattr(person, "ensure_animation_ready"):
        return False

    try:
        person.ensure_animation_ready(world)
        if getattr(person, "animation_ready", False):
            log_event(
                env_state.LOGGER,
                logging.INFO,
                "person_animation_confirmed",
                "Person animation is loaded and ready",
            )
            return True
    except Exception as exc:
        log_event(
            env_state.LOGGER,
            logging.ERROR,
            "person_animation_failed_fatal",
            "Person animation readiness failed fatally; exiting.",
            error=str(exc),
        )
        raise

    log_event(
        env_state.LOGGER,
        logging.ERROR,
        "person_animation_not_ready",
        "Person animation did not become ready; refusing to run without the real animation graph",
    )
    raise RuntimeError("Person animation graph did not become ready in strict animation mode")

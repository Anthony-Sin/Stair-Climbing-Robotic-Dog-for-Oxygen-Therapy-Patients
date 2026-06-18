"""Generate the realistic staircase ``.usda`` visual for the final scene.

Pure Python -- run with the *system* Python, no Isaac Sim required:

    python -m final_scene.build_assets          # from sim/isaac
    # or
    python sim/isaac/final_scene/build_assets.py

Output (overwritten each run) into ``final_scene/assets/``:

    staircase.usda   solid concrete treads/risers + nosing lips + a top landing +
                     a sloped metal handrail (posts + rail) on both sides.

This is a VISUAL mesh only: the climb collision is the (invisible) box treads that
``isaac_env.spawn_obstacles`` spawns from the active StairSpec, so the visual is
authored from the SAME geometry numbers (``final_scene.spec.StairVisualSpec``) and
overlaid at the staircase base by ``isaac_mount.attach_final_scene``.

Origin convention: X=0 at the staircase base (StairSpec.start_x_m), Z=0 at the
floor, Y centred -- so the Isaac side drops it at translate (start_x, 0, 0).
"""

from __future__ import annotations

import os
import sys

# Allow `python build_assets.py` as well as `python -m final_scene.build_assets`.
# Both need sim/isaac on the path so the dependency-free o2_payload mesh/USDA
# helpers can be reused (no duplication).
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from o2_payload import geometry as geo
    from o2_payload.usda import UsdaScene, mesh_prim, xform
    from final_scene.spec import SPEC
else:
    from o2_payload import geometry as geo
    from o2_payload.usda import UsdaScene, mesh_prim, xform
    from .spec import SPEC

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _beam(p0, p1, half_y: float, half_z: float) -> "geo.Mesh":
    """A straight rectangular bar between two points (cross-section 2*half_y in Y
    by 2*half_z in Z). Used for the sloped handrail that follows the nosing line."""
    (x0, y0, z0) = p0
    (x1, y1, z1) = p1
    m = geo.Mesh()

    def ring(x, y, z):
        return [
            m.add_point((x, y - half_y, z - half_z)),
            m.add_point((x, y - half_y, z + half_z)),
            m.add_point((x, y + half_y, z + half_z)),
            m.add_point((x, y + half_y, z - half_z)),
        ]

    r0 = ring(x0, y0, z0)
    r1 = ring(x1, y1, z1)
    for k in range(4):
        j = (k + 1) % 4
        m.add_face([r0[k], r0[j], r1[j], r1[k]])
    m.add_face([r0[3], r0[2], r0[1], r0[0]])  # cap end0
    m.add_face([r1[0], r1[1], r1[2], r1[3]])  # cap end1
    return m


def build_staircase() -> UsdaScene:
    s = SPEC.stair
    run = s.step_depth_m
    rise = s.step_height_m
    n = s.step_count
    w = s.width_m
    top = s.top_height_m
    total_run = n * run
    slope = rise / run  # nosing line gradient

    scene = UsdaScene(
        "Staircase",
        doc=f"Realistic staircase ({n} steps, rise={rise*1000:.0f} mm, "
            f"run={run*1000:.0f} mm, top={top*1000:.0f} mm). Origin at base "
            f"(X=0=StairSpec.start_x_m, Z=0=floor, Y centred).",
    )
    root = scene.add(xform("Staircase"))

    # --- solid concrete steps (tread + riser as one block per step) ---
    steps = geo.Mesh()
    nosings = geo.Mesh()
    for i in range(n):
        cx = i * run + run / 2.0
        h = (i + 1) * rise
        steps.merge(geo.box((cx, 0.0, h / 2.0), (run, w, h)))
        # nosing lip overhanging the riser at the front edge of each tread
        nosings.merge(geo.box((i * run, 0.0, h - 0.012), (0.04, w, 0.022)))
    root.add_child(mesh_prim("Steps", steps, s.tread_color, roughness=0.85))
    root.add_child(mesh_prim("Nosings", nosings, s.nosing_color, roughness=0.6))

    # --- top landing slab ---
    landing = geo.box((total_run + s.landing_depth_m / 2.0, 0.0, top / 2.0),
                      (s.landing_depth_m, w, top))
    root.add_child(mesh_prim("Landing", landing, s.landing_color, roughness=0.85))

    # --- handrails (posts + sloped rail) on both sides ---
    if s.handrail:
        rail_y = w / 2.0 + 0.06
        hand_h = 0.9                      # rail height above the nosing line
        post_r = 0.028
        rail_hy, rail_hz = 0.03, 0.03
        # post x positions: base, every ~3 treads, and the landing end
        xs = sorted(set([0.15] + [k * run for k in range(0, n + 1, 3)] + [total_run, total_run + s.landing_depth_m - 0.15]))
        for side, sy in (("L", rail_y), ("R", -rail_y)):
            posts = geo.Mesh()
            for px in xs:
                nose_z = min(top, px * slope)
                rail_top = nose_z + hand_h
                posts.merge(geo.cylinder((px, sy, rail_top / 2.0), post_r, rail_top,
                                         axis="z", seg=12))
            root.add_child(mesh_prim(f"Posts{side}", posts, s.post_color, roughness=0.5))
            # sloped rail following the nosing line, then flat over the landing
            rail = _beam((0.0, sy, hand_h), (total_run, sy, top + hand_h), rail_hy, rail_hz)
            rail.merge(_beam((total_run, sy, top + hand_h),
                             (total_run + s.landing_depth_m, sy, top + hand_h),
                             rail_hy, rail_hz))
            root.add_child(mesh_prim(f"Rail{side}", rail, s.rail_color,
                                     metallic=0.6, roughness=0.3, double_sided=True))
    return scene


def _summary(scene: UsdaScene) -> str:
    meshes = [p for p in _walk(scene) if p.type_name == "Mesh"]
    pts = 0
    for p in meshes:
        for line in p.attr_lines:
            if line.startswith("point3f[] points"):
                pts += line.count("(")
                break
    return f"{len(meshes)} meshes, {pts} points"


def _walk(scene: UsdaScene):
    stack = list(scene.roots)
    while stack:
        p = stack.pop()
        yield p
        stack.extend(p.children)


def main() -> None:
    SPEC.validate()
    os.makedirs(ASSETS_DIR, exist_ok=True)
    scene = build_staircase()
    path = os.path.join(ASSETS_DIR, "staircase.usda")
    scene.write(path)
    print(f"[build_assets] wrote {path}  ({_summary(scene)}, {os.path.getsize(path)} bytes)")
    print("[build_assets] done.")


if __name__ == "__main__":
    main()

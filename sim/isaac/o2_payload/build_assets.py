"""Generate the oxygen-concentrator + rail-cradle ``.usda`` visual assets.

Pure Python -- run with the *system* Python, no Isaac Sim required:

    python -m o2_payload.build_assets            # from sim/isaac
    # or
    python sim/isaac/o2_payload/build_assets.py

Outputs (overwritten each run) into ``o2_payload/assets/``:

    o2_concentrator.usda   mock-up Rhythm Healthcare P2-E6 shell + details.
                           Authored with the ORIGIN AT THE TANK'S GEOMETRIC
                           CENTRE so the Isaac side can drop it at
                           ``SPEC.tank_center_m`` in the trunk frame.
    o2_rails.usda          3D-printed rail cradle + retaining straps.
                           Authored with the ORIGIN AT THE TANK REST PLANE
                           (top of the base plate == top of the robot's back),
                           placed at ``SPEC.mount.holder_center_m``.

These are *visual* meshes only. Mass, collision and the breakable strap joint are
applied on top of them at spawn time by ``isaac_mount.py``.
"""

from __future__ import annotations

import os
import sys

# Allow running as a plain script (python build_assets.py) as well as a module.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from o2_payload import geometry as geo
    from o2_payload.spec import SPEC
    from o2_payload.usda import Prim, UsdaScene, mesh_prim, xform
else:
    from . import geometry as geo
    from .spec import SPEC
    from .usda import Prim, UsdaScene, mesh_prim, xform

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


# ---------------------------------------------------------------------------
# Concentrator (origin = tank geometric centre)
# ---------------------------------------------------------------------------
def build_concentrator() -> UsdaScene:
    c = SPEC.concentrator
    L, W, H = c.length_m, c.width_m, c.height_m
    hx, hy, hz = L / 2.0, W / 2.0, H / 2.0
    seg = 7  # segments per rounded corner

    scene = UsdaScene(
        "Concentrator",
        doc="Mock-up Rhythm Healthcare P2-E6 portable oxygen concentrator "
            f"({L*1000:.0f}x{W*1000:.0f}x{H*1000:.0f} mm). Origin at tank centre.",
    )
    root = scene.add(xform("Concentrator"))

    # --- shell: grey base trim + off-white rounded body ---
    base_h = min(0.018, H * 0.12)
    base = geo.rounded_rect_prism(
        (0.0, 0.0, -hz + base_h / 2.0), L, W, base_h, c.corner_radius_m, seg
    )
    root.add_child(mesh_prim("BaseTrim", base, c.base_color, roughness=0.7))

    body_h = H - base_h
    body = geo.rounded_rect_prism(
        (0.0, 0.0, -hz + base_h + body_h / 2.0), L, W, body_h, c.corner_radius_m, seg
    )
    root.add_child(mesh_prim("Shell", body, c.body_color, roughness=0.35))

    # --- front face (+Y): cannula outlet port (light-blue bezel + dark hole) ---
    front_y = hy
    port_cx, port_cz = 0.018, -hz + H * 0.27
    port_r = min(0.026, hy * 0.7)
    bezel = geo.cylinder((port_cx, front_y + 0.004, port_cz), port_r, 0.014, axis="y", seg=28)
    root.add_child(mesh_prim("OutletBezel", bezel, c.port_color, roughness=0.25))
    hole = geo.cylinder((port_cx, front_y + 0.009, port_cz), port_r * 0.5, 0.010, axis="y", seg=24)
    root.add_child(mesh_prim("OutletHole", hole, (0.10, 0.13, 0.16), roughness=0.4))

    # row of dose buttons along the port
    btn_mesh = geo.Mesh()
    n_btn = 6
    span = port_r * 1.1
    for i in range(n_btn):
        t = (i / (n_btn - 1) - 0.5) * 2.0 * span
        btn_mesh.merge(
            geo.cylinder((port_cx + t, front_y + 0.010, port_cz - port_r * 0.55),
                         0.0035, 0.006, axis="y", seg=12)
        )
    root.add_child(mesh_prim("DoseButtons", btn_mesh, (0.20, 0.22, 0.25), roughness=0.5))

    # --- front face (+Y): Rhythm Healthcare logo block (orange mark + grey word) ---
    logo_cz = hz - H * 0.22
    mark = geo.box((-hx * 0.18, front_y + 0.0025, logo_cz), (0.020, 0.005, 0.020))
    root.add_child(mesh_prim("LogoMark", mark, c.accent_color, roughness=0.3))
    word = geo.box((-hx * 0.18 + 0.044, front_y + 0.0022, logo_cz), (0.060, 0.004, 0.012))
    root.add_child(mesh_prim("LogoWord", word, c.logo_grey, roughness=0.4))

    # --- side face (-X): recessed circular intake filter cap ---
    side_x = -hx
    cap_cz = hz - H * 0.30
    cap = geo.cylinder((side_x - 0.003, 0.0, cap_cz), min(0.030, hy * 0.92), 0.012,
                       axis="x", seg=32)
    root.add_child(mesh_prim("IntakeCap", cap, c.intake_color, roughness=0.45))
    cap_ring = geo.cylinder((side_x - 0.001, 0.0, cap_cz), min(0.030, hy * 0.92) * 0.62,
                            0.013, axis="x", seg=28)
    root.add_child(mesh_prim("IntakeCenter", cap_ring, (0.55, 0.58, 0.62), roughness=0.5))

    # --- top: small carry-handle recess hint + status LCD on front-top ---
    lcd = geo.box((hx * 0.16, front_y + 0.0025, hz - H * 0.50), (0.038, 0.005, 0.026))
    root.add_child(mesh_prim("Display", lcd, (0.12, 0.16, 0.20), roughness=0.15))
    return scene


# ---------------------------------------------------------------------------
# Rail cradle (origin = tank rest plane, centred on the robot back)
# ---------------------------------------------------------------------------
def build_rails() -> UsdaScene:
    c = SPEC.concentrator
    r = SPEC.rail
    L, W, H = c.length_m, c.width_m, c.height_m

    plate_t = r.base_plate_thickness_m
    rail_t = r.rail_thickness_m
    wall_h = r.wall_height_m
    gap = r.side_gap_m
    over = r.fore_aft_overhang_m

    plate_l = L + 2.0 * over
    plate_w = W + 2.0 * (gap + rail_t)

    scene = UsdaScene(
        "O2Rails",
        doc="3D-printed adjustable rail cradle + retaining straps "
            f"({r.mass_kg*1000:.0f} g). Origin at tank rest plane (robot back top).",
    )
    root = scene.add(xform("O2Rails"))

    # --- base plate (top face at z=0, body hangs just below) ---
    plate = geo.box((0.0, 0.0, -plate_t / 2.0), (plate_l, plate_w, plate_t))
    root.add_child(mesh_prim("BasePlate", plate, r.rail_color, roughness=0.6))

    # --- two side rails hugging the tank's long sides (run along X) ---
    rail_y = W / 2.0 + gap + rail_t / 2.0
    for side, sy in (("L", rail_y), ("R", -rail_y)):
        wall = geo.box((0.0, sy, wall_h / 2.0), (plate_l, rail_t, wall_h))
        root.add_child(mesh_prim(f"SideRail{side}", wall, r.rail_color, roughness=0.6))
        # adjustable-slot bolt holes along each rail (visual detail)
        holes = geo.Mesh()
        for i in range(5):
            hx = (i / 4.0 - 0.5) * (plate_l * 0.72)
            holes.merge(geo.cylinder((hx, sy + (rail_t / 2.0) * (1 if sy > 0 else -1),
                                      wall_h * 0.55), 0.005, rail_t * 1.1,
                                     axis="y", seg=12))
        root.add_child(mesh_prim(f"RailHoles{side}", holes, (0.06, 0.06, 0.07),
                                 roughness=0.7))

    # --- front + rear end stops (low walls preventing fore-aft slide) ---
    stop_x = L / 2.0 + r.upright_thickness_m / 2.0
    stop_h = wall_h * 0.75
    for side, sx in (("F", stop_x), ("B", -stop_x)):
        stop = geo.box((sx, 0.0, stop_h / 2.0),
                       (r.upright_thickness_m, plate_w * 0.9, stop_h))
        root.add_child(mesh_prim(f"EndStop{side}", stop, r.rail_color, roughness=0.6))

    # --- retaining straps arcing over the tank top (the "secure" element) ---
    strap_z = H + 0.006   # just above the tank top
    strap_w = plate_w + 0.01
    for side, sx in (("Front", L * 0.22), ("Rear", -L * 0.22)):
        strap = geo.box((sx, 0.0, strap_z), (0.026, strap_w, 0.006))
        root.add_child(mesh_prim(f"Strap{side}", strap, r.strap_color, roughness=0.55))
        # short posts connecting strap ends down to the rails
        for sy in (rail_y, -rail_y):
            post = geo.box((sx, sy, (strap_z + wall_h) / 2.0),
                           (0.022, rail_t * 0.8, strap_z - wall_h))
            root.add_child(mesh_prim(f"Post{side}{'L' if sy>0 else 'R'}", post,
                                     r.strap_color, roughness=0.55))
    return scene


# ---------------------------------------------------------------------------
def _summ(scene: UsdaScene) -> str:
    n_meshes = sum(1 for p in _walk(scene) if p.type_name == "Mesh")
    n_pts = sum(_count_points(p) for p in _walk(scene))
    return f"{n_meshes} meshes, {n_pts} points"


def _walk(scene: UsdaScene):
    stack = list(scene.roots)
    while stack:
        p = stack.pop()
        yield p
        stack.extend(p.children)


def _count_points(prim: Prim) -> int:
    for line in prim.attr_lines:
        if line.startswith("point3f[] points"):
            return line.count("(")
    return 0


def main() -> None:
    SPEC  # touch the validated spec (raises if the numbers are inconsistent)
    os.makedirs(ASSETS_DIR, exist_ok=True)

    targets = [
        ("o2_concentrator.usda", build_concentrator()),
        ("o2_rails.usda", build_rails()),
    ]
    for name, scene in targets:
        path = os.path.join(ASSETS_DIR, name)
        scene.write(path)
        size = os.path.getsize(path)
        print(f"[build_assets] wrote {path}  ({_summ(scene)}, {size} bytes)")

    print("[build_assets] done.")


if __name__ == "__main__":
    main()

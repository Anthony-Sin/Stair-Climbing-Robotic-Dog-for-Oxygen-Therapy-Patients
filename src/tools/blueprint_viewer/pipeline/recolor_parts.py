#!/usr/bin/env python
"""Surgically re-color the baked robot.glb per material group, WITHOUT a re-bake.

The shipped bake MERGES each Go2 link's authored material groups (from the source
COLLADA / USD GeomSubsets) into ONE material-less mesh, so a whole link renders a
single flat color. This tool recovers that split so the viewer can paint the real
two-tone Go2 (silver shell + black hardware) -- and it does so by replacing ONLY
the geometry of the affected meshes, leaving every node + animation channel
byte-identical (no re-sim, no re-rig).

Links recolored, and how each source subset routes to a color:

  base   -> robot_base mesh
    深色橡胶  (dark-rubber shell, spans the whole body)      -> SHELL (silver)
    白色logo  (the embossed printed logo WORDS on the body)  -> BLACK  (so the
                                                                robot's own text
                                                                reads, per user)
    黑色贴纸 / 黑色金属 / 黑色塑料  (all at the head, x>0.25) -> BLACK  (lidar +
                                                                sensor cluster)
  {FL,FR,RL,RR}_calf  -> *_calf meshes
    深色橡胶  (the shin)                                      -> SHELL (silver)
    黑色足端  (the foot CONTACT PAD, at the calf's bottom)    -> BLACK  (only the
                                                                ground-contact
                                                                pad, not the shin)

The hip "connector" housings and the tiny foot collision ball are painted flat
black by the VIEWER (robotBlackMaterial) -- they're uniform, so they need no
per-vertex split here.

Colors are LINEAR RGB (three.js multiplies vertex colors in linear space). SHELL
matches the viewer's flat silver robotMaterial; BLACK matches robotBlackMaterial,
so the baked-COLOR_0 blacks and the flat-material blacks share one tone (no seam
where a black calf-pad meets the black foot ball).

mesh_cache/ is gitignored, so pass --usd to wherever go2.usd lives if the default
(pipeline/mesh_cache/go2.usd) is absent.

Usage:
  python recolor_parts.py --usd <go2.usd>           # DRY RUN: extract + print stats
  python recolor_parts.py --usd <go2.usd> --apply    # write models/robot.glb in place
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE_DIR))

import usd_mesh as um  # reuse the exact link-local transform helpers + cache paths

# --- category colors (LINEAR RGB) ---------------------------------------------
# SHELL = linear(sRGB 0xc0c0c0, "Silver") so the baked body/shin match the
# viewer's flat-silver robotMaterial (three converts 0xc0c0c0 to this internally).
SHELL = (0.5271, 0.5271, 0.5271)
# BLACK = linear(sRGB 0x232629): near-black, lifted just enough that PBR shading
# still reads form on the lidar/sensors/logo/foot-pad. Matches robotBlackMaterial.
BLACK = (0.0170, 0.0194, 0.0222)
CAT_COLOR = {"shell": SHELL, "black": BLACK}


def cat_base(subset_name: str) -> str:
    """Route a base-link subset name to a color category."""
    n = subset_name
    if "白色logo" in n:  # the embossed printed logo WORDS -> black so they read
        return "black"
    if "黑色" in n:  # black sticker / metal / plastic == head lidar + sensors
        return "black"
    return "shell"  # 深色橡胶 (shell) + any unnamed default


def cat_calf(subset_name: str) -> str:
    """Route a calf-link subset name to a color category."""
    if "黑色" in subset_name:  # 黑色足端 == the ground-contact foot pad
        return "black"
    return "shell"  # 深色橡胶 == the shin


def belly_black(ctr, nrm):
    """Geometric override for the base link: paint the downward-facing underside
    (the belly panel) black. The model tags the whole body shell as ONE material,
    so without this the belly inherits the silver body color. ctr/nrm are
    link-local; base z is UP (bbox z[-0.097,+0.089]), so 'downward-facing + in the
    lower half' selects the belly floor without touching the silver sides."""
    return "black" if (nrm[2] < -0.5 and ctr[2] < -0.02) else None


def cat_shell(subset_name: str) -> str:
    """Everything -> shell. For links whose black region is picked GEOMETRICALLY
    (via geom_override) rather than by material tag."""
    return "shell"


# The leg's round pitch joints (axis = y) are DISCS in the x-z plane, so select
# them RADIALLY so the black follows the round edge (a flat z-cut would slice the
# circle with a straight chord). THIGH has two: the big TOP hip-pitch housing (the
# circle where the upper leg meets the hip) and the smaller BOTTOM knee pivot. The
# knee straddles two meshes, so it's also painted on the CALF's TOP. Centers/radii
# are link-local (thigh z[-0.227,+0.051]; calf z[-0.236,+0.031]). Grow a *_R to
# cover more of that joint, shrink it to cover less.
THIGH_HOUSING_CZ, THIGH_HOUSING_R = 0.003, 0.052   # top hip-pitch housing
THIGH_KNEE_CZ, THIGH_KNEE_R = -0.205, 0.034        # bottom knee pivot
CALF_KNEE_CZ, CALF_KNEE_R = 0.006, 0.030           # top knee pivot


def _in_disc(ctr, cz, r):
    dz = ctr[2] - cz
    return (ctr[0] * ctr[0] + dz * dz) < (r * r)


def thigh_black(ctr, nrm):
    return "black" if (_in_disc(ctr, THIGH_HOUSING_CZ, THIGH_HOUSING_R)
                       or _in_disc(ctr, THIGH_KNEE_CZ, THIGH_KNEE_R)) else None


def calf_knee_black(ctr, nrm):
    # foot pad at the calf bottom is handled by cat_calf's subset routing; this
    # adds the round knee pivot at the calf top.
    return "black" if _in_disc(ctr, CALF_KNEE_CZ, CALF_KNEE_R) else None


# (usd_link_name, glb_mesh_name, category_fn, geom_override_or_None)
LINKS = [
    ("base", "robot_base", cat_base, None),  # belly override OFF -- user wanted a LEG part black (the hip), not the underbody
    ("FL_thigh", "FL_thigh", cat_shell, thigh_black),
    ("FR_thigh", "FR_thigh", cat_shell, thigh_black),
    ("RL_thigh", "RL_thigh", cat_shell, thigh_black),
    ("RR_thigh", "RR_thigh", cat_shell, thigh_black),
    ("FL_calf", "FL_calf", cat_calf, calf_knee_black),
    ("FR_calf", "FR_calf", cat_calf, calf_knee_black),
    ("RL_calf", "RL_calf", cat_calf, calf_knee_black),
    ("RR_calf", "RR_calf", cat_calf, calf_knee_black),
]


def _weld(points, tol=1e-7):
    remap = {}
    new_points = []
    index_of = []
    inv = 1.0 / tol
    for p in points:
        key = (int(round(p[0] * inv)), int(round(p[1] * inv)), int(round(p[2] * inv)))
        idx = remap.get(key)
        if idx is None:
            idx = len(new_points)
            remap[key] = idx
            new_points.append(p)
        index_of.append(idx)
    return new_points, index_of


def _bbox(pts, sel=None):
    if sel is not None:
        pts = [pts[i] for i in sel]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def extract_link_colored(stage, xcache, link_name: str, cat_fn, geom_override=None):
    """Return (positions, normals, colors, indices, counts_by_cat, cat_of_name) for
    a link, link-local, smooth-normaled, with per-category COLOR_0. Categories are
    unwelded from one another so colors don't bleed across a material seam.

    geom_override, if given, is a callable (centroid, normal) -> cat|None applied
    per FACE after the subset routing, to reclassify faces by geometry (e.g. belly)."""
    from pxr import Usd, UsdGeom

    link_prim = None
    for p in stage.Traverse():
        if p.GetName() == link_name:
            link_prim = p
            break
    if link_prim is None:
        raise RuntimeError(f"no '{link_name}' link prim")

    def under_link(prim):
        p = prim
        while p and p.GetPath() != "/":
            if p == link_prim:
                return True
            p = p.GetParent()
        return False

    mesh_prim = None
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if prim.IsA(UsdGeom.Mesh) and under_link(prim):
            mesh_prim = prim
            break
    if mesh_prim is None:
        raise RuntimeError(f"no mesh prim under '{link_name}'")

    m = UsdGeom.Mesh(mesh_prim)
    points = m.GetPointsAttr().Get()
    counts = list(m.GetFaceVertexCountsAttr().Get())
    fvindices = list(m.GetFaceVertexIndicesAttr().Get())

    # mesh-local -> link-local
    mesh_to_world = um._matrix4_to_rows(xcache.GetLocalToWorldTransform(mesh_prim))
    link_to_world = xcache.GetLocalToWorldTransform(link_prim)
    world_to_link = um._matrix4_to_rows(link_to_world.GetInverse())
    mesh_to_link = um._mat_mul_rowvec(mesh_to_world, world_to_link)
    pts = [um._transform_point_rowvec(mesh_to_link, (float(p[0]), float(p[1]), float(p[2])))
           for p in points]

    # face -> starting offset in fvindices
    face_offset = []
    off = 0
    for c in counts:
        face_offset.append(off)
        off += c

    # GeomSubsets: face-index -> category
    face_cat = ["shell"] * len(counts)  # default before subsets
    subsets = [c for c in mesh_prim.GetChildren() if c.IsA(UsdGeom.Subset)]
    cat_of_name = {}
    for sub in subsets:
        name = sub.GetName()
        cat = cat_fn(name)
        cat_of_name[name] = cat
        for f in UsdGeom.Subset(sub).GetIndicesAttr().Get() or []:
            if 0 <= f < len(face_cat):
                face_cat[f] = cat

    # Optional GEOMETRIC override: reclassify whole faces by their link-local
    # centroid + Newell normal. Runs AFTER subsets so it wins where it applies.
    # Used to paint the belly black -- the model tags the underside as shell, so
    # without this the downward body panel inherits the silver body color.
    if geom_override is not None:
        import math as _m
        for f, c in enumerate(counts):
            if c < 3:
                continue
            s = face_offset[f]
            fv = [pts[int(fvindices[s + k])] for k in range(c)]
            cx = sum(p[0] for p in fv) / c
            cy = sum(p[1] for p in fv) / c
            cz = sum(p[2] for p in fv) / c
            nx = ny = nz = 0.0
            for i in range(c):
                x0, y0, z0 = fv[i]
                x1, y1, z1 = fv[(i + 1) % c]
                nx += (y0 - y1) * (z0 + z1)
                ny += (z0 - z1) * (x0 + x1)
                nz += (x0 - x1) * (y0 + y1)
            nl = _m.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
            cat2 = geom_override((cx, cy, cz), (nx / nl, ny / nl, nz / nl))
            if cat2:
                face_cat[f] = cat2

    # Triangulate (fan) -> global triangles (indices into pts) + per-tri category
    tris = []
    for f, c in enumerate(counts):
        if c < 3:
            continue
        s = face_offset[f]
        face = [int(fvindices[s + k]) for k in range(c)]
        for k in range(1, c - 1):
            tris.append((face[0], face[k], face[k + 1], face_cat[f]))

    # Weld points globally (positions) for smooth normals
    import math
    welded_pts, remap = _weld(pts)
    acc = [[0.0, 0.0, 0.0] for _ in welded_pts]
    for (a, b, c, _cat) in tris:
        wa, wb, wc = remap[a], remap[b], remap[c]
        p0, p1, p2 = welded_pts[wa], welded_pts[wb], welded_pts[wc]
        ux, uy, uz = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        vx, vy, vz = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        nx, ny, nz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        for wi in (wa, wb, wc):
            acc[wi][0] += nx; acc[wi][1] += ny; acc[wi][2] += nz
    wnormals = []
    for aI in acc:
        ln = math.sqrt(aI[0] * aI[0] + aI[1] * aI[1] + aI[2] * aI[2])
        wnormals.append((aI[0] / ln, aI[1] / ln, aI[2] / ln) if ln > 1e-15 else (0.0, 0.0, 1.0))

    # Emit per-category: unweld across categories (own copies) but reuse smooth
    # normals from the global weld. Local per-(category, welded-vert) dedup.
    out_pos, out_nrm, out_col, out_idx = [], [], [], []
    local = {}
    counts_by_cat = {}
    for (a, b, c, cat) in tris:
        col = CAT_COLOR[cat]
        tri_out = []
        for wi in (remap[a], remap[b], remap[c]):
            key = (cat, wi)
            oi = local.get(key)
            if oi is None:
                oi = len(out_pos)
                local[key] = oi
                out_pos.append(welded_pts[wi])
                out_nrm.append(wnormals[wi])
                out_col.append(col)
            tri_out.append(oi)
        out_idx.extend(tri_out)
        counts_by_cat[cat] = counts_by_cat.get(cat, 0) + 1

    return out_pos, out_nrm, out_col, out_idx, counts_by_cat, cat_of_name


def replace_mesh_geometry(g, blob, mesh_name, out_pos, out_nrm, out_col, out_idx):
    """Append POSITION/NORMAL/COLOR_0/indices for one mesh and repoint its primitive."""
    import pygltflib as pg

    mesh_idx = next((i for i, mm in enumerate(g.meshes) if mm.name == mesh_name), None)
    if mesh_idx is None:
        raise RuntimeError(f"mesh '{mesh_name}' not found in glb")

    def pad4():
        while len(blob) % 4:
            blob.append(0)

    def add_view(data_bytes, target):
        pad4()
        off = len(blob)
        blob.extend(data_bytes)
        g.bufferViews.append(pg.BufferView(buffer=0, byteOffset=off, byteLength=len(data_bytes), target=target))
        return len(g.bufferViews) - 1

    ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
    FLOAT, UINT = 5126, 5125

    pos_bytes = b"".join(struct.pack("<3f", *p) for p in out_pos)
    nrm_bytes = b"".join(struct.pack("<3f", *n) for n in out_nrm)
    col_bytes = b"".join(struct.pack("<3f", *c) for c in out_col)
    idx_bytes = b"".join(struct.pack("<I", i) for i in out_idx)

    pos_bv = add_view(pos_bytes, ARRAY_BUFFER)
    nrm_bv = add_view(nrm_bytes, ARRAY_BUFFER)
    col_bv = add_view(col_bytes, ARRAY_BUFFER)
    idx_bv = add_view(idx_bytes, ELEMENT_ARRAY_BUFFER)

    xs = [p[0] for p in out_pos]; ys = [p[1] for p in out_pos]; zs = [p[2] for p in out_pos]
    pos_acc = pg.Accessor(bufferView=pos_bv, componentType=FLOAT, count=len(out_pos), type="VEC3",
                          min=[min(xs), min(ys), min(zs)], max=[max(xs), max(ys), max(zs)])
    nrm_acc = pg.Accessor(bufferView=nrm_bv, componentType=FLOAT, count=len(out_nrm), type="VEC3")
    col_acc = pg.Accessor(bufferView=col_bv, componentType=FLOAT, count=len(out_col), type="VEC3")
    idx_acc = pg.Accessor(bufferView=idx_bv, componentType=UINT, count=len(out_idx), type="SCALAR")

    base_accs = len(g.accessors)
    g.accessors.extend([pos_acc, nrm_acc, col_acc, idx_acc])

    prim = g.meshes[mesh_idx].primitives[0]
    prim.attributes.POSITION = base_accs + 0
    prim.attributes.NORMAL = base_accs + 1
    prim.attributes.COLOR_0 = base_accs + 2
    prim.indices = base_accs + 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=str(um.GO2_USD_LOCAL), help="path to go2.usd (mesh_cache is gitignored -- point at a real copy)")
    ap.add_argument("--glb", default=str(_PIPELINE_DIR.parent / "models" / "robot.glb"))
    ap.add_argument("--apply", action="store_true", help="write the glb (default: dry run)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(args.usd), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"could not open {args.usd}")
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    import pygltflib as pg
    g = pg.GLTF2().load(str(args.glb)) if args.apply else None
    blob = bytearray(g.binary_blob()) if args.apply else None

    for link_name, mesh_name, cat_fn, geom in LINKS:
        pos, nrm, col, idx, counts_by_cat, cat_of_name = extract_link_colored(stage, xcache, link_name, cat_fn, geom)
        black_idx = [i for i, c in enumerate(col) if c == BLACK]
        print(f"=== {link_name} -> mesh '{mesh_name}' ===")
        print("  subsets:", cat_of_name)
        print(f"  tris/cat={counts_by_cat}  verts={len(pos):,} tris={len(idx)//3:,}  black_verts={len(black_idx):,}")
        if black_idx:
            bb = _bbox(pos, black_idx)
            print(f"  BLACK bbox x[{bb[0]:+.3f},{bb[1]:+.3f}] y[{bb[2]:+.3f},{bb[3]:+.3f}] z[{bb[4]:+.3f},{bb[5]:+.3f}]")
        if args.apply:
            replace_mesh_geometry(g, blob, mesh_name, pos, nrm, col, idx)

    if args.apply:
        while len(blob) % 4:
            blob.append(0)
        g.buffers[0].byteLength = len(blob)
        g.set_binary_blob(bytes(blob))
        g.save(str(args.glb))
        print(f"APPLIED -> {args.glb}  ({len(blob):,} bytes)")
    else:
        print("(dry run -- pass --apply to write the glb)")


if __name__ == "__main__":
    main()

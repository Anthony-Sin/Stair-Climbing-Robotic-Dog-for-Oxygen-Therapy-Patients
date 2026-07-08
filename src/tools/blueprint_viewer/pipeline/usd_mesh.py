"""Extracts per-link visual meshes from the EXACT Isaac Sim Go2 asset (go2.usd).

The runtime sim loads
  https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd
(a small USD-crate wrapper that composes sibling ``configuration/*.usd`` layers,
which in turn reference instanceable mesh / material layers). This module:

  1. ``fetch_asset_tree()``  -- recursively downloads the wrapper and every
     composition dependency (resolved RELATIVE to each referencing layer, mirroring
     the S3 prefix's directory structure) into ``pipeline/mesh_cache/``, iterating
     until the stage composes without unresolved-asset warnings.
  2. ``extract_link_meshes()`` -- opens the composed stage, finds every
     ``UsdGeom.Mesh`` prim, associates it to a robot LINK by walking up the prim
     path to the link-named ancestor (base, FL_hip, FL_thigh, ... Head_upper etc.),
     bakes the mesh points into the LINK-LOCAL frame (mesh world transform composed
     with the inverse of the link Xform's world transform, at Default time),
     triangulates (faceVertexCounts may include quads/ngons -- fan triangulation),
     and merges all of a link's mesh islands into one geometry.Mesh.

Meshes come out in the link's own frame, so they drop directly onto the existing
URDF-driven SceneNode tree as visuals (robot_build attaches them; the joint tree,
node names, and animation channels are unchanged). USD-extracted meshes need no
URDF <visual> origin correction -- the link frames in the Isaac asset coincide with
the URDF link frames (verified numerically by the assembled-bbox check in
robot_build's mesh-source selection).
"""
from __future__ import annotations

import math
import posixpath
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as geo

S3_PREFIX = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/"
)
MESH_CACHE_DIR = Path(__file__).resolve().parent / "mesh_cache"
GO2_USD_LOCAL = MESH_CACHE_DIR / "go2.usd"


def fetch_asset_tree(verbose: bool = True) -> List[str]:
    """Recursively download go2.usd + every composition/external dependency from the
    S3 prefix into mesh_cache/, preserving relative directory structure. Returns the
    list of layer-relative paths fetched/present. Idempotent (skips existing files).
    """
    from pxr import Sdf

    fetched: List[str] = []
    pending: List[str] = ["go2.usd"]
    seen: set = set()
    while pending:
        rel = pending.pop(0)
        if rel in seen:
            continue
        seen.add(rel)
        local = MESH_CACHE_DIR / Path(rel)
        if local.is_dir() or not Path(rel).suffix:
            # A dep that resolves to a bare directory / suffix-less path (some layers
            # carry directory-only asset refs) -- nothing to fetch or scan.
            continue
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            url = S3_PREFIX + rel
            try:
                urllib.request.urlretrieve(url, str(local))
                if verbose:
                    print(f"  DL   {rel}  ({local.stat().st_size:,} B)")
            except Exception as e:  # 404s on optional layers are tolerable; report and move on
                if verbose:
                    print(f"  MISS {rel}: {e}")
                continue
        else:
            if verbose:
                print(f"  have {rel}  ({local.stat().st_size:,} B)")
        fetched.append(rel)

        try:
            layer = Sdf.Layer.FindOrOpen(str(local))
        except Exception:
            layer = None  # unrecognized format (texture, etc.)
        if layer is None:
            continue  # not a USD layer -- no deps to scan
        base_dir = posixpath.dirname(rel)
        deps = list(layer.GetCompositionAssetDependencies()) + list(layer.GetExternalAssetDependencies())
        for dep in deps:
            dep_posix = dep.replace("\\", "/")
            if dep_posix.startswith(("http:", "https:", "omniverse:")) or dep_posix.startswith("/"):
                if verbose:
                    print(f"  skip absolute/remote dep: {dep_posix}")
                continue
            dep_rel = posixpath.normpath(posixpath.join(base_dir, dep_posix))
            if dep_rel not in seen:
                pending.append(dep_rel)
    return fetched


def _matrix4_to_rows(m) -> List[List[float]]:
    """pxr Gf.Matrix4d -> row-major 4x4 nested list (Gf matrices are row vectors:
    p' = p @ M, i.e. translation lives in the FOURTH ROW m[3][0..2])."""
    return [[float(m[i][j]) for j in range(4)] for i in range(4)]


def _transform_point_rowvec(m: List[List[float]], p: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Row-vector convention (USD/Gf): p' = [px py pz 1] @ M."""
    x, y, z = p
    return (
        x * m[0][0] + y * m[1][0] + z * m[2][0] + m[3][0],
        x * m[0][1] + y * m[1][1] + z * m[2][1] + m[3][1],
        x * m[0][2] + y * m[1][2] + z * m[2][2] + m[3][2],
    )


def _mat_mul_rowvec(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _weld_vertices(
    points: List[Tuple[float, float, float]], tri_indices: List[int], tol: float = 1e-7,
) -> Tuple[List[Tuple[float, float, float]], List[int]]:
    """Merge positionally-identical vertices (several Go2 sub-meshes -- calf, foot --
    are authored fully UNWELDED: verts == 3x tris). Welding shrinks the glb ~4 MB and
    lets the smooth-normal pass actually smooth across former duplicate seams,
    matching the sculpted look of the sim render. Exact-position hash (quantized by
    ``tol``), so genuinely distinct-but-close vertices are untouched."""
    remap: Dict[Tuple[int, int, int], int] = {}
    new_points: List[Tuple[float, float, float]] = []
    index_of: List[int] = []
    inv = 1.0 / tol
    for p in points:
        key = (int(round(p[0] * inv)), int(round(p[1] * inv)), int(round(p[2] * inv)))
        idx = remap.get(key)
        if idx is None:
            idx = len(new_points)
            remap[key] = idx
            new_points.append(p)
        index_of.append(idx)
    new_indices = [index_of[i] for i in tri_indices]
    return new_points, new_indices


def _indexed_mesh_with_smooth_normals(
    points: List[Tuple[float, float, float]], tri_indices: List[int],
) -> geo.Mesh:
    """Build a geo.Mesh reusing the AUTHORED shared vertices, with area-weighted
    smooth vertex normals (unnormalized face-normal accumulation = area weighting,
    the standard scheme). Degenerate triangles contribute nothing."""
    n_verts = len(points)
    acc = [[0.0, 0.0, 0.0] for _ in range(n_verts)]
    for f in range(0, len(tri_indices), 3):
        i0, i1, i2 = tri_indices[f], tri_indices[f + 1], tri_indices[f + 2]
        p0, p1, p2 = points[i0], points[i1], points[i2]
        ux, uy, uz = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        vx, vy, vz = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        nx, ny, nz = (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)
        for i in (i0, i1, i2):
            acc[i][0] += nx
            acc[i][1] += ny
            acc[i][2] += nz
    normals: List[Tuple[float, float, float]] = []
    for a in acc:
        ln = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        if ln < 1e-15:
            normals.append((0.0, 0.0, 1.0))
        else:
            normals.append((a[0] / ln, a[1] / ln, a[2] / ln))
    mesh = geo.Mesh()
    mesh.positions = list(points)
    mesh.normals = normals
    mesh.indices = list(tri_indices)
    return mesh


def decimate_mesh(mesh: geo.Mesh, target_tris: int) -> geo.Mesh:
    """Light quadric decimation via trimesh/fast_simplification, then smooth-normal
    rebuild. Used only to trim the heaviest links ~20% to fit the 350k assembled-robot
    budget -- NOT the aggressive 10x reduction that (verified earlier) floors out on
    these multi-island meshes. Falls back to the input mesh on any failure."""
    try:
        import numpy as np
        import trimesh

        verts = np.asarray(mesh.positions, dtype=float)
        faces = np.asarray(mesh.indices, dtype=int).reshape(-1, 3)
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        dec = tm.simplify_quadric_decimation(face_count=target_tris)
        pts = [tuple(map(float, p)) for p in dec.vertices]
        idx = [int(i) for i in dec.faces.reshape(-1)]
        if not idx:
            return mesh
        return _indexed_mesh_with_smooth_normals(pts, idx)
    except Exception as e:
        print(f"  decimation failed ({e!r}); keeping full-resolution mesh")
        return mesh


def extract_stage_meshes(usd_path: Path, *, verbose: bool = False) -> Optional[geo.Mesh]:
    """Open ANY usd/usda stage and return every visible UsdGeom.Mesh baked into the
    STAGE-ROOT frame, welded + smooth-normaled, combined into one geo.Mesh. Used for
    the authored O2 payload assets (o2_concentrator.usda / o2_rails.usda -- plain
    non-instanced text layers whose doc strings pin their origins to the exact mount
    reference points the viewer already uses). Returns None if nothing loads."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path), Usd.Stage.LoadAll)
    if stage is None:
        return None
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    combined = geo.Mesh()
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
            continue
        if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        usd_mesh_prim = UsdGeom.Mesh(prim)
        points_attr = usd_mesh_prim.GetPointsAttr().Get()
        counts = usd_mesh_prim.GetFaceVertexCountsAttr().Get()
        indices = usd_mesh_prim.GetFaceVertexIndicesAttr().Get()
        if not points_attr or not counts or not indices:
            continue
        to_world = _matrix4_to_rows(xcache.GetLocalToWorldTransform(prim))
        pts = [_transform_point_rowvec(to_world, (float(p[0]), float(p[1]), float(p[2])))
               for p in points_attr]
        tri_indices: List[int] = []
        vidx = 0
        for fc in counts:
            face = [int(indices[vidx + k]) for k in range(fc)]
            vidx += fc
            if fc < 3:
                continue
            for k in range(1, fc - 1):
                tri_indices.extend([face[0], face[k], face[k + 1]])
        if not tri_indices:
            continue
        w_pts, w_idx = _weld_vertices(pts, tri_indices)
        piece = _indexed_mesh_with_smooth_normals(w_pts, w_idx)
        combined.extend(piece)
        if verbose:
            print(f"  {prim.GetPath()}: +{piece.triangle_count()} tris")
    return combined if combined.indices else None


# Links heavier than this get decimated toward DECIMATE_FACTOR of their original
# count so the ASSEMBLED robot (all 17 link instances) stays under the 350k budget:
# extracted full-res total is 397k; trimming base/hips/thighs ~22% lands ~321k.
DECIMATE_THRESHOLD_TRIS = 20_000
DECIMATE_FACTOR = 0.78


def load_link_visual_meshes(
    link_names: List[str], *, decimate: bool = True, verbose: bool = True,
) -> Dict[str, geo.Mesh]:
    """Public entry for robot_build: fetch the asset tree (idempotent), extract
    per-link meshes, lightly decimate the heavy ones. Raises on total failure
    (caller falls back to dae/primitives per link)."""
    fetch_asset_tree(verbose=verbose)
    meshes = extract_link_meshes(link_names, verbose=verbose)
    if decimate:
        for link, m in list(meshes.items()):
            n = m.triangle_count()
            if n > DECIMATE_THRESHOLD_TRIS:
                target = int(n * DECIMATE_FACTOR)
                dec = decimate_mesh(m, target)
                if verbose:
                    print(f"  decimated {link}: {n:,} -> {dec.triangle_count():,} tris")
                meshes[link] = dec
    return meshes


def extract_link_meshes(
    link_names: List[str], *, usd_path: Optional[Path] = None, verbose: bool = True,
) -> Dict[str, geo.Mesh]:
    """Open the composed go2.usd stage and return {link_name: merged Mesh} with
    points baked into each link's LOCAL frame. Links with no mesh prims are simply
    absent from the result (caller falls back per-link)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path or GO2_USD_LOCAL), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError("could not open go2.usd stage")

    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # Locate the link Xform prim for each link name: prefer an exact prim-name match
    # anywhere in the hierarchy (Isaac robot layout is /go2_description/<link>/...).
    link_prims: Dict[str, "Usd.Prim"] = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in link_names and name not in link_prims:
            link_prims[name] = prim
    if verbose:
        print(f"  link prims found: {sorted(link_prims)}")

    def owning_link(prim) -> Optional[str]:
        p = prim
        while p and p.GetPath() != "/":
            if p.GetName() in link_prims and p == link_prims[p.GetName()]:
                return p.GetName()
            p = p.GetParent()
        return None

    out: Dict[str, geo.Mesh] = {}
    skipped: List[str] = []
    # CRITICAL: the Isaac Go2 keeps each link's geometry under an INSTANCEABLE
    # `<link>/visuals` prim; plain stage.Traverse() does not descend into instance
    # proxies and reports ZERO meshes (observed) -- TraverseInstanceProxies is required.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        link = owning_link(prim)
        if link is None:
            skipped.append(str(prim.GetPath()))
            continue
        usd_mesh = UsdGeom.Mesh(prim)

        # Respect visibility/purpose: skip invisible or guide/proxy-purpose meshes
        # (Isaac assets keep collision proxies under purpose=guide or invisible prims).
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
            continue
        purpose = img.ComputePurpose()
        if purpose in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue

        points_attr = usd_mesh.GetPointsAttr().Get()
        counts = usd_mesh.GetFaceVertexCountsAttr().Get()
        indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
        if not points_attr or not counts or not indices:
            continue

        # mesh-local -> world, then world -> link-local.
        mesh_to_world = _matrix4_to_rows(xcache.GetLocalToWorldTransform(prim))
        link_to_world = xcache.GetLocalToWorldTransform(link_prims[link])
        world_to_link = _matrix4_to_rows(link_to_world.GetInverse())
        mesh_to_link = _mat_mul_rowvec(mesh_to_world, world_to_link)

        local_pts = [
            _transform_point_rowvec(mesh_to_link, (float(p[0]), float(p[1]), float(p[2])))
            for p in points_attr
        ]

        # Triangulate (fan) each face keeping the AUTHORED shared-vertex indexing --
        # per-face flat normals would triple the vertex count (~1.2M verts for the
        # full robot => a ~33 MB glb, over the 25 MB budget); indexed vertices with
        # area-weighted SMOOTH normals match the sculpted look of the sim render and
        # keep the file ~12 MB.
        tri_indices: List[int] = []
        vidx = 0
        for fc in counts:
            face = [int(indices[vidx + k]) for k in range(fc)]
            vidx += fc
            if fc < 3:
                continue
            for k in range(1, fc - 1):
                tri_indices.extend([face[0], face[k], face[k + 1]])
        if not tri_indices:
            continue
        welded_pts, welded_idx = _weld_vertices(local_pts, tri_indices)
        mesh = _indexed_mesh_with_smooth_normals(welded_pts, welded_idx)
        if link in out:
            out[link].extend(mesh)
        else:
            out[link] = mesh
        if verbose:
            print(f"  {prim.GetPath()}  -> link {link}: +{mesh.triangle_count()} tris "
                  f"(link total {out[link].triangle_count()})")

    if verbose and skipped:
        print(f"  mesh prims with no owning link (skipped): {len(skipped)}")
        for s in skipped[:10]:
            print(f"    {s}")
    return out


if __name__ == "__main__":
    LINKS = ["base", "Head_upper", "Head_lower",
             "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
             "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
             "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
             "RR_hip", "RR_thigh", "RR_calf", "RR_foot"]
    print("=== load_link_visual_meshes (fetch + extract + light decimation) ===")
    meshes = load_link_visual_meshes(LINKS)
    total = 0
    total_verts = 0
    print("\nper-link summary:")
    for link in LINKS:
        m = meshes.get(link)
        if m is None:
            print(f"  {link:12s}: NO MESH")
            continue
        xs = [p[0] for p in m.positions]; ys = [p[1] for p in m.positions]; zs = [p[2] for p in m.positions]
        total += m.triangle_count()
        total_verts += len(m.positions)
        print(f"  {link:12s}: {m.triangle_count():7,} tris {len(m.positions):7,} verts  "
              f"bbox x[{min(xs):+.3f},{max(xs):+.3f}] y[{min(ys):+.3f},{max(ys):+.3f}] "
              f"z[{min(zs):+.3f},{max(zs):+.3f}]")
    print(f"\nTOTAL: {total:,} tris, {total_verts:,} verts "
          f"({'OK, under' if total <= 350_000 else 'OVER'} the 350k assembled budget)")

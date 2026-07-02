"""Minimal USDA (ASCII USD) writer -- dependency-free.

Just enough to author static, coloured polygon meshes grouped under Xforms, so
the oxygen-tank + rail visuals can be produced without the ``pxr`` USD libraries
or Isaac Sim. The output is plain ``#usda 1.0`` text that Isaac Sim / usdview /
``Usd.Stage.Open`` load natively.

Per-face flat normals are computed automatically (Newell's method) and authored
as ``faceVarying`` so the rounded shell shades cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .geometry import Mesh

Pt = Tuple[float, float, float]


def _f(x: float) -> str:
    """Compact float formatting (no needless trailing zeros / sci-notation)."""
    s = f"{float(x):.6g}"
    return "0" if s in ("-0", "-0.0") else s


def _vec(p: Sequence[float]) -> str:
    return "(" + ", ".join(_f(v) for v in p) + ")"


def _newell_normal(points: List[Pt], face: Sequence[int]) -> Pt:
    nx = ny = nz = 0.0
    n = len(face)
    for k in range(n):
        x0, y0, z0 = points[face[k]]
        x1, y1, z1 = points[face[(k + 1) % n]]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag < 1e-12:
        return (0.0, 0.0, 1.0)
    return (nx / mag, ny / mag, nz / mag)


@dataclass
class Prim:
    name: str
    type_name: str
    attr_lines: List[str] = field(default_factory=list)
    children: List["Prim"] = field(default_factory=list)

    def add_child(self, child: "Prim") -> "Prim":
        self.children.append(child)
        return child


def xform(name: str, translate: Optional[Pt] = None) -> Prim:
    p = Prim(name, "Xform")
    if translate is not None:
        p.attr_lines.append(f"double3 xformOp:translate = {_vec(translate)}")
        p.attr_lines.append('uniform token[] xformOpOrder = ["xformOp:translate"]')
    return p


def mesh_prim(
    name: str,
    mesh: Mesh,
    color: Pt,
    *,
    double_sided: bool = False,
    metallic: Optional[float] = None,
    roughness: Optional[float] = None,
) -> Prim:
    """Build a coloured Mesh prim from a :class:`~geometry.Mesh`."""
    counts = [len(f) for f in mesh.faces]
    indices: List[int] = [i for f in mesh.faces for i in f]
    normals: List[Pt] = []
    for f in mesh.faces:
        nrm = _newell_normal(mesh.points, f)
        normals.extend([nrm] * len(f))
    (mn, mx) = mesh.extent()

    p = Prim(name, "Mesh")
    a = p.attr_lines
    a.append(f"point3f[] points = [{', '.join(_vec(pt) for pt in mesh.points)}]")
    a.append(f"int[] faceVertexCounts = [{', '.join(str(c) for c in counts)}]")
    a.append(f"int[] faceVertexIndices = [{', '.join(str(i) for i in indices)}]")
    a.append(
        f"normal3f[] normals = [{', '.join(_vec(nv) for nv in normals)}] "
        f'(interpolation = "faceVarying")'
    )
    a.append(f"float3[] extent = [{_vec(mn)}, {_vec(mx)}]")
    a.append('uniform token subdivisionScheme = "none"')
    a.append(f"bool doubleSided = {'true' if double_sided else 'false'}")
    a.append(
        f"color3f[] primvars:displayColor = [{_vec(color)}] "
        f'(interpolation = "constant")'
    )
    if metallic is not None:
        a.append(f"float[] primvars:displayMetallic = [{_f(metallic)}] "
                 f'(interpolation = "constant")')
    if roughness is not None:
        a.append(f"float[] primvars:displayRoughness = [{_f(roughness)}] "
                 f'(interpolation = "constant")')
    return p


class UsdaScene:
    def __init__(
        self,
        default_prim: str,
        *,
        meters_per_unit: float = 1.0,
        up_axis: str = "Z",
        doc: str = "",
    ) -> None:
        self.default_prim = default_prim
        self.meters_per_unit = meters_per_unit
        self.up_axis = up_axis
        self.doc = doc
        self.roots: List[Prim] = []

    def add(self, prim: Prim) -> Prim:
        self.roots.append(prim)
        return prim

    # -- serialization ---------------------------------------------------
    def _render_prim(self, prim: Prim, indent: int) -> List[str]:
        pad = "    " * indent
        lines = [f'{pad}def {prim.type_name} "{prim.name}"', f"{pad}{{"]
        inner = "    " * (indent + 1)
        for line in prim.attr_lines:
            lines.append(f"{inner}{line}")
        if prim.attr_lines and prim.children:
            lines.append("")
        for child in prim.children:
            lines.extend(self._render_prim(child, indent + 1))
        lines.append(f"{pad}}}")
        return lines

    def to_string(self) -> str:
        head = [
            "#usda 1.0",
            "(",
            f'    defaultPrim = "{self.default_prim}"',
            f"    metersPerUnit = {_f(self.meters_per_unit)}",
            f'    upAxis = "{self.up_axis}"',
        ]
        if self.doc:
            esc = self.doc.replace('"', '\\"')
            head.append(f'    doc = "{esc}"')
        head.append(")")
        head.append("")
        body: List[str] = []
        for root in self.roots:
            body.extend(self._render_prim(root, 0))
            body.append("")
        return "\n".join(head + body) + "\n"

    def write(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_string())

"""Load the on-robot O2 payload spec and derive the numbers RL training needs.

The payload's mass / centre-of-mass / bounding box live in ONE place --
``sim/isaac/o2_payload/spec.py`` -- and are imported here through the same path-shim
``fine_tuning.sim_model_source`` uses for the depth-encoder code. Keeping a single definition means
the URDF link and the domain-randomisation bounds can never silently disagree with the
payload that the sim actually mounts.

This module is pure Python (no torch / Isaac / pxr) so it imports on any box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .. import sim_model_source

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class PayloadNumbers:
    """The payload facts RL training consumes, in the Go2 trunk/base frame (SI)."""

    mass_kg: float                 # tank + holder
    com_m: Vec3                    # payload CoM (trunk frame)
    extents_m: Vec3               # bounding box (X fore-aft, Y lateral, Z up)
    inertia_diag: Vec3            # box inertia about the CoM (Ixx, Iyy, Izz)
    trunk_mass_kg: float
    com_shift_m: Vec3             # how far the payload moves the combined CoM
    mass_fraction: float          # payload / trunk
    orientation: str

    # --- derived bounds for domain randomisation -----------------------------
    def added_mass_range(self, margin_kg: float = 1.25) -> Tuple[float, float]:
        """A DR band that ALWAYS carries at least the tank, plus headroom.

        robot_lab's IsaacLab ``randomize_rigid_body_mass`` base event samples a uniform
        value from ``mass_distribution_params`` and ADDS it to the trunk link at startup
        (``operation="add"``). Centring this band on the real payload (rather than the
        stock (-1, 3) around nothing) means every episode trains *with* the load.
        """
        lo = max(0.0, self.mass_kg - margin_kg * 0.5)
        hi = self.mass_kg + margin_kg
        return (round(lo, 3), round(hi, 3))

    def com_range(
        self, jitter_m: float
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """Per-axis ``(lo, hi)`` CoM-shift bands centred on ``com_shift_m`` +/- jitter.

        IsaacLab's ``randomize_rigid_body_com`` event samples a per-axis offset from a
        ``com_range`` dict and ADDS it to the body's nominal CoM. To train carrying the
        rearward/elevated tank load, we centre that band on how far the payload actually
        shifts the combined CoM (``com_shift_m``, rearward -x + elevated +z) and jitter
        each axis by +/- ``jitter_m`` for domain-randomisation robustness. Returns three
        ``(lo, hi)`` tuples in (x, y, z) order, rounded to mm precision.
        """
        j = abs(float(jitter_m))
        return tuple(
            (round(c - j, 6), round(c + j, 6)) for c in self.com_shift_m
        )  # type: ignore[return-value]


def box_inertia(mass_kg: float, extents_m: Vec3) -> Vec3:
    """Solid-box principal inertia (diagonal) about the box centre.

    Matches ``o2_payload.isaac_mount._box_inertia`` so the trained dynamics line up
    with what the Isaac runtime mounts.
    """
    ex, ey, ez = (float(extents_m[0]), float(extents_m[1]), float(extents_m[2]))
    m = float(mass_kg)
    ixx = m / 12.0 * (ey * ey + ez * ez)
    iyy = m / 12.0 * (ex * ex + ez * ez)
    izz = m / 12.0 * (ex * ex + ey * ey)
    return (ixx, iyy, izz)


def load_payload_numbers() -> PayloadNumbers:
    """Import ``o2_payload.spec.SPEC`` (single source of truth) and reduce it.

    Raises ImportError (with the underlying cause) if the spec cannot be imported --
    callers in preflight catch this and report a red check rather than crashing.
    """
    sim_model_source.ensure_sim_on_path()
    from o2_payload.spec import SPEC  # noqa: E402  (path-shim import)

    mass = float(SPEC.total_payload_mass_kg)
    com = tuple(round(float(v), 6) for v in SPEC.payload_com_m)
    ext = tuple(round(float(v), 6) for v in SPEC.mounted_extents_m)
    return PayloadNumbers(
        mass_kg=round(mass, 6),
        com_m=com,                                   # type: ignore[arg-type]
        extents_m=ext,                               # type: ignore[arg-type]
        inertia_diag=tuple(round(v, 8) for v in box_inertia(mass, ext)),  # type: ignore[arg-type]
        trunk_mass_kg=round(float(SPEC.trunk_mass_kg), 6),
        com_shift_m=tuple(round(float(v), 6) for v in SPEC.com_shift_m()),  # type: ignore[arg-type]
        mass_fraction=round(float(SPEC.payload_mass_fraction), 4),
        orientation=str(SPEC.mount.orientation),
    )

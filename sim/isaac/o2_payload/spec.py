"""Single source of truth for the on-robot oxygen-concentrator payload.

This module is intentionally dependency-free (pure Python, no ``pxr`` / Isaac /
``numpy``) so it can be imported by:

  * the pure-Python asset generator (``build_assets.py``) that emits the ``.usda``
    visual models without Isaac Sim, and
  * the Isaac-Sim runtime modules (``isaac_mount.py`` / ``isaac_monitor.py``)
    that spawn the physics bodies and watch them.

ALL geometric values live here in SI units (metres, kilograms, radians). The
human-facing source values (inches / pounds) are kept alongside as comments and
converted with the exact factors below so the numbers are auditable.

------------------------------------------------------------------------------
Modelled hardware
------------------------------------------------------------------------------
We model the *mock-up* of the Rhythm Healthcare P2-E6 portable oxygen
concentrator that physically rides on this robot -- NOT the production P2-E6.
The developer measured the mock-up as:

    9.1 in (L) x 3.5 in (W) x 7.2 in (H),  4.6 lb

(The production spec sheet says 8.7 x 3.4 x 6.3 in / 4.37 lb; we deliberately do
NOT use those here -- the mock-up is what is actually on the robot.)

The 3D-printed tank holder / rail cradle adds 0.3 lb.

The tank slides on rails on the robot's back and must stay >= 3.3 in clear of the
XT16 LiDAR so the LiDAR cable is not pinched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

# ---------------------------------------------------------------------------
# Exact unit conversions
# ---------------------------------------------------------------------------
IN_TO_M: float = 0.0254          # 1 inch  -> metres   (exact)
LB_TO_KG: float = 0.45359237     # 1 pound -> kilograms (exact)
G: float = 9.80665               # standard gravity (m/s^2)


def inches(value_in: float) -> float:
    """Inches -> metres."""
    return value_in * IN_TO_M


def pounds(value_lb: float) -> float:
    """Pounds -> kilograms."""
    return value_lb * LB_TO_KG


Vec3 = Tuple[float, float, float]


# ---------------------------------------------------------------------------
# Concentrator (the oxygen tank that rides on the robot)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConcentratorSpec:
    """Mock-up Rhythm Healthcare P2-E6 portable oxygen concentrator."""

    # Outer bounding box. The robot's fore-aft axis is X, lateral is Y, up is Z.
    # The concentrator is mounted "tall", long axis (L) along the robot's X.
    length_m: float = inches(9.1)   # 9.1 in  -> 0.231140 m  (along robot +X)
    width_m: float = inches(3.5)    # 3.5 in  -> 0.088900 m  (along robot  Y)
    height_m: float = inches(7.2)   # 7.2 in  -> 0.182880 m  (along robot +Z)

    mass_kg: float = pounds(4.6)    # 4.6 lb  -> 2.086525 kg

    # Visual styling for the generated mesh (Rhythm Healthcare aesthetic).
    corner_radius_m: float = inches(0.55)   # rounded vertical edges
    body_color: Vec3 = (0.94, 0.95, 0.96)   # off-white shell
    base_color: Vec3 = (0.62, 0.65, 0.68)   # grey bottom trim strip
    accent_color: Vec3 = (0.95, 0.45, 0.12)  # Rhythm Healthcare orange logo block
    logo_grey: Vec3 = (0.50, 0.53, 0.57)    # logo wordmark block
    port_color: Vec3 = (0.62, 0.80, 0.90)   # light-blue cannula outlet bezel
    intake_color: Vec3 = (0.86, 0.88, 0.90)  # recessed circular intake cap

    @property
    def half_extents_m(self) -> Vec3:
        return (self.length_m / 2.0, self.width_m / 2.0, self.height_m / 2.0)

    @property
    def volume_m3(self) -> float:
        return self.length_m * self.width_m * self.height_m

    @property
    def bulk_density_kgpm3(self) -> float:
        """Effective average density (sanity check ~ a dense electronics box)."""
        return self.mass_kg / self.volume_m3


# ---------------------------------------------------------------------------
# Rail / cradle holder (3D-printed, bolts to the robot back)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RailSpec:
    """3D-printed adjustable rail cradle that clamps the concentrator down."""

    mass_kg: float = pounds(0.3)        # 0.3 lb -> 0.136078 kg (whole printed part)

    rail_thickness_m: float = inches(0.5)   # square cross-section of each side rail
    base_plate_thickness_m: float = inches(0.28)
    wall_height_m: float = inches(2.6)      # cradle side-wall height that hugs the tank
    upright_thickness_m: float = inches(0.4)
    # Side walls/rails sit just outside the tank footprint with a small print gap.
    side_gap_m: float = inches(0.12)
    # The base plate / rails extend a little beyond the tank footprint fore-aft.
    fore_aft_overhang_m: float = inches(0.45)

    rail_color: Vec3 = (0.18, 0.19, 0.22)   # matte dark printed PLA/PETG
    strap_color: Vec3 = (0.10, 0.10, 0.11)  # retaining strap across the top


# ---------------------------------------------------------------------------
# Mounting geometry -- where the payload sits in the robot trunk frame
# ---------------------------------------------------------------------------
UPRIGHT = "upright"
CROSSWISE = "crosswise"
FLAT = "flat"
_ORIENTATIONS = (UPRIGHT, CROSSWISE, FLAT)


@dataclass(frozen=True)
class MountSpec:
    """Placement of the payload relative to the Go2 trunk/base prim origin.

    The tank rides on the robot's back, behind the LiDAR. Its fore-aft position is
    *computed* (see ``O2PayloadSpec.cradle_base_x_m``) so the tank's nearest face
    is always exactly the LiDAR-cable clearance behind the LiDAR -- i.e. as far
    toward centre as the 3.3 in rule allows for the chosen orientation.
    """

    # XT16 LiDAR mount in the trunk frame (from sim_lidar_xt16.Xt16Config:
    # mount_x/y/z = 0,0,0.10) plus the small forward bias noted in isaac_env
    # (LiDAR centre ~ X = 0.05). Used purely for the clearance check.
    lidar_center_m: Vec3 = (0.05, 0.0, 0.10)

    # Required minimum horizontal clearance from the LiDAR (cable protection).
    lidar_clearance_min_m: float = inches(3.3)   # 3.3 in -> 0.083820 m

    # Trunk-local cradle base (top face of the robot back). X is computed from the
    # orientation + clearance; only Y (centred) and Z (back-plate height) are fixed.
    cradle_base_y_m: float = 0.0
    cradle_base_z_m: float = 0.075   # Go2 base top plate ~0.075 m above body origin

    # Mount ORIENTATION (concentrator is 9.1 L x 3.5 W x 7.2 H in). Sets which box
    # dimension points where -- (X fore-aft, Y lateral, Z up):
    #   * "upright"   tall, long axis fore-aft  -> (L, W, H): narrow (89 mm) across
    #                 the back, but rides high (CoM +40 mm) and far back (the 9.1 in
    #                 length is pinned at the LiDAR clearance).
    #   * "crosswise" (developer's choice) tall, long axis SIDE-TO-SIDE -> (W, L, H):
    #                 only 3.5 in deep fore-aft, so it sits ~70 mm CLOSER to centre
    #                 (CoM X -42 -> -24 mm, pitch torque ~halved). Trade-off: the
    #                 9.1 in length overhangs the robot's sides ~2 cm each; still
    #                 tall (CoM +40 mm).
    #   * "flat"      lies on its 9.1 x 7.2 in face -> (L, H, W): CoM ~47 mm lower
    #                 (steadiest on stairs) but 183 mm wide across the back.
    orientation: str = CROSSWISE


# ---------------------------------------------------------------------------
# Breakable attachment ("strap + quick-release clip" model)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrapSpec:
    """Breakable fixed joint that secures the tank to the rails.

    Normal locomotion (and even brisk turns / single stairs) keeps the joint
    well below threshold; a hard fall or collision spikes the reaction past it,
    the joint releases, and the tank tumbles free -- which the monitor reports.

    Sizing rationale (tank mass 2.087 kg):
      * static weight load                ~ 20.5 N
      * worst-case sustained locomotion   ~ 4 g  -> ~80 N
      * domain-randomization push impulse ~ m*dv/dt = 2.087*0.4/0.005 -> ~167 N
        (isaac_env shoves the base by ~0.4 m/s every 4 s by default), which a
        300 N strap could clip during a step + gait transient -- so the tank
        used to pop off "too easily". The threshold is set well above that
        combined disturbance but still below a genuine fall/slam impact.
    """

    # ~30x static weight: survives the per-step push impulse + foot-impact
    # transients of a full multi-step climb, yet still releases in a real crash
    # so the monitor's "tank fell off" detection stays meaningful.
    break_force_n: float = 600.0     # N   (~30x static weight; survives ~29 g)
    break_torque_nm: float = 120.0   # N.m (~37x the 3.27 N.m static pitch torque)


# ---------------------------------------------------------------------------
# Aggregate spec + derived robot-effect helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class O2PayloadSpec:
    concentrator: ConcentratorSpec = field(default_factory=ConcentratorSpec)
    rail: RailSpec = field(default_factory=RailSpec)
    mount: MountSpec = field(default_factory=MountSpec)
    strap: StrapSpec = field(default_factory=StrapSpec)

    # Go2 trunk reference (from assets/go2.../payloads/Physics/physics.usda).
    # Used only to *report* how the payload shifts the combined centre of mass.
    trunk_mass_kg: float = 6.921
    trunk_com_m: Vec3 = (0.021112, 0.0, -0.005366)

    # ---- mounted bounding-box extents (depend on orientation) ----
    @property
    def mounted_extents_m(self) -> Vec3:
        """Tank bounding-box extents in the trunk frame -- (X fore-aft, Y lateral,
        Z up) -- for the chosen mount orientation (see MountSpec.orientation):

        * "upright"   -> (L, W, H): stands tall, long axis fore-aft.
        * "crosswise" -> (W, L, H): stands tall, long axis side-to-side (only 3.5 in
                         deep fore-aft, so it sits closer to centre).
        * "flat"      -> (L, H, W): lies on its large face, short W vertical (lowest
                         CoM).
        """
        c = self.concentrator
        o = self.mount.orientation
        if o == FLAT:
            return (c.length_m, c.height_m, c.width_m)
        if o == CROSSWISE:
            return (c.width_m, c.length_m, c.height_m)
        return (c.length_m, c.width_m, c.height_m)  # upright

    # ---- placement of each rigid piece in the trunk frame ----
    @property
    def cradle_base_x_m(self) -> float:
        """Fore-aft cradle position: the forward-most spot that still keeps the
        tank's nearest face >= the LiDAR-cable clearance behind the LiDAR. A
        smaller fore-aft extent (e.g. the crosswise orientation, 3.5 in deep)
        therefore sits closer to the robot's centre."""
        fa = self.mounted_extents_m[0]
        m = self.mount
        return m.lidar_center_m[0] - m.lidar_clearance_min_m - fa / 2.0

    @property
    def lidar_clearance_actual_m(self) -> float:
        """Clearance from the LiDAR centre to the tank's nearest (front) face --
        the real space the LiDAR cable has. Edge-based (accounts for the tank's
        fore-aft extent), so it can never silently let the tank overlap the LiDAR."""
        fa = self.mounted_extents_m[0]
        front_edge_x = self.cradle_base_x_m + fa / 2.0
        return self.mount.lidar_center_m[0] - front_edge_x

    @property
    def tank_center_m(self) -> Vec3:
        """Tank centre: cradle base + half the tank's VERTICAL extent (it rests in
        the cradle, so the centre height follows the mount orientation)."""
        m = self.mount
        return (
            self.cradle_base_x_m,
            m.cradle_base_y_m,
            m.cradle_base_z_m + self.mounted_extents_m[2] / 2.0,
        )

    @property
    def holder_center_m(self) -> Vec3:
        """Holder/cradle reference point (its base plate, on the trunk top)."""
        m = self.mount
        return (self.cradle_base_x_m, m.cradle_base_y_m, m.cradle_base_z_m)

    # ---- aggregate payload mass ----
    @property
    def total_payload_mass_kg(self) -> float:
        return self.concentrator.mass_kg + self.rail.mass_kg

    @property
    def payload_com_m(self) -> Vec3:
        """Centre of mass of (tank + holder), trunk frame."""
        mt, mh = self.concentrator.mass_kg, self.rail.mass_kg
        ct, ch = self.tank_center_m, self.holder_center_m
        total = mt + mh
        return tuple((mt * ct[i] + mh * ch[i]) / total for i in range(3))  # type: ignore[return-value]

    # ---- effect of the payload on the robot ----
    def combined_com_m(self, *, tank_attached: bool = True) -> Vec3:
        """Combined trunk+payload centre of mass in the trunk frame.

        When ``tank_attached`` is False (tank fell off) only the rail/holder
        mass remains bolted to the robot.
        """
        masses = [self.trunk_mass_kg, self.rail.mass_kg]
        coms = [self.trunk_com_m, self.holder_center_m]
        if tank_attached:
            masses.append(self.concentrator.mass_kg)
            coms.append(self.tank_center_m)
        total = sum(masses)
        return tuple(
            sum(m * c[i] for m, c in zip(masses, coms)) / total for i in range(3)
        )  # type: ignore[return-value]

    def com_shift_m(self, *, tank_attached: bool = True) -> Vec3:
        """How far the payload moves the combined CoM vs. the bare trunk."""
        base = self.trunk_com_m
        new = self.combined_com_m(tank_attached=tank_attached)
        return tuple(new[i] - base[i] for i in range(3))  # type: ignore[return-value]

    @property
    def payload_weight_n(self) -> float:
        """Downward weight the robot must carry while the tank is attached."""
        return self.total_payload_mass_kg * G

    @property
    def pitch_torque_nm(self) -> float:
        """Static pitch torque about the trunk origin from the offset payload.

        Negative X (rearward) load => nose-up / rear-down pitching tendency,
        which raises tip-over risk during a climb. Reported as a magnitude.
        """
        com = self.payload_com_m
        return abs(self.total_payload_mass_kg * G * com[0])

    @property
    def payload_mass_fraction(self) -> float:
        """Payload mass as a fraction of the bare trunk mass."""
        return self.total_payload_mass_kg / self.trunk_mass_kg


def validate(spec: O2PayloadSpec | None = None) -> O2PayloadSpec:
    """Sanity-check the spec and return it. Raises AssertionError on a mistake."""
    s = spec or O2PayloadSpec()

    # Source measurements round-trip to the documented metric values.
    assert abs(s.concentrator.length_m - 0.231140) < 1e-6
    assert abs(s.concentrator.width_m - 0.088900) < 1e-6
    assert abs(s.concentrator.height_m - 0.182880) < 1e-6
    assert abs(s.concentrator.mass_kg - 2.086525) < 1e-6
    assert abs(s.rail.mass_kg - 0.136078) < 1e-6

    # Orientation must be one we know how to build.
    assert s.mount.orientation in _ORIENTATIONS, (
        f"Unknown mount orientation {s.mount.orientation!r}; expected one of {_ORIENTATIONS}"
    )

    # LiDAR clearance requirement is satisfied by the chosen placement. The
    # placement is computed to sit exactly at the limit, so this holds by
    # construction (with float slack) for every orientation.
    assert s.lidar_clearance_actual_m >= s.mount.lidar_clearance_min_m - 1e-9, (
        f"Tank too close to LiDAR: {s.lidar_clearance_actual_m*1000:.0f} mm "
        f"< required {s.mount.lidar_clearance_min_m*1000:.0f} mm"
    )

    # Density should look like a dense electronics box, not foam or lead.
    assert 400.0 < s.concentrator.bulk_density_kgpm3 < 1500.0
    return s


SPEC: O2PayloadSpec = validate()


if __name__ == "__main__":
    s = SPEC
    c = s.concentrator
    print("=== O2 payload spec (mock-up Rhythm Healthcare P2-E6) ===")
    print(f"tank  : {c.length_m*1000:.1f} x {c.width_m*1000:.1f} x {c.height_m*1000:.1f} mm"
          f"  ({c.length_m/IN_TO_M:.1f} x {c.width_m/IN_TO_M:.1f} x {c.height_m/IN_TO_M:.1f} in)")
    ext = s.mounted_extents_m
    print(f"mount : {s.mount.orientation.upper()}"
          f"  -> trunk-frame extents X(fore-aft) x Y(lateral) x Z(up) = "
          f"{ext[0]*1000:.0f} x {ext[1]*1000:.0f} x {ext[2]*1000:.0f} mm")
    print(f"tank  : {c.mass_kg:.3f} kg ({c.mass_kg/LB_TO_KG:.1f} lb), "
          f"bulk density {c.bulk_density_kgpm3:.0f} kg/m^3")
    print(f"holder: {s.rail.mass_kg:.3f} kg ({s.rail.mass_kg/LB_TO_KG:.1f} lb)")
    print(f"payload total: {s.total_payload_mass_kg:.3f} kg "
          f"({s.payload_mass_fraction*100:.1f}% of trunk)")
    print(f"tank centre (trunk frame): {tuple(round(v,4) for v in s.tank_center_m)}")
    print(f"cradle base x (computed): {s.cradle_base_x_m:.4f} m")
    print(f"LiDAR edge clearance: {s.lidar_clearance_actual_m*1000:.0f} mm "
          f"(required >= {s.mount.lidar_clearance_min_m*1000:.0f} mm)")
    print(f"payload CoM (trunk frame): {tuple(round(v,4) for v in s.payload_com_m)}")
    print(f"combined CoM shift (attached): "
          f"{tuple(round(v*1000,1) for v in s.com_shift_m())} mm")
    print(f"payload weight: {s.payload_weight_n:.1f} N, "
          f"static pitch torque: {s.pitch_torque_nm:.2f} N.m")
    print(f"strap break: {s.strap.break_force_n:.0f} N / {s.strap.break_torque_nm:.0f} N.m")

"""Tuning constants for the person-following controller.

These module-level constants are read-only tuning parameters consumed by the
PersonFollower controller's lost-target recovery, gap-rate clamp, and
zone-transition gating logic. They are defined here once and imported where
needed; they are never reassigned at runtime.
"""

# How long the LiDAR-bearing bridge may keep turning the dog toward a patient the 69 deg RGB
# camera has lost but the 360 deg LiDAR still tracks. Bounded so a lock onto a moving
# distractor cannot spin the dog forever; YOLO normally re-acquires well within this.
_LIDAR_BRIDGE_MAX_SEC = 12.0
# Proportional yaw gain (rad/s per rad of bearing) used while the LiDAR bridge tracks a live
# bearing: a 20 deg off-axis patient -> ~0.35 rad/s, a 45 deg -> ~0.79 rad/s, capped at the
# tracking yaw limit. Far more responsive than the slow fixed blind-search speed.
_LIDAR_BRIDGE_YAW_GAIN = 1.0
# When the last bbox glimpse was within this bearing of the axis but lateral MOTION points the
# other way, the patient was crossing/reversing (a zigzag apex) -- trust the motion (where they
# are heading), not the stale last-seen side.
_REVERSAL_BEARING_DEG = 20.0
# Seconds to sweep ONE +/- arc leg of the in-place re-acquire scan. The scan rate is derived
# from this and lost_search_arc_deg so the scan is brisk (reaches the arc in ~this long) rather
# than the slow fixed lost_search_yaw_speed used for a one-frame-glimpse correction.
_LOST_SCAN_LEG_SEC = 2.5

# Max plausible patient walking speed (m/s) used to clamp a single frame's gap jump:
# at the variable ~283 ms follow loop, the fused gap cannot physically move faster
# than this between accepted frames, so a larger jump is a fusion glitch, not motion.
_MAX_PERSON_SPEED_MPS = 1.5
# Minimum confidence a single fused frame needs to CHANGE the stop/brake zone. Below
# this the zone only changes after two consecutive frames agree, killing phantom
# stops/cruises from one low-confidence (disagreeing) frame.
_ZONE_CHANGE_MIN_CONFIDENCE = 0.5

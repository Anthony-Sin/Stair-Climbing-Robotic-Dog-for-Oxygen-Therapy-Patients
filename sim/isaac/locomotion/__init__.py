"""Go2 locomotion controllers and the shared articulation helpers they build on.

Members:
  - go2_locomotion_utils    shared joint maps, PD torque law, default-pose tables
  - pgtt_locomotion_policy   default walker (PGTT phase-guided heightmap MLP)
  - pgtt_policy_net          torch loader/runner for the converted PGTT .npz net
  - pgtt_heightmap           body-aligned heightmap sensor for the PGTT policy
  - parkour_locomotion_policy Extreme-Parkour depth/vision RL policy + parkour climb backend
  - parkour_depth_backbone   vendored depth-encoder nn.Modules for the parkour policy
  - rl_locomotion_policy     rl_sar proprioceptive RL net -- the `blind_rl` climb backend
  - closed_loop_stair_climber deterministic IK + balance FSM climber (`ik` backend)
  - scripted_stair_gait      legacy open-loop scripted stair gait
  - pgtt_stair_handoff       dual-policy walk<->climb handoff FSM

This package is a `sys.path`-root subpackage: callers add ``sim/isaac`` to
``sys.path`` and import e.g. ``from locomotion.pgtt_locomotion_policy import ...``.
"""

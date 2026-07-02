"""One-off asset/engine preparation scripts. Not imported at runtime.

Members:
  - go2_usd_setup      downloads the Go2 URDF and converts it to USD (run via Isaac python)
  - build_reid_engine  builds the OSNet ReID TensorRT engine (run inside the robot Docker)

These are standalone scripts, not part of the simulation import graph.
"""

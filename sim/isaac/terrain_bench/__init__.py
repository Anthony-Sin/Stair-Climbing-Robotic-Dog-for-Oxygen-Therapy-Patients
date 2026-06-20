"""terrain_bench -- multi-terrain locomotion benchmark for the Go2.

A boot-once test harness: it drives the frozen parkour policy across a battery of
varied terrains (flat, ramps, stair presets) inside ONE warm Isaac Kit, records the
existing headless videos per terrain, and rolls the per-terrain results into
perf_tracker. Docker / vision / TensorRT are never involved.

Two halves:
  * ``terrain_registry`` -- pure-Python terrain catalogue + ``build_terrain``. The
    module top imports nothing from Isaac, so it is importable both inside Kit
    (to spawn prims) and host-side via plain ``python3`` (the launcher reads the
    battery with ``--emit-json``).
  * ``bench_metrics`` -- host-side aggregator that reuses perf_tracker.

The Kit-side wiring (a per-episode terrain in the warm command file + a Docker-free
drive) lives behind isaac_env.py's ``--bench`` flag; the default one-shot run is
unaffected.
"""

"""Real-hardware telemetry that re-emits the sim's run-dir schema.

The sim's perf_tracker parses a run folder (logs/status.jsonl, debug/isaac_env.jsonl
fall_diag stream, reports/stair_demo_report.json). These writers produce the SAME
layout sourced from real ROS 2 data, so ``perf_tracker.update_table.extract_metrics``
+ ``record_run`` ingest a real run UNCHANGED -- the leaderboard, classification, and
charts all work without a sim-specific code path.
"""

"""Standalone pre-run verification + real-run ingestion.

``preflight`` runs the offline-checkable sanity checks (weights load, CRC works,
default pose within joint limits) before any motion; the live-topic checks
(/lowstate fresh, sport released, camera/LiDAR alive) are enforced by the control
node's startup gate. ``extract_real_run`` ingests a finished real run through the
sim's public perf_tracker API.
"""

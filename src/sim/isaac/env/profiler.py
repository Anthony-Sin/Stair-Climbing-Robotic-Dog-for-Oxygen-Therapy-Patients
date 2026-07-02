"""isaac_env.py extraction (Phase 2 split): profiler. Verbatim bodies; only env_state requalification added."""
import logging
import time

# ---------------------------------------------------------------------------
# Step-phase profiler: accumulate per-phase wall-milliseconds over the Isaac step
# loop and log ONE summary line every ~200 steps. The sim runs at RTF ~0.117, so
# this pins down which subsystem (physics / GPU render readback / recorder encode /
# parkour depth / LiDAR raycast / frame publish) eats the wall time. Cheap: a couple
# of perf_counter() reads per phase, no allocation on the hot path.
# ---------------------------------------------------------------------------
_PROFILE_PHASES = (
    "physics", "render_readback", "recorder", "parkour_depth", "lidar", "publisher", "other"
)

class _StepProfiler:
    """Accumulate per-phase wall-time and emit a mean-ms/%-of-loop summary periodically.

    Usage per step::

        prof.step_begin()
        with prof.phase("physics"):
            world.step(...)
        ...
        prof.step_end(sim_dt_this_step)

    ``phase()`` is a context manager; a phase that never runs on a given step simply
    contributes 0 that step. ``step_end`` folds the un-attributed remainder of the
    loop into ``other`` and, every ``interval`` steps, logs the summary and resets.
    """

    def __init__(self, logger, log_event_fn, *, interval: int = 200, enabled: bool = True):
        self._logger = logger
        self._log_event = log_event_fn
        self._interval = max(1, int(interval))
        self.enabled = bool(enabled)
        self._accum = {p: 0.0 for p in _PROFILE_PHASES}
        self._steps = 0
        self._loop_accum = 0.0          # summed full-step wall time (window)
        self._sim_accum = 0.0           # summed sim-time advanced (window), for RTF
        self._step_t0 = 0.0
        self._attributed = 0.0          # phase time attributed within the current step

    class _Ctx:
        __slots__ = ("_prof", "_key", "_t0")

        def __init__(self, prof, key):
            self._prof = prof
            self._key = key
            self._t0 = 0.0

        def __enter__(self):
            if self._prof.enabled:
                self._t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            if self._prof.enabled:
                dt = time.perf_counter() - self._t0
                self._prof._accum[self._key] += dt
                self._prof._attributed += dt
            return False

    def phase(self, key: str):
        return _StepProfiler._Ctx(self, key)

    def step_begin(self) -> None:
        if not self.enabled:
            return
        self._step_t0 = time.perf_counter()
        self._attributed = 0.0

    def step_end(self, sim_dt: float = 0.0) -> None:
        if not self.enabled:
            return
        loop = time.perf_counter() - self._step_t0
        # Un-attributed remainder of the loop (command read, telemetry, evaluation,
        # scene motion, etc.) folds into "other" -- never negative.
        self._accum["other"] += max(0.0, loop - self._attributed)
        self._loop_accum += loop
        self._sim_accum += max(0.0, float(sim_dt))
        self._steps += 1
        if self._steps >= self._interval:
            self._emit()
            self._reset()

    def _emit(self) -> None:
        n = max(1, self._steps)
        loop_ms = (self._loop_accum / n) * 1000.0
        means_ms = {p: (self._accum[p] / n) * 1000.0 for p in _PROFILE_PHASES}
        denom = self._loop_accum if self._loop_accum > 1e-9 else 1e-9
        pcts = {p: round(100.0 * self._accum[p] / denom, 1) for p in _PROFILE_PHASES}
        # Measured RTF over the window (sim seconds advanced / wall seconds spent).
        rtf = None
        if self._sim_accum > 0.0 and self._loop_accum > 1e-9:
            rtf = round(self._sim_accum / self._loop_accum, 4)
        fields = {"steps": int(self._steps), "loop_ms": round(loop_ms, 2), "rtf": rtf}
        for p in _PROFILE_PHASES:
            fields[f"{p}_ms"] = round(means_ms[p], 2)
            fields[f"{p}_pct"] = pcts[p]
        try:
            self._log_event(self._logger, logging.INFO, "step_profile",
                            "Per-phase step timing (mean ms and pct of loop over window)", **fields)
        except Exception:
            pass

    def _reset(self) -> None:
        for p in _PROFILE_PHASES:
            self._accum[p] = 0.0
        self._steps = 0
        self._loop_accum = 0.0
        self._sim_accum = 0.0

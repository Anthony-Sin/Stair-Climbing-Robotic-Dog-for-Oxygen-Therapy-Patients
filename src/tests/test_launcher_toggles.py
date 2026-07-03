"""§9 launcher placebo-toggle test: every launcher Option actually changes the child command.

A "placebo toggle" is a launcher menu option the user can flip that has NO effect on what
actually gets run -- e.g. the O2-payload toggle that (per the review) did nothing. This test
drives the REAL command-assembly path (``launcher_lib.config.build_command``) and asserts that
flipping each Option off its default changes the resulting child command SIGNATURE -- either
the argv passed to the child process OR the environment overlay ``build_command`` sets on top
of ``os.environ`` (``vision_preview`` toggles ``SHOW_RECORDINGS``, not argv).

Host-safe: only builds arg lists; never spawns the child (no Isaac / Docker / bash).
"""
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from launcher_lib.config import _real_config, _sim_config, build_command  # noqa: E402


def _signature(cfg):
    """The observable child-command signature: (argv tuple, sorted env OVERLAY items).

    The env overlay is what build_command adds/changes ON TOP of the ambient os.environ --
    isolating the toggle's effect from the inherited environment (which is huge and noisy).
    """
    display, argv, cwd, env, supported = build_command(cfg)
    overlay = tuple(sorted(
        (k, v) for k, v in env.items()
        if os.environ.get(k) != v  # only keys build_command set/changed
    ))
    return tuple(argv), overlay


def _toggle_to_change(opt):
    """Flip ``opt`` off its current value, returning True if the value actually changed.

    Bools flip; choices/numerics cycle +1, falling back to -1 if +1 was a no-op (already at a
    boundary). Returns False only if neither direction moves the value (e.g. a 1-choice list).
    """
    before = opt.value
    opt.cycle(1)
    if opt.value != before:
        return True
    opt.cycle(-1)
    return opt.value != before


def _configs():
    return [("sim", _sim_config), ("real", _real_config)]


def _visible_option_ids():
    ids = []
    for target, factory in _configs():
        cfg = factory()
        for opt in cfg.visible():
            ids.append((target, opt.key))
    return ids


@pytest.mark.parametrize("target,opt_key", _visible_option_ids())
def test_toggling_option_changes_child_command(target, opt_key):
    factory = dict(_configs())[target]

    # Fresh config so each option is toggled from the pristine default state.
    cfg = factory()
    opt = cfg.get(opt_key)
    # The option must be visible in this pristine config to be user-togglable here.
    if opt not in cfg.visible():
        pytest.skip(f"{target}:{opt_key} not visible in the default config")

    baseline = _signature(cfg)

    changed_value = _toggle_to_change(opt)
    assert changed_value, (
        f"{target}:{opt_key} value could not be toggled at all (cycle is a no-op) -- "
        "the menu option is inert."
    )

    after = _signature(cfg)
    assert after != baseline, (
        f"PLACEBO TOGGLE: flipping {target}:{opt_key} (value {baseline!r} -> {opt.value!r}) did "
        f"NOT change the child command signature. argv and env overlay are identical, so the "
        f"launcher menu lies to the user about this option (the O2-payload-does-nothing class).\n"
        f"  baseline argv: {baseline[0]}\n  baseline env overlay: {baseline[1]}\n"
        f"  toggled  argv: {after[0]}\n  toggled  env overlay: {after[1]}"
    )


def test_o2_payload_toggle_is_not_placebo():
    """Explicit regression pin for the named case: the sim O2-payload toggle must add its flag
    to the child argv (this is the toggle the review called out as doing nothing)."""
    cfg = _sim_config()
    before_argv, _ = _signature(cfg)
    o2 = cfg.get("o2")
    assert _toggle_to_change(o2), "o2 option would not toggle"
    after_argv, _ = _signature(cfg)
    added = [a for a in after_argv if a not in before_argv]
    assert added == [o2.flag], (
        f"O2 payload toggle should add exactly {o2.flag!r} to argv, got delta {added!r}"
    )


def test_discovered_enough_options():
    """Sanity: the parametrization is non-empty (a silently-empty option list would make the
    placebo test pass vacuously)."""
    ids = _visible_option_ids()
    assert len(ids) >= 5, f"expected several launcher options, found {len(ids)}: {ids}"

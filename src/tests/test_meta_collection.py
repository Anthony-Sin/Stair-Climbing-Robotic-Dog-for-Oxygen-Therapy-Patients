"""Meta-test: every ``test_*.py`` in this suite must expose >= 1 collectable item.

WHY (the invisibility class this guards against): ``test_parkour_contract.py`` shipped
with its checks named ``_test_*`` behind a ``__main__`` runner. It passed when run by
hand (~80 s) yet pytest collected ZERO items from it -- so it was green-but-unrun in CI
and any regression it "covered" was actually uncovered. This meta-test makes that failure
mode loud: a test file that collects nothing (leading-underscore names, a bare
``if __name__ == '__main__'`` runner with no ``test_*`` functions, an all-skipped file
that is never even importable, etc.) fails HERE with the offending path named.

Implementation: run ``pytest --collect-only`` in a SUBPROCESS per file. This is the exact
mechanism CI uses, so it catches everything real collection catches (import errors,
naming, markers) without importing the target into this interpreter (which could pollute
state or crash on a heavy import). A file whose items are ALL environment-skipped still
"collects" fine -- importorskip / skipif produce collected-then-skipped items, not zero
items -- so this does not punish host-unsafe files that degrade cleanly.

Host-safe: no torch / Isaac / hardware needed; it only shells out to pytest collection.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Files this meta-test must not recurse into itself (avoid a fork bomb: collecting THIS
# file spawns a subprocess that would collect this file again...). Also skip the shared
# conftest, which is not a test module.
_SELF = os.path.basename(__file__)


def _env_ignored_files():
    """The set of test files conftest.py ignores on THIS host (Isaac/GPU-only files whose
    runtime modules are absent). conftest's ``collect_ignore`` only applies to directory
    collection, NOT to a file named explicitly on the command line -- so when we shell out
    ``pytest <file>`` we'd otherwise hit their import errors. We honor the same list here so
    the meta-test stays host-safe and only checks files this host can actually collect."""
    try:
        spec = importlib.util.spec_from_file_location(
            "_meta_conftest", os.path.join(_TESTS_DIR, "conftest.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return set(getattr(mod, "collect_ignore", []) or [])
    except Exception:
        return set()


def _test_files():
    ignored = _env_ignored_files()
    return sorted(
        f
        for f in os.listdir(_TESTS_DIR)
        if f.startswith("test_") and f.endswith(".py")
        and f != _SELF and f not in ignored
    )


def _collect_count(path: str) -> tuple[int, str]:
    """Return (num_items_collected, combined_output) for one test file.

    Uses ``--collect-only -q``; the final summary line reads e.g. ``12 tests collected``
    or ``no tests ran``. We parse the machine-stable ``collected`` count pytest prints,
    falling back to counting ``::`` node lines. A collection ERROR (import failure) is
    surfaced as a negative count so the caller can distinguish "0 items" from "errored".
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", os.path.join(_TESTS_DIR, path),
         "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=os.path.dirname(_TESTS_DIR),
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Count node ids (lines containing "::"); robust across pytest versions and to the
    # trailing "N tests collected" / "no tests ran" summary wording.
    node_lines = [ln for ln in out.splitlines() if "::" in ln]
    n = len(node_lines)
    # A hard collection error (bad import) shows "errors" in the summary while emitting
    # zero node lines -- flag it distinctly (-1) rather than reporting a plain 0.
    if n == 0 and ("error" in out.lower() and "collected" not in out.lower()):
        return -1, out
    return n, out


@pytest.mark.parametrize("test_file", _test_files())
def test_file_collects_at_least_one_item(test_file):
    n, out = _collect_count(test_file)
    assert n != -1, (
        f"{test_file}: pytest hit a COLLECTION ERROR (import failure) -- it collects "
        f"nothing and would sit green-but-unrun. Fix the import or add a clean skip guard.\n"
        f"----- pytest --collect-only output -----\n{out}"
    )
    assert n >= 1, (
        f"{test_file}: pytest collected ZERO items. A test file with no `test_*`-named "
        f"function/method is invisible to CI (the `_test_*` + `__main__`-runner trap that "
        f"hid test_parkour_contract.py). Rename its checks to `test_*`.\n"
        f"----- pytest --collect-only output -----\n{out}"
    )


def test_meta_discovers_files():
    """Sanity: the discovery glob itself finds the known-good files (guards against a
    silently-empty parametrization that would make this whole meta-test pass vacuously)."""
    files = _test_files()
    assert len(files) >= 10, f"expected many test_*.py files, found {len(files)}: {files}"
    assert "test_meta_collection.py" not in files, "meta-test must not recurse into itself"

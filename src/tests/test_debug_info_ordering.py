"""Static ``debug_info`` first-write-vs-read ordering contract for the frame loop.

Incident 8.5 (from CLAUDE.md): ``debug_info`` is rebuilt every frame and filled in CALL
ORDER. A consumer that ``.get()``s a key its producer writes LATER in the same frame
silently receives the DEFAULT -- a dead gate. This has killed safety gates three times.
Unit tests hide it because they hand-build the dict in the "right" order.

This test parses ``src/core/main.py`` with ``ast`` (current content, at test time -- so it
tracks edits by sibling agents) and FAILS on any literal-keyed read whose EARLIEST producer
is strictly downstream of the read.

Producer model (kept conservative to avoid false positives -- ``debug_info`` is passed by
reference into helper functions that populate it, AND is re-bound from helper return values):
  * A literal write ``debug_info["k"] = ...`` produces key ``k`` at its line.
  * ``debug_info.update(...)`` produces an UNKNOWN set of keys -> a BARRIER (it may produce
    any key) at its line.
  * Any call passing ``debug_info`` as an argument (e.g. ``_apply_stair_command_policy(...,
    debug_info)``) may populate keys inside the callee -> a BARRIER at its line.
  * ``... , debug_info = person_follower.update(...)`` -- re-BINDING ``debug_info`` from a
    call's return value produces an unknown set of keys -> a BARRIER at its line. (The follow
    dict is built entirely inside the follower and returned here.)

A read of ``k`` is flagged ONLY when every producer of ``k`` -- its first literal write AND
every barrier -- lies strictly AFTER the read line (i.e. nothing could have populated it
yet). That is exactly the dead-gate shape; a key populated by an earlier helper call or
``.update`` is correctly NOT flagged.

Reads of keys NEVER written literally in main.py and with no barrier before them are treated
as external/default reads (the caller seeded them, or they legitimately default) and are NOT
flagged -- unless a downstream literal write exists, which is the dead-gate case.
"""
import ast
import os

import pytest

_MAIN = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "core", "main.py")
)

# Intentional cross-frame reads: keys deliberately read this frame that were produced on a
# PREVIOUS frame (carried in a local, not re-read from a downstream write). Seed minimally.
# Each entry needs a comment explaining WHY it is a legitimate cross-frame read, not a bug.
_CROSS_FRAME_WHITELIST: set[str] = {
    # (none required today: the producer model already absorbs helper-call population, so
    #  main.py currently has zero first-write-downstream reads. Add a key here -- with a
    #  reason -- only if a genuine previous-frame read is introduced.)
}


def _analyze(src: str):
    """Return (first_literal_write: {key: line}, barrier_lines: sorted[int], reads: [(key,line)])."""
    tree = ast.parse(src)
    first_write: dict[str, int] = {}
    barriers: list[int] = []
    reads: list[tuple[str, int]] = []

    def _binds_debug_info(target) -> bool:
        """True if ``target`` (an assignment target) binds the whole ``debug_info`` name,
        directly (``debug_info = ...``) or inside a tuple unpack
        (``a, debug_info = f()``)."""
        if isinstance(target, ast.Name):
            return target.id == "debug_info"
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(isinstance(e, ast.Name) and e.id == "debug_info" for e in target.elts)
        return False

    class V(ast.NodeVisitor):
        def visit_Assign(self, node):
            rhs_is_call = isinstance(node.value, ast.Call)
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                        and t.value.id == "debug_info"
                        and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)):
                    k = t.slice.value
                    first_write[k] = min(node.lineno, first_write.get(k, 10 ** 9))
                # Re-binding the whole debug_info name from a CALL return (the follower builds
                # and returns the dict) can produce an unknown key set -> barrier.
                elif rhs_is_call and _binds_debug_info(t):
                    barriers.append(node.lineno)
            self.generic_visit(node)

        def visit_Call(self, node):
            f = node.func
            # reads: debug_info.get("k"[, default])
            if (isinstance(f, ast.Attribute) and f.attr == "get"
                    and isinstance(f.value, ast.Name) and f.value.id == "debug_info"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                reads.append((node.args[0].value, node.lineno))
            # debug_info.update(...) -> barrier (produces an unknown set of keys)
            if (isinstance(f, ast.Attribute) and f.attr == "update"
                    and isinstance(f.value, ast.Name) and f.value.id == "debug_info"):
                barriers.append(node.lineno)
            # debug_info passed as an argument to a helper -> barrier (callee may populate it)
            passes_di = any(
                isinstance(a, ast.Name) and a.id == "debug_info" for a in node.args
            ) or any(
                isinstance(kw.value, ast.Name) and kw.value.id == "debug_info"
                for kw in node.keywords
            )
            if passes_di:
                barriers.append(node.lineno)
            self.generic_visit(node)

        def visit_Subscript(self, node):
            # reads: debug_info["k"] in a load context
            if (isinstance(node.value, ast.Name) and node.value.id == "debug_info"
                    and isinstance(node.ctx, ast.Load)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)):
                reads.append((node.slice.value, node.lineno))
            self.generic_visit(node)

    V().visit(tree)
    return first_write, sorted(set(barriers)), reads


def _find_violations(src: str):
    first_write, barriers, reads = _analyze(src)
    violations = []
    for key, read_line in reads:
        if key in _CROSS_FRAME_WHITELIST:
            continue
        producers = list(barriers)
        if key in first_write:
            producers.append(first_write[key])
        if not producers:
            continue  # never written literally, no barrier -> external/default read, fine
        # Flagged only when NOTHING could have produced the key at or before the read line.
        if not any(p <= read_line for p in producers):
            earliest = min(producers)
            violations.append((key, read_line, earliest))
    violations.sort(key=lambda v: v[1])
    return violations


def test_main_py_is_analyzable():
    assert os.path.exists(_MAIN), f"main.py not found at {_MAIN}"
    with open(_MAIN, encoding="utf-8") as f:
        src = f.read()
    # Must parse and expose the mechanism we are guarding (sanity: it uses debug_info).
    first_write, barriers, reads = _analyze(src)
    assert reads, "no debug_info reads found in main.py -- parser or file changed shape"
    assert first_write, "no literal debug_info writes found in main.py -- unexpected"


def test_no_debug_info_read_before_first_write():
    with open(_MAIN, encoding="utf-8") as f:
        src = f.read()
    violations = _find_violations(src)
    if violations:
        lines = "\n".join(
            f"  debug_info[{k!r}] READ at main.py:{rl} but earliest producer is "
            f"main.py:{wl} (downstream) -- dead gate (incident 8.5)."
            for k, rl, wl in violations
        )
        pytest.fail(
            "debug_info keys read before anything could have produced them this frame "
            "(incident-8.5 dead gate):\n" + lines +
            "\n\nFix: move the producer above the reader, pass the value as an argument, or "
            "-- if it is an intentional previous-frame read -- add the key to "
            "_CROSS_FRAME_WHITELIST with a reason."
        )

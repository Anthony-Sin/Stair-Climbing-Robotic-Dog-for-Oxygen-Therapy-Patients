"""§9 config-contract tests: cross-process defaults agree, and no dead CLI flags.

Two independent guards over ``src/core/args_parser.py``:

(a) CONTRACT AGREEMENT -- ``src/shared/config_contract.py`` is the single source of truth for
    the handful of knobs that MUST match numerically across the separately-launched
    processes (core controller, sim isaac_env, real ROS 2 sidecars). This asserts the CORE
    argparse defaults equal the canonical contract values, so the two ends can't silently
    drift (e.g. controller standoff 0.6 m while the sim spawns the patient at 1.5 m).

(b) NO DEAD FLAGS -- every ``add_argument`` flag defined in the core parser must have >= 1
    consumer somewhere under ``src/`` (its destination name is read via ``args.<dest>`` or
    ``getattr(args, "<dest>", ...)``). This is how the review found the dead
    ``--stair-follow-bearing-scale`` (a core copy consumed by nothing; the sim used its own).

Host-safe: parses source with ``ast`` and scans files as text -- no argparse invocation
(which would consume sys.argv), no torch / Isaac / hardware.
"""
import ast
import os

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
_ARGS_PARSER = os.path.join(_SRC, "core", "args_parser.py")


# ---------------------------------------------------------------------------
# Parse the core argparse defaults out of args_parser.py (statically, no execution).
# ---------------------------------------------------------------------------
def _parse_add_arguments():
    """Return a list of dicts: {flag, dest, default, is_store, lineno} for every
    ``add_argument`` call in the core parser."""
    with open(_ARGS_PARSER, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = []

    class V(ast.NodeVisitor):
        def visit_Call(self, node):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "add_argument":
                optstrings = [
                    a.value for a in node.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)
                ]
                kw = {k.arg: k.value for k in node.keywords}
                dest = None
                if "dest" in kw and isinstance(kw["dest"], ast.Constant):
                    dest = kw["dest"].value
                action = None
                if "action" in kw and isinstance(kw["action"], ast.Constant):
                    action = kw["action"].value
                default = _UNSET = object()
                # If the default is sourced straight from the contract, i.e.
                # ``default=contract_value("some_key")``, record that key -- this is the
                # STRONGEST form of agreement (the value literally IS the contract value).
                default_contract_key = None
                if "default" in kw:
                    dnode = kw["default"]
                    if (isinstance(dnode, ast.Call) and isinstance(dnode.func, ast.Name)
                            and dnode.func.id == "contract_value" and dnode.args
                            and isinstance(dnode.args[0], ast.Constant)
                            and isinstance(dnode.args[0].value, str)):
                        default_contract_key = dnode.args[0].value
                        default = "<contract_value>"
                    else:
                        try:
                            default = ast.literal_eval(dnode)
                        except Exception:
                            default = "<expr>"
                longflags = [o for o in optstrings if o.startswith("--")]
                primary = longflags[0] if longflags else (optstrings[0] if optstrings else None)
                if primary is not None:
                    d = dest if dest else primary.lstrip("-").replace("-", "_")
                    out.append({
                        "flag": primary,
                        "dest": d,
                        "default": (None if default is _UNSET else default),
                        "has_default": default is not _UNSET,
                        "default_contract_key": default_contract_key,
                        "action": action,
                        "lineno": node.lineno,
                    })
            self.generic_visit(node)

    V().visit(tree)
    return out


def _arg_for(dest):
    for a in _parse_add_arguments():
        if a["dest"] == dest and a["has_default"]:
            return a
    raise AssertionError(f"no add_argument with a default found for dest {dest!r}")


# ---------------------------------------------------------------------------
# (a) contract agreement
# ---------------------------------------------------------------------------
def _load_contract():
    import importlib
    try:
        mod = importlib.import_module("shared.config_contract")
    except Exception as exc:  # pragma: no cover - depends on sibling-agent file
        pytest.skip(f"shared.config_contract not importable yet ({exc}); "
                    "sibling agent owns it -- rerun once it lands")
    return mod


#: Cross-process knobs where the CORE argparse ``dest`` maps 1:1 to a CONTRACT key and the
#: default MUST equal the canonical value. Kept minimal + explicit (not every contract key
#: has a core flag: follow_standoff_sim / stair_preset_demo / stair_bearing_scale are owned
#: by the sim/real parsers, not core).
_CORE_DEFAULT_VS_CONTRACT = {
    # core dest              # contract key
    "cmd_port":              "cmd_port",
    "follow_standoff_band_out": "standoff_band_out",
    "target_distance":       "follow_standoff_real",  # core --target-distance is the real standoff
}


def test_core_defaults_match_contract():
    mod = _load_contract()
    contract = mod.CONTRACT
    mismatches = []
    for core_dest, contract_key in _CORE_DEFAULT_VS_CONTRACT.items():
        assert contract_key in contract, f"contract missing key {contract_key!r}"
        arg = _arg_for(core_dest)
        want = mod.contract_value(contract_key)
        sourced_key = arg["default_contract_key"]
        if sourced_key is not None:
            # Strongest form: the default IS contract_value("..."). Just verify it points at
            # the RIGHT contract key (a typo'd key would sail through a value comparison).
            if sourced_key != contract_key:
                mismatches.append(
                    f"--{core_dest.replace('_', '-')} default=contract_value({sourced_key!r}) "
                    f"but should source {contract_key!r}"
                )
        else:
            core_default = arg["default"]
            if core_default != want:
                mismatches.append(
                    f"--{core_dest.replace('_', '-')} default={core_default!r} != "
                    f"CONTRACT[{contract_key!r}]={want!r}"
                )
    assert not mismatches, (
        "core argparse defaults drifted from the cross-process contract:\n  "
        + "\n  ".join(mismatches)
    )


def test_contract_value_fails_loud_on_typo():
    mod = _load_contract()
    with pytest.raises(KeyError):
        mod.contract_value("definitely_not_a_contract_key")


# ---------------------------------------------------------------------------
# (b) no dead flags: every flag's dest is consumed somewhere under src/
# ---------------------------------------------------------------------------

# Flags whose dest is intentionally NOT consumed in-process (pure passthrough / reserved).
# Seed minimally -- each needs a reason. A flag added here is exempt from the dead-flag check.
_DEAD_FLAG_WHITELIST = {
    # --sim / --follow / --debug etc. are consumed; nothing is currently a pure passthrough
    # from the CORE parser. (The dead --stair-follow-bearing-scale that the review found has
    # since been removed from core; if a genuine passthrough flag is added, list it here.)
}


def _consumer_index():
    """Concatenated text of every .py under src/ EXCEPT the parser itself and tests, for a
    cheap 'is this dest referenced anywhere' scan."""
    blobs = []
    for root, _dirs, files in os.walk(_SRC):
        # skip caches + the tests dir (a dest referenced only by a test is still 'unconsumed'
        # by production) + the parser file (definitions, not consumers)
        if "__pycache__" in root:
            continue
        rel = os.path.relpath(root, _SRC)
        if rel.split(os.sep)[0] == "tests":
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            if os.path.abspath(path) == os.path.abspath(_ARGS_PARSER):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    blobs.append(f.read())
            except Exception:
                pass
    return "\n".join(blobs)


def test_no_dead_flags():
    args = _parse_add_arguments()
    index = _consumer_index()
    dead = []
    for a in args:
        dest = a["dest"]
        if dest in _DEAD_FLAG_WHITELIST:
            continue
        # A consumer reads args.<dest> or getattr(args, "<dest>", ...). Match either the
        # attribute access ".<dest>" or the string literal '"<dest>"' / "'<dest>'".
        needle_attr = "." + dest
        needle_str1 = '"' + dest + '"'
        needle_str2 = "'" + dest + "'"
        if needle_attr in index or needle_str1 in index or needle_str2 in index:
            continue
        dead.append(f"--{a['flag'].lstrip('-')} (dest={dest}, args_parser.py:{a['lineno']})")
    assert not dead, (
        "core CLI flags with NO consumer anywhere under src/ (dead flags -- the "
        "--stair-follow-bearing-scale class the review found):\n  "
        + "\n  ".join(dead)
        + "\nEither wire the flag up, delete it, or (if a genuine passthrough) add its dest "
          "to _DEAD_FLAG_WHITELIST with a reason."
    )


def test_parser_has_many_flags():
    """Sanity: the static parse actually found the parser's flags (guards a silently-empty
    scan that would make the dead-flag test pass vacuously)."""
    args = _parse_add_arguments()
    assert len(args) >= 50, f"expected many add_argument flags, found {len(args)}"

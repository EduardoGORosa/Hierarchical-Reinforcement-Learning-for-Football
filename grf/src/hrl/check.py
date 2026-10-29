"""The test runner: one node at a time, or the ladder of everything below one.

From grf/src (host for layers 1-3 and 6-7 maths; container for everything: ./run.sh hrl check):

    python3 -m hrl.check                 the whole ladder, one line per node
    python3 -m hrl.check 24              node 24: every unit test, then the stack test (or 37a, 37b)
    python3 -m hrl.check 24 --unit       only the unit tests (this node alone, lower nodes faked)
    python3 -m hrl.check 24 --demo       node 24's demo: predict its output first, then run it
    python3 -m hrl.check --upto 29       every node up to 29 in teaching order, units and stacks
    python3 -m hrl.check --layer 2       one layer of the build map
    python3 -m hrl.check --asks          list every `# ASK:` comment you've left in hrl/

Results:  PASS · FAIL (with the numbers) · TODO (still raises NotImplementedError; says which
node's TODO it hit) · SKIP (needs the container).

Reading a node's results:
    units FAIL                 fix this node's own code first; its stack result means nothing yet
    units PASS, stack FAIL     your code for this node is right on its own. The bug is in a lower
                               node, or in how the pieces join. Run --upto <the node before>
    units PASS, stack TODO     the stack is waiting on a lower node you haven't written yet
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
import traceback
from pathlib import Path

from hrl.tests._support import Skip

# Teaching order, from the build map. 41 and 42 are theory nodes, with no code. A node with its
# own test file and a letter (37a, test_n37a_*.py) is a string; one file shared by 30a/30b is 30.
ORDER: list[int | str] = [23, 24, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                          "37a", "37b", 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
LAYERS = {1: "PPO pieces", 2: "SMDP arithmetic", 3: "options", 4: "execution (toy pitch)",
          5: "two phases (toy pitch)", 6: "style", 7: "evaluation"}
THEORY = {41: "a constraint can be substituted away", 42: "lambda is the constraint's price"}


def _node(s: str) -> int | str:
    """'37' -> 37, '37a' -> '37a': the key ORDER uses."""
    return int(s) if s.isdigit() else s


def _test_modules() -> dict[int | str, object]:
    import hrl.tests as pkg
    mods = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        m = re.match(r"test_n(\d+[a-z]?)_", info.name)
        if m:
            mods[_node(m.group(1))] = importlib.import_module(f"hrl.tests.{info.name}")
    return mods


def _run(fn) -> tuple[str, str]:
    try:
        fn()
    except Skip as e:
        return "SKIP", str(e)
    except NotImplementedError as e:
        return "TODO", str(e)
    except AssertionError as e:
        return "FAIL", str(e)
    except Exception:
        return "FAIL", "crashed:\n" + traceback.format_exc(limit=3)
    return "PASS", ""


def run_node(n: int | str, mod, unit_only: bool = False, quiet: bool = False) -> dict:
    units = [(name, getattr(mod, name)) for name in dir(mod) if name.startswith("test_unit")]
    units.sort(key=lambda t: t[1].__code__.co_firstlineno)
    res = {"units": [(name[len("test_unit_"):] or "unit", *_run(fn)) for name, fn in units]}
    if not unit_only and hasattr(mod, "test_stack"):
        res["stack"] = _run(mod.test_stack)
    if quiet:
        return res
    needs = ", ".join(map(str, mod.NEEDS)) or "nothing"
    print(f"\nnode {n} · {mod.TITLE}   [layer {mod.LAYER} · needs {needs}]")
    for name, status, msg in res["units"]:
        print(f"  unit   {name:28s} {status}" + (f"\n         {msg}" if msg and status != "PASS" else ""))
    if "stack" in res:
        status, msg = res["stack"]
        print(f"  stack  {'(your code for ' + needs + ' underneath)':28s} {status}"
              + (f"\n         {msg}" if msg and status != "PASS" else ""))
        u = {s for _, s, _ in res["units"]}
        if u == {"PASS"} and status == "FAIL":
            before = ORDER[ORDER.index(n) - 1] if ORDER.index(n) else n
            print(f"  -> units pass, stack fails: the bug is BELOW this node, or in the JOIN (how this\n"
                  f"     node calls the ones below). Run --upto {before}: if that is all +, it's the join.\n"
                  f"     Check every call this node makes against the called function's signature.")
        elif u == {"PASS"} and status == "TODO":
            print(f"  -> the stack is waiting on a lower node: {msg}")
    return res


def _mark(res) -> str:
    statuses = [s for _, s, _ in res["units"]] + ([res["stack"][0]] if "stack" in res else [])
    for s, m in (("FAIL", "x"), ("TODO", "."), ("SKIP", "s")):
        if s in statuses:
            return m
    return "+"


def ladder(nodes, mods, verbose: bool) -> int:
    fails = 0
    print("\n  + all pass   x a failure   . TODO left   s needs the container\n")
    layer = None
    for n in nodes:
        if n in THEORY:
            print(f"  {n:>3}  (theory) {THEORY[n]}")
            continue
        mod = mods.get(n)
        if mod is None:
            continue
        if mod.LAYER != layer:
            layer = mod.LAYER
            print(f" layer {layer} · {LAYERS[layer]}")
        res = run_node(n, mod, quiet=True)
        mark = _mark(res)
        fails += mark == "x"
        print(f"  {n:>3} {mark} {mod.TITLE}")
        if verbose and mark == "x":
            run_node(n, mod)
    return fails


def asks() -> None:
    root = Path(__file__).resolve().parent
    found = 0
    for p in sorted(root.rglob("*.py")):
        if p.name == "check.py":           # this file mentions the marker; it isn't a question
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.lstrip().startswith("# ASK:") or "  # ASK:" in line:
                found += 1
                print(f"{p.relative_to(root.parent)}:{i}: {line.strip()}")
    print(f"\n{found} question(s). Say 'read my ASKs' in the session to get them answered.")


def main(argv: list[str]) -> int:
    mods = _test_modules()
    if "--asks" in argv:
        asks()
        return 0
    if "--upto" in argv:
        n = _node(argv[argv.index("--upto") + 1])
        return 1 if ladder(ORDER[:ORDER.index(n) + 1], mods, verbose=True) else 0
    if "--layer" in argv:
        k = int(argv[argv.index("--layer") + 1])
        return 1 if ladder([n for n in ORDER if n in mods and mods[n].LAYER == k], mods, True) else 0
    nums = [_node(a) for a in argv if re.fullmatch(r"\d+[a-z]?", a)]
    if not nums:
        return 1 if ladder(ORDER, mods, verbose=False) else 0
    for n in nums:
        if n in THEORY:
            print(f"\nnode {n} is a theory node ({THEORY[n]}): nothing to run.")
            continue
        mod = mods[n]
        if "--demo" in argv:
            if not hasattr(mod, "demo"):
                print(f"node {n} has no demo")
            else:
                mod.demo()
            continue
        res = run_node(n, mod, unit_only="--unit" in argv)
        if _mark(res) == "x":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

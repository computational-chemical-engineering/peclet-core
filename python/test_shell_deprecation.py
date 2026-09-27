"""The peclet-core compatibility shell warns on the old spelling; the canonical spelling is silent.

    python test_shell_deprecation.py {geom|mpi} <core>/packaging/shell

The ladder step 2 of suite/docs/CORE_BOUNDARY.md, gated rather than trusted. The three shell files
are staged as ``peclet/core/`` in a temporary directory -- the layout the peclet-core wheel installs
(packaging/pyproject-core.toml) -- and each check runs in a fresh interpreter:

1. ``python -W error::DeprecationWarning -c "import peclet.core.<old>"`` must exit NON-ZERO, and
   its stderr must carry the shell's message (so an ImportError cannot pass for the warning);
2. ``python -c "import peclet.core.<old>"`` with the default filters must exit 0 and PRINT the
   warning -- stacklevel=2 attributes it to the importing ``__main__``, which is what makes a
   user see it rather than only a test runner;
3. ``python -W error::DeprecationWarning -c "import peclet.<canonical>"`` must exit 0.

Exits 77 (ctest SKIP) when the canonical module is not importable at all: peclet.geom is a separate
distribution (peclet-geom), and without it check 1 would "fail" by ImportError for the wrong reason.
"""
import os
import shutil
import subprocess
import sys
import tempfile

CANONICAL = {"geom": "geom", "mpi": "halo"}


def run(args, env):
    return subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True)


def main():
    old, shell_dir = sys.argv[1], sys.argv[2]
    canon = CANONICAL[old]
    message = f"peclet.core.{old} is peclet.{canon} since peclet 1.2.0"

    with tempfile.TemporaryDirectory() as tmp:
        pkg = os.path.join(tmp, "peclet", "core")   # `peclet` stays a PEP 420 namespace: no __init__
        os.makedirs(pkg)
        for src, dst in (("core_init.py", "__init__.py"), ("core_geom.py", "geom.py"),
                         ("core_mpi.py", "mpi.py")):
            shutil.copy(os.path.join(shell_dir, src), os.path.join(pkg, dst))
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (tmp, env.get("PYTHONPATH", "")) if p)
        env.pop("PYTHONWARNINGS", None)

        r = run(["-c", f"import peclet.{canon}"], env)
        if r.returncode != 0:
            print(f"SKIP: peclet.{canon} is not importable here:\n{r.stderr}")
            return 77

        failures = []
        r = run(["-W", "error::DeprecationWarning", "-c", f"import peclet.core.{old}"], env)
        if r.returncode == 0:
            failures.append(f"`import peclet.core.{old}` under -W error exited 0: no DeprecationWarning")
        elif f"DeprecationWarning: {message}" not in r.stderr:
            failures.append(f"`import peclet.core.{old}` failed, but not with the shell's warning:\n{r.stderr}")

        r = run(["-c", f"import peclet.core.{old}"], env)
        if r.returncode != 0:
            failures.append(f"`import peclet.core.{old}` (default filters) failed:\n{r.stderr}")
        elif message not in r.stderr:
            failures.append(f"`import peclet.core.{old}` (default filters) did not show the warning; "
                            f"stderr was:\n{r.stderr!r}")

        r = run(["-W", "error::DeprecationWarning", "-c", f"import peclet.{canon}"], env)
        if r.returncode != 0:
            failures.append(f"canonical `import peclet.{canon}` is not silent:\n{r.stderr}")

    for f in failures:
        print("FAIL:", f)
    if not failures:
        print(f"OK: peclet.core.{old} warns ({message}...); peclet.{canon} is silent")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

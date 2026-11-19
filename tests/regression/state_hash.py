"""Bitwise reference of the code-health gate (.codehealth/config.toml, tier state_hash).

Runs the fixed-input cases of ``state_hash_driver`` (tests/regression/state_hash_driver.cpp) and the
per-test iterate digests of the Anderson tests, and prints one SHA-256 per case:

    OMP_NUM_THREADS=1 python tests/regression/state_hash.py --build <Kokkos build tree> --save h.json

- body, cutcell, scene, amg: serial, rank 0's output bytes.
- halo, migrate: under ``mpirun -np 1, 2, 4, 8``; the per-rank output files are hashed in rank
  order. halo runs on the default transport (on-node shared memory), with the shared-memory
  transport off (PECLET_CORE_HALO_SHM=0) and, at np 4, split into groups of two
  (PECLET_CORE_HALO_SHM_GROUP=2). The bytes moved are the same by contract, so the three must hash
  alike; a mismatch fails the run.
- anderson: the ``digest <test> <hex>`` lines of test_anderson (FNV-1a over every iterate, gamma,
  residual and decision) and of test_anderson_mpi at np 1, 2, 4.

Exits non-zero when any run fails (an MPI case that does not run is a failure, never a skip).
Prints one ``[deps] morton=<dir> @ <commit>`` line first: the gate requires it equal at base and
head, so a shared morton checkout that moved between the two runs fails instead of hashing two
different programs.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

MPIRUN = ["/usr/bin/mpirun", "--bind-to", "none"]
NPS = (1, 2, 4, 8)


def run(cmd, env=None):
    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:])
        raise SystemExit(f"FAILED ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def sha_files(prefix, nranks):
    h = hashlib.sha256()
    for r in range(nranks):
        with open(f"{prefix}.r{r}", "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def morton_line(build):
    with open(os.path.join(build, "CMakeCache.txt")) as f:
        m = re.search(r"^PECLET_CORE_MORTON_DIR:PATH=(.*)$", f.read(), re.M)
    d = m.group(1) if m else ""
    rev = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True).stdout.strip() if d else ""
    return f"[deps] morton={d} @ {rev or 'none'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--build", required=True, help="core build tree with PECLET_CORE_ENABLE_KOKKOS")
    ap.add_argument("--save", help="write the hashes as JSON")
    a = ap.parse_args()
    build = os.path.abspath(a.build)
    driver = os.path.join(build, "tests", "state_hash_driver")
    print(morton_line(build), flush=True)

    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for case in ("body", "cutcell", "scene", "amg"):
            pre = os.path.join(tmp, case)
            run([driver, case, pre], env)
            out[case] = sha_files(pre, 1)
        for case in ("halo", "migrate"):
            for np_ in NPS:
                pre = os.path.join(tmp, f"{case}{np_}")
                run(MPIRUN + ["-np", str(np_), driver, case, pre], env)
                out[f"{case}.np{np_}"] = sha_files(pre, np_)
        for np_ in NPS:
            pre = os.path.join(tmp, f"halo_noshm{np_}")
            run(MPIRUN + ["-np", str(np_), driver, "halo", pre], dict(env, PECLET_CORE_HALO_SHM="0"))
            if sha_files(pre, np_) != out[f"halo.np{np_}"]:
                raise SystemExit(f"FAILED: halo np{np_}: PECLET_CORE_HALO_SHM=0 moved other bytes")
        pre = os.path.join(tmp, "halo_group4")
        run(MPIRUN + ["-np", "4", driver, "halo", pre], dict(env, PECLET_CORE_HALO_SHM_GROUP="2"))
        if sha_files(pre, 4) != out["halo.np4"]:
            raise SystemExit("FAILED: halo np4: PECLET_CORE_HALO_SHM_GROUP=2 moved other bytes")

    def digests(text):
        lines = [ln for ln in text.splitlines() if ln.startswith("digest ")]
        if not lines:
            raise SystemExit("FAILED: no digest lines")
        return hashlib.sha256("\n".join(lines).encode()).hexdigest()

    out["anderson"] = digests(run([os.path.join(build, "tests", "test_anderson")], env))
    for np_ in (1, 2, 4):
        text = run(MPIRUN + ["-np", str(np_), os.path.join(build, "tests", "test_anderson_mpi")], env)
        out[f"anderson_mpi.np{np_}"] = digests(text)

    for k in sorted(out):
        print(f"{k} {out[k]}")
    if a.save:
        with open(a.save, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()

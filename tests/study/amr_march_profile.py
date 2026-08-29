"""M1 — the march-time attribution matrix (docs/amr_march_perf_and_distributed_plan.md Phase M).

P3c observation 3: the gap-graded bed saves 1.62x in CELLS and ~0% in STEP TIME (d8 1.85 vs
1.81 s/step, d7 0.89 vs 0.92). This script produces the per-phase table that says where the
missing 1.62x goes, on the two arms of two geometry families, so the four pre-registered
hypotheses can be separated:

  H-iters   the graded mesh needs more pressure/momentum ITERATIONS per step
  H-mg      its MG hierarchy has more levels engaged / worse per-level shapes
  H-launch  many small kernels put it in launch-latency-bound territory
  H-band    step cost tracks CUT-BAND size, not leaf count (both arms share a band
            at matched depth, which would explain equal step times exactly)

It drives the M0 step profiler (`PECLET_CORE_PROFILE_STEP`), which prints one fenced per-phase
table per window of steps. The FIRST window is the warm-up and is labelled as such; the
following `--repeats` windows are the measurement.

  core/tests/study/amr_march_profile.py [--geom bed|zh] [--depth 7] [--arms ug]
                                        [--steps 200] [--repeats 3] [--gap-file F]

  --geom bed   arms u = uniform finest band, g = gap-graded (the P3c pair)
  --geom zh    arms a = uniform control, b = two-level latitude (the P2b pair; --depth 8 is N=256)
"""
import importlib.util
import math
import os
import sys
import time

ROOT = __file__.rsplit("/tests/", 1)[0]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def opt(args, nm, d):
    return args[args.index(nm) + 1] if nm in args else d


def main():
    args = sys.argv[1:]
    geom = opt(args, "--geom", "bed")
    depth = int(opt(args, "--depth", 7))
    steps = int(opt(args, "--steps", 200))
    repeats = int(opt(args, "--repeats", 3))
    gapfile = opt(args, "--gap-file", None)
    arms = opt(args, "--arms", "ug" if geom == "bed" else "ab")
    N = 2 ** depth

    # The profiler is read when the Flow is CONSTRUCTED, so set it before importing the driver.
    os.environ["PECLET_CORE_PROFILE_STEP"] = "1"
    os.environ["PECLET_CORE_PROFILE_STEP_WINDOW"] = str(steps)

    import numpy as np

    if geom == "bed":
        drv = _load("bed", ROOT + "/tests/study/amr_bed_graded.py")
        gapfn, _ = drv.make_gap_lookup(N, gapfile=gapfile)
    else:
        drv = _load("zh", ROOT + "/tests/study/amr_zh_ladder.py")
    amr = drv.amr

    print(f"# M1 march profile: geom={geom} N={N} (depth {depth}) arms={arms} "
          f"steps/window={steps} windows={1 + repeats} (first = warm-up)", flush=True)
    for arm in arms:
        t0 = time.time()
        tree = drv.build(arm, N, gapfn) if geom == "bed" else drv.build(arm, N)
        lev = np.asarray(tree.levels())
        lmax = int(math.log2(N))
        hist = np.bincount(lev, minlength=lmax + 1).tolist()
        print(f"\n## arm {arm}: leaves {tree.num_leaves} levels {hist} "
              f"(mesh {time.time() - t0:.1f}s)", flush=True)
        if geom == "bed":
            fl = amr.Flow(tree, 1.0, drv.MU, 60.0)
            fl.set_body_force(drv.FX, 0.0, 0.0)
            fl.set_advection(False)
            fl.set_ghost_sampled(True)
            fl.set_cf_scheme(1)
            C, R = drv.load_pack(N)
            t0 = time.time()
            fl.set_solid_spheres(C, np.array([R]), True)
        else:
            fl = amr.Flow(tree, 1.0, 0.1, 60.0)
            fl.set_body_force(1e-3, 0.0, 0.0)
            fl.set_advection(False)
            fl.set_ghost_sampled(True)
            fl.set_cf_scheme(1)
            t0 = time.time()
            fl.set_solid(drv.make_sdf(N))
        print(f"   set_solid {time.time() - t0:.2f}s", flush=True)
        for w in range(1 + repeats):
            t0 = time.time()
            for _ in range(steps):
                fl.step(100, 60)
            tag = "warm-up" if w == 0 else f"window {w}/{repeats}"
            print(f"   [{arm}] {tag}: {steps} steps in {time.time() - t0:.1f}s "
                  f"({(time.time() - t0) / steps * 1e3:.1f} ms/step wall), "
                  f"pres_iters(last)={fl.last_pres_iters()}", flush=True)


if __name__ == "__main__":
    main()

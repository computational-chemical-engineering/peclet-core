"""M1 — the march-time attribution matrix (docs/amr_march_perf_and_distributed_plan.md Phase M).

P3c observation 3: the gap-graded bed saves 1.62x in CELLS and ~0% in STEP TIME (d8 1.85 vs
1.81 s/step, d7 0.89 vs 0.92). This script produces the numbers that say where the missing 1.62x
goes, so the four pre-registered hypotheses can be separated:

  H-iters   the graded mesh needs more pressure/momentum ITERATIONS per step
  H-mg      its MG hierarchy has more levels engaged / worse per-level shapes
  H-launch  many small kernels put it in launch-latency-bound territory
  H-band    step cost tracks CUT-BAND size, not leaf count (both arms share a band
            at matched depth, which would explain equal step times exactly)

Two modes.

`--structure` (default) is CONTENTION-FREE: leaf counts, level histograms, the pressure MG's
per-level cell counts, the overlay row census (the cut band), and a short march for the
per-step ITERATION counts. None of it is a wall-clock claim, so it is valid on a shared GPU —
which matters, because these boxes are shared and a neighbouring job moved a fixed workload's
step time by 3x while its iteration counts stayed put.

`--time` drives the M0 step profiler (`PECLET_CORE_PROFILE_STEP`) for the per-phase breakdown.
It INTERLEAVES the arms window by window (both Flows resident at once) so that whatever else is
on the GPU hits both arms equally: the arm RATIO stays meaningful even when the absolute numbers
do not. Every window's wall ms/step is printed so the spread is visible; treat a set of windows
whose spread exceeds the arm difference as inconclusive on absolute time.

  core/tests/study/amr_march_profile.py [--geom bed|zh] [--depth 7] [--arms ug]
       [--structure | --time] [--steps 200] [--repeats 3] [--probe-steps 20] [--gap-file F]

  --geom bed   arms u = uniform finest band, g = gap-graded (the P3c pair)
  --geom zh    arms a = uniform control, b = two-level latitude (the P2b pair; --depth 8 is N=256)
"""
import importlib.util
import math
import os
import re
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


def parse_arms(spec, depth):
    """`ug` or `u,g` or `g@6,g@7` — an arm letter, optionally pinned to its own depth. The
    depth-pinned form interleaves two RESOLUTIONS of the same arm, which is how the
    mesh-independent part of a step is measured (H-launch) without trusting absolute times."""
    items = spec.split(",") if ("," in spec or "@" in spec) else re.findall(r"[a-z]\d*", spec)
    out = []
    for it in items:
        if "@" in it:
            a, d = it.split("@")
            out.append((a, int(d)))
        else:
            out.append((it, depth))
    return out


def make_flow(drv, geom, arm, N, gapfn):
    """Mesh + configured, solid-set Flow for one arm (identical settings across arms)."""
    import numpy as np

    t0 = time.time()
    if geom == "bed":
        # `g<K>` = the graded arm at gap floor n = K (the policy dial of P3a's table). The
        # shipped policy is n = 4; a larger n coarsens harder, which is how a LARGE cell-count
        # lever is obtained at a depth whose uniform arm still fits in one GPU.
        ngap0 = drv.NGAP
        if arm[0] == "g" and len(arm) > 1:
            drv.NGAP = float(arm[1:])
        tree = drv.build(arm[0], N, gapfn)
        drv.NGAP = ngap0
    else:
        tree = drv.build(arm, N)
    tmesh = time.time() - t0
    lmax = int(math.log2(N))
    hist = np.bincount(np.asarray(tree.levels()), minlength=lmax + 1).tolist()
    amr = drv.amr
    t0 = time.time()
    if geom == "bed":
        fl = amr.Flow(tree, 1.0, drv.MU, 60.0)
        fl.set_body_force(drv.FX, 0.0, 0.0)
    else:
        fl = amr.Flow(tree, 1.0, 0.1, 60.0)
        fl.set_body_force(1e-3, 0.0, 0.0)
    fl.set_advection(False)
    fl.set_ghost_sampled(True)
    fl.set_cf_scheme(1)
    if geom == "bed":
        C, R = drv.load_pack(N)
        fl.set_solid_spheres(C, np.array([R]), True)
    else:
        fl.set_solid(drv.make_sdf(N))
    return dict(arm=arm, tree=tree, fl=fl, leaves=tree.num_leaves, hist=hist, mesh_s=tmesh,
                solid_s=time.time() - t0)


def main():
    args = sys.argv[1:]
    geom = opt(args, "--geom", "bed")
    depth = int(opt(args, "--depth", 7))
    steps = int(opt(args, "--steps", 200))
    repeats = int(opt(args, "--repeats", 3))
    probe = int(opt(args, "--probe-steps", 20))
    gapfile = opt(args, "--gap-file", None)
    arms = opt(args, "--arms", "ug" if geom == "bed" else "ab")
    timed = "--time" in args
    N = 2 ** depth

    # The profiler flag is read when a Flow is CONSTRUCTED. Structure mode turns it on too, with a
    # window wider than the probe, so it prints its MG-hierarchy header (a static shape, valid on a
    # busy box) and never reaches a timing table.
    os.environ["PECLET_CORE_PROFILE_STEP"] = "1"
    os.environ["PECLET_CORE_PROFILE_STEP_WINDOW"] = str(steps if timed else probe + 1)

    plan = parse_arms(arms, depth)
    if geom == "bed":
        drv = _load("bed", ROOT + "/tests/study/amr_bed_graded.py")
        gaps = {d: drv.make_gap_lookup(2 ** d, gapfile=gapfile)[0] for _, d in plan}
    else:
        drv = _load("zh", ROOT + "/tests/study/amr_zh_ladder.py")
        gaps = {d: None for _, d in plan}

    mode = "timed (interleaved windows)" if timed else "structure (contention-free)"
    print(f"# M1 march profile [{mode}]: geom={geom} arms="
          + " ".join(f"{a}@N{2 ** d}" for a, d in plan), flush=True)

    if not timed:
        # One arm at a time: the depth-8 uniform bed does not fit beside anything else.
        for arm, d in plan:
            a = make_flow(drv, geom, arm, 2 ** d, gaps[d])
            print(f"\n## arm {arm} N={2 ** d}: leaves {a['leaves']} levels {a['hist']} "
                  f"(mesh {a['mesh_s']:.1f}s, set_solid {a['solid_s']:.2f}s = "
                  f"{1e6 * a['solid_s'] / a['leaves']:.1f} us/leaf)", flush=True)
            mom, pres = [], []
            for _ in range(probe):
                a["fl"].step(100, 60)
                pres.append(a["fl"].last_pres_iters())
                mom.append(a["fl"].last_mom_iters())
            print(f"   [{arm}] {probe} steps: pressure iters "
                  f"min/mean/max {min(pres)}/{sum(pres) / len(pres):.1f}/{max(pres)}"
                  f" | momentum iters mean {sum(mom) / len(mom):.1f}", flush=True)
            del a
        return

    built = [make_flow(drv, geom, arm, 2 ** d, gaps[d]) for arm, d in plan]
    for a, (arm, d) in zip(built, plan):
        a["arm"] = f"{arm}@N{2 ** d}"
        print(f"## arm {a['arm']}: leaves {a['leaves']} levels {a['hist']} "
              f"(mesh {a['mesh_s']:.1f}s, set_solid {a['solid_s']:.2f}s)", flush=True)
    print(f"# {1 + repeats} windows of {steps} steps per arm, arms INTERLEAVED "
          f"(first window each = warm-up)", flush=True)
    for w in range(1 + repeats):
        for a in built:
            t0 = time.time()
            for _ in range(steps):
                a["fl"].step(100, 60)
            dt = time.time() - t0
            tag = "warm-up" if w == 0 else f"window {w}/{repeats}"
            print(f"   [{a['arm']}] {tag}: {steps} steps in {dt:.1f}s "
                  f"({dt / steps * 1e3:.1f} ms/step wall), pres_iters(last)="
                  f"{a['fl'].last_pres_iters()}", flush=True)


if __name__ == "__main__":
    main()

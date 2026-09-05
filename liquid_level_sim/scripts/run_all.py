"""End-to-end driver: generate scenes -> render with pbrt-v4 -> analyze -> HTML report.

Examples
  python scripts/run_all.py --smoke                     # 2 tiny samples, ~1 min, sanity check
  python scripts/run_all.py --n 16 --spp 128            # default study
  python scripts/run_all.py --n 9 --fill-sweep --containers cylinder --seed 3
  python scripts/run_all.py --skip-gen --skip-render    # re-run analysis/report only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


def step(name: str, args: list[str]) -> None:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.time()
    subprocess.run([PY, str(HERE / args[0]), *args[1:]], check=True)
    print(f"=== {name} done in {time.time() - t0:.1f}s ===", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--containers", default="cylinder,box")
    ap.add_argument("--res", default="640x480")
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--mask-spp", type=int, default=16)
    ap.add_argument("--fill-sweep", action="store_true")
    ap.add_argument("--nthreads", type=int, default=None)
    ap.add_argument("--tonemap", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-render existing outputs")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 samples at 320x240, 16 spp")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.res, a.spp, a.mask_spp = 2, "320x240", 16, 8

    t0 = time.time()
    if not a.skip_gen:
        g = ["gen_scenes.py", "--n", str(a.n), "--seed", str(a.seed), "--containers", a.containers,
             "--res", a.res, "--spp", str(a.spp), "--mask-spp", str(a.mask_spp)]
        if a.fill_sweep:
            g.append("--fill-sweep")
        step("generate scenes", g)
    if not a.skip_render:
        r = ["render.py"]
        if a.nthreads:
            r += ["--nthreads", str(a.nthreads)]
        if a.tonemap:
            r.append("--tonemap")
        if a.force:
            r.append("--force")
        step("render (pbrt-v4)", r)
    step("analyze", ["analyze.py"])
    step("report", ["build_report.py"])
    print(f"\nall done in {time.time() - t0:.1f}s -> {HERE.parent / 'outputs' / 'report.html'}")


if __name__ == "__main__":
    main()

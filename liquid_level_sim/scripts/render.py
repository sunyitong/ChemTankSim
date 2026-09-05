"""Render generated scenes with pbrt-v4 (CPU) and convert EXR beauty passes to PNG."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import (IMGTOOL_EXE, MASK_DIR, PASSES, PBRT_EXE, RENDER_DIR, SCENES_DIR,
                    check_pbrt, ensure_dirs, load_json, run)


def render_pass(scene: Path, out: Path, spp: int | None, nthreads: int | None, quiet: bool) -> float:
    cmd = [PBRT_EXE, "--outfile", out]
    if spp:
        cmd += ["--spp", spp]
    if nthreads:
        cmd += ["--nthreads", nthreads]
    if quiet:
        cmd += ["--quiet"]
    cmd.append(scene)
    t0 = time.time()
    run(cmd, cwd=scene.parent, quiet=quiet)
    return time.time() - t0


def exr_to_png(exr: Path, png: Path, tonemap: bool) -> None:
    cmd = [IMGTOOL_EXE, "convert"]
    if tonemap:
        cmd += ["--tonemap"]
    cmd += ["--outfile", png, exr]
    run(cmd, quiet=True)


def render_sample(meta: dict, spp: int | None, mask_spp: int | None, nthreads: int | None,
                  passes=PASSES, quiet=True, tonemap=False, force=False) -> dict:
    sid = meta["sid"]
    timings = {}
    for p in passes:
        scene = SCENES_DIR / meta["passes"][p]
        if p == "beauty":
            out = RENDER_DIR / f"{sid}_beauty.exr"
            png = RENDER_DIR / f"{sid}_beauty.png"
            if force or not png.exists():
                timings[p] = render_pass(scene, out, spp, nthreads, quiet)
                exr_to_png(out, png, tonemap)
        else:
            out = MASK_DIR / f"{sid}_{p}.png"
            if force or not out.exists():
                timings[p] = render_pass(scene, out, mask_spp, nthreads, quiet)
    return timings


def render_all(metas: list[dict], spp=None, mask_spp=None, nthreads=None, passes=PASSES, quiet=True,
               tonemap=False, force=False) -> dict:
    check_pbrt()
    ensure_dirs()
    all_t = {}
    for i, m in enumerate(metas):
        t = render_sample(m, spp, mask_spp, nthreads, passes, quiet, tonemap, force)
        all_t[m["sid"]] = t
        tot = sum(t.values())
        if t:
            print(f"[{i + 1}/{len(metas)}] {m['sid']}: " + ", ".join(f"{k}={v:.1f}s" for k, v in t.items())
                  + f"  (total {tot:.1f}s)", flush=True)
        else:
            print(f"[{i + 1}/{len(metas)}] {m['sid']}: cached", flush=True)
    return all_t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spp", type=int, default=None, help="override beauty spp from the scene file")
    ap.add_argument("--mask-spp", type=int, default=None)
    ap.add_argument("--nthreads", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated sample ids")
    ap.add_argument("--passes", default=",".join(PASSES))
    ap.add_argument("--tonemap", action="store_true", help="apply Reinhard tonemap when converting EXR->PNG")
    ap.add_argument("--force", action="store_true", help="re-render even if outputs exist")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    metas = load_json(SCENES_DIR / "dataset.json")
    if a.only:
        keep = set(a.only.split(","))
        metas = [m for m in metas if m["sid"] in keep]
    render_all(metas, a.spp, a.mask_spp, a.nthreads, a.passes.split(","), not a.verbose, a.tonemap, a.force)


if __name__ == "__main__":
    main()

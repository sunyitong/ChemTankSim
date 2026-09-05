"""Render the MVP scenes with pbrt-v4 (CPU) and convert EXR -> sRGB PNG with imgtool."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from mvp_common import (IMGTOOL_EXE, MANIFEST, MVP_OUT, PBRT_EXE, RENDER_DIR, SCENE_DIR, check_pbrt,
                        ensure_dirs, load_json, run, save_json)


def render(scene: Path, out_exr: Path, nthreads: int | None, spp: int | None, quiet: bool,
           seed: int | None = None) -> float:
    cmd = [PBRT_EXE, "--outfile", out_exr]
    if nthreads:
        cmd += ["--nthreads", nthreads]
    if spp:
        cmd += ["--spp", spp]
    if seed is not None:
        cmd += ["--seed", seed]
    if quiet:
        cmd += ["--quiet"]
    cmd.append(scene.name)
    t0 = time.time()
    run(cmd, cwd=scene.parent, quiet=quiet)
    return time.time() - t0


def to_png(exr: Path, png: Path) -> None:
    run([IMGTOOL_EXE, "convert", "--outfile", png, exr], quiet=True)


def render_manifest(manifest: dict, nthreads=None, spp=None, force=False, quiet=True, overview=True,
                    out_path: Path | None = None) -> dict:
    check_pbrt()
    ensure_dirs()
    MANIFEST = out_path or globals()["MANIFEST"]      # noqa: N806  (per-run manifest, e.g. manifest_smoke.json)
    jobs = manifest["jobs"] + ([manifest["overview"]] if overview else [])
    for i, j in enumerate(jobs):
        scene = SCENE_DIR / j["scene"]
        exr, png = RENDER_DIR / j["exr"], RENDER_DIR / j["png"]
        if png.exists() and exr.exists() and not force:
            print(f"[{i + 1}/{len(jobs)}] {j['scene']}: cached", flush=True)
            continue
        t = render(scene, exr, nthreads, spp, quiet, j.get("seed"))
        to_png(exr, png)
        j["render_seconds"] = round(t, 1)
        print(f"[{i + 1}/{len(jobs)}] {j['name'] if 'name' in j else j['scene']}: {t:.1f}s -> {png.name}", flush=True)
        save_json(manifest, MANIFEST)          # progress is visible while a long batch runs
    save_json(manifest, MANIFEST)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(SCENE_DIR / "manifest.json"))
    ap.add_argument("--nthreads", type=int, default=None)
    ap.add_argument("--spp", type=int, default=None, help="override spp in the scene files")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-overview", action="store_true")
    a = ap.parse_args()
    manifest = load_json(Path(a.manifest))
    render_manifest(manifest, a.nthreads, a.spp, a.force, not a.verbose, not a.no_overview)


if __name__ == "__main__":
    main()

"""Build, render and post-process the Phase-1 test set (baseline/compare pairs for the web app).

  python scripts/testset_run.py --smoke               # 300x400, 24 spp, geometry check
  python scripts/testset_run.py                       # 900x1200, 160 spp, ~1 min per image
  python scripts/testset_run.py --only A1,B2 --force  # subset
  python scripts/testset_run.py --skip-render         # redo post-processing + contact sheet only
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import mvp_patterns
from mvp_common import IMGTOOL_EXE, PBRT_EXE, Setup, check_pbrt, load_json, run, save_json
from testset_post import process
from testset_scenes import COMBOS, TS_IMAGES, TS_OUT, TS_PATTERNS, TS_RENDERS, TS_SCENES, TS_TEX, VESSELS, baseline_name, build_scene
from testset_textures import make_maps


def ensure():
    for d in (TS_OUT, TS_SCENES, TS_TEX, TS_PATTERNS, TS_RENDERS, TS_IMAGES):
        d.mkdir(parents=True, exist_ok=True)


def patterns_for_panel():
    """Panel textures for the 60 x 80 cm panel used here (separate from the MVP's 50 x 60 set)."""
    mvp_patterns.PATTERN_DIR = TS_PATTERNS
    cfg = Setup(panel_w=60.0, panel_h=80.0)
    return mvp_patterns.generate_all(cfg)


def render(scene: Path, exr: Path, spp: int | None, nthreads: int | None) -> float:
    cmd = [PBRT_EXE, "--quiet", "--outfile", exr]
    if spp:
        cmd += ["--spp", spp]
    if nthreads:
        cmd += ["--nthreads", nthreads]
    cmd.append(scene.name)
    t0 = time.time(); run(cmd, cwd=scene.parent, quiet=True); return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--res", default="900x1200")
    ap.add_argument("--spp", type=int, default=160)
    ap.add_argument("--only", default=None, help="comma-separated combo name prefixes, e.g. A1,B2")
    ap.add_argument("--nthreads", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--no-jitter", action="store_true", help="no exposure/shift jitter on compare frames")
    ap.add_argument("--dirt", action="store_true", help="enable the wall-deposit textures listed in COMBOS (default: clean glass)")
    a = ap.parse_args()
    w, h = (int(v) for v in a.res.lower().split("x"))
    spp, tag = a.spp, ""
    if a.smoke:
        w, h, spp, tag = 300, 400, 24, "_smoke"
    combos = [c for c in COMBOS if not a.only or any(c.name.startswith(p) for p in a.only.split(","))]
    if not a.dirt:
        combos = [replace(c, dirt=None) for c in combos]

    ensure(); check_pbrt()
    print("=== panel patterns ==="); patterns_for_panel()
    print("=== deposit textures ===")
    tex_cache = {}
    for c in combos:
        if c.dirt and (c.dirt, c.dirt_seed) not in tex_cache:
            tex_cache[(c.dirt, c.dirt_seed)] = make_maps(c.dirt, c.dirt_seed, TS_TEX)
            print(f"  {c.dirt}/{c.dirt_seed}: coverage mean {tex_cache[(c.dirt, c.dirt_seed)]['coverage_mean']:.2f}")

    print("=== scenes ===")
    jobs, seen = [], set()
    for c in combos:
        tex = tex_cache.get((c.dirt, c.dirt_seed)) if c.dirt else None
        bname = baseline_name(c) + tag
        if bname not in seen:
            seen.add(bname)
            text, gt = build_scene(c, False, w, h, spp, tex, bname + ".exr")
            (TS_SCENES / f"{bname}.pbrt").write_text(text, encoding="utf-8")
            jobs.append({"name": bname, "role": "baseline", "combo": None, "part": c.part, "vessel": c.vessel, "camera": c.camera,
                         "pattern": c.pattern, "dirt": c.dirt, "gt": gt, "jitter": False})
        fname = c.name + tag
        text, gt = build_scene(c, True, w, h, spp, tex, fname + ".exr")
        (TS_SCENES / f"{fname}.pbrt").write_text(text, encoding="utf-8")
        jobs.append({"name": fname, "role": "compare", "combo": c.name, "baseline": bname, "part": c.part, "vessel": c.vessel,
                     "camera": c.camera, "pattern": c.pattern, "dirt": c.dirt, "layers": c.layers, "note": c.note, "gt": gt,
                     "jitter": not a.no_jitter})
    print(f"  {len(jobs)} scenes ({sum(j['role'] == 'baseline' for j in jobs)} baselines) at {w}x{h}, {spp} spp")

    if not a.skip_render:
        print("=== render ===")
        for i, j in enumerate(jobs):
            exr = TS_RENDERS / f"{j['name']}.exr"
            if exr.exists() and not a.force:
                print(f"[{i + 1}/{len(jobs)}] {j['name']}: cached"); continue
            t = render(TS_SCENES / f"{j['name']}.pbrt", exr, None, a.nthreads)
            j["render_seconds"] = round(t, 1)
            print(f"[{i + 1}/{len(jobs)}] {j['name']}: {t:.1f}s", flush=True)

    print("=== post-process (sensor noise, jitter) ===")
    rng = np.random.default_rng(2026)
    for j in jobs:
        exr, png = TS_RENDERS / f"{j['name']}.exr", TS_IMAGES / f"{j['name']}.png"
        if not exr.exists():
            print(f"  missing render {exr.name}"); continue
        j["post"] = process(exr, png, rng, j["jitter"])
        j["png"] = png.name

    # contact sheet: one row per compare job: baseline | compare
    print("=== contact sheet ===")
    rows = []
    for j in jobs:
        if j["role"] != "compare" or "png" not in j:
            continue
        b = cv2.imread(str(TS_IMAGES / f"{j['baseline']}.png")); c = cv2.imread(str(TS_IMAGES / j["png"]))
        if b is None or c is None:
            continue
        s = 300 / c.shape[1]
        b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for im, txt in ((b, "baseline"), (c, j["combo"])):
            cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        layers = " + ".join(f"{l} {f:.2f}" for l, f in j["layers"])
        cv2.putText(c, f"{j['vessel']} | {j['camera']} | {j['pattern']} | dirt {j['dirt']} | {layers}", (6, c.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(c, f"{j['vessel']} | {j['camera']} | {j['pattern']} | dirt {j['dirt']} | {layers}", (6, c.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([b, c], axis=1))
    if rows:
        per_row = 3
        rows += [np.zeros_like(rows[0])] * ((-len(rows)) % per_row)
        sheet = np.concatenate([np.concatenate(rows[i:i + per_row], axis=1) for i in range(0, len(rows), per_row)], axis=0)
        cv2.imwrite(str(TS_OUT / f"contact_sheet{tag}.png"), sheet)
    save_json({"resolution": [w, h], "spp": spp, "jobs": jobs}, TS_OUT / f"manifest{tag}.json")
    print("done ->", TS_OUT)


if __name__ == "__main__":
    main()

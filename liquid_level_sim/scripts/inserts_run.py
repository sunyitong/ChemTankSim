"""Render vessels with fixtures that PARTLY occlude the backlight pattern — stirrer shaft with blade,
glass thermometer, PTFE dosing tube — as baseline (empty) / compare (filled) pairs, to test whether
the level detector survives occluders that are not liquid.

  python scripts/inserts_run.py --smoke
  python scripts/inserts_run.py                 # 900x1200, 160 spp
Outputs: outputs/inserts/{renders,images,manifest.json,contact_sheet.png,cases.json}
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

import mvp_patterns
from env_light import env_light
from mvp_common import PBRT_EXE, PROJ_ROOT, Setup, check_pbrt, run, save_json
from testset_post import process
from testset_scenes import CAMERAS, TS_PATTERNS, TS_SCENES, VESSELS, Combo, build_scene

OUT = PROJ_ROOT / "outputs" / "inserts"
RENDERS, IMAGES = OUT / "renders", OUT / "images"
AMBIENT = 0.2                         # lit room, so fixtures are not pure silhouettes

ROD = dict(type="rod", material="steel", x=0.0, y=0.0, radius=0.4, z0_frac=0.08, above_rim=6.0, paddle=2.2)
THERMO = dict(type="rod", material="glass", x=4.6, y=1.0, radius=0.3, z0_frac=0.15, above_rim=8.0)
DIP = dict(type="tube", material="ptfe", x=-4.4, y=-0.8, radius=0.4, z0_frac=0.22, above_rim=5.0)

SCENARIOS = [
    (Combo("R1_cyl_stirrer_water50", "cylinder", "normal", "rg_checker", None, 0, [("water", 0.50)], part="R",
           note="stainless stirrer shaft (8 mm) with blade on the axis, fixture present in both frames", inserts=[ROD]),
     "stirrer shaft"),
    (Combo("R2_cyl_dosing_tube_water45_mosaic", "cylinder", "normal", "mosaic", None, 0, [("water", 0.45)], part="R",
           note="PTFE dosing tube (8 mm) inserted for the compare frame only, dipping below the level",
           inserts=[{**DIP, "x": -3.0, "z0_frac": 0.30, "frames": "compare"}]),
     "dosing tube (compare only)"),
    (Combo("R3_cyl_reactor_water60", "cylinder", "normal", "rg_checker", None, 0, [("water", 0.60)], part="R",
           note="reactor set-up: stirrer shaft + glass thermometer + PTFE dip tube, all fixtures in both frames",
           inserts=[ROD, THERMO, DIP]),
     "stirrer + thermometer + dip tube"),
    (Combo("R4_erl_rod_water40_grid", "erlenmeyer", "normal", "grid", None, 0, [("water", 0.40)], part="R",
           note="steel rod (6 mm) off-axis through the Erlenmeyer neck, both frames",
           inserts=[dict(type="rod", material="steel", x=0.9, y=0.0, radius=0.3, z0_frac=0.10, above_rim=6.0)]),
     "steel rod"),
    (Combo("R5_cyl_stirrer_water_oil", "cylinder", "normal", "rg_checker", None, 0, [("water", 0.40), ("sunflower_oil", 0.25)], part="R",
           note="water + sunflower oil with the stirrer shaft: two interfaces behind a partial occluder", inserts=[ROD]),
     "stirrer shaft, two liquids"),
]


def fixture_key(combo: Combo) -> str:
    fixed = [i for i in combo.inserts if i.get("frames", "both") == "both"]
    return "-".join(f"{i.get('type', 'rod')}{i.get('material', 'steel')[0]}{i.get('x', 0):+g}" for i in fixed) or "none"


def render(scene: Path, exr: Path, nthreads):
    cmd = [PBRT_EXE, "--quiet", "--outfile", exr] + (["--nthreads", nthreads] if nthreads else []) + [scene.name]
    t0 = time.time(); run(cmd, cwd=scene.parent, quiet=True); return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--spp", type=int, default=160)
    ap.add_argument("--res", default="900x1200"); ap.add_argument("--only", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--nthreads", type=int, default=None); ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--env", default=None, help="Poly Haven HDRI id prepared by env_map.py (default: constant ambient)")
    a = ap.parse_args()
    light = {"env": env_light(a.env)} if a.env else {"ambient": AMBIENT}
    w, h = (int(v) for v in a.res.lower().split("x")); spp, tag = a.spp, ""
    if a.smoke:
        w, h, spp, tag = 300, 400, 24, "_smoke"
    for d in (OUT, RENDERS, IMAGES, TS_SCENES, TS_PATTERNS):
        d.mkdir(parents=True, exist_ok=True)
    check_pbrt()
    mvp_patterns.PATTERN_DIR = TS_PATTERNS; mvp_patterns.generate_all(Setup(panel_w=60.0, panel_h=80.0))
    combos = [(c, f) for c, f in SCENARIOS if not a.only or any(c.name.startswith(p) for p in a.only.split(","))]

    print("=== scenes ===")
    jobs, seen = [], {}
    for c, fixture in combos:
        bkey = (c.vessel, c.camera, c.pattern, fixture_key(c))
        if bkey not in seen:                               # baseline: empty vessel with the same fixtures
            bname = f"base_{c.vessel}_{c.camera}_{c.pattern}_{fixture_key(c)}{tag}"
            text, gt = build_scene(c, False, w, h, spp, light, bname + ".exr")
            (TS_SCENES / f"{bname}.pbrt").write_text(text, encoding="utf-8")
            jobs.append({"name": bname, "role": "baseline", "combo": None, "vessel": c.vessel, "camera": c.camera, "pattern": c.pattern,
                         "fixtures": [i for i in c.inserts if i.get("frames", "both") == "both"], "layers": [], "gt": gt, "jitter": False})
            seen[bkey] = bname
        name = f"{c.name}{tag}"
        text, gt = build_scene(c, True, w, h, spp, light, name + ".exr")
        (TS_SCENES / f"{name}.pbrt").write_text(text, encoding="utf-8")
        jobs.append({"name": name, "role": "compare", "combo": c.name, "baseline": seen[bkey], "vessel": c.vessel, "camera": c.camera,
                     "pattern": c.pattern, "fixtures": c.inserts, "layers": c.layers, "note": c.note, "gt": gt, "jitter": True})
    print(f"  {len(jobs)} scenes ({sum(j['role'] == 'baseline' for j in jobs)} baselines) at {w}x{h}, {spp} spp")

    if not a.skip_render:
        print("=== render ===")
        for i, j in enumerate(jobs):
            exr = RENDERS / f"{j['name']}.exr"
            if exr.exists() and not a.force:
                print(f"[{i + 1}/{len(jobs)}] {j['name']}: cached"); continue
            t = render(TS_SCENES / f"{j['name']}.pbrt", exr, a.nthreads); j["render_seconds"] = round(t, 1)
            print(f"[{i + 1}/{len(jobs)}] {j['name']}: {t:.1f}s", flush=True)

    print("=== post-process ===")
    rng = np.random.default_rng(91)
    for j in jobs:
        exr, png = RENDERS / f"{j['name']}.exr", IMAGES / f"{j['name']}.png"
        if exr.exists():
            j["post"] = process(exr, png, rng, j["jitter"]); j["png"] = png.name

    print("=== contact sheet + cases ===")
    rows, cases = [], []
    for c, fixture in combos:
        k = next((j for j in jobs if j["combo"] == c.name and j["role"] == "compare"), None)
        b = next((j for j in jobs if k and j["name"] == k.get("baseline")), None)
        if not (b and k and "png" in b and "png" in k):
            continue
        ib, ik = cv2.imread(str(IMAGES / b["png"])), cv2.imread(str(IMAGES / k["png"]))
        s = 320 / ik.shape[1]
        ib = cv2.resize(ib, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); ik = cv2.resize(ik, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for im, txt in ((ib, "baseline (empty, fixtures)"), (ik, c.name)):
            cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([ib, ik], axis=1))
        g = k["gt"]; v = VESSELS[c.vessel]; cam = CAMERAS[c.camera]
        r_max = max(v.r_out(z) for z in np.linspace(0, v.H, 200)) if v.kind == "revolved" else v.params["R"]
        half = math.tan(math.asin(min(r_max / cam["dist"], 0.99))) / math.tan(math.radians(cam["fov"]) / 2) * w / 2 * 0.94
        y0, y1 = max(0, g["rim_row_front"] + 4), min(h, g["bottom_row_front"] - 4)
        cases.append({"case": c.name, "baseline": b["name"], "compare": k["name"],
                      "roi_frac": [max(0.0, (w / 2 - half) / w), y0 / h, min(1.0, 2 * half / w), (y1 - y0) / h],
                      "gt_rows_frac": [l["top_row_front"] / h for l in g["layers"]], "gt_kinds": [l["kind"] for l in g["layers"]],
                      "gt_liquids": [l["liquid"] for l in g["layers"]],
                      "label": f"{c.vessel} · {c.camera} · {c.pattern} · {fixture}", "note": c.note})
    if rows:
        per = 2; rows += [np.zeros_like(rows[0])] * ((-len(rows)) % per)
        cv2.imwrite(str(OUT / f"contact_sheet{tag}.png"), np.concatenate([np.concatenate(rows[i:i + per], axis=1) for i in range(0, len(rows), per)], axis=0))
    save_json({"resolution": [w, h], "spp": spp, "env": a.env, "jobs": jobs}, OUT / f"manifest{tag}.json")
    save_json({"resolution": [w, h], "cases": cases, "image_dir": str(IMAGES)}, OUT / f"cases{tag}.json")
    print("done ->", OUT)


if __name__ == "__main__":
    main()

"""K group: two more vessels with a METAL fixture AND limescale, lit by a laboratory HDRI environment.
The compare frame's environment is rotated and dimmed a little (someone moved, a door opened): ambient
interference that the detector must ignore.

  python scripts/lab_run.py --smoke
  python scripts/lab_run.py --env childrens_hospital
Outputs: outputs/lab/{renders,images,manifest.json,contact_sheet.png,cases.json}
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
from scale_run import scale_textures
from testset_post import process
from testset_scenes import CAMERAS, TS_PATTERNS, TS_SCENES, TS_TEX, VESSELS, Combo, build_scene

OUT = PROJ_ROOT / "outputs" / "lab"
RENDERS, IMAGES = OUT / "renders", OUT / "images"
ROD_NECK = dict(type="rod", material="steel", x=0.9, y=0.2, radius=0.4, z0_frac=0.08, above_rim=7.0)
THERMO_NECK = dict(type="rod", material="glass", x=-0.9, y=-0.3, radius=0.25, z0_frac=0.15, above_rim=9.0)
SCENARIOS = [
    (Combo("K1_reagent_rod_scale_tide_water45", "reagent", "normal", "rg_checker", "scale:tide", 0, [("water", 0.45)], part="K",
           note="wide-mouth reagent bottle, steel stirrer rod through the neck, tide-mark limescale, lab HDRI (compare frame: room light rotated 12 deg, dimmed 15 %)",
           inserts=[ROD_NECK]), "steel rod + tide marks"),
    (Combo("K2_gourd_rod_thermo_scale_vase_water40_mosaic", "gourd", "normal", "mosaic", "scale:vase", 0, [("water", 0.40)], part="K",
           note="gourd flask, steel rod + glass thermometer through the neck, water-line ring at 60 % with film, lab HDRI (compare frame: room light rotated 12 deg, dimmed 15 %)",
           inserts=[dict(ROD_NECK, radius=0.3), THERMO_NECK]), "rod + thermometer + limescale ring"),
]
COMPARE_ENV = dict(rotate=12.0, scale_mult=0.85)


def render(scene: Path, exr: Path, nthreads):
    cmd = [PBRT_EXE, "--quiet", "--outfile", exr] + (["--nthreads", nthreads] if nthreads else []) + [scene.name]
    t0 = time.time(); run(cmd, cwd=scene.parent, quiet=True); return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--spp", type=int, default=160)
    ap.add_argument("--res", default="900x1200"); ap.add_argument("--env", default="childrens_hospital"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--nthreads", type=int, default=None)
    a = ap.parse_args()
    w, h = (int(v) for v in a.res.lower().split("x")); spp, tag = a.spp, ""
    if a.smoke:
        w, h, spp, tag = 300, 400, 24, "_smoke"
    for d in (OUT, RENDERS, IMAGES, TS_SCENES, TS_PATTERNS, TS_TEX):
        d.mkdir(parents=True, exist_ok=True)
    check_pbrt()
    mvp_patterns.PATTERN_DIR = TS_PATTERNS; mvp_patterns.generate_all(Setup(panel_w=60.0, panel_h=80.0))
    env0, env1 = env_light(a.env), env_light(a.env, **COMPARE_ENV)
    print("=== scenes ===")
    jobs = []
    for c, fixture in SCENARIOS:
        tex = scale_textures(c.vessel, c.dirt.split(":")[1]); tex["env"] = env1
        bname = f"base_{c.name}{tag}"
        text, gt = build_scene(c, False, w, h, spp, {"env": env0}, bname + ".exr")
        (TS_SCENES / f"{bname}.pbrt").write_text(text, encoding="utf-8")
        jobs.append({"name": bname, "role": "baseline", "combo": c.name, "vessel": c.vessel, "camera": c.camera, "pattern": c.pattern,
                     "fixtures": c.inserts, "env": env0, "layers": [], "gt": gt, "jitter": False})
        name = f"{c.name}{tag}"
        text, gt = build_scene(c, True, w, h, spp, tex, name + ".exr")
        (TS_SCENES / f"{name}.pbrt").write_text(text, encoding="utf-8")
        jobs.append({"name": name, "role": "compare", "combo": c.name, "baseline": bname, "vessel": c.vessel, "camera": c.camera,
                     "pattern": c.pattern, "fixtures": c.inserts, "deposit": tex["preset"], "env": env1, "layers": c.layers, "note": c.note,
                     "gt": gt, "jitter": True})
    print(f"  {len(jobs)} scenes at {w}x{h}, {spp} spp, env {a.env}")
    print("=== render ===")
    for i, j in enumerate(jobs):
        exr = RENDERS / f"{j['name']}.exr"
        if exr.exists() and not a.force:
            print(f"[{i + 1}/{len(jobs)}] {j['name']}: cached"); continue
        t = render(TS_SCENES / f"{j['name']}.pbrt", exr, a.nthreads); j["render_seconds"] = round(t, 1)
        print(f"[{i + 1}/{len(jobs)}] {j['name']}: {t:.1f}s", flush=True)
    print("=== post-process ===")
    rng = np.random.default_rng(123)
    for j in jobs:
        exr, png = RENDERS / f"{j['name']}.exr", IMAGES / f"{j['name']}.png"
        if exr.exists():
            j["post"] = process(exr, png, rng, j["jitter"]); j["png"] = png.name
    print("=== contact sheet + cases ===")
    rows, cases = [], []
    for c, fixture in SCENARIOS:
        k = next((j for j in jobs if j["combo"] == c.name and j["role"] == "compare"), None)
        b = next((j for j in jobs if j["combo"] == c.name and j["role"] == "baseline"), None)
        if not (b and k and "png" in b and "png" in k):
            continue
        ib, ik = cv2.imread(str(IMAGES / b["png"])), cv2.imread(str(IMAGES / k["png"]))
        s = 320 / ik.shape[1]
        ib = cv2.resize(ib, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); ik = cv2.resize(ik, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for im, txt in ((ib, "baseline (clean, fixtures, lab HDRI)"), (ik, c.name)):
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
                      "label": f"{c.vessel} · {c.camera} · {c.pattern} · {fixture} · lab HDRI", "note": c.note})
    if rows:
        cv2.imwrite(str(OUT / f"contact_sheet{tag}.png"), np.concatenate(rows, axis=1))
    save_json({"resolution": [w, h], "spp": spp, "env": a.env, "jobs": jobs}, OUT / f"manifest{tag}.json")
    save_json({"resolution": [w, h], "cases": cases, "image_dir": str(IMAGES)}, OUT / f"cases{tag}.json")
    print("done ->", OUT)


if __name__ == "__main__":
    main()

"""Render limescale reference samples: vessels whose INNER wall carries the physically motivated
deposit of scale_model.py (water-line ring + creep, standing film, tide marks, droplets, drips),
as baseline (empty, deposit dry) / compare (filled, submerged deposit index-matched) pairs, plus
macro close-ups of the water line.

  python scripts/scale_run.py --smoke
  python scripts/scale_run.py                 # 900x1200, 192 spp
Outputs: outputs/scale/{textures,renders,images,manifest.json,contact_sheet.png,cases.json}
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import mvp_patterns
from mvp_common import PBRT_EXE, PROJ_ROOT, Setup, check_pbrt, run, save_json
from scale_model import PRESETS, generate_tau, write_maps
from testset_post import process
from testset_scenes import CAMERAS, TS_PATTERNS, TS_SCENES, TS_TEX, VESSELS, Combo, build_scene

OUT = PROJ_ROOT / "outputs" / "scale"
RENDERS, IMAGES = OUT / "renders", OUT / "images"

SCENARIOS = [
    # name, vessel, preset, camera, pattern, current fill, cam_dz mode
    Combo("S1_cyl_vase_line60_water45", "cylinder", "normal", "rg_checker", "scale:vase", 0, [("water", 0.45)],
          note="long-standing line at 60 %, film below; now filled to 45 %: film below 45 % is submerged"),
    Combo("S2_cyl_tide_water30_mosaic", "cylinder", "normal", "mosaic", "scale:tide", 0, [("water", 0.30)],
          note="level fell 60 -> 45 % in steps (tide marks), splashes and pour streaks; now 30 %"),
    Combo("S3_gourd_vase_line60_water40", "gourd", "normal", "rg_checker", "scale:vase", 0, [("water", 0.40)],
          note="same deposit history on the gourd flask"),
    Combo("S4_cyl_vase_macro_checker", "cylinder", "macro", "rg_checker", "scale:vase", 0, [("water", 0.45)],
          note="macro view of the water-line ring, checker panel"),
    Combo("S5_cyl_vase_macro_white", "cylinder", "macro", "white", "scale:vase", 0, [("water", 0.45)],
          note="macro view of the water-line ring, white panel"),
    # ---- position / form / extent variants
    Combo("V1_cyl_line30_water55", "cylinder", "normal", "rg_checker", "scale:line30", 0, [("water", 0.55)],
          note="old line at 30 %, now filled above it: the ring and film are submerged (wet, index-matched)"),
    Combo("V2_cyl_line50thin_water35_grid", "cylinder", "normal", "grid", "scale:line50_thin", 0, [("water", 0.35)],
          note="a single thin ring at 50 %, no film"),
    Combo("V3_erl_line75heavy_water40_mosaic", "erlenmeyer", "normal", "mosaic", "scale:line75_heavy", 0, [("water", 0.40)],
          note="heavy iron-tinted ring at 75 % on the Erlenmeyer cone, film below"),
    Combo("V4_reagent_line92full_water60", "reagent", "normal", "rg_checker", "scale:line92_full", 0, [("water", 0.60)],
          note="bottle kept nearly full: film on the whole wall, ring at 92 %"),
    Combo("V5_cyl_band4060_water50", "cylinder", "normal", "rg_checker", "scale:band4060", 0, [("water", 0.50)],
          note="level fluctuated 40-60 %: dense tide-mark band; current level inside the band"),
    Combo("V6_gourd_tilted_water45_mosaic", "gourd", "normal", "mosaic", "scale:tilted", 0, [("water", 0.45)],
          note="gourd stored tilted while drying: inclined line and film boundary"),
    Combo("V7_cyl_splash_water30_wide", "cylinder", "wide", "rg_checker", "scale:splash", 0, [("water", 0.30)],
          note="splash/pour dominated: many droplet rings and streaks, faint line at 45 %"),
    Combo("V8_cyl_sector_water50", "cylinder", "normal", "rg_checker", "scale:sector", 0, [("water", 0.50)],
          note="deposit only on one third of the circumference, current level at the old line"),
    Combo("V9_erl_dripsonly_water45_tele", "erlenmeyer", "tele", "rg_checker", "scale:drips_only", 0, [("water", 0.45)],
          note="no water line: pour streaks from the rim and scattered droplets"),
    Combo("V10_cyl_line50thin_water50_white", "cylinder", "normal", "white", "scale:line50_thin", 0, [("water", 0.50)],
          note="thin old ring exactly at the current level, white panel (hardest case)"),
]


def scale_textures(vessel_name: str, preset: str) -> dict:
    v = VESSELS[vessel_name]
    H_mm = v.inner_height * 10
    if v.kind == "revolved":
        r_max = max(v.r_in(z) for z in np.linspace(v.tb, v.H, 200))
        circ = 2 * math.pi * r_max * 10
    else:
        circ = 2 * math.pi * (v.params["R"] - v.t) * 10
    p = PRESETS[preset](H_mm, H_mm - 3)
    tau, info = generate_tau(p, circ, H_mm)
    tag = f"{preset}_{vessel_name}"
    files = write_maps(tau, TS_TEX, tag, p.iron)
    files.update(kind="scale", info=info, preset=preset, line_fill=p.line_mm / H_mm, ambient=0.25)
    return files


def render(scene: Path, exr: Path, nthreads):
    cmd = [PBRT_EXE, "--quiet", "--outfile", exr] + (["--nthreads", nthreads] if nthreads else []) + [scene.name]
    t0 = time.time(); run(cmd, cwd=scene.parent, quiet=True); return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--spp", type=int, default=192)
    ap.add_argument("--res", default="900x1200"); ap.add_argument("--only", default=None); ap.add_argument("--force", action="store_true")
    ap.add_argument("--nthreads", type=int, default=None); ap.add_argument("--skip-render", action="store_true")
    a = ap.parse_args()
    w, h = (int(v) for v in a.res.lower().split("x")); spp, tag = a.spp, ""
    if a.smoke:
        w, h, spp, tag = 300, 400, 32, "_smoke"
    for d in (OUT, RENDERS, IMAGES, TS_TEX, TS_SCENES, TS_PATTERNS):
        d.mkdir(parents=True, exist_ok=True)
    check_pbrt()
    mvp_patterns.PATTERN_DIR = TS_PATTERNS; mvp_patterns.generate_all(Setup(panel_w=60.0, panel_h=80.0))
    combos = [c for c in SCENARIOS if not a.only or any(c.name.startswith(p) for p in a.only.split(","))]

    print("=== deposit textures ===")
    tex_cache = {}
    for c in combos:
        key = (c.vessel, c.dirt.split(":")[1])
        if key not in tex_cache:
            tex_cache[key] = scale_textures(*key)
            print(f"  {key}: {tex_cache[key]['info']}")

    print("=== scenes ===")
    # Baselines are the vessel as it was photographed when NEW: clean glass, same camera / panel /
    # room light, no liquid. Only the compare frame carries liquid AND the deposit. Baselines are
    # shared between scenarios with the same vessel, camera (incl. aim offset) and pattern.
    jobs, seen = [], {}
    for c in combos:
        tex = tex_cache[(c.vessel, c.dirt.split(":")[1])]
        v = VESSELS[c.vessel]
        cc = c
        if c.camera == "macro":                            # aim the macro camera at the water-line ring
            line_z = v.tb + tex["line_fill"] * v.inner_height
            cc = replace(c, cam_dz=line_z - v.H / 2)
        bkey = (c.vessel, c.camera, c.pattern, round(cc.cam_dz, 3))
        if bkey not in seen:
            bname = f"base_{c.vessel}_{c.camera}_{c.pattern}_dz{cc.cam_dz:g}{tag}"
            text, gt = build_scene(cc, False, w, h, spp, {"ambient": tex["ambient"]}, bname + ".exr")
            (TS_SCENES / f"{bname}.pbrt").write_text(text, encoding="utf-8")
            jobs.append({"name": bname, "role": "baseline", "combo": None, "vessel": c.vessel, "camera": c.camera, "pattern": c.pattern,
                         "deposit": None, "layers": [], "note": "clean vessel, no liquid", "gt": gt, "jitter": False, "cam_dz": cc.cam_dz})
            seen[bkey] = bname
        name = f"{c.name}{tag}"
        text, gt = build_scene(cc, True, w, h, spp, tex, name + ".exr")
        (TS_SCENES / f"{name}.pbrt").write_text(text, encoding="utf-8")
        jobs.append({"name": name, "role": "compare", "combo": c.name, "baseline": seen[bkey], "vessel": c.vessel, "camera": c.camera,
                     "pattern": c.pattern, "deposit": tex["preset"], "deposit_line_fill": tex["line_fill"], "layers": c.layers,
                     "note": c.note, "gt": gt, "jitter": True, "cam_dz": cc.cam_dz})
    print(f"  {len(jobs)} scenes ({sum(j['role'] == 'baseline' for j in jobs)} clean baselines) at {w}x{h}, {spp} spp")

    if not a.skip_render:
        print("=== render ===")
        for i, j in enumerate(jobs):
            exr = RENDERS / f"{j['name']}.exr"
            if exr.exists() and not a.force:
                print(f"[{i + 1}/{len(jobs)}] {j['name']}: cached"); continue
            t = render(TS_SCENES / f"{j['name']}.pbrt", exr, a.nthreads); j["render_seconds"] = round(t, 1)
            print(f"[{i + 1}/{len(jobs)}] {j['name']}: {t:.1f}s", flush=True)

    print("=== post-process ===")
    rng = np.random.default_rng(77)
    for j in jobs:
        exr, png = RENDERS / f"{j['name']}.exr", IMAGES / f"{j['name']}.png"
        if exr.exists():
            j["post"] = process(exr, png, rng, j["jitter"]); j["png"] = png.name

    print("=== contact sheet + cases ===")
    rows, cases = [], []
    for c in combos:
        k = next((j for j in jobs if j["combo"] == c.name and j["role"] == "compare"), None)
        b = next((j for j in jobs if k and j["name"] == k.get("baseline")), None)
        if not (b and k and "png" in b and "png" in k):
            continue
        ib, ik = cv2.imread(str(IMAGES / b["png"])), cv2.imread(str(IMAGES / k["png"]))
        s = 320 / ik.shape[1]
        ib = cv2.resize(ib, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); ik = cv2.resize(ik, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for im, txt in ((ib, "baseline (clean, empty)"), (ik, c.name)):
            cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.concatenate([ib, ik], axis=1))
        # web-app case (ROI = vessel interior; GT = current level + deposit line)
        g = k["gt"]; v = VESSELS[c.vessel]; cam = CAMERAS[c.camera]
        r_max = max(v.r_out(z) for z in np.linspace(0, v.H, 200)) if v.kind == "revolved" else v.params["R"]
        half = math.tan(math.asin(min(r_max / cam["dist"], 0.99))) / math.tan(math.radians(cam["fov"]) / 2) * w / 2 * 0.94
        y0, y1 = max(0, g["rim_row_front"] + 4), min(h, g["bottom_row_front"] - 4)
        cases.append({"case": c.name, "baseline": b["name"], "compare": k["name"],
                      "roi_frac": [max(0.0, (w / 2 - half) / w), y0 / h, min(1.0, 2 * half / w), (y1 - y0) / h],
                      "gt_rows_frac": [l["top_row_front"] / h for l in g["layers"]], "gt_kinds": [l["kind"] for l in g["layers"]],
                      "gt_liquids": [l["liquid"] for l in g["layers"]],
                      "label": f"{c.vessel} · {c.camera} · {c.pattern} · limescale {tex_cache[(c.vessel, c.dirt.split(':')[1])]['preset']}",
                      "note": c.note})
    if rows:
        per = 2; rows += [np.zeros_like(rows[0])] * ((-len(rows)) % per)
        cv2.imwrite(str(OUT / f"contact_sheet{tag}.png"), np.concatenate([np.concatenate(rows[i:i + per], axis=1) for i in range(0, len(rows), per)], axis=0))
    save_json({"resolution": [w, h], "spp": spp, "jobs": jobs}, OUT / f"manifest{tag}.json")
    save_json({"resolution": [w, h], "cases": cases, "image_dir": str(IMAGES)}, OUT / f"cases{tag}.json")
    print("done ->", OUT)


if __name__ == "__main__":
    main()

"""Diagnostic: does render quality limit the F2 (Erlenmeyer) pouring-sequence accuracy? Re-render a few
frames of a sequence at much higher quality and run the detector on them:
  hq   : 900x1200, 128 spp (4x pixels, 4x samples)   -> detected at full resolution and downscaled to 450x600
  spp  : 450x600, 256 spp (8x samples, same pixels)  -> isolates sensor/path-tracing noise from resolution
The sequence frames (450x600, 32 spp) are the reference. Writes outputs/pour/quality/<seq>_quality.json.

  python scripts/diag_pour_quality.py F2 8 24 36 60 76
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import mvp_patterns
import pour_video as pv
from env_light import env_light
from level_detect import detect
from mvp_common import PBRT_EXE, PROJ_ROOT, Setup, run
from scale_run import scale_textures
from testset_post import process
from testset_scenes import TS_PATTERNS, TS_SCENES, VESSELS, Combo, build_scene

sid = sys.argv[1] if len(sys.argv) > 1 else "F2"
frames = [int(a) for a in sys.argv[2:]] or [8, 24, 36, 60, 76]
seq = pv.SEQUENCES[sid]
OUT = PROJ_ROOT / "outputs" / "pour" / "quality"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(PROJ_ROOT / f"outputs/pour/{sid}/video_meta.json", encoding="utf-8"))
W0, H0 = meta["resolution"]; fx, fy, fw, fh = meta["roi_frac"]
vessel = VESSELS[seq["vessel"]]; tube = seq["tube"]
env0 = {"env": env_light("childrens_hospital")}
env1 = {"env": env_light("childrens_hospital", **pv.COMPARE_ENV)} if seq["env_change"] else env0
tex1 = dict(env1)
if seq["dirt"]:
    tex1.update(scale_textures(seq["vessel"], seq["dirt"].split(":")[1])); tex1["env"] = env1["env"]
mvp_patterns.PATTERN_DIR = TS_PATTERNS; mvp_patterns.generate_all(Setup(panel_w=60.0, panel_h=80.0))
srgb = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def render_png(name, combo, filled, w, h, spp, tex):
    text, gt = build_scene(replace(combo, name=name), filled, w, h, spp, tex, name + ".exr")
    (TS_SCENES / f"{name}.pbrt").write_text(text, encoding="utf-8")
    exr = OUT / f"{name}.exr"
    if not exr.exists():
        run([PBRT_EXE, "--quiet", "--outfile", exr, f"{name}.pbrt"], cwd=TS_SCENES, quiet=True)
    png = OUT / f"{name}.png"; process(exr, png, np.random.default_rng(3), jitter=False)
    return png, gt


def roi(img, W, H):
    x0, y0, x1, y1 = round(fx * W), round(fy * H), round(fx * W + fw * W), round(fy * H + fh * H)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def level_pct(det, W, H, box, gt_frac):
    x0, y0, x1, y1 = box
    est = (y1 - (y0 + det.levels[0]["row"] * 2)) / (y1 - y0) * 100 if det.levels else None
    gt = (y1 - gt_frac * H) / (y1 - y0) * 100
    hyp = {k: (None if v is None else (y1 - (y0 + v * 2)) / (y1 - y0) * 100) for k, v in
           {"top": det.surface, "dip": det.dip, "first": det.first_jump}.items()}
    return est, gt, hyp


results = {"sequence": sid, "frames": []}
variants = {"hq": (900, 1200, 128), "spp": (450, 600, 256)}
base_pngs = {}
for vname, (w, h, spp) in variants.items():
    combo0 = Combo(f"Q{sid}_{vname}_base", seq["vessel"], "normal", seq["pattern"], None, 0, [("water", seq["f0"])], part="Q", inserts=[tube])
    base_pngs[vname], _ = render_png(f"Q{sid}_{vname}_base", combo0, False, w, h, spp, env0)
    print(f"{vname}: baseline done", flush=True)
ref_rows = {r["i"]: r for r in json.load(open(PROJ_ROOT / f"outputs/pour/{sid}/eval_video.json"))["frames"]}
for i in frames:
    fr = meta["frames"][i]; fill, t = fr["fill"], fr["t"]
    z_lvl = vessel.tb + fill * vessel.inner_height
    rng = np.random.default_rng(5)
    for _ in range(i + 1):                                     # same droplet randomness as the sequence frame
        stream = pv.stream_for(vessel, tube, z_lvl, t, rng)
    row = {"i": i, "t": t, "fill": fill, "seq_450x600_32spp": {"est": ref_rows[i]["est_pct"], "gt": ref_rows[i]["gt_pct"], "hyp": ref_rows[i]["hyp"]}}
    for vname, (w, h, spp) in variants.items():
        combo = Combo(f"Q{sid}_{vname}_f{i:03d}", seq["vessel"], "normal", seq["pattern"], seq["dirt"], 0, [("water", fill)], part="Q", inserts=[tube, stream])
        png, gt = render_png(f"Q{sid}_{vname}_f{i:03d}", combo, True, w, h, spp, tex1)
        gt_frac = gt["layers"][0]["top_row_front"] / h
        B, C = srgb(base_pngs[vname]), srgb(png)
        Br, box = roi(B, w, h); Cr, _ = roi(C, w, h)
        det = detect(Br, Cr, horizon_row=(h / 2 - box[1]) / 2)
        est, gtp, hyp = level_pct(det, w, h, box, gt_frac)
        row[f"{vname}_{w}x{h}_{spp}spp"] = {"est": est, "gt": gtp, "err": None if est is None else est - gtp, "mode": det.mode, "hyp": hyp}
        if w != W0:                                            # the same high-quality render downscaled to the sequence size
            Bs, Cs = (cv2.resize(x, (W0, H0), interpolation=cv2.INTER_AREA) for x in (B, C))
            Br, box = roi(Bs, W0, H0); Cr, _ = roi(Cs, W0, H0)
            det = detect(Br, Cr, horizon_row=(H0 / 2 - box[1]) / 2)
            est, gtp, hyp = level_pct(det, W0, H0, box, gt_frac)
            row[f"{vname}_down_{W0}x{H0}"] = {"est": est, "gt": gtp, "err": None if est is None else est - gtp, "mode": det.mode, "hyp": hyp}
        print(f"frame {i} {vname}: done", flush=True)
    results["frames"].append(row)
json.dump(results, open(OUT / f"{sid}_quality.json", "w"), indent=1)
print()
print(f"{'frame':>5} {'GT':>6} | {'seq 32spp':>10} | {'spp256':>8} | {'hq full':>8} | {'hq->450':>8}")
for r in results["frames"]:
    f = lambda d: "   -  " if d["est"] is None else f"{d['est'] - d['gt']:+6.1f}"
    print(f"{r['i']:5d} {r['seq_450x600_32spp']['gt']:6.1f} | {r['seq_450x600_32spp']['est'] - r['seq_450x600_32spp']['gt']:+10.1f} | {f(r['spp_450x600_256spp']):>8} | {f(r['hq_900x1200_128spp']):>8} | {f(r['hq_down_450x600']):>8}")

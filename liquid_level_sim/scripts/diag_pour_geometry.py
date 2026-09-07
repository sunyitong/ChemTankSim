"""Diagnostic: where do the detector's surface hypotheses fall relative to the FRONT and BACK rim of the
free surface? build_scene projects both rims (top_row_front = ground truth, top_row_back = far rim as seen
from the camera). Joins them with the per-frame hypotheses of eval_video.json for a pouring sequence.

  python scripts/diag_pour_geometry.py F2
"""
import json
import sys
from dataclasses import replace

import numpy as np

import pour_video as pv
from env_light import env_light
from mvp_common import PROJ_ROOT
from scale_run import scale_textures
from testset_scenes import VESSELS, Combo, build_scene

sid = sys.argv[1] if len(sys.argv) > 1 else "F2"
seq = pv.SEQUENCES[sid]
meta = json.load(open(PROJ_ROOT / f"outputs/pour/{sid}/video_meta.json", encoding="utf-8"))
ev = {r["i"]: r for r in json.load(open(PROJ_ROOT / f"outputs/pour/{sid}/eval_video.json"))["frames"]}
W, H = meta["resolution"]; fx, fy, fw, fh = meta["roi_frac"]
y0, y1 = round(fy * H), round(fy * H + fh * H)
pct = lambda row: (y1 - row) / (y1 - y0) * 100
vessel = VESSELS[seq["vessel"]]; tube = seq["tube"]
tex1 = {"env": env_light("childrens_hospital", **pv.COMPARE_ENV)} if seq["env_change"] else {"env": env_light("childrens_hospital")}
if seq["dirt"]:
    env = tex1["env"]; tex1.update(scale_textures(seq["vessel"], seq["dirt"].split(":")[1])); tex1["env"] = env
base = Combo(seq["name"], seq["vessel"], "normal", seq["pattern"], seq["dirt"], 0, [("water", seq["f0"])], part="F", inserts=[tube])
rows = []
for fr in meta["frames"]:
    c = replace(base, name=f"{sid}_f{fr['i']:03d}", layers=[("water", fr["fill"])])
    _, gt = build_scene(c, True, W, H, 32, tex1, f"{c.name}.exr")
    lay = gt["layers"][0]
    e = ev[fr["i"]]; h = e["hyp"]
    rows.append(dict(i=fr["i"], t=fr["t"], fill=fr["fill"], front=pct(lay["top_row_front"]), back=pct(lay["top_row_back"]),
                     raw=h["raw"], dip=h["dip"], first=h["first"], top=h["top"], tracked=e["tracked_pct"]))
cam_z = vessel.H / 2
print(f"{sid}: camera height {cam_z:.2f} cm above the vessel base; level passes the camera height at fill "
      f"{(cam_z - vessel.tb) / vessel.inner_height:.3f}")
print(f"{'i':>3} {'fill':>5} {'front(GT)':>9} {'back':>6} {'band':>5} | {'top':>6} {'dip':>6} {'first':>6} {'raw':>6} {'trk':>6} | top-back raw-front raw-back")
f = lambda v: "   -  " if v is None else f"{v:6.1f}"
for r in rows[::4]:
    band = r["back"] - r["front"]
    tb = "   -  " if r["top"] is None else f"{r['top'] - r['back']:+6.1f}"
    print(f"{r['i']:3d} {r['fill']:5.3f} {r['front']:9.1f} {r['back']:6.1f} {band:+5.1f} | {f(r['top'])} {f(r['dip'])} {f(r['first'])} {f(r['raw'])} {f(r['tracked'])} | {tb:>8} {r['raw'] - r['front']:+9.1f} {r['raw'] - r['back']:+8.1f}")
above = [r for r in rows if r["back"] > r["front"] + 0.5]          # camera above the level: far rim projects higher
below = [r for r in rows if r["back"] < r["front"] - 0.5]
for name, sub in (("camera above the level", above), ("camera below the level", below)):
    if not sub:
        continue
    tb = [r["top"] - r["back"] for r in sub if r["top"] is not None]
    rf = [r["raw"] - r["front"] for r in sub]; rb = [r["raw"] - r["back"] for r in sub]
    tf = [r["tracked"] - r["front"] for r in sub]
    print(f"\n{name}: {len(sub)} frames, far-rim band {np.mean([r['back'] - r['front'] for r in sub]):+.1f} pt on average")
    print(f"  identity end - back rim : median {np.median(tb):+.2f} pt (MAD {np.median(np.abs(np.array(tb) - np.median(tb))):.2f}), n={len(tb)}")
    print(f"  raw - front rim (= error): median {np.median(rf):+.2f} pt;  raw - back rim: median {np.median(rb):+.2f} pt")
    print(f"  tracked - front rim      : median {np.median(tf):+.2f} pt")
json.dump(rows, open(PROJ_ROOT / f"outputs/pour/{sid}/geometry_rims.json", "w"), indent=1)

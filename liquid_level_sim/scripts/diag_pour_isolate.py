"""Diagnostic (2026-09-07): isolate what breaks F2 frame 50 (fill 0.40): render variants of the same frame and run the detector on each.
  v_full    : as in the sequence (deposit + env change + tube/stream)
  v_noenvch : deposit, env unchanged (same light as baseline)
  v_nodep   : no deposit, env change
  v_ambient : deposit, constant ambient 0.25 (as the static V3 case), baseline also ambient
usage: exp_f2.py [frame index, default 50]"""
import sys, json, math
sys.path.insert(0, "scripts")
from dataclasses import replace
from pathlib import Path
import cv2, numpy as np
import mvp_patterns
from env_light import env_light
from mvp_common import PBRT_EXE, PROJ_ROOT, Setup, run
from scale_run import scale_textures
from testset_post import process
from testset_scenes import TS_PATTERNS, TS_SCENES, VESSELS, Combo, build_scene
import pour_video as pv
import level_detect as ld

i = int(sys.argv[1]) if len(sys.argv) > 1 else 50
sid, seq = "F2", pv.SEQUENCES["F2"]
OUT = PROJ_ROOT / "outputs" / "pour" / "exp"; OUT.mkdir(parents=True, exist_ok=True)
meta = json.load(open(PROJ_ROOT / f"outputs/pour/{sid}/video_meta.json", encoding="utf-8"))
W, H = meta["resolution"]; fx, fy, fw, fh = meta["roi_frac"]
x0, y0, x1, y1 = round(fx * W), round(fy * H), round(fx * W + fw * W), round(fy * H + fh * H)
fr = meta["frames"][i]; fill, t = fr["fill"], fr["t"]
vessel = VESSELS[seq["vessel"]]; tube = seq["tube"]
z_lvl = vessel.tb + fill * vessel.inner_height
rng = np.random.default_rng(5)
stream = pv.stream_for(vessel, tube, z_lvl, t, rng)
env0 = {"env": env_light("childrens_hospital")}; env1 = {"env": env_light("childrens_hospital", **pv.COMPARE_ENV)}
dep = scale_textures(seq["vessel"], "line75_heavy")


def tex_with(light, deposit):
    tx = dict(light)
    if deposit:
        tx.update(dep); tx["env"] = light.get("env"); tx.pop("ambient", None) if "env" in light else None
        if "env" not in light:
            tx.pop("env", None); tx["ambient"] = light["ambient"]
    return tx


variants = {
    "v_full": (env0, tex_with(env1, True), "scale:line75_heavy"),
    "v_noenvch": (env0, tex_with(env0, True), "scale:line75_heavy"),
    "v_nodep": (env0, dict(env1), None),
    "v_ambient": ({"ambient": 0.25}, tex_with({"ambient": 0.25}, True), "scale:line75_heavy"),
}
mvp_patterns.PATTERN_DIR = TS_PATTERNS; mvp_patterns.generate_all(Setup(panel_w=60.0, panel_h=80.0))
srgb = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
for name, (light0, tex1, dirt) in variants.items():
    combo = Combo(f"X_{name}", seq["vessel"], "normal", seq["pattern"], dirt, 0, [("water", fill)], part="X", inserts=[tube, stream])
    outs = {}
    for tag, filled, tx in (("base", False, light0), ("cmp", True, tex1)):
        nm = f"X_{name}_{tag}"
        text, gt = build_scene(replace(combo, name=nm), filled, W, H, 32, tx, nm + ".exr")
        (TS_SCENES / f"{nm}.pbrt").write_text(text, encoding="utf-8")
        exr = OUT / f"{nm}.exr"
        if not exr.exists():
            run([PBRT_EXE, "--quiet", "--outfile", exr, f"{nm}.pbrt"], cwd=TS_SCENES, quiet=True)
        png = OUT / f"{nm}.png"; process(exr, png, np.random.default_rng(3), jitter=False); outs[tag] = png
    base = srgb(outs["base"])[y0:y1, x0:x1]; img = srgb(outs["cmp"])[y0:y1, x0:x1]
    det = ld.detect(base, img, horizon_row=(H / 2 - y0) / 2); n = len(det.dE)
    pct = lambda r: None if r is None else round((n - r) / n * 100, 1)
    gt_pct = (y1 - fr["gt_rows_frac"][0] * H) / (y1 - y0) * 100
    cls = "".join({0: ".", 1: "I", 2: "L"}[int(v)] for v in det.cls)
    print(f"{name:10s} GT {gt_pct:.1f} % -> {det.mode} levels={[pct(l['row']) for l in det.levels]} top={pct(det.surface)} dip={pct(det.dip)} first={pct(det.first_jump)}")
    print("   cls:", cls)

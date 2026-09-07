"""Filling sequences: a PTFE dosing tube pours water into a vessel at a constant rate; the level rises
linearly from f0 to f1 over SECONDS at FPS. The BASELINE of each sequence is the empty, clean vessel with
the tube already in place. Every frame carries the pour stream (liquid column from the tube tip to the
surface), satellite droplets falling beside it and splash droplets at the impact. Sequences F2 / F3 add a
limescale deposit (dry above the level, index-matched where submerged, so it changes as the level rises)
and the laboratory HDRI, with the room light rotated / dimmed relative to the baseline frame.

  python scripts/pour_video.py --smoke                 # 3 frames per sequence, 300x400, 16 spp (separate dirs)
  python scripts/pour_video.py --only F2,F3            # 450x600, 32 spp, 20 fps x 4 s = 81 frames each
  python scripts/pour_video.py --encode-only           # re-encode the MP4s from existing frames
Outputs: outputs/pour/<seq>/{frames,<name>.mp4,video_meta.json,contact_sheet.png}, outputs/pour/cases_video.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

import mvp_patterns
from env_light import env_light
from mvp_common import PBRT_EXE, PROJ_ROOT, Setup, check_pbrt, run, save_json
from scale_run import scale_textures
from testset_post import process
from testset_scenes import CAMERAS, TS_PATTERNS, TS_SCENES, VESSELS, Combo, build_scene

OUT = PROJ_ROOT / "outputs" / "pour"
COMPARE_ENV = dict(rotate=12.0, scale_mult=0.85)   # room light differs a little from the baseline photograph
MP4_CRF = 26                                       # three embedded sequences must stay well under the 16 MB page budget

SEQUENCES = {
    "F1": dict(name="F1_cyl_pour", vessel="cylinder", pattern="rg_checker", dirt=None, env_change=False,
               tube=dict(type="tube", material="ptfe", x=-2.8, y=0.0, radius=0.4, z0_frac=0.80, above_rim=6.0), f0=0.12, f1=0.55,
               label="cylinder · normal · rg_checker · pouring, 20 fps",
               note="PTFE dosing tube pours water at a constant rate; level rises from 12 % to 55 % in 4 s"),
    "F2": dict(name="F2_erl_pour_scale_mosaic", vessel="erlenmeyer", pattern="mosaic", dirt="scale:line75_heavy", env_change=True,
               tube=dict(type="tube", material="ptfe", x=0.8, y=0.2, radius=0.4, z0_frac=0.82, above_rim=7.0), f0=0.15, f1=0.55,
               label="erlenmeyer · normal · mosaic · limescale line75 · dosing tube · lab HDRI · pouring",
               note="Erlenmeyer with a heavy iron-tinted water-line ring at 75 % and film below it; PTFE dosing tube through the neck pours water 15 -> 55 % in 4 s; laboratory HDRI, room light rotated 12 deg and dimmed 15 % vs the baseline"),
    "F3": dict(name="F3_reagent_pour_scale_tide", vessel="reagent", pattern="rg_checker", dirt="scale:tide", env_change=True,
               tube=dict(type="tube", material="ptfe", x=-1.2, y=0.3, radius=0.4, z0_frac=0.82, above_rim=7.0), f0=0.15, f1=0.55,
               label="reagent bottle · normal · rg_checker · limescale tide marks · dosing tube · lab HDRI · pouring",
               note="Wide-mouth reagent bottle with tide-mark limescale (level fell 60 -> 45 % in steps, splashes, streaks); PTFE dosing tube through the neck pours water 15 -> 55 % in 4 s; laboratory HDRI, room light rotated 12 deg and dimmed 15 % vs the baseline"),
}


def render(scene: Path, exr: Path, nthreads):
    cmd = [PBRT_EXE, "--quiet", "--outfile", exr] + (["--nthreads", nthreads] if nthreads else []) + [scene.name]
    t0 = time.time(); run(cmd, cwd=scene.parent, quiet=True); return time.time() - t0


def stream_for(vessel, tube: dict, z_lvl: float, t: float, rng: np.random.Generator) -> dict:
    z_tip = vessel.tb + tube["z0_frac"] * vessel.inner_height
    x, y = tube["x"], tube["y"]
    drops = []
    for k in range(2):                                     # satellite droplets falling beside the stream
        phase = (t * 2.2 + 0.5 * k) % 1.0
        drops.append((x + (0.6 if k else -0.6), y + 0.2, z_tip - phase * (z_tip - z_lvl - 0.3), 0.2))
    for _ in range(3):                                     # splash droplets around the impact point
        ang, rad = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.2)
        drops.append((x + rad * math.cos(ang), y + rad * math.sin(ang), z_lvl + rng.uniform(0.25, 1.1), rng.uniform(0.12, 0.18)))
    return dict(type="stream", x=x, y=y, r=0.30 + 0.03 * math.sin(2 * math.pi * 3.0 * t), z_top=z_tip - 0.05,
                z_bot=z_lvl - 0.4, droplets=drops)


def encode(frames_dir: Path, fps: int, mp4: Path, crf: int = MP4_CRF):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "frame_%03d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium", "-g", "1", "-bf", "0",   # intra-only: cheap exact seeks
           "-movflags", "+faststart", str(mp4)]
    subprocess.run(cmd, check=True)
    return mp4.stat().st_size


def run_sequence(sid: str, seq: dict, a, w: int, h: int, spp: int, fps: int, seconds: float, n: int, env_id: str) -> dict | None:
    tag = "_smoke" if a.smoke else ""
    sdir, renders = OUT / f"{sid}{tag}", OUT / f"renders{tag}"
    frames_dir = sdir / "frames"
    for d in (sdir, renders, frames_dir):
        d.mkdir(parents=True, exist_ok=True)
    vessel = VESSELS[seq["vessel"]]; cam = CAMERAS["normal"]; tube = seq["tube"]
    base_combo = Combo(seq["name"], seq["vessel"], "normal", seq["pattern"], seq["dirt"], 0, [("water", seq["f0"])], part="F",
                       inserts=[tube], note=seq["note"])
    light0 = {"env": env_light(env_id)}
    light1 = {"env": env_light(env_id, **COMPARE_ENV)} if seq["env_change"] else light0
    tex1 = dict(light1)
    if seq["dirt"]:
        tex1.update(scale_textures(seq["vessel"], seq["dirt"].split(":")[1]))
        tex1["env"] = light1["env"]
    meta_path = sdir / "video_meta.json"
    bname = f"{sid}_baseline"
    if not a.encode_only:
        print(f"=== {sid}: scenes ===")
        rng = np.random.default_rng(5)
        jobs = []
        text, gt0 = build_scene(replace(base_combo, name=bname), False, w, h, spp, light0, bname + ".exr")
        (TS_SCENES / f"{bname}.pbrt").write_text(text, encoding="utf-8")
        jobs.append({"name": bname, "frame": None, "t": None, "fill": 0.0, "gt": gt0})
        for i in range(n):
            t = i / fps if not a.smoke else i * seconds / max(1, n - 1)
            fill = seq["f0"] + (seq["f1"] - seq["f0"]) * (t / seconds)
            z_lvl = vessel.tb + fill * vessel.inner_height
            c = replace(base_combo, name=f"{sid}_f{i:03d}", layers=[("water", fill)], inserts=[tube, stream_for(vessel, tube, z_lvl, t, rng)])
            text, gt = build_scene(c, True, w, h, spp, tex1, f"{c.name}.exr")
            (TS_SCENES / f"{c.name}.pbrt").write_text(text, encoding="utf-8")
            jobs.append({"name": c.name, "frame": i, "t": t, "fill": fill, "gt": gt})
        print(f"  {len(jobs)} scenes at {w}x{h}, {spp} spp")
        for k, j in enumerate(jobs):
            exr = renders / f"{j['name']}.exr"
            if exr.exists() and not a.force:
                continue
            tt = render(TS_SCENES / f"{j['name']}.pbrt", exr, a.nthreads)
            print(f"[{sid} {k + 1}/{len(jobs)}] {j['name']}: {tt:.1f}s", flush=True)
        prng = np.random.default_rng(3)
        frames = []
        for j in jobs:
            exr = renders / f"{j['name']}.exr"
            png = frames_dir / (f"{bname}.png" if j["frame"] is None else f"frame_{j['frame']:03d}.png")
            process(exr, png, prng, jitter=False)              # fixed camera: sensor noise only
            if j["frame"] is not None:
                g = j["gt"]
                frames.append({"i": j["frame"], "t": round(j["t"], 4), "fill": round(j["fill"], 5),
                               "gt_rows_frac": [l["top_row_front"] / h for l in g["layers"]]})
        g = jobs[-1]["gt"]
        r_max = max(vessel.r_out(z) for z in np.linspace(0, vessel.H, 200))
        half = math.tan(math.asin(min(r_max / cam["dist"], 0.99))) / math.tan(math.radians(cam["fov"]) / 2) * w / 2 * 0.94
        y0, y1 = max(0, g["rim_row_front"] + 4), min(h, g["bottom_row_front"] - 4)
        meta = {"case": seq["name"], "kind": "video", "fps": fps, "seconds": seconds, "n_frames": n, "resolution": [w, h], "spp": spp,
                "baseline": bname, "video": f"{seq['name']}.mp4", "image_dir": str(frames_dir), "video_path": str(sdir / f"{seq['name']}.mp4"),
                "roi_frac": [max(0.0, (w / 2 - half) / w), y0 / h, min(1.0, 2 * half / w), (y1 - y0) / h],
                "frames": frames, "label": seq["label"], "note": seq["note"], "tube": tube, "fill_range": [seq["f0"], seq["f1"]],
                "deposit": seq["dirt"], "env": env_id, "env_change": COMPARE_ENV if seq["env_change"] else None}
        save_json(meta, meta_path)
        pick = [frames_dir / f"{bname}.png"] + [frames_dir / f"frame_{i:03d}.png" for i in np.linspace(0, n - 1, 5).astype(int)]
        ims = [cv2.imread(str(p)) for p in pick if p.exists()]
        if ims:
            s = 240 / ims[0].shape[1]
            ims = [cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for im in ims]
            cv2.imwrite(str(sdir / "contact_sheet.png"), np.concatenate(ims, axis=1))
    if not meta_path.exists():
        return None
    meta = json.load(open(meta_path, encoding="utf-8"))
    if not a.smoke:
        size = encode(frames_dir, fps, sdir / f"{seq['name']}.mp4")
        print(f"  {sid} mp4: {size / 1e6:.2f} MB")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--res", default="450x600"); ap.add_argument("--fps", type=int, default=20); ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--nthreads", type=int, default=None); ap.add_argument("--force", action="store_true"); ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated sequence ids (default: all)")
    ap.add_argument("--env", default="childrens_hospital", help="Poly Haven HDRI id prepared by env_map.py")
    a = ap.parse_args()
    w, h = (int(v) for v in a.res.lower().split("x")); spp, fps, seconds = a.spp, a.fps, a.seconds
    n = round(fps * seconds) + 1
    if a.smoke:
        w, h, spp, n = 300, 400, 16, 3
    for d in (OUT, TS_SCENES, TS_PATTERNS):
        d.mkdir(parents=True, exist_ok=True)
    if not a.encode_only:
        check_pbrt()
        mvp_patterns.PATTERN_DIR = TS_PATTERNS; mvp_patterns.generate_all(Setup(panel_w=60.0, panel_h=80.0))
    ids = [s for s in SEQUENCES if not a.only or s in a.only.split(",")]
    for sid in ids:
        run_sequence(sid, SEQUENCES[sid], a, w, h, spp, fps, seconds, n, a.env)
    if not a.smoke:   # every full sequence with metadata (also ones not rendered this run) goes into the bundle list
        allm = [json.load(open(OUT / sid / "video_meta.json", encoding="utf-8")) for sid in SEQUENCES if (OUT / sid / "video_meta.json").exists()]
        save_json({"cases": allm}, OUT / "cases_video.json")
        print(f"cases_video.json: {len(allm)} sequences")
    print("done ->", OUT)


if __name__ == "__main__":
    main()

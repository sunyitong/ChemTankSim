"""Filling sequence: a PTFE dosing tube pours water into the 15 cm cylinder; the level rises linearly
from F0 to F1 over SECONDS at FPS (constant feed). Frame 0 of the pair is the BASELINE: the empty
vessel with the tube already in place. Every frame carries the pour stream (liquid column from the
tube tip to the surface), satellite droplets falling beside it and a few splash droplets at the impact.

  python scripts/pour_video.py --smoke                 # 3 frames, 300x400, 16 spp
  python scripts/pour_video.py                         # 450x600, 32 spp, 20 fps x 4 s = 81 frames
  python scripts/pour_video.py --encode-only           # re-encode the MP4 from existing frames
Outputs: outputs/pour/{renders,frames,F1_cyl_pour.mp4,video_meta.json,cases_video.json,contact_sheet.png}
"""
from __future__ import annotations

import argparse
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
from testset_post import process
from testset_scenes import CAMERAS, TS_PATTERNS, TS_SCENES, VESSELS, Combo, build_scene

OUT = PROJ_ROOT / "outputs" / "pour"
RENDERS, FRAMES = OUT / "renders", OUT / "frames"
TUBE = dict(type="tube", material="ptfe", x=-2.8, y=0.0, radius=0.4, z0_frac=0.80, above_rim=6.0)
AMBIENT = 0.2
F0, F1 = 0.12, 0.55                    # fill fraction of the inner height at t = 0 and t = SECONDS


def render(scene: Path, exr: Path, nthreads):
    cmd = [PBRT_EXE, "--quiet", "--outfile", exr] + (["--nthreads", nthreads] if nthreads else []) + [scene.name]
    t0 = time.time(); run(cmd, cwd=scene.parent, quiet=True); return time.time() - t0


def stream_for(vessel, z_lvl: float, t: float, rng: np.random.Generator) -> dict:
    z_tip = vessel.tb + TUBE["z0_frac"] * vessel.inner_height
    x, y = TUBE["x"], TUBE["y"]
    drops = []
    for k in range(2):                                     # satellite droplets falling beside the stream
        phase = (t * 2.2 + 0.5 * k) % 1.0
        drops.append((x + (0.6 if k else -0.6), y + 0.2, z_tip - phase * (z_tip - z_lvl - 0.3), 0.2))
    for _ in range(3):                                     # splash droplets around the impact point
        ang, rad = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 1.5)
        drops.append((x + rad * math.cos(ang), y + rad * math.sin(ang), z_lvl + rng.uniform(0.25, 1.1), rng.uniform(0.12, 0.18)))
    return dict(type="stream", x=x, y=y, r=0.30 + 0.03 * math.sin(2 * math.pi * 3.0 * t), z_top=z_tip - 0.05,
                z_bot=z_lvl - 0.4, droplets=drops)


def encode(frames_dir: Path, fps: int, mp4: Path):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "frame_%03d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21", "-preset", "medium", "-g", "1", "-bf", "0",   # intra-only: cheap exact seeks
           "-movflags", "+faststart", str(mp4)]
    subprocess.run(cmd, check=True)
    return mp4.stat().st_size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--spp", type=int, default=32)
    ap.add_argument("--res", default="450x600"); ap.add_argument("--fps", type=int, default=20); ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--nthreads", type=int, default=None); ap.add_argument("--force", action="store_true"); ap.add_argument("--encode-only", action="store_true")
    ap.add_argument("--env", default=None, help="Poly Haven HDRI id prepared by env_map.py (default: constant ambient)")
    a = ap.parse_args()
    light = {"env": env_light(a.env)} if a.env else {"ambient": AMBIENT}
    w, h = (int(v) for v in a.res.lower().split("x")); spp, fps, seconds = a.spp, a.fps, a.seconds
    n = round(fps * seconds) + 1
    if a.smoke:
        w, h, spp, n = 300, 400, 16, 3
    for d in (OUT, RENDERS, FRAMES, TS_SCENES, TS_PATTERNS):
        d.mkdir(parents=True, exist_ok=True)
    vessel = VESSELS["cylinder"]; cam = CAMERAS["normal"]
    base_combo = Combo("F1_cyl_pour", "cylinder", "normal", "rg_checker", None, 0, [("water", F0)], part="F", inserts=[TUBE],
                       note="PTFE dosing tube pours water at a constant rate; level rises from 12 % to 55 % in 4 s")
    if not a.encode_only:
        check_pbrt()
        mvp_patterns.PATTERN_DIR = TS_PATTERNS; mvp_patterns.generate_all(Setup(panel_w=60.0, panel_h=80.0))
        print("=== scenes ===")
        rng = np.random.default_rng(5)
        jobs = []
        text, gt0 = build_scene(replace(base_combo, name="F1_baseline"), False, w, h, spp, light, "F1_baseline.exr")
        (TS_SCENES / "F1_baseline.pbrt").write_text(text, encoding="utf-8")
        jobs.append({"name": "F1_baseline", "frame": None, "t": None, "fill": 0.0, "gt": gt0})
        for i in range(n):
            t = i / fps if not a.smoke else i * seconds / max(1, n - 1)
            fill = F0 + (F1 - F0) * (t / seconds)
            z_lvl = vessel.tb + fill * vessel.inner_height
            c = replace(base_combo, name=f"F1_f{i:03d}", layers=[("water", fill)], inserts=[TUBE, stream_for(vessel, z_lvl, t, rng)])
            text, gt = build_scene(c, True, w, h, spp, light, f"{c.name}.exr")
            (TS_SCENES / f"{c.name}.pbrt").write_text(text, encoding="utf-8")
            jobs.append({"name": c.name, "frame": i, "t": t, "fill": fill, "gt": gt})
        print(f"  {len(jobs)} scenes at {w}x{h}, {spp} spp")
        print("=== render ===")
        for k, j in enumerate(jobs):
            exr = RENDERS / f"{j['name']}.exr"
            if exr.exists() and not a.force:
                continue
            tt = render(TS_SCENES / f"{j['name']}.pbrt", exr, a.nthreads)
            print(f"[{k + 1}/{len(jobs)}] {j['name']}: {tt:.1f}s", flush=True)
        print("=== post-process ===")
        prng = np.random.default_rng(3)
        frames = []
        for j in jobs:
            exr = RENDERS / f"{j['name']}.exr"
            png = FRAMES / ("F1_baseline.png" if j["frame"] is None else f"frame_{j['frame']:03d}.png")
            process(exr, png, prng, jitter=False)              # fixed camera: sensor noise only, no per-frame jitter
            if j["frame"] is not None:
                g = j["gt"]
                frames.append({"i": j["frame"], "t": round(j["t"], 4), "fill": round(j["fill"], 5),
                               "gt_rows_frac": [l["top_row_front"] / h for l in g["layers"]]})
        g = jobs[-1]["gt"]
        r_max = max(vessel.r_out(z) for z in np.linspace(0, vessel.H, 200))
        half = math.tan(math.asin(min(r_max / cam["dist"], 0.99))) / math.tan(math.radians(cam["fov"]) / 2) * w / 2 * 0.94
        y0, y1 = max(0, g["rim_row_front"] + 4), min(h, g["bottom_row_front"] - 4)
        meta = {"case": "F1_cyl_pour", "kind": "video", "fps": fps, "seconds": seconds, "n_frames": n, "resolution": [w, h], "spp": spp,
                "baseline": "F1_baseline", "video": "F1_cyl_pour.mp4",
                "roi_frac": [max(0.0, (w / 2 - half) / w), y0 / h, min(1.0, 2 * half / w), (y1 - y0) / h],
                "frames": frames, "label": "cylinder · normal · rg_checker · pouring, 20 fps",
                "note": base_combo.note, "tube": TUBE, "fill_range": [F0, F1], "env": a.env}
        save_json(meta, OUT / "video_meta.json")
        save_json({"resolution": [w, h], "cases": [meta], "image_dir": str(FRAMES)}, OUT / "cases_video.json")
        # contact sheet: baseline + 5 frames
        pick = [FRAMES / "F1_baseline.png"] + [FRAMES / f"frame_{i:03d}.png" for i in np.linspace(0, n - 1, 5).astype(int)]
        ims = [cv2.imread(str(p)) for p in pick if p.exists()]
        if ims:
            s = 240 / ims[0].shape[1]
            ims = [cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for im in ims]
            cv2.imwrite(str(OUT / "contact_sheet.png"), np.concatenate(ims, axis=1))
    if not a.smoke:
        size = encode(FRAMES, fps, OUT / "F1_cyl_pour.mp4")
        print(f"mp4: {size / 1e6:.2f} MB -> {OUT / 'F1_cyl_pour.mp4'}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()

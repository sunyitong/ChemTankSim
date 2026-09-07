"""Real footage cases (2026-09-07): a glass tumbler in front of a backlit red/green checker panel.
  P1  photo pair  — empty vs filled tumbler (two stills from the same tripod position)
  P2  pouring video — 1920x1080 @ 30 fps, 51 s; the level rises from ~12 s to ~42 s. Sampled at SAMPLE_FPS over
      [T0, T1] and played back at PLAY_FPS (time-lapse), baseline = a frame before the stream starts.
Both are cropped to the same portrait window around the glass. No ground truth (real footage).

  python scripts/real_cases.py            -> outputs/real/{images, cases.json, P2/(frames, mp4, video_meta.json), cases_video.json}
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from mvp_common import PROJ_ROOT, save_json

SRC = Path(r"C:\Users\ysun13\Desktop")
PHOTO_EMPTY, PHOTO_FULL, VIDEO = SRC / "20260907-140026.JPG", SRC / "20260907-140152.JPG", SRC / "20260907-140051.mp4"
OUT = PROJ_ROOT / "outputs" / "real"
CROP = (555, 0, 1365, 1080)                    # x0, y0, x1, y1 in the 1920x1080 source: 3:4 portrait around the glass
ROI_FRAC = [0.265, 0.218, 0.395, 0.588]        # inside the glass: below the rim ellipse, above the base (fraction of the crop)
VIDEO_SIZE = (450, 600)                        # frames are downscaled to the size of the synthetic sequences
T_BASE, T0, T1 = 1.0, 8.0, 46.0                # baseline time, sampled window (s, source time)
SAMPLE_FPS, PLAY_FPS = 4.0, 20.0               # 4 source frames per second, played at 20 fps = 5x time-lapse
CRF = 29


def crop(im):
    x0, y0, x1, y1 = CROP
    return im[y0:y1, x0:x1]


def encode(frames_dir: Path, fps: float, mp4: Path, crf: int):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "frame_%03d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium", "-g", "1", "-bf", "0",
                    "-movflags", "+faststart", str(mp4)], check=True)
    return mp4.stat().st_size


def main():
    img_dir = OUT / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    # ---- P1: photo pair
    e, f = cv2.imread(str(PHOTO_EMPTY)), cv2.imread(str(PHOTO_FULL))
    ce, cf = crop(e), crop(f)
    cv2.imwrite(str(img_dir / "P1_real_empty.png"), ce); cv2.imwrite(str(img_dir / "P1_real_full.png"), cf)
    h, w = ce.shape[:2]
    p1 = {"case": "P1_real_tumbler_photo", "baseline": "P1_real_empty", "compare": "P1_real_full", "roi_frac": ROI_FRAC,
          "gt_rows_frac": [], "gt_kinds": [], "gt_liquids": [], "label": "real photo · glass tumbler · rg checker panel · empty vs filled",
          "note": "Two photographs from the same tripod position: empty tumbler, then filled to about three quarters. Backlit red/green checker "
                  "panel, room lights off. No ground truth; the reading is the front-rim contact line as judged by eye.",
          "source": {"baseline": PHOTO_EMPTY.name, "compare": PHOTO_FULL.name, "crop": CROP}}
    save_json({"resolution": [w, h], "image_dir": str(img_dir), "cases": [p1]}, OUT / "cases.json")
    print(f"P1 stills {w}x{h}")
    # ---- P2: pouring video
    sdir = OUT / "P2"; frames_dir = sdir / "frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(VIDEO)); fps_src = cap.get(cv2.CAP_PROP_FPS)
    want = [T_BASE] + [T0 + k / SAMPLE_FPS for k in range(int(round((T1 - T0) * SAMPLE_FPS)) + 1)]
    idx_want = [int(round(t * fps_src)) for t in want]
    grabbed = {}
    i = 0; need = set(idx_want)
    while need:
        ok, fr = cap.read()
        if not ok:
            break
        if i in need:
            grabbed[i] = cv2.resize(crop(fr), VIDEO_SIZE, interpolation=cv2.INTER_AREA); need.discard(i)
        i += 1
    cap.release()
    assert idx_want[0] in grabbed, "baseline frame missing"
    cv2.imwrite(str(frames_dir / "P2_baseline.png"), grabbed[idx_want[0]])
    frames = []
    for k, (t, fi) in enumerate(zip(want[1:], idx_want[1:])):
        if fi not in grabbed:
            continue
        cv2.imwrite(str(frames_dir / f"frame_{len(frames):03d}.png"), grabbed[fi])
        frames.append({"i": len(frames), "t": round(len(frames) / PLAY_FPS, 4), "src_t": round(t, 3), "gt_rows_frac": []})
    n = len(frames)
    size = encode(frames_dir, PLAY_FPS, sdir / "P2_real_tumbler_pour.mp4", CRF)
    meta = {"case": "P2_real_tumbler_pour", "kind": "video", "fps": PLAY_FPS, "seconds": n / PLAY_FPS, "n_frames": n, "resolution": list(VIDEO_SIZE),
            "spp": None, "baseline": "P2_baseline", "video": "P2_real_tumbler_pour.mp4", "image_dir": str(frames_dir),
            "video_path": str(sdir / "P2_real_tumbler_pour.mp4"), "roi_frac": ROI_FRAC, "frames": frames,
            "label": f"real footage · glass tumbler · rg checker panel · pouring, {SAMPLE_FPS / PLAY_FPS * 100:.0f} % → {PLAY_FPS / SAMPLE_FPS:.0f}× time-lapse",
            "note": f"Phone video (1920x1080, 30 fps) of water poured into a tumbler from a spout above the frame; {T1 - T0:.0f} s of the pour sampled at "
                    f"{SAMPLE_FPS:.0f} frames per second and played back at {PLAY_FPS:.0f} fps ({PLAY_FPS / SAMPLE_FPS:.0f}x). Baseline = the empty tumbler "
                    f"{T0 - T_BASE:.0f} s before the stream starts. Camera slightly above the rim, so the surface is seen from above. No ground truth.",
            "source": {"file": VIDEO.name, "fps": fps_src, "window_s": [T0, T1], "baseline_s": T_BASE, "sample_fps": SAMPLE_FPS, "crop": CROP},
            "deposit": None, "env": "real"}
    save_json(meta, sdir / "video_meta.json")
    save_json({"cases": [meta]}, OUT / "cases_video.json")
    print(f"P2 {n} frames, mp4 {size / 1e6:.2f} MB")
    # contact sheet: baseline + 5 frames, with the ROI drawn
    pick = [frames_dir / "P2_baseline.png"] + [frames_dir / f"frame_{i:03d}.png" for i in np.linspace(0, n - 1, 5).astype(int)]
    ims = []
    for p in pick:
        im = cv2.imread(str(p)); H, W = im.shape[:2]; fx, fy, fw, fh = ROI_FRAC
        cv2.rectangle(im, (round(fx * W), round(fy * H)), (round((fx + fw) * W), round((fy + fh) * H)), (255, 255, 255), 1)
        ims.append(cv2.resize(im, (240, 320), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(sdir / "contact_sheet.png"), np.concatenate(ims, axis=1))
    stills = []
    for p in (img_dir / "P1_real_empty.png", img_dir / "P1_real_full.png"):
        im = cv2.imread(str(p)); H, W = im.shape[:2]; fx, fy, fw, fh = ROI_FRAC
        cv2.rectangle(im, (round(fx * W), round(fy * H)), (round((fx + fw) * W), round((fy + fh) * H)), (255, 255, 255), 2)
        stills.append(cv2.resize(im, (240, 320), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(OUT / "photos_contact_sheet.png"), np.concatenate(stills, axis=1))
    print("done ->", OUT)


if __name__ == "__main__":
    main()

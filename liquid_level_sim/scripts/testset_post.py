"""Camera-like post-processing for the test set: EXR (linear) -> noisy sRGB PNG.

  * exposure jitter   global gain (flicker / auto-exposure drift), compare frames only by default
  * sub-pixel shift   bilinear translation of up to +-2.5 px (small camera vibration), compare only
  * sensor noise      shot noise (variance proportional to signal) + read noise, per channel
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from mvp_compare import read_linear, to_srgb


def process(exr: Path, png: Path, rng: np.random.Generator, jitter: bool, shot: float = 2.5e-4,
            read: float = 1.0e-5, max_shift: float = 2.5, gain_range=(0.95, 1.05)) -> dict:
    lin = read_linear(exr)
    info = {"gain": 1.0, "shift_px": [0.0, 0.0]}
    if jitter:
        g = float(rng.uniform(*gain_range)); lin = lin * g; info["gain"] = g
        dx, dy = (float(v) for v in rng.uniform(-max_shift, max_shift, 2))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        lin = cv2.warpAffine(lin, M, (lin.shape[1], lin.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        info["shift_px"] = [dx, dy]
    sigma = np.sqrt(np.clip(lin, 0, None) * shot + read)
    noisy = lin + rng.normal(0.0, 1.0, lin.shape).astype(np.float32) * sigma
    srgb = to_srgb(noisy)
    cv2.imwrite(str(png), cv2.cvtColor((srgb * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    info["noise_sigma_at_0.8"] = float(np.sqrt(0.8 * shot + read))
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exr"); ap.add_argument("png"); ap.add_argument("--jitter", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(process(Path(a.exr), Path(a.png), np.random.default_rng(a.seed), a.jitter)))

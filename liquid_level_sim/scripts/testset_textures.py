"""Procedural deposit textures for the vessel walls (u = angle around the vessel, v = height).

Three maps per (level, seed):
  dirt_amount_*.png   coverage 0..1 -> "amount" of a pbrt mix(glass, deposit) material
  dirt_color_*.png    deposit colour (whitish lime scale vs. brownish grime)
  rough_*.png         glass micro-roughness 0..1 (scaled in the scene), frosting around deposits
  rings_*.png         inner-wall scale rings only (old water lines), used on the dry inner wall
All levels keep clear windows so the backlight pattern stays visible through the wall.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SIZE = 1024
LEVELS = {
    #        rings drips blotches haze  max   haze-threshold (lower = more film)
    "light":  (2,   3,    6,      0.15, 0.35, 0.52),
    "medium": (3,   8,    14,     0.35, 0.60, 0.44),
    "heavy":  (4,   14,   24,     0.50, 0.75, 0.38),
}
SCALE_RGB = np.array([0.86, 0.85, 0.78])
GRIME_RGB = np.array([0.42, 0.31, 0.19])


def value_noise(rng, cells: int, size: int = SIZE) -> np.ndarray:
    g = rng.random((cells, cells)).astype(np.float32)
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)


def fbm(rng, octaves=(6, 12, 24, 48), weights=(0.5, 0.25, 0.15, 0.1)) -> np.ndarray:
    n = sum(w * value_noise(rng, c) for c, w in zip(octaves, weights))
    n -= n.min(); n /= (n.max() + 1e-6)
    return n


def make_maps(level: str, seed: int, out_dir: Path) -> dict:
    n_rings, n_drips, n_blobs, haze_amp, vmax, haze_th = LEVELS[level]
    rng = np.random.default_rng(seed)
    v = np.linspace(1, 0, SIZE)[:, None]          # row 0 = top of the vessel (v = 1)
    u = np.linspace(0, 1, SIZE, endpoint=False)[None, :]
    scale = np.zeros((SIZE, SIZE), np.float32)   # whitish deposits
    grime = np.zeros((SIZE, SIZE), np.float32)   # brownish deposits
    rings = np.zeros((SIZE, SIZE), np.float32)

    # old water lines: horizontal rings, broken and modulated around the vessel
    for _ in range(n_rings):
        vc = rng.uniform(0.25, 0.85); sig = rng.uniform(0.004, 0.012); amp = rng.uniform(0.35, 0.9)
        band = np.exp(-((v - vc) / sig) ** 2)
        mod = 0.55 + 0.45 * value_noise(rng, 5)[:1, :]                    # 1 x SIZE modulation along u
        gaps = (value_noise(rng, 9)[:1, :] > rng.uniform(0.15, 0.4)).astype(np.float32)
        ring = amp * band * mod * (0.3 + 0.7 * gaps)
        rings += ring
        scale += ring
        # drips hanging from the ring
        for _ in range(max(1, n_drips // max(n_rings, 1))):
            uc = rng.uniform(0, 1); w = rng.uniform(0.003, 0.009); L = rng.uniform(0.05, 0.3); a = rng.uniform(0.2, 0.7)
            du = np.minimum(np.abs(u - uc), 1 - np.abs(u - uc))           # wrap around
            below = np.clip((vc - v) / L, 0, 1)
            drip = a * np.exp(-(du / w) ** 2) * np.where((v < vc) & (v > vc - L), 1 - below ** 1.5, 0)
            drip *= 1 + 0.6 * np.sin(v * rng.uniform(60, 200) + rng.uniform(0, 6))  # beaded streak
            scale += np.clip(drip, 0, None)

    # blotches: splashes, fingerprints, dried droplets
    for _ in range(n_blobs):
        uc, vc = rng.uniform(0, 1), rng.uniform(0.05, 0.95)
        su, sv = rng.uniform(0.015, 0.09), rng.uniform(0.015, 0.09)
        du = np.minimum(np.abs(u - uc), 1 - np.abs(u - uc))
        blob = np.exp(-((du / su) ** 2 + ((v - vc) / sv) ** 2) * rng.uniform(1.0, 2.5))
        blob *= 0.6 + 0.4 * value_noise(rng, 40)                          # mottled interior
        if rng.random() < 0.5:
            grime += rng.uniform(0.3, 0.9) * blob
        else:
            scale += rng.uniform(0.3, 0.9) * blob

    # haze: thin film, low-frequency
    haze = haze_amp * np.clip((fbm(rng) - haze_th) * 2.5, 0, 1)
    scale += haze

    # guaranteed clear windows
    win = np.ones((SIZE, SIZE), np.float32)
    for _ in range(3):
        uc, vc = rng.uniform(0, 1), rng.uniform(0.2, 0.8)
        du = np.minimum(np.abs(u - uc), 1 - np.abs(u - uc))
        win *= 1 - 0.85 * np.exp(-((du / 0.12) ** 2 + ((v - vc) / 0.22) ** 2))
    scale *= win; grime *= win

    total = scale + grime
    amount = np.clip(total, 0, vmax).astype(np.float32)
    wgt = np.where(total > 1e-6, grime / (total + 1e-6), 0.0)[..., None]
    color = (1 - wgt) * SCALE_RGB + wgt * GRIME_RGB
    color *= 0.85 + 0.15 * value_noise(rng, 64)[..., None]
    # micro-roughness only where deposits are substantial (thin films and clean glass stay clear)
    rough = np.clip((amount - 0.12) / max(vmax - 0.12, 1e-3), 0, 1) * (0.6 + 0.4 * value_noise(rng, 48))
    rings_map = np.clip(rings, 0, min(0.7, vmax)).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{level}_{seed}"
    files = {}
    for name, arr, rgb in (("dirt_amount", amount, False), ("dirt_color", color, True), ("rough", rough, False), ("rings", rings_map, False)):
        p = out_dir / f"{name}_{tag}.png"
        img = np.clip(arr, 0, 1)
        if rgb:
            cv2.imwrite(str(p), cv2.cvtColor((img * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(p), (img * 255 + 0.5).astype(np.uint8))
        files[name] = p.name
    files["coverage_mean"] = float(amount.mean())
    files["coverage_over_0.3"] = float((amount > 0.3).mean())
    return files


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/testset/textures")
    for lv in LEVELS:
        print(lv, make_maps(lv, 1, out))

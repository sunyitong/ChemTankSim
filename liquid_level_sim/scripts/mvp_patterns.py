"""Generate the emission textures for the light panel (outputs/mvp/patterns/*.png).

The panel is panel_w x panel_h cm; textures are rendered at PX_PER_CM so that a 2 cm checker cell
is 100 px wide. pbrt reads PNG as sRGB and converts to linear radiance, so colours below are sRGB.
u = 0 is the image LEFT of the camera view, v = 1 is the top (see mvp_scene.panel()).
"""
from __future__ import annotations

import argparse
import colorsys

import cv2
import numpy as np

from mvp_common import PATTERN_DIR, Setup, ensure_dirs

PX_PER_CM = 50

# sRGB colours of the panel LEDs / print
RED = (0.85, 0.08, 0.06)
GREEN = (0.10, 0.75, 0.12)
WHITE = (0.95, 0.95, 0.95)
BLACK = (0.02, 0.02, 0.02)


def _canvas(cfg: Setup) -> tuple[np.ndarray, int, int]:
    w = int(round(cfg.panel_w * PX_PER_CM))
    h = int(round(cfg.panel_h * PX_PER_CM))
    return np.zeros((h, w, 3), np.float32), w, h


def _write(img: np.ndarray, name: str) -> None:
    bgr = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", (bgr * 255 + 0.5).astype(np.uint8))
    assert ok
    path = PATTERN_DIR / f"{name}.png"
    blob = buf.tobytes()
    if not path.exists() or path.read_bytes() != blob:      # don't touch a file a running render may be reading
        path.write_bytes(blob)


def checker(cfg: Setup, cell_cm: float = 2.0, c1=RED, c2=GREEN) -> np.ndarray:
    img, w, h = _canvas(cfg)
    cell = cell_cm * PX_PER_CM
    # anchor the grid on the panel centre so a cell corner sits on the cylinder axis
    xs = np.floor((np.arange(w) - w / 2) / cell).astype(int)
    ys = np.floor((np.arange(h) - h / 2) / cell).astype(int)
    parity = (xs[None, :] + ys[:, None]) & 1
    img[parity == 0] = c1
    img[parity == 1] = c2
    return img


def mosaic(cfg: Setup, cell_cm: float = 1.5, seed: int = 0) -> np.ndarray:
    """Random saturated colours per cell: a locally unique colour code, so the left-right flip
    produced by the liquid 'lens' is visible (a symmetric checkerboard cannot show it)."""
    img, w, h = _canvas(cfg)
    cell = cell_cm * PX_PER_CM
    nx, ny = int(np.ceil(w / cell)) + 2, int(np.ceil(h / cell)) + 2
    rng = np.random.default_rng(seed)
    cols = np.zeros((ny, nx, 3), np.float32)
    for j in range(ny):
        for i in range(nx):
            hue = rng.uniform(0, 1)
            sat = rng.uniform(0.75, 1.0)
            val = rng.uniform(0.65, 1.0)
            cols[j, i] = colorsys.hsv_to_rgb(hue, sat, val)
    xi = np.floor((np.arange(w) - w / 2) / cell).astype(int) + nx // 2
    yi = np.floor((np.arange(h) - h / 2) / cell).astype(int) + ny // 2
    img[:] = cols[yi[:, None], xi[None, :]]
    return img


def grid(cfg: Setup, spacing_cm: float = 1.0, line_mm: float = 1.0, major_every: int = 5,
         major_mm: float = 2.0) -> np.ndarray:
    """Graph paper: thin black lines on white, thicker line every `major_every` cells."""
    img, w, h = _canvas(cfg)
    img[:] = WHITE
    sp = spacing_cm * PX_PER_CM
    lw = max(1, int(round(line_mm / 10 * PX_PER_CM)))
    mw = max(1, int(round(major_mm / 10 * PX_PER_CM)))
    for axis, n in ((1, w), (0, h)):
        k = 0
        c = n / 2
        while c - k * sp >= 0:
            for pos in {c + k * sp, c - k * sp}:
                width = mw if k % major_every == 0 else lw
                a, b = int(round(pos - width / 2)), int(round(pos + width / 2))
                a, b = max(a, 0), min(b, n)
                if axis == 1:
                    img[:, a:b] = BLACK
                else:
                    img[a:b, :] = BLACK
            k += 1
    return img


GENERATORS = {"rg_checker": checker, "mosaic": mosaic, "grid": grid}


def generate_all(cfg: Setup) -> dict:
    ensure_dirs()
    out = {"white": None}
    for name, fn in GENERATORS.items():
        _write(fn(cfg), name)
        out[name] = f"{name}.png"
        print(f"  pattern {name}: {PATTERN_DIR / (name + '.png')}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    generate_all(Setup())

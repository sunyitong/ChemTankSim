"""Physically motivated limescale (CaCO3) deposit model for the INNER wall of a glass vessel.

The deposit is described by an optical depth map tau(s, z) on the unrolled inner wall
(s = circumferential position, z = height, both in mm). tau is built from the processes seen in
reference photographs and in the literature (docs/RESEARCH.md §12):

  1. standing-water film     hard water stood at level Z1 for a long time: a thin, nearly uniform
                             translucent film everywhere below Z1, slightly denser towards the
                             bottom (evaporation concentrates the solution), with faint drainage
                             streaks and micro-crystal granularity
  2. water-line ring         repeated wetting/drying of the meniscus at Z1 deposits a dense band
                             just above the line; it "creeps" upward with an exponential tail
                             (salt creeping, Qazi et al. 2019) and is broken into fingers
  3. tide marks              if the level fell in steps while evaporating, fainter rings remain at
                             the intermediate levels
  4. dried droplets          splashes above the line dry into coffee rings (thin centre, dense rim)
  5. drips / pour streaks    narrow beaded trails running down from the rim or from droplets
  6. tilt and asymmetry      the line is slightly tilted and its thickness varies around the vessel

Rendering: a thin scattering layer of optical depth tau interacts with a photon with probability
1 - exp(-tau); that probability is used as the `amount` of a pbrt mix(glass, deposit) material, so
partially covered pixels still transmit the backlight. The deposit is a diffuse-transmission layer
whose reflectance rises and transmittance falls with tau (thin film: milky, thick crust: chalky
white). Where the deposit is submerged in water, pores are index-matched (1.59 vs 1.33 instead of
1.59 vs 1.0): tau is divided by WET_FACTOR and the layer becomes much more transparent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WET_FACTOR = 4.5              # scattering reduction of a water-soaked CaCO3 layer
SCALE_RGB = np.array([0.88, 0.86, 0.80])     # dry deposit colour (slightly warm off-white)
IRON_RGB = np.array([0.72, 0.58, 0.40])      # tint of thick deposits in iron-bearing water


@dataclass
class ScaleParams:
    line_mm: float                      # long-standing water line height Z1 (mm above inner bottom)
    bottom_mm: float = 0.0              # inner bottom height (mm) — film extends down to here
    film_tau: float = 0.30              # optical depth of the standing-water film just below the line
    film_gradient: float = 0.8          # film_tau * (1 + gradient * depth fraction) towards the bottom
    ring_tau: float = 2.8               # peak optical depth of the water-line ring
    ring_up_mm: float = 4.0             # creeping tail length above the line (e-folding)
    ring_down_mm: float = 1.2           # decay below the line
    finger_strength: float = 0.7        # how strongly the upward creep is broken into fingers
    tilt_mm: float = 1.5                # tilt of the line across the vessel (peak-to-peak / 2)
    tide_marks: list = field(default_factory=list)   # [(z_mm, tau, width_mm), ...]
    n_droplets: int = 60
    droplet_zone_mm: float = 35.0       # droplets concentrate within this height above the line
    n_drips: int = 4
    rim_mm: float = 200.0               # rim height (drips start near here)
    iron: float = 0.15                  # 0..1 warm tint of thick deposits
    seed: int = 1
    sector: tuple | None = None         # (centre_frac, width_frac) of the circumference carrying deposit (one-sided)
    film_below_only: bool = True


def _noise(rng, shape, cells_y, cells_x, octaves=3):
    """Smooth value noise with decreasing-amplitude octaves, normalised to 0..1."""
    h, w = shape
    out = np.zeros(shape, np.float32); amp, tot = 1.0, 0.0
    for o in range(octaves):
        cy, cx = min(h, cells_y * 2 ** o), min(w, cells_x * 2 ** o)
        g = rng.random((cy, cx)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= 0.5
    out /= tot
    out -= out.min(); out /= (out.max() + 1e-6)
    return out


def generate_tau(p: ScaleParams, circumference_mm: float, height_mm: float, res_mm: float = 0.1,
                 max_px: int = 4096) -> tuple[np.ndarray, dict]:
    """Return tau(rows, cols) with row 0 at the TOP of the wall (pbrt v = 1) and cols around the wall."""
    rng = np.random.default_rng(p.seed)
    W = int(min(max_px, round(circumference_mm / res_mm))); H = int(min(max_px // 2, round(height_mm / res_mm)))
    mm_x, mm_y = circumference_mm / W, height_mm / H
    z = (height_mm - (np.arange(H) + 0.5) * mm_y)[:, None]           # mm above the inner bottom, top row first
    theta = (np.arange(W) + 0.5) / W * 2 * np.pi
    tau = np.zeros((H, W), np.float32)

    # --- 6. tilted, modulated water line
    zc = p.line_mm + p.tilt_mm * np.cos(theta - rng.uniform(0, 2 * np.pi))[None, :]
    around = 0.75 + 0.5 * _noise(rng, (1, W), 1, 6, 2)                 # thickness varies around the vessel

    # --- 1. standing-water film below the line
    depth_frac = np.clip((zc - z) / max(zc.mean() - p.bottom_mm, 1.0), 0, 1)
    film = p.film_tau * (1 + p.film_gradient * depth_frac)
    streaks = 1 + 0.35 * (_noise(rng, (H, W), 4, 160, 2) - 0.5) * 2    # drainage streaks: fine in s, long in z
    grain = 1 + 0.25 * (_noise(rng, (H, W), H // 3, W // 3, 1) - 0.5) * 2   # micro-crystal granularity (~0.3 mm)
    below = 1 / (1 + np.exp((z - zc) / 0.6))                            # smooth step at the line
    tau += film * streaks * grain * below * around

    # --- 2. water-line ring with upward creep broken into fingers
    dz = z - zc
    # creep fingers: irregular tail length around the vessel (multi-octave), 0.4 .. 1.8 x the mean tail
    fing = _noise(rng, (1, W), 1, W // 40, 3) * 0.6 + _noise(rng, (1, W), 1, W // 8, 2) * 0.4
    fingers = 0.4 + 1.4 * np.clip((fing - 0.5) * p.finger_strength * 2 + 0.5, 0, 1)
    up = np.exp(-np.clip(dz, 0, None) / (p.ring_up_mm * fingers))
    down = np.exp(np.clip(dz, None, 0) / p.ring_down_mm)
    ring = p.ring_tau * np.where(dz >= 0, up, down) * around * (0.85 + 0.3 * grain - 0.15)
    tau += ring

    # --- 3. tide marks at intermediate levels
    for zt, tt, wt in p.tide_marks:
        zct = zt + p.tilt_mm * np.cos(theta - rng.uniform(0, 2 * np.pi))[None, :]
        band = np.exp(-0.5 * ((z - zct) / wt) ** 2) * (0.7 + 0.6 * _noise(rng, (1, W), 1, 8, 2))
        tau += tt * band

    # --- 4. dried droplets (coffee rings) above the line
    ys, xs = np.mgrid[0:H, 0:W]
    for _ in range(p.n_droplets):
        zd = p.line_mm + rng.exponential(p.droplet_zone_mm) + 1.0
        if zd > height_mm - 2 or rng.random() < 0.15:
            zd = rng.uniform(p.line_mm, height_mm - 2)
        sd = rng.uniform(0, circumference_mm)
        R = float(np.clip(rng.lognormal(np.log(1.4), 0.45), 0.4, 4.5))          # mm
        ell = rng.uniform(1.0, 1.8) if rng.random() < 0.4 else 1.0             # some ran a little before drying
        r0, c0 = int((height_mm - zd) / mm_y), int(sd / mm_x)
        rr, cc = int(R * ell / mm_y) + 3, int(R / mm_x) + 3
        sl = (slice(max(0, r0 - rr), min(H, r0 + rr)), slice(max(0, c0 - cc), min(W, c0 + cc)))
        dy = (ys[sl] - r0) * mm_y / ell; dx = (xs[sl] - c0) * mm_x
        r = np.sqrt(dx * dx + dy * dy)
        rim = np.exp(-0.5 * ((r - R) / (0.15 * R)) ** 2)
        inside = 1 / (1 + np.exp((r - R) / (0.05 * R)))
        t_rim = rng.uniform(0.8, 2.5); t_centre = t_rim * rng.uniform(0.08, 0.25)
        tau[sl] += t_rim * rim + t_centre * inside * (0.6 + 0.8 * grain[sl])

    # --- 5. drips / pour streaks from the rim, beaded (Rayleigh-Plateau)
    for _ in range(p.n_drips):
        s0 = rng.uniform(0, circumference_mm); z0 = min(height_mm - 1, p.rim_mm - rng.uniform(0, 8))
        L = rng.uniform(15, 90); wmm = rng.uniform(0.6, 1.3); t = rng.uniform(0.6, 1.8)
        wob = _noise(rng, (H, 1), 12, 1, 2)[:, 0] - 0.5                          # lateral wobble
        sc = s0 + 2.0 * wob                                                      # mm
        dxs = (xs * mm_x - sc[:, None]); dxs = np.minimum(np.abs(dxs), circumference_mm - np.abs(dxs))
        along = np.clip((z0 - z[:, 0]) / L, 0, 1)[:, None]
        active = ((z[:, 0] < z0) & (z[:, 0] > z0 - L))[:, None]
        bead = 1 + 0.7 * np.sin(z[:, 0] * 2 * np.pi / rng.uniform(2.0, 4.0) + rng.uniform(0, 6))[:, None]
        prof = np.exp(-0.5 * (dxs / wmm) ** 2) * (1 - along ** 2) * bead
        tau += t * prof * active
        # terminal droplet at the end of the run
        r0, c0 = int((height_mm - (z0 - L)) / mm_y), int(s0 / mm_x) % W
        rr = int(2.2 * wmm / mm_y) + 2; cc = int(2.2 * wmm / mm_x) + 2
        sl = (slice(max(0, r0 - rr), min(H, r0 + rr)), slice(max(0, c0 - cc), min(W, c0 + cc)))
        dyy = (ys[sl] - r0) * mm_y; dxx = (xs[sl] - c0) * mm_x
        tau[sl] += t * 1.5 * np.exp(-0.5 * ((np.sqrt(dxx ** 2 + dyy ** 2) - 1.6 * wmm) / (0.35 * wmm)) ** 2)

    # --- one-sided deposit (vessel stored tilted / splashed from one side): smooth angular window
    if p.sector:
        c, wdt = p.sector
        d = np.abs(((np.arange(W) + 0.5) / W - c + 0.5) % 1.0 - 0.5)          # circular distance in turns
        win = 1 / (1 + np.exp((d - wdt / 2) / 0.02))
        win = win * (0.85 + 0.3 * _noise(rng, (1, W), 1, 10, 2)[0])
        tau *= win[None, :]

    tau = np.clip(tau, 0, 8).astype(np.float32)
    info = {"px": [W, H], "mm_per_px": [mm_x, mm_y], "tau_mean": float(tau.mean()), "tau_p99": float(np.percentile(tau, 99)),
            "coverage_tau_gt_0.3": float((tau > 0.3).mean()), "coverage_tau_gt_1": float((tau > 1.0).mean())}
    return tau, info


def write_maps(tau: np.ndarray, out_dir: Path, tag: str, iron: float = 0.15) -> dict:
    """amount (dry/wet) = 1 - exp(-tau); deposit reflectance/transmittance rise/fall with tau;
    glass micro-roughness follows the deposit. All 8-bit PNG, linear encoding in pbrt."""
    out_dir.mkdir(parents=True, exist_ok=True)
    a_dry = 1 - np.exp(-tau); a_wet = 1 - np.exp(-tau / WET_FACTOR)
    thick = 1 - np.exp(-tau / 1.5)                                     # 0 thin film .. 1 crust
    refl = 0.22 + 0.38 * thick; trans = 0.72 - 0.38 * thick             # albedo ~0.94; forward-scattering film, chalky crust
    col = SCALE_RGB[None, None, :] * (1 - iron * thick[..., None]) + IRON_RGB[None, None, :] * (iron * thick[..., None])
    rough = np.clip(tau, 0, 1) * 0.85
    files = {}
    def w8(name, arr):
        p = out_dir / f"{name}_{tag}.png"
        img = np.clip(arr, 0, 1)
        if img.ndim == 3:
            cv2.imwrite(str(p), cv2.cvtColor((img * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(p), (img * 255 + 0.5).astype(np.uint8))
        files[name] = p.name
    w8("scale_amount_dry", a_dry); w8("scale_amount_wet", a_wet)
    w8("scale_refl", col * refl[..., None]); w8("scale_trans", col * trans[..., None])
    w8("scale_rough", rough); w8("scale_thick", thick)                  # thick: share of fully diffuse scattering
    return files


PRESETS = {
    # long-standing line at 60 % of a 20 cm inner height, slow evaporation, few splashes
    "vase": lambda H, rim: ScaleParams(line_mm=0.60 * H, film_tau=0.16, film_gradient=0.6, ring_tau=2.2, ring_up_mm=3.5,
                                        tilt_mm=1.2, n_droplets=35, n_drips=3, rim_mm=rim, iron=0.12, seed=7),
    # level fell in steps (tide marks 60 -> 45 %), more splashes and pour streaks
    "tide": lambda H, rim: ScaleParams(line_mm=0.60 * H, film_tau=0.12, film_gradient=1.0, ring_tau=1.8, ring_up_mm=3.0,
                                        tilt_mm=1.8, tide_marks=[(0.56 * H, 0.8, 1.0), (0.52 * H, 0.6, 0.9), (0.48 * H, 0.8, 1.1), (0.45 * H, 1.3, 1.3)],
                                        n_droplets=90, droplet_zone_mm=45, n_drips=6, rim_mm=rim, iron=0.25, seed=11),
    # light: a faint line and film only
    "light": lambda H, rim: ScaleParams(line_mm=0.55 * H, film_tau=0.07, film_gradient=0.5, ring_tau=1.0, ring_up_mm=2.5,
                                         tilt_mm=0.8, n_droplets=15, n_drips=1, rim_mm=rim, iron=0.08, seed=3),
    # ---- position / form / extent variants
    "line30": lambda H, rim: ScaleParams(line_mm=0.30 * H, film_tau=0.10, film_gradient=0.5, ring_tau=2.0, ring_up_mm=3.0,
                                          tilt_mm=1.0, n_droplets=20, n_drips=1, rim_mm=rim, iron=0.10, seed=21),
    # thin line only, no film: a single short stay of water
    "line50_thin": lambda H, rim: ScaleParams(line_mm=0.50 * H, film_tau=0.0, ring_tau=1.6, ring_up_mm=2.2, ring_down_mm=0.8,
                                               tilt_mm=0.6, n_droplets=10, n_drips=0, rim_mm=rim, iron=0.05, seed=22),
    "line75_heavy": lambda H, rim: ScaleParams(line_mm=0.75 * H, film_tau=0.22, film_gradient=0.8, ring_tau=3.0, ring_up_mm=5.0,
                                                tilt_mm=1.5, n_droplets=40, n_drips=4, rim_mm=rim, iron=0.30, seed=23),
    # filled almost to the rim for a long time: film on the whole wall, ring near the top
    "line92_full": lambda H, rim: ScaleParams(line_mm=0.92 * H, film_tau=0.18, film_gradient=0.5, ring_tau=2.2, ring_up_mm=3.0,
                                               tilt_mm=1.0, n_droplets=8, droplet_zone_mm=10, n_drips=2, rim_mm=rim, iron=0.12, seed=24),
    # level fluctuated between 40 and 60 %: a dense band of tide marks
    "band4060": lambda H, rim: ScaleParams(line_mm=0.60 * H, film_tau=0.14, film_gradient=0.6, ring_tau=1.2, ring_up_mm=2.5,
                                            tilt_mm=1.2, tide_marks=[(f * H, 0.5 + 0.5 * ((i * 7) % 3) / 2, 0.8 + 0.3 * (i % 2)) for i, f in enumerate(np.linspace(0.40, 0.585, 8))],
                                            n_droplets=30, n_drips=2, rim_mm=rim, iron=0.15, seed=25),
    # vessel stored tilted while drying: strongly inclined line and film boundary
    "tilted": lambda H, rim: ScaleParams(line_mm=0.55 * H, film_tau=0.14, film_gradient=0.7, ring_tau=2.0, ring_up_mm=3.5,
                                          tilt_mm=18.0, n_droplets=25, n_drips=2, rim_mm=rim, iron=0.12, seed=26),
    # heavy splashing / pouring: droplets and streaks dominate, faint line
    "splash": lambda H, rim: ScaleParams(line_mm=0.45 * H, film_tau=0.05, film_gradient=0.3, ring_tau=0.8, ring_up_mm=2.5,
                                          tilt_mm=1.0, n_droplets=160, droplet_zone_mm=70, n_drips=10, rim_mm=rim, iron=0.10, seed=27),
    # deposit only on one third of the circumference
    "sector": lambda H, rim: ScaleParams(line_mm=0.50 * H, film_tau=0.16, film_gradient=0.6, ring_tau=2.2, ring_up_mm=3.5,
                                          tilt_mm=1.0, n_droplets=40, n_drips=2, rim_mm=rim, iron=0.12, seed=28, sector=(0.5, 0.34)),
    # no water line at all: pour streaks from the rim and a few droplets
    "drips_only": lambda H, rim: ScaleParams(line_mm=0.05 * H, film_tau=0.0, ring_tau=0.0, n_droplets=30, droplet_zone_mm=150,
                                              n_drips=12, rim_mm=rim, iron=0.10, seed=29),
}


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/scale/textures")
    H, circ = 200.0, 2 * np.pi * 75.0
    for name, mk in PRESETS.items():
        p = mk(H, H + 3)
        tau, info = generate_tau(p, circ, H)
        files = write_maps(tau, out, name, p.iron)
        print(name, info, files["scale_amount_dry"])

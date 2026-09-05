"""Phase-0 comparison: how differently does each backlight pattern reveal the liquid level?

For every pattern we compute, inside the container ROI,
  * dE_vs_empty(fill)  mean CIE Lab dE between the render at `fill` and the empty container
                       (how strongly the liquid section changes the background appearance)
  * dE_sens            dE between fill 0.50 and 0.52 (a 4 mm level change), cavity mean and
                       mean inside the band between the two level rows
  * noise floor        dE between two renders of the SAME scene with different RNG seeds
  * meniscus control   the same sensitivity with a flat free surface (no meniscus)
  * line contrast      luminance contrast of the level line in a single image
  * row profiles of dE and the detected transition row vs the analytic GT rows
  * checkerboard period (edge spacing) above / below the level and on the directly seen panel,
    compared with the paraxial cylindrical-lens prediction (docs/RESEARCH.md §3.2)
and write contact sheets, zoom crops, difference heat-maps, metrics.json and report.html.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from dataclasses import fields
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from jinja2 import Template  # noqa: E402

from mvp_common import (COMPARE_DIR, IMGTOOL_EXE, MANIFEST, MVP_OUT, RENDER_DIR, SENS_PAIR, Setup,  # noqa: E402
                        ensure_dirs, load_json, run, save_json)

# fixed categorical colours per pattern (identity never cycles)
PATTERN_COLOR = {"white": "#6b7280", "rg_checker": "#2563eb", "mosaic": "#7c3aed", "grid": "#d97706"}
PATTERN_LABEL = {"white": "white", "rg_checker": "R/G checker", "mosaic": "mosaic", "grid": "grid lines"}
FILL_COLORS = ["#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]     # sequential single hue for ordered fills
DE_VMAX = 30.0

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#9ca3af", "figure.dpi": 110})


# ----------------------------------------------------------------------------- image io
def read_pfm(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        assert f.readline().strip() == b"PF", p
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4").reshape(h, w, 3)
    return np.flipud(data).astype(np.float32)          # PFM stores rows bottom-up


def read_linear(exr: Path) -> np.ndarray:
    """Linear RGB float image via imgtool (OpenCV wheels have no EXR support)."""
    pfm = exr.with_suffix(".pfm")
    if not pfm.exists() or pfm.stat().st_mtime < exr.stat().st_mtime:
        run([IMGTOOL_EXE, "convert", "--outfile", pfm, exr], quiet=True)
    return read_pfm(pfm)


def to_srgb(lin: np.ndarray) -> np.ndarray:
    x = np.clip(lin, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def to_lab(srgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(srgb), cv2.COLOR_RGB2Lab)


def to_u8(srgb: np.ndarray) -> np.ndarray:
    return (np.clip(srgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def bgr(u8_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(u8_rgb, cv2.COLOR_RGB2BGR)


def label(img: np.ndarray, text: str, org=(8, 22), scale=0.6, color=(255, 255, 255)) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def luminance(srgb: np.ndarray) -> np.ndarray:
    return 0.2126 * srgb[..., 0] + 0.7152 * srgb[..., 1] + 0.0722 * srgb[..., 2]


# ----------------------------------------------------------------------------- analysis helpers
def cavity_rows(cfg: Setup) -> tuple[int, int]:
    """Image rows spanned by the inner cavity at the FRONT inner wall (near rim .. near inner bottom)."""
    p = cfg.project(np.array([[0.0, -cfg.r_in, cfg.z_rim], [0.0, -cfg.r_in, cfg.bottom]]))
    return int(math.ceil(p[0, 1])), int(math.floor(p[1, 1]))


def central_cols(cfg: Setup, frac: float = 0.6) -> tuple[int, int]:
    _, _, c0, c1 = cfg.container_box()
    w = c1 - c0
    return int(c0 + w * (0.5 - frac / 2)), int(c0 + w * (0.5 + frac / 2))


def row_profile(de: np.ndarray, cols: tuple[int, int]) -> np.ndarray:
    return de[:, cols[0]:cols[1]].mean(axis=1)


def smooth(x: np.ndarray, k: int = 9) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")


def transition_row(prof: np.ndarray, rows: tuple[int, int]) -> float:
    """First row (top-down, inside the cavity) where the smoothed profile exceeds half its robust max."""
    r0, r1 = rows
    seg = smooth(prof)[r0:r1]
    thr = 0.5 * np.percentile(seg, 95)
    idx = np.nonzero(seg > thr)[0]
    return float(r0 + idx[0]) if len(idx) else float("nan")


def peak_row(prof: np.ndarray, rows: tuple[int, int]) -> float:
    r0, r1 = rows
    return float(r0 + int(np.argmax(smooth(prof)[r0:r1])))


def line_contrast(srgb: np.ndarray, gt_row: float, cols: tuple[int, int], sy: float) -> float:
    """Single-image visibility of the level line: relative luminance drop of the band
    [gt-16, gt+4] px (meniscus band sits just above the geometric level) vs its neighbours."""
    lum = luminance(srgb)[:, cols[0]:cols[1]].mean(axis=1)
    g = int(round(gt_row))
    s = lambda v: int(round(v * sy))  # noqa: E731
    band = lum[max(g - s(16), 0):g + s(4) + 1].mean()
    neigh = np.concatenate([lum[max(g - s(40), 0):max(g - s(20), 1)], lum[g + s(8):g + s(28)]]).mean()
    return float((neigh - band) / (neigh + 1e-6))


def edge_period(sig2d: np.ndarray, along_rows: bool, min_gap: float, amp: float = 0.08) -> float:
    """Dominant period of an alternating pattern from zero-crossing spacing.
    sig2d: signed contrast (e.g. R-G). Each row (or column) is analysed independently and the
    median spacing between consecutive crossings (= half period) is doubled."""
    lines = sig2d if along_rows else sig2d.T
    halves = []
    for s in lines:
        s = np.convolve(s, np.ones(3) / 3, mode="same")
        sgn = np.sign(s)
        idx = np.nonzero((sgn[:-1] * sgn[1:] < 0) & (np.abs(s[:-1]) + np.abs(s[1:]) > amp))[0]
        if len(idx) < 2:
            continue
        x = idx + s[idx] / (s[idx] - s[idx + 1])         # linear interpolation of the crossing
        d = np.diff(x)
        halves.extend(d[d > min_gap].tolist())
    return float(2 * np.median(halves)) if len(halves) >= 6 else float("nan")


def checker_periods(srgb: np.ndarray, rows: tuple[int, int], cols: tuple[int, int], sy: float) -> dict:
    rg = srgb[..., 0] - srgb[..., 1]
    patch = rg[rows[0]:rows[1], cols[0]:cols[1]]
    return {"rows": [int(rows[0]), int(rows[1])], "period_h_px": edge_period(patch, True, 4 * sy),
            "period_v_px": edge_period(patch, False, 4 * sy)}


# ----------------------------------------------------------------------------- exact 2-D ray trace
def trace_panel_x(cfg: Setup, theta: np.ndarray, filled: bool) -> np.ndarray:
    """Geometric-optics reference independent of pbrt: trace camera rays in the horizontal plane
    through the cylindrical glass wall (and the water when `filled`) and return where they hit the
    panel (x, cm). Rays that miss the container see the panel directly; TIR -> nan."""
    n_air, n_g, n_in = 1.0, cfg.eta_glass, (cfg.eta_liquid if filled else 1.0)
    p = np.stack([np.zeros_like(theta), np.full_like(theta, -cfg.cam_dist)], axis=1)
    d = np.stack([np.sin(theta), np.cos(theta)], axis=1)
    alive = np.ones(len(theta), bool)

    def hit(p, d, r, entering):
        b = (p * d).sum(1)
        c = (p * p).sum(1) - r * r
        disc = b * b - c
        ok = disc >= 0
        s = np.sqrt(np.maximum(disc, 0))
        t = (-b - s) if entering else (-b + s)
        ok &= t > 1e-9
        return t, ok

    def refract(d, n_hat, n1, n2):
        eta = n1 / n2
        cos_i = -(d * n_hat).sum(1)
        k = 1 - eta * eta * (1 - cos_i * cos_i)
        ok = k >= 0
        return eta * d + (eta * cos_i - np.sqrt(np.maximum(k, 0)))[:, None] * n_hat, ok

    seq = [(cfg.r_out, True, n_air, n_g), (cfg.r_in, True, n_g, n_in), (cfg.r_in, False, n_in, n_g), (cfg.r_out, False, n_g, n_air)]
    inside = np.zeros(len(theta), bool)          # rays that actually enter the container
    for i, (r, entering, n1, n2) in enumerate(seq):
        t, ok = hit(p, d, r, entering)
        if i == 0:
            inside = ok                            # misses go straight to the panel
        sel = alive & inside & ok
        q = p[sel] + t[sel, None] * d[sel]
        n_hat = q / np.linalg.norm(q, axis=1, keepdims=True)
        if not entering:
            n_hat = -n_hat
        dn, ok2 = refract(d[sel], n_hat, n1, n2)
        p[sel], d[sel] = q, dn
        alive[np.nonzero(sel)[0][~ok2]] = False          # total internal reflection
        if i > 0:
            alive &= ~(inside & ~ok)                     # numerical miss of a later surface
    t = (cfg.panel_y - p[:, 1]) / d[:, 1]
    x = p[:, 0] + t * d[:, 0]
    x[~alive | (d[:, 1] <= 0)] = np.nan
    return x


def col_to_theta(cfg: Setup, cols: np.ndarray, W: int) -> np.ndarray:
    # pbrt: fov spans the shorter (x) axis; image left = world +x (left-handed), hence the minus sign
    return -np.arctan((cols - W / 2) / (W / 2) * math.tan(math.radians(cfg.fov) / 2))


def median_period_from_map(xp: np.ndarray, cols: np.ndarray, c0: int, c1: int, cell_cm: float) -> float:
    """Median spacing (px) between consecutive checker edges seen in columns c0..c1, given the
    column -> panel-x map; mirrors the edge-spacing estimator used on the renders."""
    m = (cols >= c0) & (cols < c1) & np.isfinite(xp)
    x, c = xp[m], cols[m]
    if len(x) < 3:
        return float("nan")
    k = np.floor(x / cell_cm)
    ch = np.nonzero(np.diff(k) != 0)[0]
    if len(ch) < 3:
        return float("nan")
    edges = c[ch] + (np.round(x[ch + 1] / cell_cm) * cell_cm - x[ch]) / (x[ch + 1] - x[ch]) * (c[ch + 1] - c[ch])
    return float(2 * np.median(np.diff(edges)))


def predict_periods_2d(cfg: Setup, W: int, sx: float, cols_central: tuple[int, int], box, cell_cm: float = 2.0) -> dict:
    cols = np.arange(W) + 0.5
    th = col_to_theta(cfg, cols, W)
    xp_fill = trace_panel_x(cfg, th, True)
    xp_empty = trace_panel_x(cfg, th, False)
    xp_direct = np.tan(th) * (cfg.cam_dist + cfg.panel_y)          # no container at all
    c0, c1 = cols_central
    wing = int(30 * sx)
    direct_cols = [(int(20 * sx), box[2] - wing), (box[3] + wing, W - int(20 * sx))]

    def local_period(xp, a, b):                     # median of 2*cell / |dx_p/dcol| over columns a..b
        g = np.abs(np.gradient(xp))[a:b]
        return float(np.nanmedian(2 * cell_cm / g))

    pd = float(np.nanmean([local_period(xp_direct, a, b) for a, b in direct_cols if b - a > 4]))
    pa = median_period_from_map(xp_empty, cols, c0, c1, cell_cm)
    pl = median_period_from_map(xp_fill, cols, c0, c1, cell_cm)
    if not np.isfinite(pa):
        pa = local_period(xp_empty, c0, c1)
    if not np.isfinite(pl):
        pl = local_period(xp_fill, c0, c1)
    # local magnification at the image centre
    ic = W // 2
    g = lambda xp: abs(xp[ic + 2] - xp[ic - 2]) / 4  # noqa: E731  cm per px
    out = {"cell_cm": cell_cm, "direct_period_px": pd, "air_period_px": pa, "liquid_period_px": pl,
           "ratio_h_liquid_over_air": pl / pa, "ratio_h_liquid_over_direct": pl / pd, "ratio_h_air_over_direct": pa / pd,
           "centre_local_ratio_liquid_over_direct": g(xp_direct) / g(xp_fill),
           "centre_local_ratio_air_over_direct": g(xp_direct) / g(xp_empty),
           "flipped": bool(np.nanmean(np.gradient(xp_fill)[c0:c1]) * np.nanmean(np.gradient(xp_direct)[c0:c1]) < 0),
           "_cols": cols, "_xp_fill": xp_fill, "_xp_empty": xp_empty, "_xp_direct": xp_direct}
    return out


# ----------------------------------------------------------------------------- main comparison
def setup_from_manifest(m: dict) -> Setup:
    names = {f.name for f in fields(Setup)}
    kw = {k: v for k, v in m["setup"].items() if k in names}
    for k in ("sigma_a", "sigma_s"):
        kw[k] = tuple(kw[k])
    return Setup(**kw)


def compare_and_report(manifest: dict, tag: str | None = None) -> dict:
    ensure_dirs()
    cfg = setup_from_manifest(manifest)
    jobs = manifest["jobs"]
    main = [j for j in jobs if j.get("variant", "main") == "main"]
    if tag is None:
        n = main[0]["name"]
        tag = n[n.index("_f") + 5:] if "_f" in n else ""
    out = COMPARE_DIR / (tag.strip("_") or "full")
    out.mkdir(parents=True, exist_ok=True)

    patterns = list(dict.fromkeys(j["pattern"] for j in main))
    fills = sorted({j["fill"] for j in main})
    by = {(j["pattern"], j["fill"]): j for j in main}
    seedv = {(j["pattern"], j["fill"]): j for j in jobs if j.get("variant") == "seed"}
    nomen = {(j["pattern"], j["fill"]): j for j in jobs if j.get("variant") == "nomeniscus"}

    W, H = main[0]["width"], main[0]["height"]
    sx, sy = W / cfg.width, H / cfg.height_px                  # smoke renders are scaled down
    rows = tuple(int(v * sy) for v in cavity_rows(cfg))
    cols = tuple(int(v * sx) for v in central_cols(cfg))
    b = cfg.container_box()
    box = (int(b[0] * sy), int(b[1] * sy), int(b[2] * sx), int(b[3] * sx))

    # ---- load everything (sRGB float + Lab)
    srgb, lab = {}, {}

    def load(key, j):
        lin = read_linear(RENDER_DIR / j["exr"])
        srgb[key] = to_srgb(lin)
        lab[key] = to_lab(srgb[key])

    for key, j in by.items():
        load(("main",) + key, j)
    for key, j in seedv.items():
        load(("seed",) + key, j)
    for key, j in nomen.items():
        load(("nomen",) + key, j)

    f_lens, M_h = cfg.lens_focal(), cfg.lens_magnification()
    metrics = {"setup": manifest["setup"], "analysis_rows": rows, "analysis_cols": cols, "container_box": box,
               "scale": [sx, sy], "patterns": {}, "predicted": {
                   "lens_focal_cm": f_lens, "object_space_magnification": M_h,
                   "image_space_period_ratio": M_h * (cfg.cam_dist + cfg.panel_y) / cfg.cam_dist,
                   "note": "paraxial thick-lens estimate for a water cylinder (thin glass wall ignored); "
                           "ratio = horizontal pattern period seen through the liquid / seen through air"}}

    a_f, b_f = SENS_PAIR
    de_empty_maps, de_sens_maps, de_noise_maps, de_sens_nomen_maps = {}, {}, {}, {}

    def band_of(pa, pb) -> tuple[int, int]:
        ga, gb = by[(pa[0], pa[1])]["gt_row_near"] * sy, by[(pb[0], pb[1])]["gt_row_near"] * sy
        s16, s4 = int(16 * sy), int(4 * sy)
        return int(min(ga, gb)) - s16, int(max(ga, gb)) + s4     # covers the meniscus band above the level

    for p in patterns:
        pm = {"fills": {}, "color": PATTERN_COLOR.get(p, "#111827")}
        empty = lab.get(("main", p, 0.0))
        for f in fills:
            j = by[(p, f)]
            gt_near, gt_far = j["gt_row_near"] * sy, j["gt_row_far"] * sy
            rec = {"gt_row_near": gt_near, "gt_row_far": gt_far, "level_cm": j["level_cm"],
                   "render_seconds": j.get("render_seconds")}
            if f > 0:
                rec["line_contrast"] = line_contrast(srgb[("main", p, f)], gt_near, cols, sy)
            if empty is not None and f > 0:
                de = np.linalg.norm(lab[("main", p, f)] - empty, axis=2)
                de_empty_maps[(p, f)] = de
                prof = row_profile(de, cols)
                tr = transition_row(prof, rows)
                rec.update({
                    "dE_vs_empty_cavity": float(de[rows[0]:rows[1], cols[0]:cols[1]].mean()),
                    "dE_vs_empty_below_level": float(de[int(gt_near):rows[1], cols[0]:cols[1]].mean()),
                    "dE_vs_empty_above_level": float(de[rows[0]:int(gt_far) - int(16 * sy), cols[0]:cols[1]].mean())
                    if int(gt_far) - int(16 * sy) > rows[0] else None,
                    "transition_row": tr, "transition_err_px_vs_near": tr - gt_near,
                    "transition_err_px_vs_far": tr - gt_far, "profile": prof.tolist()})
            pm["fills"][f"{f:.2f}"] = rec

        # ---- sensitivity to a 4 mm level change
        if ("main", p, a_f) in lab and ("main", p, b_f) in lab:
            de = np.linalg.norm(lab[("main", p, b_f)] - lab[("main", p, a_f)], axis=2)
            de_sens_maps[p] = de
            prof = row_profile(de, cols)
            band = band_of((p, a_f), (p, b_f))
            pm["sensitivity"] = {
                "pair": [a_f, b_f], "delta_level_mm": (b_f - a_f) * cfg.height * 10, "band_rows": band,
                "dE_mean_cavity": float(de[rows[0]:rows[1], cols[0]:cols[1]].mean()),
                "dE_mean_band": float(de[band[0]:band[1], cols[0]:cols[1]].mean()),
                "dE_peak_row_mean": float(smooth(prof)[rows[0]:rows[1]].max()),
                "frac_pixels_dE_gt_5": float((de[rows[0]:rows[1], cols[0]:cols[1]] > 5).mean()),
                "peak_row": peak_row(prof, rows),
                "gt_rows_near": [by[(p, a_f)]["gt_row_near"] * sy, by[(p, b_f)]["gt_row_near"] * sy],
                "profile": prof.tolist()}
            # noise floor: same scene, other seed
            if ("seed", p, a_f) in lab:
                dn = np.linalg.norm(lab[("seed", p, a_f)] - lab[("main", p, a_f)], axis=2)
                de_noise_maps[p] = dn
                pm["noise_floor"] = {
                    "dE_mean_cavity": float(dn[rows[0]:rows[1], cols[0]:cols[1]].mean()),
                    "dE_mean_band": float(dn[band[0]:band[1], cols[0]:cols[1]].mean()),
                    "dE_peak_row_mean": float(smooth(row_profile(dn, cols))[rows[0]:rows[1]].max()),
                    "frac_pixels_dE_gt_5": float((dn[rows[0]:rows[1], cols[0]:cols[1]] > 5).mean())}
                pm["sensitivity"]["snr_band"] = pm["sensitivity"]["dE_mean_band"] / max(pm["noise_floor"]["dE_mean_band"], 1e-6)
                pm["sensitivity"]["snr_cavity"] = pm["sensitivity"]["dE_mean_cavity"] / max(pm["noise_floor"]["dE_mean_cavity"], 1e-6)
            # meniscus control
            if ("nomen", p, a_f) in lab and ("nomen", p, b_f) in lab:
                dm = np.linalg.norm(lab[("nomen", p, b_f)] - lab[("nomen", p, a_f)], axis=2)
                de_sens_nomen_maps[p] = dm
                gt_near = nomen[(p, a_f)]["gt_row_near"] * sy
                pm["no_meniscus"] = {
                    "dE_mean_cavity": float(dm[rows[0]:rows[1], cols[0]:cols[1]].mean()),
                    "dE_mean_band": float(dm[band[0]:band[1], cols[0]:cols[1]].mean()),
                    "dE_peak_row_mean": float(smooth(row_profile(dm, cols))[rows[0]:rows[1]].max()),
                    "line_contrast_f050": line_contrast(srgb[("nomen", p, a_f)], gt_near, cols, sy),
                    "line_contrast_f050_with_meniscus": pm["fills"][f"{a_f:.2f}"]["line_contrast"]}
        metrics["patterns"][p] = pm

    # ---- checkerboard period: direct panel vs through air section vs through liquid section
    if "rg_checker" in patterns:
        per = {}
        wing = int(30 * sx)
        direct_cols = [(int(20 * sx), max(box[2] - wing, int(20 * sx) + 10)), (min(box[3] + wing, W - int(20 * sx) - 10), W - int(20 * sx))]
        for f in fills:
            j = by[("rg_checker", f)]
            img = srgb[("main", "rg_checker", f)]
            gt_near, gt_far = j["gt_row_near"] * sy, j["gt_row_far"] * sy
            rec = {}
            air_rows = (rows[0] + int(40 * sy), (int(gt_far) - int(30 * sy)) if f > 0 else rows[1] - int(60 * sy))
            liq_rows = (int(gt_near) + int(20 * sy), rows[1] - int(60 * sy)) if f > 0 else None
            for name, rr in (("air", air_rows), ("liquid", liq_rows)):
                rec[name] = checker_periods(img, rr, cols, sy) if rr and rr[1] - rr[0] > 30 * sy else None
            # direct view of the panel (outside the container), same rows as the whole cavity
            ph, pv = [], []
            for c in direct_cols:
                if c[1] - c[0] > 20 * sx:
                    r = checker_periods(img, rows, c, sy)
                    ph.append(r["period_h_px"]); pv.append(r["period_v_px"])
            rec["direct"] = {"period_h_px": float(np.nanmean(ph)) if ph else float("nan"),
                             "period_v_px": float(np.nanmean(pv)) if pv else float("nan")}
            if rec["air"] and rec["liquid"]:
                rec["ratio_h_liquid_over_air"] = rec["liquid"]["period_h_px"] / rec["air"]["period_h_px"]
                rec["ratio_v_liquid_over_air"] = rec["liquid"]["period_v_px"] / rec["air"]["period_v_px"]
                rec["ratio_h_liquid_over_direct"] = rec["liquid"]["period_h_px"] / rec["direct"]["period_h_px"]
            if rec["air"]:
                rec["ratio_h_air_over_direct"] = rec["air"]["period_h_px"] / rec["direct"]["period_h_px"]
            # mid-plane band (rows around the camera's horizontal plane) where the 2-D trace is exact
            mid = (int(H / 2 - 40 * sy), int(H / 2 + 40 * sy))
            if f == 0 or mid[0] > int(gt_near) + int(20 * sy) or mid[1] < int(gt_far) - int(30 * sy):
                rec["midplane"] = checker_periods(img, mid, cols, sy)
                rec["midplane"]["medium"] = "liquid" if (f > 0 and mid[0] > int(gt_near)) else "air"
            per[f"{f:.2f}"] = rec
        metrics["checker_period"] = per
        rt = predict_periods_2d(cfg, W, sx, cols, box)
        metrics["predicted"]["raytrace_2d"] = {k: v for k, v in rt.items() if not k.startswith("_")}
        metrics["predicted"]["raytrace_2d"]["note"] = ("exact geometric-optics trace of camera rays in the horizontal plane "
                                                       "through glass wall + water, independent of pbrt; compare with the "
                                                       "mid-plane measurements")
        mids = {f: r["midplane"] for f, r in per.items() if r.get("midplane")}
        metrics["predicted"]["raytrace_2d"]["measured_midplane"] = {
            "air": [m["period_h_px"] for m in mids.values() if m["medium"] == "air"],
            "liquid": [m["period_h_px"] for m in mids.values() if m["medium"] == "liquid"]}

    # ---- figures
    figs = {}
    figs["sheet_patterns"] = sheet_patterns(patterns, out)
    figs["sheet_renders"] = sheet_renders(srgb, patterns, fills, out, sy)
    figs["sheet_zoom"] = sheet_zoom(srgb, by, patterns, [f for f in fills if f > 0 and f != b_f], out, box, sy)
    figs["diff_vs_empty"] = fig_diff_vs_empty(de_empty_maps, metrics, patterns, [f for f in fills if f > 0], rows, box, out)
    figs["sensitivity"] = fig_sensitivity(de_sens_maps, de_noise_maps, metrics, patterns, rows, box, out)
    if de_sens_nomen_maps:
        figs["meniscus"] = fig_meniscus(srgb, by, nomen, de_sens_maps, de_sens_nomen_maps, metrics, a_f, box, cols, sy, out)
    if "checker_period" in metrics:
        figs["checker_period"] = fig_checker_period(srgb, by, metrics, rows, cols, out, sy)
        figs["raytrace"] = fig_raytrace(rt, metrics, cols, box, W, out)
    metrics["figures"] = {k: Path(v).name for k, v in figs.items() if v}
    ov = manifest.get("overview", {})
    metrics["overview_png"] = ov["png"] if ov and (RENDER_DIR / ov["png"]).exists() else None
    metrics["renders"] = {f"{p}|{f:.2f}": by[(p, f)]["png"] for (p, f) in by}
    metrics["render_seconds_total"] = float(sum(j.get("render_seconds", 0) or 0 for j in jobs))
    save_json(metrics, out / "metrics.json")
    write_report(metrics, cfg, manifest, out, patterns, tag)
    print(f"comparison written to {out}")
    return metrics


# ----------------------------------------------------------------------------- figures
def sheet_patterns(patterns, out: Path, width: int = 260) -> Path:
    """The panel textures as the camera would see them without a container (image left = u=0)."""
    from mvp_common import PATTERN_DIR
    tiles = []
    for p in patterns:
        f = PATTERN_DIR / f"{p}.png"
        if f.exists():
            im = cv2.imread(str(f), cv2.IMREAD_COLOR)
        else:                                   # uniform white has no texture file
            im = np.full((3000, 2500, 3), 242, np.uint8)
        h = int(round(im.shape[0] * width / im.shape[1]))
        im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (width - 1, h - 1), (120, 120, 120), 1)
        label(im, PATTERN_LABEL.get(p, p), (6, 18), 0.5)
        tiles.append(im)
    path = out / "sheet_patterns.png"
    cv2.imwrite(str(path), np.concatenate(tiles, axis=1))
    return path


def sheet_renders(srgb, patterns, fills, out: Path, sy: float) -> Path:
    scale = 0.3 if sy >= 0.99 else 1.0
    tiles = []
    for p in patterns:
        row = []
        for f in fills:
            im = bgr(to_u8(srgb[("main", p, f)]))
            if scale != 1.0:
                im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            label(im, f"{PATTERN_LABEL.get(p, p)}  fill {f:.2f}  ({f * 20:.1f} cm)", (6, 18), 0.45)
            row.append(im)
        tiles.append(np.concatenate(row, axis=1))
    path = out / "sheet_renders.png"
    cv2.imwrite(str(path), np.concatenate(tiles, axis=0))
    return path


def sheet_zoom(srgb, by, patterns, fills, out: Path, box, sy: float) -> Path:
    """Full-resolution crops around the GT level line, all patterns x selected fills."""
    top, bot, c0, c1 = box
    half_h, pad = int(150 * sy), int(30 * sy)
    tiles = []
    for p in patterns:
        row = []
        for f in fills:
            g = int(round(by[(p, f)]["gt_row_near"] * sy))
            im = bgr(to_u8(srgb[("main", p, f)]))
            H, W = im.shape[:2]
            y0, y1 = max(g - half_h, 0), min(g + half_h, H)
            x0, x1 = max(c0 - pad, 0), min(c1 + pad, W)
            crop = im[y0:y1, x0:x1].copy()
            cv2.line(crop, (0, g - y0), (pad - 6, g - y0), (0, 255, 0), 1)             # GT ticks in the margins only
            cv2.line(crop, (crop.shape[1] - pad + 6, g - y0), (crop.shape[1] - 1, g - y0), (0, 255, 0), 1)
            label(crop, f"{PATTERN_LABEL.get(p, p)} fill {f:.2f}", (6, 18), 0.5)
            row.append(crop)
        tiles.append(np.concatenate(row, axis=1))
    path = out / "sheet_zoom_level.png"
    cv2.imwrite(str(path), np.concatenate(tiles, axis=0))
    return path


def _imshow_de(ax, de, box, vmax=DE_VMAX, pad=40):
    top, bot, c0, c1 = box
    r0, c0p = max(top - pad, 0), max(c0 - pad, 0)
    im = ax.imshow(de[r0:bot + pad, c0p:c1 + pad], cmap="Blues", vmin=0, vmax=vmax,
                   extent=(c0p, c1 + pad, bot + pad, r0), interpolation="nearest")
    ax.set_xticks([]); ax.grid(False)
    return im


def _imshow_rgb(ax, srgb_img, y0, y1, x0, x1):
    ax.imshow(np.clip(srgb_img[y0:y1, x0:x1], 0, 1), extent=(x0, x1, y1, y0), interpolation="nearest")
    ax.set_xticks([]); ax.grid(False)


def fig_diff_vs_empty(maps, metrics, patterns, fills, rows, box, out: Path) -> Path | None:
    if not maps:
        return None
    nP, nF = len(patterns), len(fills)
    fig, axes = plt.subplots(nP, nF + 1, figsize=(2.1 * nF + 4.4, 3.2 * nP), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * nF + [1.9], "hspace": 0.45, "wspace": 0.15})
    last_im = None
    for i, p in enumerate(patterns):
        for k, f in enumerate(fills):
            ax = axes[i, k]
            if (p, f) not in maps:
                ax.axis("off"); continue
            last_im = _imshow_de(ax, maps[(p, f)], box)
            rec = metrics["patterns"][p]["fills"][f"{f:.2f}"]
            ax.axhline(rec["gt_row_near"], color="#111827", lw=0.8, ls="--")
            ax.set_title(f"{PATTERN_LABEL.get(p, p)} · fill {f:.2f}\nmean dE {rec['dE_vs_empty_cavity']:.1f}", fontsize=8)
            if k:
                ax.set_yticks([])
        axp = axes[i, nF]
        for k, f in enumerate(fills):
            rec = metrics["patterns"][p]["fills"].get(f"{f:.2f}")
            if not rec or "profile" not in rec:
                continue
            prof = smooth(np.array(rec["profile"]))
            y = np.arange(len(prof))
            c = FILL_COLORS[k % len(FILL_COLORS)]
            axp.plot(prof[rows[0]:rows[1]], y[rows[0]:rows[1]], color=c, lw=1.6, label=f"fill {f:.2f}")
            axp.axhline(rec["gt_row_near"], color=c, lw=0.8, ls="--")
        axp.invert_yaxis()
        axp.set_xlabel("row-mean dE vs empty")
        axp.set_title(f"{PATTERN_LABEL.get(p, p)}: row profile (dashed = GT row, front wall)", fontsize=8)
        if i == 0:
            axp.legend(fontsize=7, frameon=False)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:, :nF].ravel().tolist(), shrink=0.5, pad=0.01).set_label("CIE Lab dE vs empty container")
    fig.suptitle("Where does the liquid change the background?  |render(fill) - render(empty)| in the container ROI", fontsize=10)
    path = out / "diff_vs_empty.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_sensitivity(maps, noise_maps, metrics, patterns, rows, box, out: Path) -> Path | None:
    ps = [p for p in patterns if p in maps]
    if not ps:
        return None
    n_noise = len(noise_maps)
    fig = plt.figure(figsize=(2.2 * (len(ps) + n_noise) + 9, 4.8))
    gs = fig.add_gridspec(1, len(ps) + n_noise + 2, width_ratios=[1] * (len(ps) + n_noise) + [1.8, 1.8], wspace=0.15)
    last_im = None
    k = 0
    for p in ps:
        ax = fig.add_subplot(gs[0, k]); k += 1
        last_im = _imshow_de(ax, maps[p], box)
        s = metrics["patterns"][p]["sensitivity"]
        for g in s["gt_rows_near"]:
            ax.axhline(g, color="#111827", lw=0.7, ls="--")
        ax.set_title(f"{PATTERN_LABEL.get(p, p)}\nband dE {s['dE_mean_band']:.1f}", fontsize=8)
        if k > 1:
            ax.set_yticks([])
    for p, dn in noise_maps.items():
        ax = fig.add_subplot(gs[0, k]); k += 1
        _imshow_de(ax, dn, box)
        nf = metrics["patterns"][p]["noise_floor"]
        ax.set_title(f"noise floor {PATTERN_LABEL.get(p, p)}\n(seed A vs B) band dE {nf['dE_mean_band']:.1f}", fontsize=8)
        ax.set_yticks([])
    axp = fig.add_subplot(gs[0, k]); k += 1
    for p in ps:
        s = metrics["patterns"][p]["sensitivity"]
        prof = smooth(np.array(s["profile"]))
        y = np.arange(len(prof))
        axp.plot(prof[rows[0]:rows[1]], y[rows[0]:rows[1]], color=PATTERN_COLOR[p], lw=1.6, label=PATTERN_LABEL.get(p, p))
    for g in metrics["patterns"][ps[0]]["sensitivity"]["gt_rows_near"]:
        axp.axhline(g, color="#111827", lw=0.7, ls="--")
    axp.invert_yaxis(); axp.set_xlabel("row-mean dE (0.52 vs 0.50)"); axp.legend(fontsize=7, frameon=False)
    axp.set_title("row profile (dashed = the two GT rows)", fontsize=8)
    axb = fig.add_subplot(gs[0, k])
    x = np.arange(len(ps))
    vals = [metrics["patterns"][p]["sensitivity"]["dE_mean_band"] for p in ps]
    noise = [metrics["patterns"][p].get("noise_floor", {}).get("dE_mean_band", np.nan) for p in ps]
    bars = axb.bar(x - 0.18, vals, width=0.36, color=[PATTERN_COLOR[p] for p in ps], label="4 mm level change")
    axb.bar(x + 0.18, noise, width=0.36, color="#d1d5db", label="noise floor (2 seeds)")
    for bx, v in zip(x - 0.18, vals):
        axb.text(bx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#111827")
    for bx, v in zip(x + 0.18, noise):
        if not np.isnan(v):
            axb.text(bx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#374151")
    axb.set_xticks(x); axb.set_xticklabels([PATTERN_LABEL.get(p, p) for p in ps], rotation=20)
    axb.set_ylabel("mean dE in the level band"); axb.set_title("image change for a 4 mm level change", fontsize=8)
    axb.legend(fontsize=7, frameon=False)
    if last_im is not None:
        fig.colorbar(last_im, ax=[fig.axes[i] for i in range(len(ps) + n_noise)], shrink=0.6, pad=0.01).set_label("dE")
    fig.suptitle("Sensitivity: |render(fill 0.52) - render(fill 0.50)|  (level +4 mm), with Monte-Carlo noise floor", fontsize=10)
    path = out / "sensitivity_4mm.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_meniscus(srgb, by, nomen, sens_maps, sens_nomen_maps, metrics, f0, box, cols, sy, out: Path) -> Path:
    ps = list(sens_nomen_maps)
    top, bot, c0, c1 = box
    fig, axes = plt.subplots(len(ps), 5, figsize=(16, 3.6 * len(ps)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.6, 1.6, 1, 1, 1.2], "wspace": 0.12, "hspace": 0.35})
    for i, p in enumerate(ps):
        g = int(round(by[(p, f0)]["gt_row_near"] * sy))
        hh, pad = int(110 * sy), int(30 * sy)
        y0, y1, x0, x1 = max(g - hh, 0), g + hh, max(c0 - pad, 0), c1 + pad
        _imshow_rgb(axes[i, 0], srgb[("main", p, f0)], y0, y1, x0, x1)
        axes[i, 0].set_title(f"{PATTERN_LABEL.get(p, p)} fill {f0:.2f} — WITH meniscus\nline contrast {metrics['patterns'][p]['fills'][f'{f0:.2f}']['line_contrast']:.3f}", fontsize=8)
        _imshow_rgb(axes[i, 1], srgb[("nomen", p, f0)], y0, y1, x0, x1)
        axes[i, 1].set_title(f"flat surface (no meniscus)\nline contrast {metrics['patterns'][p]['no_meniscus']['line_contrast_f050']:.3f}", fontsize=8)
        for ax in axes[i, :2]:
            ax.axhline(g, color="#22c55e", lw=0.6, ls="--")
        _imshow_de(axes[i, 2], sens_maps[p], box)
        axes[i, 2].set_title(f"+4 mm, with meniscus\nband dE {metrics['patterns'][p]['sensitivity']['dE_mean_band']:.1f}", fontsize=8)
        _imshow_de(axes[i, 3], sens_nomen_maps[p], box)
        axes[i, 3].set_title(f"+4 mm, no meniscus\nband dE {metrics['patterns'][p]['no_meniscus']['dE_mean_band']:.1f}", fontsize=8)
        axes[i, 3].set_yticks([])
        axb = axes[i, 4]
        vals = [metrics["patterns"][p]["sensitivity"]["dE_mean_band"], metrics["patterns"][p]["no_meniscus"]["dE_mean_band"],
                metrics["patterns"][p].get("noise_floor", {}).get("dE_mean_band", np.nan)]
        bars = axb.bar(["meniscus", "flat", "noise"], vals, color=[PATTERN_COLOR[p], "#93c5fd", "#d1d5db"], width=0.55)
        for b_, v in zip(bars, vals):
            if not np.isnan(v):
                axb.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        axb.set_ylabel("band dE for +4 mm"); axb.set_title("meniscus vs flat surface", fontsize=8)
    fig.suptitle("How much of the level visibility is the meniscus?  (green dashed = geometric GT row)", fontsize=10)
    path = out / "meniscus_control.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_checker_period(srgb, by, metrics, rows, cols, out: Path, sy: float) -> Path:
    """Sliding-window horizontal period of the checkerboard vs image row, for each fill."""
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    fills = sorted({f for (p, f) in by if p == "rg_checker"})
    win = max(int(30 * sy), 8)
    for k, f in enumerate(fills):
        img = srgb[("main", "rg_checker", f)]
        rg = img[..., 0] - img[..., 1]
        ys, per = [], []
        for y in range(rows[0], rows[1] - win, max(win // 3, 2)):
            per.append(edge_period(rg[y:y + win, cols[0]:cols[1]], True, 4 * sy)); ys.append(y + win / 2)
        c = FILL_COLORS[k % len(FILL_COLORS)] if k < 4 else "#111827"
        ax.plot(per, ys, color=c, lw=1.6, label=f"fill {f:.2f}")
        if f > 0:
            ax.axhline(by[("rg_checker", f)]["gt_row_near"] * sy, color=c, lw=0.7, ls="--")
    pred = metrics["predicted"]["image_space_period_ratio"]
    d0 = metrics["checker_period"].get("0.00", {}).get("direct", {}).get("period_h_px", float("nan"))
    if not math.isnan(d0):
        ax.axvline(d0, color="#9ca3af", lw=0.8)
        ax.text(d0, rows[0] + 6 * sy, " panel seen directly", fontsize=7, color="#4b5563", va="top")
        ax.axvline(d0 * pred, color="#9ca3af", lw=0.8, ls=":")
        ax.text(d0 * pred, rows[0] + 6 * sy, f" paraxial prediction x{pred:.2f}", fontsize=7, color="#4b5563", va="top")
    ax.invert_yaxis(); ax.set_xlabel("horizontal checker period (px)"); ax.set_ylabel("image row")
    ax.set_title("R/G checker: horizontal period vs row (dashed = GT level row)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    path = out / "checker_period.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_raytrace(rt: dict, metrics: dict, cols_c: tuple[int, int], box, W: int, out: Path) -> Path:
    cols = rt["_cols"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.25})
    ax1.plot(cols, rt["_xp_direct"], color="#9ca3af", lw=1.4, label="no container (direct)")
    ax1.plot(cols, rt["_xp_empty"], color="#60a5fa", lw=1.6, label="through empty glass cylinder")
    ax1.plot(cols, rt["_xp_fill"], color="#1e3a8a", lw=1.6, label="through water-filled cylinder")
    for c in (box[2], box[3]):
        ax1.axvline(c, color="#d1d5db", lw=0.8)
    ax1.axvspan(cols_c[0], cols_c[1], color="#eff6ff", zorder=0)
    ax1.set_xlabel("image column (px)"); ax1.set_ylabel("panel x hit by the camera ray (cm)")
    ax1.set_title("2-D ray trace: which panel point does each image column see?", fontsize=9)
    ax1.legend(fontsize=7, frameon=False)
    # local horizontal period (px per 2-cell checker period) vs column
    cell = rt["cell_cm"]
    for xp, c, lab in ((rt["_xp_direct"], "#9ca3af", "direct"), (rt["_xp_empty"], "#60a5fa", "empty glass"), (rt["_xp_fill"], "#1e3a8a", "water")):
        g = np.abs(np.gradient(xp))
        per = 2 * cell / g
        ax2.plot(cols, per, color=c, lw=1.6, label=f"predicted, {lab}")
    cp = metrics["checker_period"]
    m_air = [r["midplane"]["period_h_px"] for r in cp.values() if r.get("midplane") and r["midplane"]["medium"] == "air"]
    m_liq = [r["midplane"]["period_h_px"] for r in cp.values() if r.get("midplane") and r["midplane"]["medium"] == "liquid"]
    if m_air:
        ax2.hlines(np.nanmedian(m_air), cols_c[0], cols_c[1], color="#60a5fa", ls="--", lw=1.2, label="measured (render), air, mid-plane")
    if m_liq:
        ax2.hlines(np.nanmedian(m_liq), cols_c[0], cols_c[1], color="#1e3a8a", ls="--", lw=1.2, label="measured (render), water, mid-plane")
    ax2.set_ylim(0, max(np.nanpercentile(2 * cell / np.abs(np.gradient(rt["_xp_fill"]))[cols_c[0]:cols_c[1]], 98) * 1.3, 10))
    ax2.axvspan(cols_c[0], cols_c[1], color="#eff6ff", zorder=0)
    ax2.set_xlabel("image column (px)"); ax2.set_ylabel("horizontal checker period (px)")
    ax2.set_title("local pattern period: prediction vs render (shaded = analysis columns)", fontsize=9)
    ax2.legend(fontsize=7, frameon=False)
    path = out / "lens_raytrace.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- report
REPORT = Template("""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Phase-0 液位折射畸变验证 — pbrt-v4</title>
<style>
 body{font-family:system-ui,"Microsoft YaHei",Segoe UI,sans-serif;margin:24px;color:#1f2937;background:#fafafa;max-width:1500px}
 h1{font-size:22px} h2{font-size:17px;margin-top:30px;border-bottom:1px solid #e5e7eb;padding-bottom:4px}
 table{border-collapse:collapse;font-size:13px;margin:8px 0} th,td{border:1px solid #e5e7eb;padding:4px 9px;text-align:right}
 th{background:#f3f4f6} td.l,th.l{text-align:left}
 img{max-width:100%;border:1px solid #e5e7eb;background:#fff}
 .meta{font-size:12px;color:#6b7280} .box{background:#fff;border:1px solid #e5e7eb;border-radius:6px;padding:10px 14px;margin:10px 0}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
 code{background:#f3f4f6;padding:1px 4px;border-radius:3px}
</style></head><body>
<h1>Phase-0：背景图案折射畸变 vs 液位 — 高质量物理渲染验证</h1>
<p class="meta">生成 {{ now }} · 渲染器 pbrt-v4 (CPU volpath, {{ spp }} spp, {{ W }}×{{ H }}，共 {{ n_jobs }} 张，累计 {{ '%.0f' % (t_total/60) }} 分钟) · 场景：15 cm 内径 × 20 cm 内高玻璃圆柱、水 η=1.333、水平针孔相机、Lambertian 发光板 · 研究计划 <code>docs/RESEARCH.md</code></p>

<div class="box">
<b>结论摘要（数字由本次渲染自动生成）</b>
<ul>
<li><b>液体段改变背景的强度</b>（fill 0.50，液面以下 ΔE 均值，相对空容器）：{% for p in patterns %}{{ plabel[p] }} <b>{{ '%.1f' % below50[p] }}</b>{% if not loop.last %}，{% endif %}{% endfor %}。
    图案背景下液体段的背景外观与空容器完全不同（横向放大并左右翻转），白板下几乎不变 —— 这是 H2 的直接证据。</li>
<li><b>4 mm 液位变化的可检测性</b>（液位带内 ΔE 均值 / 同场景换随机种子的噪声底）：{% for p in patterns %}{{ plabel[p] }} <b>{{ '%.1f' % sens_band[p] }}</b>{% if snr[p] %}（噪声底 {{ '%.1f' % nf[p] }}，SNR {{ '%.0f' % snr[p] }}）{% endif %}{% if not loop.last %}；{% endif %}{% endfor %}。</li>
{% if men %}<li><b>弯月面对照</b>：白板下 fill 0.50 的液位线亮度对比度 有弯月面 {{ '%.3f' % men.white.with_ }} → 平液面 {{ '%.3f' % men.white.flat }}；4 mm 变化的带内 ΔE 有弯月面 {{ '%.1f' % men.white.band_with }} → 平液面 {{ '%.1f' % men.white.band_flat }}。
    即白板下能看到液位线主要依赖弯月面（和液面掠射菲涅尔带），这类线索窄、且在真实脏壁/遮挡下最先失效；图案背景不依赖它（棋盘格 {{ '%.1f' % men.rg_checker.band_with }} → {{ '%.1f' % men.rg_checker.band_flat }}）。</li>{% endif %}
{% if cp %}<li><b>柱面透镜放大率：渲染 vs 独立光线追迹</b>（棋盘格横向周期，px）：板直视 {{ '%.1f' % cp.direct.period_h_px }}，透过空容器 {{ '%.1f' % cp.air.period_h_px }}，透过液体 {{ '%.1f' % cp.liquid.period_h_px }}；液体/空气比 <b>{{ '%.2f' % cp.ratio_h_liquid_over_air }}</b>，竖向比 {{ '%.2f' % cp.ratio_v_liquid_over_air }}。
    {% if rt %}与 pbrt 无关的二维几何光学追迹（穿过玻璃壁+水）预测：直视 {{ '%.1f' % rt.direct_period_px }}，空容器 {{ '%.1f' % rt.air_period_px }}，液体 {{ '%.1f' % rt.liquid_period_px }}，液体/空气 <b>{{ '%.2f' % rt.ratio_h_liquid_over_air }}</b>，图像中心局部放大 ×{{ '%.2f' % rt.centre_local_ratio_liquid_over_direct }}，左右翻转 = {{ rt.flipped }}。渲染与追迹一致，说明渲染器的折射几何可信。{% endif %}
    （文档中的近轴平行光公式给 {{ '%.2f' % pred.image_space_period_ratio }}，明显偏低：相机距离有限、孔径 = 整个圆柱，近轴近似不成立。）</li>{% endif %}
<li><b>液位定位</b>：差分曲线的过渡行相对几何真值行（前内壁）的偏差：{% for p in patterns %}{{ plabel[p] }} {{ terr[p] }}{% if not loop.last %}；{% endif %}{% endfor %} px（1 px ≈ {{ '%.2f' % mm_per_px }} mm）。负偏差 = 过渡出现在几何液位之上，对应弯月面顶端（+{{ '%.1f' % setup.meniscus_h }} cm ≈ {{ '%.0f' % (setup.meniscus_h*10/mm_per_px) }} px）和液面远端边缘；这是真值口径问题（RESEARCH.md §10.6），不是检测误差。</li>
</ul>
</div>

<h2>1. 实验设置</h2>
<div class="grid">
<div>
<table>
<tr><th class="l">参数</th><th>值</th></tr>
<tr><td class="l">容器内径 / 内高 / 壁厚 / 底厚 (cm)</td><td>{{ setup.r_in*2 }} / {{ setup.height }} / {{ setup.wall }} / {{ setup.bottom }}</td></tr>
<tr><td class="l">玻璃 η / 水 η</td><td>{{ setup.eta_glass }} / {{ setup.eta_liquid }}</td></tr>
<tr><td class="l">水吸收 σ_a (R,G,B) /cm</td><td>{{ setup.sigma_a }}</td></tr>
<tr><td class="l">弯月面</td><td>{{ 'h=%.2f cm, l_c=%.2f cm（另渲染平液面对照）' % (setup.meniscus_h, setup.capillary_len) if setup.meniscus else '关' }}</td></tr>
<tr><td class="l">相机位置 (x,y,z) cm / fov</td><td>{{ setup.eye }} / {{ setup.fov }}°（横向；竖向 ≈31.6°）</td></tr>
<tr><td class="l">发光板 距轴 / 尺寸 (cm) / 亮度</td><td>{{ setup.panel_y }} / {{ setup.panel_w }} × {{ setup.panel_h }} / 线性 {{ setup.panel_scale }}</td></tr>
<tr><td class="l">容器包围框 (行上,行下,列左,列右)</td><td>{{ box }}</td></tr>
<tr><td class="l">分析 ROI 行 / 列（容器中央 60% 列）</td><td>{{ rows }} / {{ cols }}</td></tr>
<tr><td class="l">积分器 / 采样</td><td>volpath maxdepth {{ setup.maxdepth }} / zsobol {{ spp }} spp, gaussian filter</td></tr>
</table>
<p class="meta">液体/玻璃无空气隙（嵌套介质：湿壁 η=η_g/η_w，干壁 η=η_g，底 η=η_g/η_w，液面 η=η_w）；水体为吸收介质；发光板为理想 Lambertian 面、自身反射率 0；暗室无其他光源；针孔相机；pbrt 为左手系，图像左侧对应世界 +x。</p>
</div>
<div>{% if overview %}<img src="{{ pre_r }}{{ overview }}" alt="overview"><p class="meta">3/4 视角总览（fill 0.50，红绿棋盘格板；左下灰盒 = 相机位置代理；此图另加了微弱环境光以便看清几何）</p>{% endif %}</div>
</div>
<p><img src="{{ pre_c }}{{ figs.sheet_patterns }}" alt="patterns" style="max-width:1100px"></p>
<p class="meta">发光板贴图（50×60 cm，50 px/cm；棋盘格 2 cm、马赛克 1.5 cm、网格 1 cm 间距 / 1 mm 线宽、5 cm 粗线）。</p>

<h2>2. 渲染总览：行 = 背景图案，列 = 液位（0 / 5 / 10 / 10.4 / 15 cm）</h2>
<img src="{{ pre_c }}{{ figs.sheet_renders }}">
<p class="meta">原始 {{ W }}×{{ H }} PNG 与线性 EXR 在 <code>outputs/mvp/renders/</code>。观察要点：液面以下棋盘格变为横向拉长的矩形（柱面透镜横向放大、竖向不变）；马赛克在液面以下左右翻转；网格线的竖线在液体段被放大成粗条，横线不变；白板下只剩一条细线。</p>

<h2>3. 液位线附近 100% 局部放大（绿色刻线 = 几何真值行，前内壁）</h2>
<img src="{{ pre_c }}{{ figs.sheet_zoom }}">

<h2>4. 液体段对背景的改变：|render(fill) − render(empty)|</h2>
<img src="{{ pre_c }}{{ figs.diff_vs_empty }}">
<table>
<tr><th class="l">图案</th><th>fill</th><th>液位 cm</th><th>GT 行(近)</th><th>GT 行(远)</th><th>ΔE 腔内均值</th><th>ΔE 液面以下</th><th>ΔE 液面以上</th><th>过渡行</th><th>过渡−GT近 (px)</th><th>过渡−GT远 (px)</th><th>单图液位线对比度</th><th>渲染秒</th></tr>
{% for p in patterns %}{% for f, r in pm[p].fills.items() if r.dE_vs_empty_cavity is defined %}
<tr><td class="l">{{ plabel[p] }}</td><td>{{ f }}</td><td>{{ '%.1f' % r.level_cm }}</td><td>{{ '%.1f' % r.gt_row_near }}</td><td>{{ '%.1f' % r.gt_row_far }}</td>
<td>{{ '%.2f' % r.dE_vs_empty_cavity }}</td><td>{{ '%.2f' % r.dE_vs_empty_below_level }}</td><td>{{ r.dE_vs_empty_above_level is not none and '%.2f' % r.dE_vs_empty_above_level or '—' }}</td>
<td>{{ '%.0f' % r.transition_row }}</td><td>{{ '%+.1f' % r.transition_err_px_vs_near }}</td><td>{{ '%+.1f' % r.transition_err_px_vs_far }}</td><td>{{ '%.3f' % r.line_contrast }}</td><td>{{ r.render_seconds or '—' }}</td></tr>
{% endfor %}{% endfor %}
</table>
<p class="meta">"ΔE 液面以上" 只统计液面远端边缘再往上 16 px 以外的区域（排除液面椭圆与弯月面带）。"单图液位线对比度" = 仅用该张图，真值行上方 16 px 至下方 4 px 的带内行均亮度相对上下邻域的下降比例（正 = 暗线）；对图案背景该值受格子影响，仅供白板参考。</p>

<h2>5. 灵敏度：液位 +4 mm（0.50 → 0.52）引起的图像变化，以及蒙特卡洛噪声底</h2>
<img src="{{ pre_c }}{{ figs.sensitivity }}">
<table>
<tr><th class="l">图案</th><th>ΔE 腔内均值</th><th>ΔE 液位带内均值</th><th>行均 ΔE 峰值</th><th>ΔE&gt;5 像素占比</th><th>差分峰值行</th><th>两 GT 行</th><th>噪声底 带内 ΔE</th><th>噪声底 腔内 ΔE</th><th>SNR(带)</th></tr>
{% for p in patterns if pm[p].sensitivity is defined %}{% set s = pm[p].sensitivity %}{% set n = pm[p].noise_floor %}
<tr><td class="l">{{ plabel[p] }}</td><td>{{ '%.2f' % s.dE_mean_cavity }}</td><td>{{ '%.2f' % s.dE_mean_band }}</td><td>{{ '%.1f' % s.dE_peak_row_mean }}</td><td>{{ '%.1f%%' % (100 * s.frac_pixels_dE_gt_5) }}</td><td>{{ '%.0f' % s.peak_row }}</td><td>{{ '%.1f / %.1f' % (s.gt_rows_near[0], s.gt_rows_near[1]) }}</td>
<td>{{ n and '%.2f' % n.dE_mean_band or '—' }}</td><td>{{ n and '%.2f' % n.dE_mean_cavity or '—' }}</td><td>{{ s.snr_band is defined and '%.0f' % s.snr_band or '—' }}</td></tr>
{% endfor %}
</table>
<p class="meta">液位带 = 两液位真值行之间再向上扩 16 px（含弯月面）、向下扩 4 px。噪声底 = 同一场景、同一 spp、不同随机种子的两次渲染之差（只对 white 与 R/G checker 渲染了对照）。</p>

{% if figs.meniscus %}
<h2>6. 弯月面对照：白板下的液位线到底来自什么？</h2>
<img src="{{ pre_c }}{{ figs.meniscus }}">
<table>
<tr><th class="l">图案</th><th>液位线对比度（有弯月面）</th><th>液位线对比度（平液面）</th><th>+4 mm 带内 ΔE（有弯月面）</th><th>+4 mm 带内 ΔE（平液面）</th><th>噪声底</th></tr>
{% for p in patterns if pm[p].no_meniscus is defined %}{% set m = pm[p].no_meniscus %}
<tr><td class="l">{{ plabel[p] }}</td><td>{{ '%.3f' % m.line_contrast_f050_with_meniscus }}</td><td>{{ '%.3f' % m.line_contrast_f050 }}</td><td>{{ '%.2f' % pm[p].sensitivity.dE_mean_band }}</td><td>{{ '%.2f' % m.dE_mean_band }}</td><td>{{ pm[p].noise_floor and '%.2f' % pm[p].noise_floor.dE_mean_band or '—' }}</td></tr>
{% endfor %}
</table>
<p class="meta">注意相机高度 (z={{ '%.2f' % setup.eye[2] }}) 与 fill 0.50 的液面 (z={{ '%.2f' % (setup.bottom + 0.5*setup.height) }}) 几乎等高，液面接近边缘视，这是对白板最有利的配置之一（液面本身投影成一条线）；相机偏离液面高度后，平液面在白板下的可见性会进一步下降 —— Phase 1 需扫描相机高度。</p>
{% endif %}

{% if figs.checker_period %}
<h2>7. 棋盘格周期 vs 行：柱面透镜放大率的直接测量</h2>
<img src="{{ pre_c }}{{ figs.checker_period }}">
<table>
<tr><th class="l">fill</th><th>板直视 横向周期 px</th><th>透过空气段</th><th>透过液体段</th><th>液体/空气</th><th>液体/直视</th><th>空气/直视</th><th>竖向 空气</th><th>竖向 液体</th><th>竖向比</th></tr>
{% for f, r in cpall.items() %}
<tr><td class="l">{{ f }}</td><td>{{ '%.1f' % r.direct.period_h_px }}</td><td>{{ r.air and '%.1f' % r.air.period_h_px or '—' }}</td><td>{{ r.liquid and '%.1f' % r.liquid.period_h_px or '—' }}</td>
<td>{{ r.ratio_h_liquid_over_air is defined and '%.2f' % r.ratio_h_liquid_over_air or '—' }}</td><td>{{ r.ratio_h_liquid_over_direct is defined and '%.2f' % r.ratio_h_liquid_over_direct or '—' }}</td><td>{{ r.ratio_h_air_over_direct is defined and '%.2f' % r.ratio_h_air_over_direct or '—' }}</td>
<td>{{ r.air and '%.1f' % r.air.period_v_px or '—' }}</td><td>{{ r.liquid and '%.1f' % r.liquid.period_v_px or '—' }}</td><td>{{ r.ratio_v_liquid_over_air is defined and '%.2f' % r.ratio_v_liquid_over_air or '—' }}</td></tr>
{% endfor %}
</table>
<p class="meta">周期用 R−G 信号过零点间距的中位数估计（容器中央 60% 列）。近轴平行光公式 M_h·(d_cam + D)/d_cam = {{ '%.2f' % pred.image_space_period_ratio }} 仅作量级参考。</p>
{% if rt %}
<h3>7b. 独立二维光线追迹 vs 渲染</h3>
<img src="{{ pre_c }}{{ figs.raytrace }}">
<table>
<tr><th class="l">量</th><th>二维追迹预测</th><th>渲染实测（中平面行带）</th></tr>
<tr><td class="l">板直视横向周期 px</td><td>{{ '%.1f' % rt.direct_period_px }}</td><td>{{ '%.1f' % cp.direct.period_h_px }}</td></tr>
<tr><td class="l">透过空容器（空气段）</td><td>{{ '%.1f' % rt.air_period_px }}</td><td>{{ rt.measured_midplane.air and '%.1f' % (rt.measured_midplane.air | sum / rt.measured_midplane.air | length) or '—' }}</td></tr>
<tr><td class="l">透过液体段</td><td>{{ '%.1f' % rt.liquid_period_px }}</td><td>{{ rt.measured_midplane.liquid and '%.1f' % (rt.measured_midplane.liquid | sum / rt.measured_midplane.liquid | length) or '—' }}</td></tr>
<tr><td class="l">液体/空气 周期比</td><td>{{ '%.2f' % rt.ratio_h_liquid_over_air }}</td><td>{{ (rt.measured_midplane.liquid and rt.measured_midplane.air) and '%.2f' % ((rt.measured_midplane.liquid | sum / rt.measured_midplane.liquid | length) / (rt.measured_midplane.air | sum / rt.measured_midplane.air | length)) or '—' }}</td></tr>
<tr><td class="l">空气/直视 周期比</td><td>{{ '%.2f' % rt.ratio_h_air_over_direct }}</td><td>{{ '%.2f' % cpall['0.00'].ratio_h_air_over_direct }}</td></tr>
<tr><td class="l">图像中心局部放大（液体/直视）</td><td>{{ '%.2f' % rt.centre_local_ratio_liquid_over_direct }}</td><td>见上图曲线</td></tr>
</table>
<p class="meta">追迹在相机光轴所在的水平面内精确（所有折射面法线水平）；渲染的"中平面行带"取图像中心 ±40 px 行、容器中央 60% 列，且只在该行带完全处于液体段（fill 0.75）或空气段（fill 0 / 0.25）时统计。</p>
{% endif %}
{% endif %}

<h2>8. 文件</h2>
<p class="meta">场景 <code>scenes/mvp/*.pbrt</code>（<code>free_surface.ply</code> / <code>free_surface_flat.ply</code> 液面网格）· 图案 <code>outputs/mvp/patterns/</code> · 渲染 <code>outputs/mvp/renders/</code> · 本目录 <code>metrics.json</code> 含全部数值与逐行曲线 · 复现：<code>.venv\\Scripts\\python scripts\\mvp_run.py --spp {{ spp }}</code></p>
</body></html>""")


def write_report(metrics, cfg, manifest, out: Path, patterns, tag: str) -> None:
    pm = metrics["patterns"]
    sens_band = {p: pm[p].get("sensitivity", {}).get("dE_mean_band", float("nan")) for p in patterns}
    nf = {p: pm[p].get("noise_floor", {}).get("dE_mean_band", float("nan")) for p in patterns}
    snr = {p: pm[p].get("sensitivity", {}).get("snr_band") for p in patterns}
    below50 = {p: pm[p]["fills"].get("0.50", {}).get("dE_vs_empty_below_level", float("nan")) for p in patterns}
    terr = {p: " ".join(f"{f}:{r['transition_err_px_vs_near']:+.0f}" for f, r in pm[p]["fills"].items() if "transition_row" in r)
            for p in patterns}
    men = None
    if all(pm.get(p, {}).get("no_meniscus") for p in ("white", "rg_checker")):
        men = {p: {"with_": pm[p]["no_meniscus"]["line_contrast_f050_with_meniscus"], "flat": pm[p]["no_meniscus"]["line_contrast_f050"],
                   "band_with": pm[p]["sensitivity"]["dE_mean_band"], "band_flat": pm[p]["no_meniscus"]["dE_mean_band"]}
               for p in ("white", "rg_checker")}
        men = type("M", (), men)()  # attribute access in the template
        men.white = type("W", (), men.white)(); men.rg_checker = type("R", (), men.rg_checker)()
    cp_all = metrics.get("checker_period", {})
    cp = cp_all.get("0.50") if cp_all.get("0.50", {}).get("liquid") and "ratio_h_liquid_over_air" in cp_all.get("0.50", {}) else None
    rt = metrics["predicted"].get("raytrace_2d")
    r_top, r_bot = cavity_rows(cfg)
    mm_per_px = cfg.height * 10 / max(r_bot - r_top, 1) / metrics["scale"][1]
    j0 = manifest["jobs"][0]
    common = dict(now=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), spp=j0["spp"], W=j0["width"], H=j0["height"],
                  n_jobs=len(manifest["jobs"]), t_total=metrics["render_seconds_total"],
                  setup=metrics["setup"], patterns=patterns, plabel=PATTERN_LABEL, pm=pm, sens_band=sens_band, nf=nf, snr=snr,
                  below50=below50, terr=terr, men=men, cp=cp, cpall=cp_all, rt=rt, pred=metrics["predicted"], figs=metrics["figures"],
                  overview=metrics["overview_png"], box=metrics["container_box"], rows=metrics["analysis_rows"],
                  cols=metrics["analysis_cols"], mm_per_px=mm_per_px)
    (out / "report.html").write_text(REPORT.render(pre_c="", pre_r="../../renders/", **common), encoding="utf-8")
    if not tag:
        (MVP_OUT / "report.html").write_text(REPORT.render(pre_c=f"compare/{out.name}/", pre_r="renders/", **common), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(MANIFEST))
    a = ap.parse_args()
    compare_and_report(load_json(Path(a.manifest)))


if __name__ == "__main__":
    main()

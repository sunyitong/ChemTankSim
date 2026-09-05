"""Evaluate level detectors on the Phase-1 test set (baseline/compare pairs + manifest GT).

Detectors
  step_raw      row-mean colour difference vs baseline, single largest step        (web app, Adaptive off)
  step_adapt    same after denoise / exposure match / shift search / threshold     (web app, Adaptive on)
  cp_v2         multi-level: per-row features [thresholded dE, log horizontal magnification M(r)
                estimated by correlating each compare row with the horizontally rescaled baseline
                row], optimal-partitioning change-point detection (up to 3 change points)

For every compare job the ROI is the vessel interior bounding box (rows rim..bottom, columns from
the silhouette half-width). Errors are reported in pixels and in % of the ROI height (the web
app's 0-100 % scale). Outputs: outputs/testset/eval/{eval.json, eval.csv, profiles_*.png, summary.md}
and outputs/testset/cases.json (ROI + GT per case, consumed by the web app bundler).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mvp_common import load_json, save_json  # noqa: E402
from testset_scenes import CAMERAS, COMBOS, TS_IMAGES, TS_OUT, VESSELS  # noqa: E402

EVAL = TS_OUT / "eval"
THR = 0.05


# ----------------------------------------------------------------------------- ROI from geometry
def vessel_roi(job: dict, W: int, H: int) -> dict:
    v = VESSELS[job["vessel"]]
    cam = CAMERAS[job["camera"]]
    if v.kind == "revolved":
        r_max = max(v.r_out(z) for z in np.linspace(0, v.H, 200))
    else:
        r_max = v.params["R"]
    half = math.tan(math.asin(min(r_max / cam["dist"], 0.99))) / math.tan(math.radians(cam["fov"]) / 2) * W / 2
    inner = half * (1 - v.t / r_max) * 0.96
    g = job["gt"]
    y0 = int(round(g["rim_row_front"])) + 4
    y1 = int(round(g["bottom_row_front"])) - 4
    return {"x0": int(round(W / 2 - inner)), "x1": int(round(W / 2 + inner)), "y0": y0, "y1": y1}


# ----------------------------------------------------------------------------- shared helpers
def srgb_f(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def box_smooth(a: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k | 1)
    return np.convolve(a, np.ones(k) / k, mode="same")


def step_from_profile(prof: np.ndarray) -> dict:
    """Largest mean(below)-mean(above) step with sub-row refinement (port of the web app)."""
    h = len(prof); ks = max(3, round(h * 0.012) | 1); sm = box_smooth(prof, ks)
    cs = np.concatenate([[0.0], np.cumsum(sm)])
    min_seg = max(2, round(h * 0.03))
    i = np.arange(min_seg, h - min_seg + 1)
    score = (cs[h] - cs[i]) / (h - i) - cs[i] / i
    bi = int(i[np.argmax(score)])
    above, below = cs[bi] / bi, (cs[h] - cs[bi]) / (h - bi); mid = 0.5 * (above + below)
    lvl = float(bi)
    for j in range(max(1, bi - ks), min(h - 1, bi + ks) + 1):
        if (sm[j - 1] - mid) * (sm[j] - mid) <= 0 and sm[j] != sm[j - 1]:
            lvl = j - 1 + (mid - sm[j - 1]) / (sm[j] - sm[j - 1]); break
    return {"row": lvl, "above": float(above), "below": float(below), "contrast": float((below - above) / (below + above + 1e-6)), "prof": sm}


# ----------------------------------------------------------------------------- detectors
def diff_raw(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.linalg.norm(b - c, axis=2) / math.sqrt(3)


def down2(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def adaptive_pair(b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Port of the web app's Adaptive pipeline: 2x2 downsample, 3x3 blur, per-channel affine match,
    integer shift search (+-3 px at half res). Returns (B, C_aligned, info) at half resolution."""
    B, C = down2(b), down2(c)
    B = cv2.blur(B, (3, 3)); C = cv2.blur(C, (3, 3))
    for ch in range(3):
        mb, sb = B[..., ch].mean(), B[..., ch].std(); mc, sc = C[..., ch].mean(), C[..., ch].std()
        a = float(np.clip(sb / sc, 0.6, 1.6)) if sc > 1e-4 else 1.0
        C[..., ch] = C[..., ch] * a + (mb - a * mc)
    Lb = B @ np.array([0.299, 0.587, 0.114], np.float32); Lc = C @ np.array([0.299, 0.587, 0.114], np.float32)
    best, bdx, bdy = np.inf, 0, 0
    h, w = Lb.shape
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            ys, xs = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
            cost = np.abs(Lb[ys, xs][::2, ::2] - Lc[ys.start + dy:ys.stop + dy, xs.start + dx:xs.stop + dx][::2, ::2]).mean()
            if cost < best - 1e-9:
                best, bdx, bdy = cost, dx, dy
    Ca = np.zeros_like(C)
    ys, xs = slice(max(0, -bdy), min(h, h - bdy)), slice(max(0, -bdx), min(w, w - bdx))
    Ca[ys, xs] = C[ys.start + bdy:ys.stop + bdy, xs.start + bdx:xs.stop + bdx]
    return B, Ca, {"shift_half_px": [bdx, bdy]}


def adaptive_diff(B: np.ndarray, Ca: np.ndarray) -> tuple[np.ndarray, float]:
    d = np.linalg.norm(B - Ca, axis=2) / math.sqrt(3)
    row_mean = d.mean(axis=1)
    quiet = np.argsort(row_mean)[:max(3, round(len(row_mean) * 0.25))]
    q = d[quiet]
    T = max(0.008, float(q.mean() + 3 * q.std()))
    d = np.where(d > T, np.minimum(d, 0.6), 0.0)
    return d, T


def magnification_profile(B: np.ndarray, C: np.ndarray, n_scale: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Per row: horizontal magnification M (and flip sign) that best maps the baseline row onto the
    compare row about the ROI centre, by normalised cross-correlation over all colour channels.
    Returns (log|M| with sign folded in as separate array, ncc)."""
    h, w, _ = B.shape
    cx = (w - 1) / 2
    x = np.arange(w) - cx
    scales = np.exp(np.linspace(math.log(0.5), math.log(8.0), n_scale))
    logM = np.full(h, np.nan); ncc_best = np.zeros(h); flip = np.zeros(h)
    for sgn in (1.0, -1.0):
        for M in scales:
            xs = cx + sgn * x / M                                  # baseline column sampled for compare column x
            valid = (xs >= 0) & (xs <= w - 1)
            if valid.sum() < 0.5 * w:
                continue
            xi = np.clip(xs, 0, w - 1)
            i0 = np.floor(xi).astype(int); t = (xi - i0)[None, :, None]; i1 = np.minimum(i0 + 1, w - 1)
            Bs = B[:, i0, :] * (1 - t) + B[:, i1, :] * t            # h x w x 3 resampled baseline
            Cv = C[:, valid, :].reshape(h, -1); Bv = Bs[:, valid, :].reshape(h, -1)
            Cv = Cv - Cv.mean(axis=1, keepdims=True); Bv = Bv - Bv.mean(axis=1, keepdims=True)
            ncc = (Cv * Bv).sum(axis=1) / (np.linalg.norm(Cv, axis=1) * np.linalg.norm(Bv, axis=1) + 1e-6)
            better = ncc > ncc_best
            ncc_best[better] = ncc[better]; logM[better] = math.log(M); flip[better] = sgn
    return logM, ncc_best, flip


def optimal_partition(X: np.ndarray, penalty: float, max_cp: int = 3) -> list[int]:
    """Exact O(K n^2) optimal partitioning of a multivariate sequence into piecewise-constant
    segments (SSE cost), with a per-change-point penalty. Returns change-point indices."""
    n, d = X.shape
    cs = np.concatenate([np.zeros((1, d)), np.cumsum(X, axis=0)])
    cs2 = np.concatenate([np.zeros((1, d)), np.cumsum(X * X, axis=0)])

    def cost(a, b):                                                  # SSE of X[a:b]
        m = b - a
        s = cs[b] - cs[a]; s2 = cs2[b] - cs2[a]
        return float((s2 - s * s / m).sum())

    min_seg = max(4, n // 40)
    INF = float("inf")
    F = np.full((max_cp + 2, n + 1), INF); F[0, 0] = 0.0
    arg = np.zeros((max_cp + 2, n + 1), int)
    C = np.full((n + 1, n + 1), INF)
    for a in range(0, n):
        for b in range(a + min_seg, n + 1):
            C[a, b] = cost(a, b)
    for k in range(1, max_cp + 2):
        for b in range(k * min_seg, n + 1):
            cands = F[k - 1, :b] + C[:b, b]
            j = int(np.argmin(cands)); F[k, b] = cands[j]; arg[k, b] = j
    totals = [F[k, n] + penalty * (k - 1) for k in range(1, max_cp + 2)]
    k = int(np.argmin(totals)) + 1
    cps, b = [], n
    for kk in range(k, 1, -1):
        b = int(arg[kk, b]); cps.append(b)
    return sorted(cps)


def median_filter1d(a: np.ndarray, w: int) -> np.ndarray:
    w = max(3, w | 1); pad = w // 2
    ap = np.pad(a, pad, mode="edge")
    return np.array([np.median(ap[i:i + w]) for i in range(len(a))])


def step_filter(f: np.ndarray, w: int) -> np.ndarray:
    """mean(f[r:r+w]) - mean(f[r-w:r]) for every row r (0 near the ends)."""
    cs = np.concatenate([[0.0], np.cumsum(f)]); n = len(f); S = np.zeros(n)
    r = np.arange(w, n - w)
    S[r] = (cs[r + w] - cs[r]) / w - (cs[r] - cs[r - w]) / w
    return S


def cp_v2(B: np.ndarray, Ca: np.ndarray, d: np.ndarray, max_levels: int = 3, tau: float = 4.0) -> dict:
    """Multi-level detector: robust step-edge peaks in [thresholded dE row mean, log M(r)].
    Both profiles are median filtered over ~5 % of the ROI height (removes the periodic dips at
    checker row boundaries), a two-sided step filter at scale w = 6 % is applied, each response is
    normalised by the robust scale of the filtered profile's first differences, and peaks above
    `tau` with separation >= 2w are returned (strongest first)."""
    n = d.shape[0]
    prof = d.mean(axis=1)
    logM, ncc, flip = magnification_profile(B, Ca)
    lm = np.where(ncc > 0.35, logM, np.nan)
    has_M = np.isfinite(lm).sum() > 0.3 * n
    if has_M:
        idx = np.arange(n); lm = np.interp(idx, idx[np.isfinite(lm)], lm[np.isfinite(lm)])
    else:
        lm = np.zeros(n)
    w_med = max(5, round(n * 0.05)); w = max(4, round(n * 0.06)); edge = max(w, round(n * 0.04))
    f1 = median_filter1d(prof, w_med); f2 = median_filter1d(lm, w_med)
    # primary level: the dE step detector (robust on every vessel)
    prim = step_from_profile(prof)["row"]
    cps, sig = [prim], [float("inf")]
    if not has_M:
        return {"cps": cps, "jumps": sig, "prof_dE": f1, "prof_logM": f2, "ncc": ncc, "flip": flip, "resp": np.zeros(n)}
    # secondary interfaces: magnification steps BELOW the primary level ...
    rng = float(np.percentile(f2, 95) - np.percentile(f2, 5))
    S = np.abs(step_filter(f2, w)) / max(rng, 0.15)
    S[: int(prim) + 2 * w] = 0; S[n - edge:] = 0
    # ... confirmed by a break in the horizontal edge structure of the compare image: the positions
    # of vertical pattern edges jump at a liquid|liquid interface (different magnification), but
    # drift smoothly where only the vessel shape changes, and are unchanged across checker row
    # boundaries (only the colours swap). g(r) = NCC of |d/dx lum| block means above vs below r.
    g = structure_continuity(Ca, k=max(3, round(n * 0.02)))
    liquid = slice(int(prim) + w, n - edge)
    g_ref = float(np.median(g[liquid])) if (n - edge) - (int(prim) + w) > 4 else 1.0
    resp = S / 0.12
    for r in np.argsort(-resp):
        if resp[r] < 1.0 or len(cps) >= max_levels:
            break
        if not all(abs(r - p) >= 2 * w for p in cps):
            continue
        lo, hi = max(edge, r - int(1.5 * w)), min(n - edge, r + int(1.5 * w))
        if hi <= lo:
            continue
        r_star = lo + int(np.argmin(g[lo:hi]))
        if g[r_star] < 0.75 * g_ref:
            cps.append(float(r_star)); sig.append(float(resp[r]))
    return {"cps": cps, "jumps": sig, "prof_dE": f1, "prof_logM": f2, "ncc": ncc, "flip": flip, "resp": resp, "g": g}


def structure_continuity(C: np.ndarray, k: int) -> np.ndarray:
    """g(r) = NCC between the mean |horizontal gradient| signature of rows [r-2k, r-k) and [r+k, r+2k)."""
    lum = 0.2126 * C[..., 0] + 0.7152 * C[..., 1] + 0.0722 * C[..., 2]
    gx = np.abs(np.diff(lum, axis=1))
    n = gx.shape[0]
    cs = np.concatenate([np.zeros((1, gx.shape[1])), np.cumsum(gx, axis=0)])
    g = np.ones(n)
    for r in range(2 * k, n - 2 * k):
        A = (cs[r - k] - cs[r - 2 * k]) / k; Bv = (cs[r + 2 * k] - cs[r + k]) / k
        A = A - A.mean(); Bv = Bv - Bv.mean()
        g[r] = float((A * Bv).sum() / (np.linalg.norm(A) * np.linalg.norm(Bv) + 1e-6))
    return g


# ----------------------------------------------------------------------------- main
def main():
    EVAL.mkdir(parents=True, exist_ok=True)
    m = load_json(TS_OUT / "manifest.json")
    W, H = m["resolution"]
    jobs = {j["name"]: j for j in m["jobs"]}
    rows, cases = [], []
    for j in m["jobs"]:
        if j["role"] != "compare":
            continue
        b = srgb_f(cv2.imread(str(TS_IMAGES / f"{j['baseline']}.png")))
        c = srgb_f(cv2.imread(str(TS_IMAGES / f"{j['name']}.png")))
        roi = vessel_roi(j, W, H)
        x0, x1, y0, y1 = roi["x0"], roi["x1"], roi["y0"], roi["y1"]
        rh = y1 - y0
        gts = [(l["top_row_front"], l["kind"], l["liquid"]) for l in j["gt"]["layers"]]
        pct = lambda row: (y1 - row) / rh * 100  # noqa: E731

        Bf, Cf = b[y0:y1, x0:x1], c[y0:y1, x0:x1]
        # 1) raw step
        d_raw = diff_raw(Bf, Cf); s_raw = step_from_profile(d_raw.mean(axis=1)); row_raw = y0 + s_raw["row"]
        # 2) adaptive step
        B2, C2, info = adaptive_pair(Bf, Cf); d_ad, T = adaptive_diff(B2, C2)
        s_ad = step_from_profile(d_ad.mean(axis=1)); row_ad = y0 + s_ad["row"] * 2
        # 3) multi-level change points
        v2 = cp_v2(B2, C2, d_ad)
        rows_v2 = [float(y0 + cp * 2) for cp in v2["cps"]]           # strongest first

        def nearest_err(est_rows, gt_row):
            if not est_rows:
                return None
            e = min(est_rows, key=lambda r: abs(r - gt_row)); return float(e - gt_row)

        rec = {"case": j["name"], "vessel": j["vessel"], "camera": j["camera"], "pattern": j["pattern"], "roi": roi,
               "roi_height_px": rh, "gt": [{"row": g[0], "pct": pct(g[0]), "kind": g[1], "liquid": g[2]} for g in gts],
               "step_raw": {"row": row_raw, "pct": pct(row_raw), "contrast": s_raw["contrast"]},
               "step_adapt": {"row": row_ad, "pct": pct(row_ad), "contrast": s_ad["contrast"], "thresh": T, "shift": info["shift_half_px"]},
               "cp_v2": {"rows": rows_v2, "pcts": [pct(r) for r in rows_v2], "jumps": v2["jumps"]}}
        top_gt = gts[-1][0]
        rec["err_px"] = {"step_raw_vs_top": row_raw - top_gt, "step_adapt_vs_top": row_ad - top_gt,
                         "cp_v2_vs_each_gt": [nearest_err(rows_v2, g[0]) for g in gts]}
        rec["err_pct"] = {k: (v / rh * 100 if isinstance(v, float) else [x / rh * 100 if x is not None else None for x in v]) for k, v in rec["err_px"].items()}
        rows.append(rec)
        cases.append({"case": j["name"], "baseline": j["baseline"], "compare": j["name"], "roi_frac": [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H],
                      "gt_rows_frac": [g[0] / H for g in gts], "gt_kinds": [g[1] for g in gts], "gt_liquids": [g[2] for g in gts],
                      "label": f"{j['vessel']} · {j['camera']} · {j['pattern']}" + (" · " + " + ".join(l for l, _ in j["layers"]) if len(j["layers"]) > 1 else ""),
                      "note": j.get("note", "")})
        print(f"{j['name']:34s} GT rows {[round(g[0]) for g in gts]} | raw {row_raw:.0f} ({rec['err_px']['step_raw_vs_top']:+.0f}px) "
              f"| adapt {row_ad:.0f} ({rec['err_px']['step_adapt_vs_top']:+.0f}px, c={s_ad['contrast']:.2f}) | v2 {[round(r) for r in rows_v2]} jumps {[round(x, 1) for x in v2['jumps']]}")

        # profile figure
        fig, axes = plt.subplots(1, 4, figsize=(15, 5), gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})
        crop = np.clip(c[y0:y1, x0:x1], 0, 1)
        axes[0].imshow(crop, extent=(x0, x1, y1, y0)); axes[0].set_title(j["name"], fontsize=8); axes[0].set_xticks([])
        yy2 = y0 + 2 * np.arange(len(v2["prof_dE"]))
        axes[1].plot(v2["prof_dE"], yy2, color="#2563eb", lw=1.2); axes[1].set_title("adaptive dE row mean", fontsize=8)
        axes[2].plot(v2["prof_logM"], yy2, color="#7c3aed", lw=1.2); axes[2].set_title("log horizontal magnification M(r)", fontsize=8)
        axes[3].plot(v2["resp"], yy2, color="#6b7280", lw=1.0); axes[3].axvline(1.0, color="#9ca3af", lw=0.8, ls=":")
        if "g" in v2:
            axes[3].plot(v2["g"] * max(1.0, float(np.nanmax(v2["resp"]))), yy2, color="#16a34a", lw=0.8)
        axes[3].set_title("M-step significance (grey, 1 = thr) / edge continuity g (green, scaled)", fontsize=7)
        for ax in axes[1:]:
            for g in gts:
                ax.axhline(g[0], color="#22c55e", lw=1.0, ls="--")
            for r in rows_v2:
                ax.axhline(r, color="#ff5a5a", lw=0.8, ls=":")
            ax.axhline(row_ad, color="#ffb454", lw=0.8)
            ax.set_ylim(y1, y0); ax.grid(alpha=.3)
        for g in gts:
            axes[0].axhline(g[0], color="#22c55e", lw=1.0, ls="--")
        fig.suptitle("green dashed = GT, amber = single-step (adaptive), red dotted = multi-level change points", fontsize=9)
        fig.savefig(EVAL / f"profiles_{j['name']}.png", dpi=100, bbox_inches="tight"); plt.close(fig)

    save_json({"resolution": [W, H], "rows": rows}, EVAL / "eval.json")
    save_json({"resolution": [W, H], "cases": cases}, TS_OUT / "cases.json")
    with open(EVAL / "eval.csv", "w", encoding="utf-8") as f:
        f.write("case,gt_rows,step_raw_err_px,step_adapt_err_px,adapt_contrast,cp_v2_rows,cp_v2_err_px_each_gt\n")
        for r in rows:
            f.write(f"{r['case']},{'/'.join(str(round(g['row'])) for g in r['gt'])},{r['err_px']['step_raw_vs_top']:.1f},{r['err_px']['step_adapt_vs_top']:.1f},"
                    f"{r['step_adapt']['contrast']:.2f},{'/'.join(str(round(x)) for x in r['cp_v2']['rows'])},{'/'.join('' if e is None else f'{e:.1f}' for e in r['err_px']['cp_v2_vs_each_gt'])}\n")
    print("->", EVAL)


if __name__ == "__main__":
    main()

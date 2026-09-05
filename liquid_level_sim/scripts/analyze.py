"""Extract ground truth from the GT passes, run a baseline liquid-level detector on the
beauty renders, and write per-sample metrics + overlay previews.

Ground truth per sample
  gt_row_near      lowest image row of the free-surface mask  = level line at the front wall
  gt_row_far       highest row of the free-surface mask       = level line at the back wall
  gt_row_analytic  same quantity from the analytic camera projection (pipeline self-check)
  container rows   rim / base rows from the container mask (for pixel -> fill-fraction mapping)

Baseline detector ("row-gradient")
  Inside the container ROI, accumulate |d/dy| of the luminance across the central columns,
  suppress the rim/base bands, and take the strongest horizontal edge as the level line.
  It is intentionally simple: the point is to have an end-to-end number to compare
  learned estimators against, not to be a good detector.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from common import MASK_DIR, OUT_DIR, PREVIEW_DIR, RENDER_DIR, SCENES_DIR, ensure_dirs, load_json, save_json


# ----------------------------------------------------------------------------- io
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_mask(p: Path, thresh: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Returns (coverage in [0,1], boolean mask)."""
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(p)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    cov = srgb_to_linear(img)
    return cov, cov >= thresh


def rows_cols(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


# ----------------------------------------------------------------------------- baseline
def _valid_rows(cont_box, n_rows: int, top_frac: float, bottom_frac: float) -> np.ndarray:
    """Rows inside the container ROI, excluding the rim ellipse band (top) and the base (bottom)."""
    top, bottom = cont_box[0], cont_box[1]
    h = bottom - top
    valid = np.zeros(n_rows, bool)
    valid[top + max(3, int(h * top_frac)): bottom - max(3, int(h * bottom_frac))] = True
    return valid


def _refine_peak(prof: np.ndarray, r: int, valid: np.ndarray) -> float:
    if 1 <= r < len(prof) - 1 and valid[r - 1] and valid[r + 1]:
        a, b, c = prof[r - 1], prof[r], prof[r + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            return float(r + 0.5 * (a - c) / denom)
    return float(r)


def baseline_row_gradient(beauty_bgr: np.ndarray, cont_box, top_frac: float = 0.18, bottom_frac: float = 0.08,
                          col_frac: float = 0.5) -> tuple[float, np.ndarray]:
    """Strongest horizontal luminance edge across the central columns of the container."""
    top, bottom, left, right = cont_box
    gray = cv2.cvtColor(beauty_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.5)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    w = right - left
    c0 = int(left + w * (0.5 - col_frac / 2))
    c1 = int(left + w * (0.5 + col_frac / 2)) + 1
    # median across columns: a real level line spans the whole width, refracted checker edges do not
    prof = np.median(np.abs(dy[:, c0:c1]), axis=1)
    valid = _valid_rows(cont_box, len(prof), top_frac, bottom_frac)
    prof_m = np.where(valid, prof, 0.0)
    r = int(prof_m.argmax())
    return _refine_peak(prof, r, valid), prof_m


def baseline_row_step(beauty_bgr: np.ndarray, cont_box, top_frac: float = 0.18, bottom_frac: float = 0.08,
                      col_frac: float = 0.5, win_frac: float = 0.06) -> tuple[float, np.ndarray]:
    """Largest step in the per-row mean colour (Lab) between a window above and below each row.
    Liquid tints / darkens the region below the level line, so the level is a colour step."""
    top, bottom, left, right = cont_box
    lab = cv2.cvtColor(beauty_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    w = right - left
    c0 = int(left + w * (0.5 - col_frac / 2))
    c1 = int(left + w * (0.5 + col_frac / 2)) + 1
    rows = lab[:, c0:c1, :].mean(axis=1)                       # (H, 3) per-row mean colour
    k = max(3, int((bottom - top) * win_frac))
    H = rows.shape[0]
    prof = np.zeros(H, np.float32)
    cs = np.cumsum(np.vstack([np.zeros((1, 3), np.float32), rows]), axis=0)
    for r in range(k, H - k):
        above = (cs[r] - cs[r - k]) / k
        below = (cs[r + k] - cs[r]) / k
        prof[r] = np.linalg.norm(below - above)
    valid = _valid_rows(cont_box, H, top_frac, bottom_frac)
    prof_m = np.where(valid, prof, 0.0)
    r = int(prof_m.argmax())
    return _refine_peak(prof, r, valid), prof_m


BASELINES = {"gradient": baseline_row_gradient, "step": baseline_row_step}
PRIMARY = "step"


# ----------------------------------------------------------------------------- main
def analyze_sample(meta: dict, make_preview: bool = True) -> dict:
    sid = meta["sid"]
    beauty = cv2.imread(str(RENDER_DIR / f"{sid}_beauty.png"), cv2.IMREAD_COLOR)
    if beauty is None:
        raise FileNotFoundError(f"missing beauty render for {sid}")
    H, W = beauty.shape[:2]
    liq_cov, liq = load_mask(MASK_DIR / f"{sid}_liquid.png")
    surf_cov, surf = load_mask(MASK_DIR / f"{sid}_surface.png")
    con_cov, con = load_mask(MASK_DIR / f"{sid}_container.png")

    res = {"sid": sid, "container": meta["container"], "liquid": meta["liquid"], "fill": meta["fill"],
           "liquid_height_cm": meta["liquid_height"], "fov": meta["fov"], "eye_z": meta["eye"][2],
           "width": W, "height": H}

    cb = rows_cols(con)
    lb = rows_cols(liq)
    sb = rows_cols(surf)
    res["container_box"] = cb
    res["liquid_box"] = lb
    res["surface_box"] = sb
    res["liquid_area_frac"] = float(liq.mean())
    res["surface_area_frac"] = float(surf.mean())

    # ---- ground truth (rendered) with sub-pixel refinement of the near edge via coverage
    if sb is not None:
        near = sb[1]
        # coverage-weighted sub-pixel position of the lowest surface row
        col = surf_cov[near, :]
        cov_next = surf_cov[near + 1, :] if near + 1 < H else np.zeros(W)
        m = col > 0.02
        sub = float((col[m].mean() + cov_next[m].mean())) if m.any() else 0.5
        res["gt_row_near"] = near + sub          # bottom of pixel `near` is near+1; partial coverage shifts it up
        res["gt_row_far"] = float(sb[0])
    else:
        res["gt_row_near"] = res["gt_row_far"] = None
    res["gt_row_analytic"] = meta["gt_row_analytic"]
    res["gt_row_far_analytic"] = meta["gt_row_far_analytic"]
    if res["gt_row_near"] is not None:
        res["gt_analytic_vs_render_px"] = res["gt_row_near"] - meta["gt_row_analytic"]
    else:
        res["gt_analytic_vs_render_px"] = None

    # ---- baseline detectors (primary = "step"; "gradient" kept for comparison)
    prof = None
    if cb is not None and res["gt_row_near"] is not None:
        top, bottom = cb[0], cb[1]
        span = max(bottom - top, 1)
        # pixel -> fill fraction using the visible container extent (perspective ignored on purpose:
        # this is what a naive 2D method would do; the GT fraction uses the same mapping so the
        # comparison isolates the detector). cm/px: inner height spans (rim - base) rows approx.
        res["cm_per_px"] = meta["height"] / span
        res["frac_gt_2d"] = (bottom - res["gt_row_near"]) / span
        for name, fn in BASELINES.items():
            est_row, p = fn(beauty, cb)
            res[f"est_row_{name}"] = est_row
            res[f"err_px_{name}"] = est_row - res["gt_row_near"]
            if name == PRIMARY:
                prof = p
        res["est_row"] = res[f"est_row_{PRIMARY}"]
        res["err_px"] = res[f"err_px_{PRIMARY}"]
        res["frac_est_2d"] = (bottom - res["est_row"]) / span
        res["err_frac_2d"] = res["frac_est_2d"] - res["frac_gt_2d"]
        res["err_mm"] = res["err_px"] * res["cm_per_px"] * 10.0
    else:
        for k in ("est_row", "err_px", "frac_est_2d", "frac_gt_2d", "err_frac_2d", "cm_per_px", "err_mm",
                  *[f"est_row_{n}" for n in BASELINES], *[f"err_px_{n}" for n in BASELINES]):
            res[k] = None

    if make_preview:
        write_preview(sid, beauty, liq, surf, con, res, prof)
    return res


def write_preview(sid, beauty, liq, surf, con, res, prof):
    H, W = beauty.shape[:2]
    over = beauty.copy()
    # liquid mask contour (cyan), surface mask (magenta tint)
    cnts, _ = cv2.findContours(liq.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, cnts, -1, (255, 255, 0), 1)
    tint = over.copy()
    tint[surf] = (0.5 * tint[surf] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    over = tint
    if res.get("gt_row_near") is not None:
        y = int(round(res["gt_row_near"]))
        cv2.line(over, (0, y), (W - 1, y), (0, 255, 0), 1)
    if res.get("est_row") is not None:
        y = int(round(res["est_row"]))
        cv2.line(over, (0, y), (W - 1, y), (0, 0, 255), 1)
    cb = res.get("container_box")
    if cb:
        cv2.rectangle(over, (cb[2], cb[0]), (cb[3], cb[1]), (0, 200, 255), 1)
    txt = f"{sid} fill={res['fill']:.2f}"
    if res.get("err_px") is not None:
        txt += f" err={res['err_px']:+.1f}px"
    cv2.putText(over, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    def m2bgr(m):
        return cv2.cvtColor((m * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    tiles = [beauty, over, m2bgr(liq), m2bgr(surf) // 2 + m2bgr(con) // 2]
    strip = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(PREVIEW_DIR / f"{sid}_overlay.png"), over)
    cv2.imwrite(str(PREVIEW_DIR / f"{sid}_strip.png"), strip)
    if prof is not None:
        # gradient profile plot as a small image next to the render
        pm = prof / (prof.max() + 1e-9)
        plot = np.full((H, 120, 3), 30, np.uint8)
        for y in range(H):
            cv2.line(plot, (0, y), (int(pm[y] * 118), y), (200, 200, 200), 1)
        if res.get("gt_row_near") is not None:
            cv2.line(plot, (0, int(res["gt_row_near"])), (119, int(res["gt_row_near"])), (0, 255, 0), 1)
        cv2.imwrite(str(PREVIEW_DIR / f"{sid}_profile.png"), np.concatenate([over, plot], axis=1))


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("err_px") is not None]
    e = np.array([r["err_px"] for r in ok]) if ok else np.zeros(0)
    ef = np.array([r["err_frac_2d"] for r in ok]) if ok else np.zeros(0)
    em = np.array([r["err_mm"] for r in ok]) if ok else np.zeros(0)
    g = np.array([r["gt_analytic_vs_render_px"] for r in rows if r.get("gt_analytic_vs_render_px") is not None])
    s = {"n": len(rows), "n_evaluated": len(ok)}
    if len(ok):
        s.update({
            "mae_px": float(np.abs(e).mean()), "median_abs_px": float(np.median(np.abs(e))),
            "rmse_px": float(math.sqrt((e ** 2).mean())), "bias_px": float(e.mean()),
            "mae_mm": float(np.abs(em).mean()), "mae_frac": float(np.abs(ef).mean()),
            "within_3px": float((np.abs(e) <= 3).mean()), "within_2pct": float((np.abs(ef) <= 0.02).mean()),
        })
        for name in BASELINES:
            en = np.array([r[f"err_px_{name}"] for r in ok])
            s[f"detector_{name}_mae_px"] = float(np.abs(en).mean())
            s[f"detector_{name}_within_3px"] = float((np.abs(en) <= 3).mean())
        for c in sorted({r["container"] for r in ok}):
            ec = np.array([r["err_px"] for r in ok if r["container"] == c])
            s[f"mae_px_{c}"] = float(np.abs(ec).mean())
        for l in sorted({r["liquid"] for r in ok}):
            el = np.array([r["err_px"] for r in ok if r["liquid"] == l])
            s[f"mae_px_{l}"] = float(np.abs(el).mean())
    if len(g):
        s["gt_check_mean_abs_px"] = float(np.abs(g).mean())
        s["gt_check_max_abs_px"] = float(np.abs(g).max())
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()
    ensure_dirs()
    metas = load_json(SCENES_DIR / "dataset.json")
    if a.only:
        keep = set(a.only.split(","))
        metas = [m for m in metas if m["sid"] in keep]
    rows = []
    for m in metas:
        try:
            r = analyze_sample(m, not a.no_preview)
        except FileNotFoundError as ex:
            print(f"skip {m['sid']}: {ex}")
            continue
        rows.append(r)
        gt = r["gt_row_near"]
        print(f"{r['sid']} {r['container']:8s} {r['liquid']:12s} fill={r['fill']:.3f} "
              f"gt_row={gt if gt is None else round(gt, 1)} analytic={r['gt_row_analytic']:.1f} "
              f"est={None if r['est_row'] is None else round(r['est_row'], 1)} "
              f"err_px={None if r['err_px'] is None else round(r['err_px'], 2)}")
    summ = summarize(rows)
    save_json({"summary": summ, "samples": rows}, OUT_DIR / "metrics.json")
    if rows:
        keys = [k for k in rows[0] if not isinstance(rows[0][k], (list, tuple, dict))]
        with open(OUT_DIR / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
    print("summary:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()})


if __name__ == "__main__":
    main()

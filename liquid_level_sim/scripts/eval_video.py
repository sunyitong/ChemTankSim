"""Evaluate the level detector on the pouring sequence: every frame against the baseline still.

  python scripts/eval_video.py [outputs/pour/video_meta.json]
Writes eval_video.json and level_vs_time.png next to the given video_meta.json (outputs/pour/<seq>/).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from level_detect import detect  # noqa: E402
from mvp_common import PROJ_ROOT, load_json  # noqa: E402


def track(frames, dt, gate=3.0, alpha=0.6):
    """Causal multi-hypothesis tracker (same rules as the web app). Each frame offers the detector's
    surface hypotheses in level percent: `dip` (contact line), `first` (located warp jump), `top`
    (identity end = far rim when the surface is seen from above) and `raw` (the detector's own choice).
    The far-rim band thickness b = top - dip is learned (EMA) whenever both exist, so `top - b` is a
    hypothesis even in frames without a contact line. The hypothesis nearest a constant-velocity
    prediction is blended in (alpha) when within `gate`; otherwise the track coasts, and after three
    misses re-acquires at the median of the last three raw values. A hypothesis that has not moved for
    two frames while the level should have is a stuck luminance feature and is ignored when another
    hypothesis is available."""
    out, lvl, vel, bad, recent, band, prev = [], None, 0.0, 0, [], None, {}
    for f in frames:
        raw, dip, first, top = f.get("raw"), f.get("dip"), f.get("first"), f.get("top")
        if top is not None and dip is not None:
            band = (top - dip) if band is None else 0.8 * band + 0.2 * (top - dip)
        if raw is None:
            out.append(lvl); continue
        recent = (recent + [raw])[-3:]
        if lvl is None:
            lvl = dip if dip is not None else raw; out.append(lvl); prev = {"dip": dip, "first": first, "top": top}; continue
        pred = lvl + vel * dt
        hyps = {}
        if dip is not None: hyps["dip"] = dip
        if first is not None: hyps["first"] = first
        if top is not None and band is not None: hyps["top"] = top - band
        hyps["raw"] = raw
        moving = abs(vel * dt) > 0.15
        live = {k: v for k, v in hyps.items() if not (moving and prev.get(k) is not None and abs(v - prev[k]) < 0.25)}
        cand = live if live else hyps
        k_best = min(cand, key=lambda k: abs(cand[k] - pred)); m = cand[k_best]; e = m - pred
        if abs(e) <= gate:
            new = pred + alpha * e; vel = 0.7 * vel + 0.3 * (new - lvl) / dt; lvl = new; bad = 0
        else:
            bad += 1
            if bad >= 3:
                lvl, vel, bad = float(np.median(recent)), 0.0, 0
            else:
                lvl, vel = pred, vel * 0.9
        prev = {"dip": dip, "first": first, "top": top}
        out.append(lvl)
    return out


def srgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def srgb_array(bgr: np.ndarray) -> np.ndarray:
    """Same conversion as srgb() for an already decoded BGR frame."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main(meta_path: Path):
    meta = load_json(meta_path); out_dir = meta_path.parent; frames_dir = out_dir / "frames"
    W, H = meta["resolution"]; fx, fy, fw, fh = meta["roi_frac"]
    x0, y0, x1, y1 = round(fx * W), round(fy * H), round(fx * W + fw * W), round(fy * H + fh * H)
    base = srgb(frames_dir / f"{meta['baseline']}.png")[y0:y1, x0:x1]
    decoded = None
    if not (frames_dir / "frame_000.png").exists():           # PNG frames are not tracked: decode the (intra-only) MP4
        cap, decoded = cv2.VideoCapture(str(out_dir / meta["video"])), []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            decoded.append(fr)
        print(f"  frames decoded from {meta['video']}: {len(decoded)} (PNG frames absent)")
    rows = []
    for f in meta["frames"]:
        img = (srgb(frames_dir / f"frame_{f['i']:03d}.png") if decoded is None else srgb_array(decoded[f["i"]]))[y0:y1, x0:x1]
        det = detect(base, img, horizon_row=(H / 2 - y0) / 2)
        gt_pct = [(y1 - g * H) / (y1 - y0) * 100 for g in f["gt_rows_frac"]]
        est_pct = [(y1 - (y0 + l["row"] * 2)) / (y1 - y0) * 100 for l in det.levels]
        pct = lambda r: (y1 - (y0 + r * 2)) / (y1 - y0) * 100 if r is not None else None
        rows.append({"i": f["i"], "t": f["t"], "fill": f["fill"], "gt_pct": gt_pct[0] if gt_pct else None,
                     "est_pct": est_pct[0] if est_pct else None, "n_levels": len(est_pct), "mode": det.mode,
                     "err_pt": (est_pct[0] - gt_pct[0]) if est_pct and gt_pct else None,
                     "hyp": {"raw": est_pct[0] if est_pct else None, "dip": pct(det.dip), "first": pct(det.first_jump), "top": pct(det.surface)}})
    tracked = track([r["hyp"] for r in rows], 1.0 / meta["fps"])
    for r, tval in zip(rows, tracked):
        r["tracked_pct"] = tval; r["tracked_err_pt"] = (tval - r["gt_pct"]) if (tval is not None and r["gt_pct"] is not None) else None
    terrs = np.array([r["tracked_err_pt"] for r in rows if r["tracked_err_pt"] is not None])
    errs = np.array([r["err_pt"] for r in rows if r["err_pt"] is not None])
    found = sum(r["est_pct"] is not None for r in rows)
    extra = sum(r["n_levels"] > 1 for r in rows)
    summary = {"frames": len(rows), "found": found, "frames_with_extra_boundary": extra,
               "median_err_pt": float(np.median(errs)) if len(errs) else None, "p90_abs_err_pt": float(np.percentile(np.abs(errs), 90)) if len(errs) else None,
               "max_abs_err_pt": float(np.max(np.abs(errs))) if len(errs) else None,
               "frames_within_3pt": int(np.sum(np.abs(errs) <= 3)), "frames_within_6pt": int(np.sum(np.abs(errs) <= 6)),
               "tracked_median_err_pt": float(np.median(terrs)) if len(terrs) else None, "tracked_p90_abs_err_pt": float(np.percentile(np.abs(terrs), 90)) if len(terrs) else None,
               "tracked_max_abs_err_pt": float(np.max(np.abs(terrs))) if len(terrs) else None, "tracked_within_3pt": int(np.sum(np.abs(terrs) <= 3))}
    json.dump({"summary": summary, "frames": rows}, open(out_dir / "eval_video.json", "w"), indent=1)
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, [r["gt_pct"] for r in rows], "--", color="#22c55e", lw=1.4, label="ground truth (front rim)")
    ax.plot(t, [r["est_pct"] if r["est_pct"] is not None else np.nan for r in rows], ".", color="#f6b545", ms=4, alpha=.6, label="per-frame detection")
    ax.plot(t, [r["tracked_pct"] if r["tracked_pct"] is not None else np.nan for r in rows], color="#f6b545", lw=1.6, label="temporal track")
    ax.set_xlabel("time (s)"); ax.set_ylabel("level (% of ROI height)"); ax.grid(alpha=.3); ax.legend(loc="lower right")
    ax.set_title(f"{meta['case']}, {meta['fps']} fps · per frame: median {summary['median_err_pt']:+.2f} pt, max {summary['max_abs_err_pt']:.1f} pt · "
                 f"tracked: median {summary['tracked_median_err_pt']:+.2f} pt, max {summary['tracked_max_abs_err_pt']:.1f} pt", fontsize=9)
    fig.tight_layout(); fig.savefig(out_dir / "level_vs_time.png", dpi=110)
    print(json.dumps(summary, indent=1))
    for r in rows[::10]:
        print(f"  t={r['t']:.2f}s fill={r['fill']:.3f} GT={r['gt_pct']:.1f}% est={r['est_pct'] if r['est_pct'] is None else round(r['est_pct'], 1)}% err={r['err_pt'] if r['err_pt'] is None else round(r['err_pt'], 2)} mode={r['mode']} n={r['n_levels']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ_ROOT / "outputs" / "pour" / "video_meta.json")

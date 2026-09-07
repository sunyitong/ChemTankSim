"""Evaluate the level detector on the pouring sequence: every frame against the baseline still.

  python scripts/eval_video.py [outputs/pour/video_meta.json]
Writes outputs/pour/eval_video.json and outputs/pour/level_vs_time.png.
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


def srgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def main(meta_path: Path):
    meta = load_json(meta_path); out_dir = meta_path.parent; frames_dir = out_dir / "frames"
    W, H = meta["resolution"]; fx, fy, fw, fh = meta["roi_frac"]
    x0, y0, x1, y1 = round(fx * W), round(fy * H), round(fx * W + fw * W), round(fy * H + fh * H)
    base = srgb(frames_dir / f"{meta['baseline']}.png")[y0:y1, x0:x1]
    rows = []
    for f in meta["frames"]:
        img = srgb(frames_dir / f"frame_{f['i']:03d}.png")[y0:y1, x0:x1]
        det = detect(base, img)
        gt_pct = [(y1 - g * H) / (y1 - y0) * 100 for g in f["gt_rows_frac"]]
        est_pct = [(y1 - (y0 + l["row"] * 2)) / (y1 - y0) * 100 for l in det.levels]
        rows.append({"i": f["i"], "t": f["t"], "fill": f["fill"], "gt_pct": gt_pct[0] if gt_pct else None,
                     "est_pct": est_pct[0] if est_pct else None, "n_levels": len(est_pct), "mode": det.mode,
                     "err_pt": (est_pct[0] - gt_pct[0]) if est_pct and gt_pct else None})
    errs = np.array([r["err_pt"] for r in rows if r["err_pt"] is not None])
    found = sum(r["est_pct"] is not None for r in rows)
    extra = sum(r["n_levels"] > 1 for r in rows)
    summary = {"frames": len(rows), "found": found, "frames_with_extra_boundary": extra,
               "median_err_pt": float(np.median(errs)) if len(errs) else None, "p90_abs_err_pt": float(np.percentile(np.abs(errs), 90)) if len(errs) else None,
               "max_abs_err_pt": float(np.max(np.abs(errs))) if len(errs) else None,
               "frames_within_3pt": int(np.sum(np.abs(errs) <= 3)), "frames_within_6pt": int(np.sum(np.abs(errs) <= 6))}
    json.dump({"summary": summary, "frames": rows}, open(out_dir / "eval_video.json", "w"), indent=1)
    t = [r["t"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(t, [r["gt_pct"] for r in rows], "--", color="#22c55e", lw=1.4, label="ground truth (front rim)")
    ax.plot(t, [r["est_pct"] if r["est_pct"] is not None else np.nan for r in rows], color="#f6b545", lw=1.4, label="detected level")
    ax.set_xlabel("time (s)"); ax.set_ylabel("level (% of ROI height)"); ax.grid(alpha=.3); ax.legend(loc="lower right")
    ax.set_title(f"pouring sequence, {meta['fps']} fps · median error {summary['median_err_pt']:+.2f} pt · max |err| {summary['max_abs_err_pt']:.2f} pt", fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "level_vs_time.png", dpi=110)
    print(json.dumps(summary, indent=1))
    for r in rows[::10]:
        print(f"  t={r['t']:.2f}s fill={r['fill']:.3f} GT={r['gt_pct']:.1f}% est={r['est_pct'] if r['est_pct'] is None else round(r['est_pct'], 1)}% err={r['err_pt'] if r['err_pt'] is None else round(r['err_pt'], 2)} mode={r['mode']} n={r['n_levels']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJ_ROOT / "outputs" / "pour" / "video_meta.json")

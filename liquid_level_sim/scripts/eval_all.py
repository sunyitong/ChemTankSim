"""Evaluate level detectors on every bundled case (test set A/B and limescale S/V) with ground truth.

  python scripts/eval_all.py outputs/testset/cases.json outputs/scale/cases.json
Writes outputs/eval_all/{eval_all.json, eval_all.md, profiles_<case>.png}.
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

from level_detect import detect, step_from_profile  # noqa: E402
from mvp_common import PROJ_ROOT, load_json  # noqa: E402
from testset_scenes import TS_IMAGES  # noqa: E402

OUT = PROJ_ROOT / "outputs" / "eval_all"


def srgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def rescale(img, scale):
    if scale == 1:
        return img
    H, W = img.shape[:2]
    return cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)


def main(case_files, scales=(1.0,)):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cf in case_files:
        data = load_json(Path(cf)); img_dir = Path(data.get("image_dir", TS_IMAGES))
        for c in data["cases"]:
          b0 = srgb(img_dir / f"{c['baseline']}.png"); k0 = srgb(img_dir / f"{c['compare']}.png")
          for scale in scales:
            b, k = rescale(b0, scale), rescale(k0, scale)
            H, W = b.shape[:2]
            fx, fy, fw, fh = c["roi_frac"]                                   # same rounding as the web app
            x0, y0 = int(round(fx * W)), int(round(fy * H))
            x1, y1 = int(round(fx * W + fw * W)), int(round(fy * H + fh * H))
            rh = y1 - y0
            gt_rows = [f * H for f in c["gt_rows_frac"]]                    # working rows (full res)
            gt_pct = [(y1 - g) / rh * 100 for g in gt_rows]
            det = detect(b[y0:y1, x0:x1], k[y0:y1, x0:x1], horizon_row=(H / 2 - y0) / 2)
            est_rows = [float(y0 + l["row"] * 2) for l in det.levels]
            est_pct = [float((y1 - r) / rh * 100) for r in est_rows]
            # photometric baseline detector for comparison (old primary)
            dE_row = y0 + step_from_profile(det.dE) * 2 if det.dE is not None else float("nan")
            top_gt = min(gt_rows)                                             # topmost GT row = free surface
            rec = {"case": c["case"], "scale": scale, "label": c["label"], "mode": det.mode, "frac_pattern": float(det.frac_pattern),
                   "gt_pct": [float(g) for g in gt_pct], "gt_kinds": c["gt_kinds"], "est_pct": est_pct,
                   "est_z": [float(l["z"]) for l in det.levels], "est_jump": [float(l["jump"]) for l in det.levels],
                   "old_dE_top_err_pct": float((y1 - dE_row) / rh * 100 - max(gt_pct)) if not np.isnan(dE_row) else None}
            # matching: each GT gets the nearest estimate; estimates not within 3 % of any GT = false positives
            matched = []
            for g in gt_pct:
                if est_pct:
                    e = min(est_pct, key=lambda v: abs(v - g)); matched.append(float(e - g) if abs(e - g) <= 6 else None)
                else:
                    matched.append(None)
            rec["err_pct_per_gt"] = matched
            rec["false_positives"] = [e for e in est_pct if all(abs(e - g) > 3 for g in gt_pct)]
            rec["top_err_pct"] = (max(est_pct) - max(gt_pct)) if est_pct else None
            rows.append(rec)
            print(f"{c['case']:38s} x{scale:<4.2f} {det.mode:18s} pat={det.frac_pattern:.2f} GT%={[round(g, 1) for g in gt_pct]} est%={[round(e, 1) for e in est_pct]} "
                  f"z={[round(z, 1) for z in rec['est_z']]} err={[None if e is None else round(e, 1) for e in matched]} FP={[round(e, 1) for e in rec['false_positives']]} old_dE_err={None if rec['old_dE_top_err_pct'] is None else round(rec['old_dE_top_err_pct'], 1)}")
            if scale != 1:
                continue
            # figure
            fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1, 1]})
            axes[0].imshow(np.clip(k[y0:y1, x0:x1], 0, 1), extent=(x0, x1, y1, y0)); axes[0].set_xticks([]); axes[0].set_title(c["case"], fontsize=8)
            yy = y0 + 2 * np.arange(len(det.dE))
            axes[1].plot(det.dE, yy, color="#2563eb", lw=1.1); axes[1].set_title("photometric dE (old primary)", fontsize=8)
            if det.logM is not None:
                axes[2].plot(det.logM, yy, color="#7c3aed", lw=1.1); axes[2].set_title("log horizontal magnification (warp)", fontsize=8)
                axes[2].fill_betweenx(yy, det.logM.min(), det.logM.max(), where=det.ncc < 0.35, color="#e5e7eb", alpha=0.6, label="no pattern")
            if det.zscore is not None:
                axes[3].plot(det.zscore, yy, color="#374151", lw=1.1); axes[3].axvline(4, color="#9ca3af", ls=":", lw=0.8); axes[3].set_title("jump significance z", fontsize=8)
            for ax in axes:
                for g in gt_rows:
                    ax.axhline(g, color="#22c55e", lw=1.0, ls="--")
                for r in est_rows:
                    ax.axhline(r, color="#ff5a5a", lw=0.9, ls=":")
                if ax is not axes[0]:
                    ax.set_ylim(y1, y0); ax.grid(alpha=.3)
            fig.suptitle("green dashed = GT, red dotted = detected (warp jumps)", fontsize=9)
            fig.savefig(OUT / f"profiles_{c['case']}.png", dpi=95, bbox_inches="tight"); plt.close(fig)

    with open(OUT / "eval_all.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    # summary table (scale 1) + robustness across scales
    lines = ["| case | mode | GT (%) | detected (%) | err per GT (pt) | false pos | old dE err |", "|---|---|---|---|---|---|---|"]
    for r in [r for r in rows if r["scale"] == 1]:
        old = r["old_dE_top_err_pct"]
        old_txt = "—" if old is None else f"{old:+.1f}"
        gt_txt = " / ".join(f"{g:.1f}" for g in r["gt_pct"])
        est_txt = " / ".join(f"{e:.1f}" for e in r["est_pct"]) or "—"
        err_txt = " / ".join("miss" if e is None else f"{e:+.1f}" for e in r["err_pct_per_gt"])
        lines.append(f"| {r['case']} | {r['mode']} | {gt_txt} | {est_txt} | {err_txt} | {len(r['false_positives'])} | {old_txt} |")
    n_gt = sum(len(r["gt_pct"]) for r in rows); n_hit = sum(sum(e is not None for e in r["err_pct_per_gt"]) for r in rows)
    n_fp = sum(len(r["false_positives"]) for r in rows)
    errs = [abs(e) for r in rows for e in r["err_pct_per_gt"] if e is not None]
    summary = f"\n**scales {sorted(set(r['scale'] for r in rows))}: {n_hit}/{n_gt} interfaces found within 6 pt, {n_fp} false positives, median |err| {np.median(errs):.2f} pt, max |err| {max(errs):.2f} pt**\n"
    if len(scales) > 1:
        lines += ["", "| case | " + " | ".join(f"x{s:.2f}" for s in scales) + " | top spread (pt) |", "|---|" + "---|" * (len(scales) + 1)]
        for case in dict.fromkeys(r["case"] for r in rows):
            rs = [next(r for r in rows if r["case"] == case and r["scale"] == s) for s in scales]
            cells = []
            for r in rs:
                e = " / ".join("miss" if v is None else f"{v:+.1f}" for v in r["err_pct_per_gt"])
                cells.append(e + (f" (+{len(r['false_positives'])}FP)" if r["false_positives"] else ""))
            tops = [r["top_err_pct"] for r in rs if r["top_err_pct"] is not None]
            spread = f"{max(tops) - min(tops):.1f}" if tops else "—"
            lines.append(f"| {case} | " + " | ".join(cells) + f" | {spread} |")
    (OUT / "eval_all.md").write_text("\n".join(lines) + summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    args = sys.argv[1:]; scales = (1.0,)
    if args and args[0].startswith("--scales="):
        scales = tuple(float(v) for v in args.pop(0).split("=", 1)[1].split(","))
    main(args or [str(PROJ_ROOT / "outputs" / "testset" / "cases.json")], scales)

"""Bundle test-set image pairs (and video sequences) into the web app.

Reads one or more cases.json files (written by testset_eval.py, scale_run.py, inserts_run.py) and
cases_video.json (pour_video.py). Still pairs are downsized to 675 px wide and JPEG-encoded; a video
case embeds its MP4 as a base64 data URL plus the baseline still at the video's native size. Writes a
`window.CASES` / `window.CASE_IMAGES` / `window.CASE_VIDEOS` snippet that level_diff_bench.html includes.

  python scripts/webapp_bundle.py <out.js> <cases.json> [<cases2.json> ...] [<cases_video.json>]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2

from mvp_common import load_json
from testset_scenes import TS_IMAGES, TS_OUT

W_OUT, Q = 675, 80   # 675 px wide = 0.75 of the 900x1200 renders (detector is scale-robust; see eval_all spread table); q80 keeps the page under 16 MB


def encode(png: Path, width: int | None = W_OUT, q: int = Q) -> str:
    im = cv2.imread(str(png))
    if im is None:
        raise FileNotFoundError(png)
    if width and im.shape[1] != width:
        h = round(im.shape[0] * width / im.shape[1])
        im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def main(out_js: Path, case_files: list[Path]):
    images, videos, out = {}, {}, []
    for cf in case_files:
        data = load_json(cf)
        img_dir = Path(data.get("image_dir", TS_IMAGES))
        for c in data["cases"]:
            if c.get("kind") == "video":
                vdir = Path(c["image_dir"]) if c.get("image_dir") else img_dir          # per-sequence frame directory
                if c["baseline"] not in images:
                    images[c["baseline"]] = encode(vdir / f"{c['baseline']}.png", width=None, q=90)
                mp4 = Path(c["video_path"]) if c.get("video_path") else cf.parent / c["video"]
                videos[c["case"]] = "data:video/mp4;base64," + base64.b64encode(mp4.read_bytes()).decode("ascii")
                out.append({"id": c["case"].split("_")[0], "case": c["case"], "kind": "video", "label": c["label"], "note": c["note"],
                            "base": c["baseline"], "video": c["case"], "fps": c["fps"], "seconds": c["seconds"], "frames": c["n_frames"],
                            "roi": c["roi_frac"], "gtFrames": [f["gt_rows_frac"] for f in c["frames"]] if any(f["gt_rows_frac"] for f in c["frames"]) else None,
                            "layers": c.get("layers", "single")})
                continue
            for key in ("baseline", "compare"):
                if c[key] not in images:
                    images[c[key]] = encode(img_dir / f"{c[key]}.png")
            out.append({"id": c["case"].split("_")[0], "case": c["case"], "label": c["label"], "note": c["note"],
                        "base": c["baseline"], "cmp": c["compare"], "roi": c["roi_frac"],
                        "gt": [{"frac": f, "kind": k, "liquid": l} for f, k, l in zip(c["gt_rows_frac"], c["gt_kinds"], c["gt_liquids"])],
                        "layers": "multi" if len(c["gt_rows_frac"]) >= 2 else "single"})   # presets the app's single / multi-level mode
    js = ("window.CASES = " + json.dumps(out, ensure_ascii=False) + ";\nwindow.CASE_IMAGES = " + json.dumps(images)
          + ";\nwindow.CASE_VIDEOS = " + json.dumps(videos) + ";\n")
    out_js.write_text(js, encoding="utf-8")
    print(f"{len(out)} cases, {len(images)} images, {len(videos)} videos, {len(js) / 1e6:.2f} MB -> {out_js}")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    out = args[0] if args else TS_OUT / "cases_embed.js"
    files = args[1:] if len(args) > 1 else [TS_OUT / "cases.json"]
    main(out, files)

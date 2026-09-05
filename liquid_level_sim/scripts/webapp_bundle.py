"""Bundle test-set image pairs into the web app.

Reads one or more cases.json files (written by testset_eval.py and scale_run.py), downsizes the
post-processed PNGs to 600 x 800, JPEG-encodes them and writes a `window.CASES` / `window.CASE_IMAGES`
JavaScript snippet that level_diff_bench.html includes inline. Shared baselines are embedded once.

  python scripts/webapp_bundle.py <out.js> <cases.json> [<cases2.json> ...]
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2

from mvp_common import load_json
from testset_scenes import TS_IMAGES, TS_OUT

W_OUT, Q = 600, 86


def encode(png: Path) -> str:
    im = cv2.imread(str(png))
    if im is None:
        raise FileNotFoundError(png)
    h = round(im.shape[0] * W_OUT / im.shape[1])
    im = cv2.resize(im, (W_OUT, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, Q])
    assert ok
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def main(out_js: Path, case_files: list[Path]):
    images, out = {}, []
    for cf in case_files:
        data = load_json(cf)
        img_dir = Path(data.get("image_dir", TS_IMAGES))
        for c in data["cases"]:
            for key in ("baseline", "compare"):
                if c[key] not in images:
                    images[c[key]] = encode(img_dir / f"{c[key]}.png")
            out.append({"id": c["case"].split("_")[0], "case": c["case"], "label": c["label"], "note": c["note"],
                        "base": c["baseline"], "cmp": c["compare"], "roi": c["roi_frac"],
                        "gt": [{"frac": f, "kind": k, "liquid": l} for f, k, l in zip(c["gt_rows_frac"], c["gt_kinds"], c["gt_liquids"])]})
    js = "window.CASES = " + json.dumps(out, ensure_ascii=False) + ";\nwindow.CASE_IMAGES = " + json.dumps(images) + ";\n"
    out_js.write_text(js, encoding="utf-8")
    print(f"{len(out)} cases, {len(images)} images, {len(js) / 1e6:.2f} MB -> {out_js}")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    out = args[0] if args else TS_OUT / "cases_embed.js"
    files = args[1:] if len(args) > 1 else [TS_OUT / "cases.json"]
    main(out, files)

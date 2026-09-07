"""Laboratory environment light: download a CC0 indoor HDRI (Poly Haven), convert it to pbrt-v4's equal-area
square map (imgtool makeequiarea), measure its mean radiance so scenes can set a physically sensible scale,
and write a tone-mapped preview.

  python scripts/env_map.py --id <polyhaven_id> [--res 2k]
Outputs: outputs/env/<id>_<res>.hdr (source), <id>_equiarea.exr (pbrt), <id>_preview.png, env_meta.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common import IMGTOOL_EXE
from mvp_common import PROJ_ROOT

OUT = PROJ_ROOT / "outputs" / "env"


def write_pfm(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    with open(path, "wb") as f:
        f.write(b"PF\n%d %d\n-1.0\n" % (w, h))
        f.write(np.ascontiguousarray(rgb[::-1].astype(np.float32)).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--res", default="2k"); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    src = OUT / f"{a.id}_{a.res}.hdr"
    if not src.exists() or a.force:
        url = f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/{a.res}/{a.id}_{a.res}.hdr"
        print("downloading", url)
        urllib.request.urlretrieve(url, src)
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)             # Radiance HDR -> float32 BGR, linear
    if img is None:
        raise SystemExit(f"cannot read {src}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    # solid-angle-weighted mean radiance of the lat-long map (rows near the poles cover less sphere)
    theta = (np.arange(rgb.shape[0]) + 0.5) / rgb.shape[0] * np.pi
    wgt = np.sin(theta)[:, None] * np.ones_like(lum)
    mean_lum = float((lum * wgt).sum() / wgt.sum())
    p99 = float(np.percentile(lum, 99.9))
    pfm = OUT / f"{a.id}_latlong.pfm"; write_pfm(pfm, rgb)
    ea = OUT / f"{a.id}_equiarea.exr"
    subprocess.run([str(IMGTOOL_EXE), "makeequiarea", str(pfm), "--outfile", str(ea)], check=True)
    prev = np.clip((rgb / (mean_lum * 4.0)) ** (1 / 2.2), 0, 1)
    prev = cv2.resize(prev, (1024, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f"{a.id}_preview.png"), cv2.cvtColor((prev * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    meta = {"id": a.id, "res": a.res, "source": f"https://polyhaven.com/a/{a.id} (CC0)", "mean_luminance": mean_lum,
            "p99_9_luminance": p99, "equiarea": ea.name, "size": list(rgb.shape[:2])}
    json.dump(meta, open(OUT / f"{a.id}_meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()

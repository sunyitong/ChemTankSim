"""One-shot driver for the Phase-0 MVP: patterns -> scenes -> render -> compare/report.

  python scripts/mvp_run.py --smoke                 # 270x360, 32 spp, geometry/exposure check
  python scripts/mvp_run.py                         # 1080x1440, 1024 spp (defaults in Setup)
  python scripts/mvp_run.py --spp 2048 --force      # re-render everything at higher quality
  python scripts/mvp_run.py --skip-render           # only rebuild comparisons + report
"""
from __future__ import annotations

import argparse
import time

from mvp_common import FILLS, MVP_OUT, PATTERNS, SCENE_DIR, Setup, load_json
from mvp_patterns import generate_all
from mvp_render import render_manifest
from mvp_scene import write_scenes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--spp", type=int, default=None)
    ap.add_argument("--res", default=None, help="WxH")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--fills", default=",".join(str(f) for f in FILLS))
    ap.add_argument("--no-meniscus", action="store_true")
    ap.add_argument("--nthreads", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-render", action="store_true")
    ap.add_argument("--skip-compare", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cfg = Setup(meniscus=not a.no_meniscus)
    spp, res, tag = a.spp, None, a.tag
    if a.res:
        res = tuple(int(v) for v in a.res.lower().split("x"))
    if a.smoke:
        spp, res, tag = spp or 32, res or (270, 360), tag or "_smoke"
    patterns = [p for p in a.patterns.split(",") if p]
    fills = [float(f) for f in a.fills.split(",") if f]

    t0 = time.time()
    mp = MVP_OUT / f"manifest{tag}.json"
    if a.skip_render and mp.exists():
        # analysis-only rerun: use the manifest of the renders on disk, don't touch scenes/patterns
        manifest = load_json(mp)
        print(f"=== using existing renders: {mp} ({len(manifest['jobs'])} jobs, spp={manifest['jobs'][0]['spp']}) ===")
    else:
        print("=== patterns ===")
        generate_all(cfg)
        print("=== scenes ===")
        manifest = write_scenes(cfg, patterns, fills, spp=spp, res=res, tag=tag)
        print(f"  {len(manifest['jobs'])} jobs, spp={manifest['jobs'][0]['spp']}, "
              f"res={manifest['jobs'][0]['width']}x{manifest['jobs'][0]['height']}, mesh={manifest['mesh']}")
        if not a.skip_render:
            print("=== render ===")
            manifest = render_manifest(manifest, a.nthreads, None, a.force, out_path=mp)
    if not a.skip_compare:
        print("=== compare ===")
        from mvp_compare import compare_and_report
        compare_and_report(manifest)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

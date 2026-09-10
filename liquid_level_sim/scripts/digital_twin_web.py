"""Add only F4/F5 to the existing Video group, preserving the original app code and layout.

Run after digital_twin_approved.py. The existing 37 cases and the detector are left intact.
Assets remain local; this script neither publishes nor modifies the real-photo cases.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from common import PROJ_ROOT

OUT=PROJ_ROOT/"outputs"/"digital_twin_approved"
WEB=PROJ_ROOT/"webapp"
DOCS=PROJ_ROOT.parent/"docs"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=OUT)
    args=parser.parse_args()
    source=args.source.resolve()
    for sid in ("D1","D2"):
        if not (source/sid/"video_meta.json").exists():
            raise SystemExit(f"Both completed video bundles are required: {source/sid}")
    html=(DOCS/"index.html").read_text(encoding="utf-8")
    def extract(name):
        return json.loads(re.search(r"window\."+name+r" = (.*?);\n",html).group(1))
    cases=extract("CASES");images=extract("CASE_IMAGES");videos=extract("CASE_VIDEOS")
    cases=[c for c in cases if c["id"] not in ("F4","F5")]
    added=[]
    for sid,case_id in [("D1","F4"),("D2","F5")]:
        c=json.loads((source/sid/"video_meta.json").read_text(encoding="utf-8"))
        src=source/sid/"frames"/(c["baseline"]+".png")
        movie=source/sid/c["video"]
        for destination in [WEB/"twin_assets",DOCS/"twin_assets"]:
            destination.mkdir(exist_ok=True)
            shutil.copy2(src,destination/src.name)
            shutil.copy2(movie,destination/c["video"])
        images[c["baseline"]]="twin_assets/"+src.name+"?v="+hashlib.sha256(src.read_bytes()).hexdigest()[:12]
        videos[c["case"]]="twin_assets/"+c["video"]+"?v="+hashlib.sha256(movie.read_bytes()).hexdigest()[:12]
        cases.append(dict(id=case_id,case=c["case"],kind="video",label=c["label"],note="",
                          base=c["baseline"],video=c["case"],fps=c["fps"],seconds=c["seconds"],frames=c["n_frames"],
                          roi=c["roi_frac"],gtFrames=[f["gt_rows_frac"] for f in c["frames"]],layers="single"))
        added.append(case_id)
    for name,value in [("CASES",cases),("CASE_IMAGES",images),("CASE_VIDEOS",videos)]:
        replacement="window."+name+" = "+json.dumps(value,ensure_ascii=False)+";\n"
        html=re.sub(r"window\."+name+r" = .*?;\n",lambda _:replacement,html,count=1)
    for path in [WEB/"level_diff_bench.html",DOCS/"index.html"]:path.write_text(html,encoding="utf-8")
    print(f"{len(cases)} cases; refreshed {added} from {source}. Layout and algorithm unchanged.")


if __name__=="__main__":main()

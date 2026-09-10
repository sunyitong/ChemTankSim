"""Verify the legacy layout/detector/tracker and the two decoded video bundles.

Video transport may change to fix loading/seeking; the original detection code may not.
The reference is pinned to the user's pre-twin version, so checks remain useful after commits.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import cv2

from common import PROJ_ROOT


def extract(html,name):
    return json.loads(re.search(r"window\."+name+r" = (.*?);\n",html).group(1))


def skeleton(html):
    for name in ["CASES","CASE_IMAGES","CASE_VIDEOS"]:
        html=re.sub(r"window\."+name+r" = .*?;\n",name+"\n",html,count=1)
    return html.replace("\r\n","\n")


def preserved_code(html):
    html=skeleton(html)
    start=html.index("let vmode = null;")
    end=html.index("function drawTrace()",start)
    tracker=html[html.index("const TRACK_GATE",start):html.index("async function showFrame",start)]
    return html[:start]+"VIDEO_TRANSPORT\n"+tracker+html[end:]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=PROJ_ROOT/"outputs"/"digital_twin_approved")
    args=parser.parse_args()
    root=PROJ_ROOT.parent;file="docs/index.html"
    old=subprocess.check_output(["git","-c","safe.directory="+root.as_posix(),"show","3f8221f:"+file],cwd=root).decode()
    new=(root/file).read_text(encoding="utf-8")
    assert preserved_code(new)==preserved_code(old),"Original markup, controls, detector or tracker changed"
    assert new==(PROJ_ROOT/"webapp"/"level_diff_bench.html").read_text(encoding="utf-8")
    old_cases=extract(old,"CASES");new_cases=extract(new,"CASES")
    assert [c for c in new_cases if c["id"] not in ["F4","F5"]]==old_cases
    assert len(new_cases)==len(old_cases)+2
    for name in ["CASE_IMAGES","CASE_VIDEOS"]:
        before,after=extract(old,name),extract(new,name)
        assert all(after[k]==v for k,v in before.items()),"An existing real or synthetic asset changed"
    report=dict(reference_commit="3f8221f",preserved_code_sha256=hashlib.sha256(preserved_code(old).encode()).hexdigest(),
                original_layout_detector_tracker_unchanged=True,legacy_cases_unchanged=len(old_cases),videos={})
    for sid,case_id in [("D1","F4"),("D2","F5")]:
        directory=args.source/sid
        meta=json.loads((directory/"video_meta.json").read_text(encoding="utf-8"))
        case=next(c for c in new_cases if c["id"]==case_id)
        cap=cv2.VideoCapture(str(directory/meta["video"]))
        fps=cap.get(cv2.CAP_PROP_FPS);count=0;hashes=set()
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert list(frame.shape[1::-1])==[720,540]
            hashes.add(hashlib.sha256(frame.tobytes()).hexdigest());count+=1
        cap.release()
        assert fps==25 and count==meta["n_frames"]==case["frames"]
        assert len(hashes)==count,"Duplicate instead of independently rendered frames"
        gt=[r["gt_rows_frac"][0] for r in meta["frames"]]
        assert all(a>b for a,b in zip(gt,gt[1:])),"Filling level must rise"
        assert len(case["gtFrames"])==count
        for key,variable in [("base","CASE_IMAGES"),("video","CASE_VIDEOS")]:
            url=extract(new,variable)[case[key]]
            asset=url.split("?",1)[0]
            assert (root/"docs"/asset).is_file() and (PROJ_ROOT/"webapp"/asset).is_file()
            assert (root/"docs"/asset).read_bytes()==(PROJ_ROOT/"webapp"/asset).read_bytes()
            expected=directory/"frames"/(meta["baseline"]+".png") if key=="base" else directory/meta["video"]
            payload=expected.read_bytes()
            assert (root/"docs"/asset).read_bytes()==payload,"Published asset differs from the rendered bundle"
            if "?v=" in url:
                assert url.rsplit("?v=",1)[1]==hashlib.sha256(payload).hexdigest()[:12],"Stale cache version"
        report["videos"][case_id]=dict(fps=fps,frames=count,unique_frames=len(hashes),resolution=[720,540],spp=meta["spp"],gt_rows_monotonic=True)
    out=args.source/"D2"/"bundle_verification.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

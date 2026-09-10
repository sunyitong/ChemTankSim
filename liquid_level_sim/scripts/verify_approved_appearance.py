"""Check the sequence midpoint against the user-approved still's exact scene and textures."""
import hashlib
import json
import re

from common import PROJ_ROOT


def normalized(text):
    text=text.replace("digital_twin_review","digital_twin_approved")
    return re.sub(r'("integer pixelsamples" \[)\d+(\])',r'\1SAMPLES\2',text)


def main():
    report={"scenes":{},"textures":{}}
    for sid in ["D1","D2"]:
        old=PROJ_ROOT/"scenes"/"digital_twin_review"/f"{sid}_051.pbrt"
        new=PROJ_ROOT/"scenes"/"digital_twin_approved"/f"{sid}_051.pbrt"
        if not new.exists():continue
        same=normalized(old.read_text())==normalized(new.read_text())
        report["scenes"][sid]={"same_except_samples_and_output_paths":same}
        assert same,f"{sid} deviates from approved scene"
    base=PROJ_ROOT/"outputs"/"digital_twin_review"/"textures"
    current=PROJ_ROOT/"outputs"/"digital_twin_approved"/"textures"
    for path in current.glob("*.png"):
        reference=base/path.name
        assert reference.exists(),f"Unexpected texture {path.name}"
        same=hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256(reference.read_bytes()).digest()
        report["textures"][path.name]=same
        assert same,f"Changed texture {path.name}"
    output=PROJ_ROOT/"outputs"/"digital_twin_approved"/"approval_consistency.json"
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

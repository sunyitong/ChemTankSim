"""Render the user-approved heavy-scale twins, with the exact review materials and lighting.

The first playback-fix push precedes this batch. Default: 64 spp, 720x540, 25 fps,
101 independently rendered frames per clip. Cache/resume uses the scene and full recipe.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import digital_twin as twin
import digital_twin_review as review
from common import PROJ_ROOT, save_json

OUT = PROJ_ROOT / "outputs" / "digital_twin_approved"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",choices=["D1","D2"])
    parser.add_argument("--spp",type=int,default=64)
    parser.add_argument("--threads",type=int,default=28)
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    config=json.loads(twin.CONFIG.read_text())
    approved=json.loads(review.CONFIG.read_text())
    config["render"].update(spp=args.spp,threads=args.threads)
    config["twins"]["D2"]["label"]="Digital twin 2 / P5 colour mosaic backlight"
    config["approved_appearance"]=approved
    config["appearance_approval"]="User confirmed heavy ring-scale/frosted-glass stills on 2026-09-10"
    config["recipe_sha256"]=hashlib.sha256(Path(review.__file__).read_bytes()+Path(twin.__file__).read_bytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    twin.OUT=review.OUT=OUT
    twin.SCENES=review.SCENES=PROJ_ROOT/"scenes"/"digital_twin_approved"
    twin.maps=lambda c: review.deposit_maps(c,approved)
    twin.materials=review.material_factory(approved)
    twin.backlight=review.board_factory(approved)
    twin.fixtures=review.fixture_factory(approved)
    original_scene=twin.scene
    def approved_scene(*a,**kw):
        path,gt=original_scene(*a,**kw)
        review.finish_scene(path,approved)
        return path,gt
    twin.scene=approved_scene
    twin.sensor=lambda png,index,c: review.camera_finish(png,approved,monochrome=png.parent.parent.name=="D1")
    save_json(config,OUT/"render_recipe.json")
    # This PBRT build can hit a rare spherical-triangle light-sampling NaN.
    # Retry only that specific numerical failure with a fresh random seed; the
    # scene, material recipe, sample count and all cached completed frames stay intact.
    run=subprocess.run
    def render_with_retry(command,*a,**kw):
        result=run(command,*a,**kw)
        if not isinstance(command,list) or Path(str(command[0])).name.lower()!="pbrt.exe":
            return result
        original_seed=int(command[command.index("--seed")+1])
        for attempt in range(1,4):
            if not result.returncode or "Check failed: !IsNaN(cosBp)" not in (result.stderr or ""):
                break
            seed=original_seed+attempt*100003
            record=OUT/"sampling_retries.json"
            entries=json.loads(record.read_text()) if record.exists() else []
            entries.append({"scene":command[-1],"retry":attempt,"seed":seed,"reason":"PBRT spherical-triangle sampling NaN"})
            save_json(entries,record)
            (OUT/f"{Path(command[-1]).stem}_retry{attempt}_log.txt").write_text(result.stderr or "")
            print(f"Retry {command[-1]}: numerical sampling failure, seed {seed}",flush=True)
            retry=list(command); retry[retry.index("--seed")+1]=str(seed)
            result=run(retry,*a,**kw)
        return result
    subprocess.run=render_with_retry
    twin.render_sequences(config,force=args.force,only=args.only,spp=args.spp)


if __name__=="__main__":
    main()

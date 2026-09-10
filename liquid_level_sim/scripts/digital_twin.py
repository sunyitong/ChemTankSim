"""Basler reactor twin: physical PBRT scenes and filling sequences.

Run from any directory: python digital_twin.py --preview / --render.
All geometry/materials are generated in 3-D; no photograph is composited into a render.
Centimetres are nominal until dimensions are measured. Source TIFF cadence is unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

from common import PROJ_ROOT, PBRT_EXE, IMGTOOL_EXE, save_json
from mvp_common import Setup
from testset_shapes import Vessel
from testset_scenes import (Combo, header, materials, revolved_body, liquids_block,
                            layer_heights, inserts_block)
from scale_model import write_maps
from pour_video import encode

OUT = PROJ_ROOT / "outputs" / "digital_twin"
SCENES = PROJ_ROOT / "scenes" / "digital_twin"
CONFIG = PROJ_ROOT / "configs" / "digital_twin.json"


@dataclass
class ReactorCamera(Setup):
    cam_x: float = 0.62

    @property
    def eye(self):
        return self.cam_x, -self.cam_dist, self.cam_z

    @property
    def look(self):
        return self.cam_x, 0.0, self.cam_z


def vessel_for(config):
    g = config["geometry"]
    def profile(z):
        q = max(0.0, (g["shoulder_z"]-z)/g["bowl_depth"])
        return max(0.85, g["outer_radius"]*math.sqrt(max(0.0, 1-q*q)))
    return Vessel("basler_reactor", profile, H=g["rim_z"], t=g["wall"], tb=g["inner_floor_z"],
                  label="Basler round-bottom reactor (nominal dimensions)")


def maps(config):
    tex = OUT / "textures"; tex.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["texture_seed"])
    h,w = 1536,2048
    yy,xx = np.mgrid[:h,:w]; z=(1-yy/h)*config["geometry"]["rim_z"]
    coarse = cv2.resize(rng.random((45,65)).astype(np.float32),(w,h),interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.random((h,w)).astype(np.float32),(0,0),.55)
    lower = np.clip((12-z)/8,0,1)
    grains = np.clip((fine-.55)*18,0,1)*(0.65+1.1*lower)
    grains *= .45+coarse
    tau = .025 + .045*coarse + grains
    # Historical water lines and diffuse mineral films, fixed in world coordinates.
    for level, strength, width in config["deposits"]["rings"]:
        wave=.07*np.sin(xx/48)+.10*np.sin(xx/147)
        tau += strength*np.exp(-((z-level-wave)/width)**2)*(.35+coarse)
    for _ in range(65):
        x0=int(rng.integers(w)); z0=rng.uniform(4,17)
        dx=np.minimum(abs(xx-x0),w-abs(xx-x0))
        tau += rng.uniform(.02,.16)*np.exp(-(dx/rng.uniform(2,7))**2)*np.exp(-((z-z0)/rng.uniform(.6,2.5))**2)
    info=write_maps(tau.astype(np.float32),tex,"reactor",iron=.02)
    # Sparse absorbing dirt on the OUTSIDE; inner mineral layer uses dry/wet index matching.
    amount=np.clip(grains*.85,0,.85)
    cv2.imwrite(str(tex/"dirt.png"),(amount*255).astype(np.uint8))
    # Printed translucent ruler: material transmission texture, attached in 3-D.
    ruler=np.full((1400,220),225,np.uint8)
    cv2.line(ruler,(65,35),(65,1370),15,3)
    for i in range(141):
        y=35+i*9
        if y>=1390:break
        cv2.line(ruler,(65,y),(205 if i%20==0 else 153 if i%10==0 else 112,y),2,4)
        if i%20==0:cv2.putText(ruler,str(i//20),(10,y+10),0,.85,18,2)
    cv2.imwrite(str(tex/"ruler.png"),ruler)
    return {**info,"kind":"scale"}


def sphere(x,y,z,r,material="black"):
    return f'AttributeBegin\nNamedMaterial "{material}"\nTranslate {x} {y} {z}\nShape "sphere" "float radius" [{r}]\nAttributeEnd'


def backlight(sid):
    """White light for twin 1; real dictionary ArUco markers on twin 2's emissive board."""
    lighting='"rgb L" [.92 .92 .92]'
    if sid=="D2":
        file=OUT/"textures"/"aruco_4x4_1000.png"
        if not file.exists():
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
            board=np.full((33*64,38*64),255,np.uint8)
            for row in range(33):
                for col in range(38):
                    code=(row*38+col)%1000
                    # PBRT camera maps world +x leftward. Reverse the board in world-space,
                    # keeping each marker readable (not mirrored) in the direct camera view.
                    marker=cv2.aruco.generateImageMarker(dictionary,code,48)
                    board[row*64+8:row*64+56,col*64+8:col*64+56]=marker
            cv2.imwrite(str(file),cv2.cvtColor(cv2.flip(board,1),cv2.COLOR_GRAY2BGR))
        lighting=f'"string filename" ["{file.as_posix()}"] "float scale" [.92]'
    return ('AttributeBegin\nAreaLightSource "diffuse" '+lighting+' "bool twosided" [true]\n'
            'Material "diffuse" "rgb reflectance" [.8 .8 .8]\n'
            'Shape "trianglemesh" "point3 P" [-38 20 -20 38 20 -20 38 20 45 -38 20 45] '
            '"point2 uv" [0 0 1 0 1 1 0 1] "integer indices" [0 1 2 0 2 3]\nAttributeEnd')


def cylinder(x,y,z0,z1,r,material="black",axis="z"):
    transform=f"Translate {x} {y} {z0}"
    if axis=="x":transform+="\nRotate 90 0 1 0"
    return f'AttributeBegin\nNamedMaterial "{material}"\n{transform}\nShape "cylinder" "float radius" [{r}] "float zmin" [0] "float zmax" [{z1-z0}]\nShape "disk" "float radius" [{r}] "float height" [{z1-z0}]\nAttributeEnd'


def fixtures(vessel,lay,ruler,config):
    parts=['MakeNamedMaterial "black" "string type" "diffuse" "rgb reflectance" [.012 .012 .012]',
           'MakeNamedMaterial "metal" "string type" "conductor" "float roughness" [.24]',
           'MakeNamedMaterial "lid" "string type" "diffuse" "rgb reflectance" [.065 .065 .065]']
    # Stirring shaft, rear return pipe and elbow, lid, clamps and side connectors.
    parts.append(inserts_block(vessel,lay,[dict(type="rod",material="black",x=0,y=0,radius=.65,z0_frac=-.04,above_rim=3)],bool(lay)))
    parts += [cylinder(-4,-.5,2,9.8,1.28),sphere(-4,-.5,9.8,1.28),
              cylinder(-9,-.5,9.8,14.8,1.15,axis="x"),
              cylinder(0,0,15.35,18.5,7.6,"lid"),
              cylinder(7.35,-6.6,16,20,1.15),cylinder(-7.5,-6.6,17,20,1.15),
              cylinder(-9,-5,6,8.5,.3,axis="x"),cylinder(-9,-5,12,14.5,.3,axis="x")]
    # A thin lid flange catches the overhead room reflection.
    parts.append(cylinder(0,0,18.25,18.5,7.85,"metal"))
    if ruler:
        file=(OUT/"textures"/"ruler.png").as_posix()
        parts += [f'Texture "rulerT" "spectrum" "imagemap" "string filename" ["{file}"]',
                  'AttributeBegin\nMaterial "diffusetransmission" "rgb reflectance" [.045 .045 .045] "texture transmittance" "rulerT"',
                  'Shape "trianglemesh" "point3 P" [5.7 -7.65 2.8 3.25 -7.65 2.8 3.65 -7.65 17.1 6.1 -7.65 17.1] "point2 uv" [0 0 1 0 1 1 0 1] "integer indices" [0 1 2 0 2 3]', 'AttributeEnd']
    return "\n".join(parts)


def scene(config,tex,sid,level,index,width,height,spp):
    vessel=vessel_for(config)
    cfg=ReactorCamera(width=width,height_px=height,**config["camera"])
    lay=layer_heights(vessel,[("water",(level-vessel.tb)/vessel.inner_height)]) if level is not None else []
    # Local roughness and wetting, with actual liquid | glass interfaces and an absorbing volume.
    for l in lay:l["h_m"]=.09
    combo=Combo(f"{sid}_{index}",vessel.name,"normal","white","scale",0,[])
    mat=materials(combo,tex,lay).replace("../../outputs/testset/textures",(OUT/"textures").as_posix())
    mat=mat.replace('MakeNamedMaterial "wall_out" "string type" "dielectric" "float eta" [1.5]', 'MakeNamedMaterial "wall_clear" "string type" "dielectric" "float eta" [1.5] "float roughness" [.006] "bool remaproughness" [false]')
    dirt=(OUT/"textures"/"dirt.png").as_posix()
    mat+=f'\nTexture "outerDirt" "float" "imagemap" "string filename" ["{dirt}"] "string encoding" "linear"\nMakeNamedMaterial "grime" "string type" "diffuse" "rgb reflectance" [.11 .10 .09]\nMakeNamedMaterial "wall_out" "string type" "mix" "string materials" ["wall_clear" "grime"] "texture amount" "outerDirt"'
    # Large white backlight with mild falloff; additional room light sets dark hardware reflectance.
    liquid_text=liquids_block(vessel,lay,vessel.r_in,SCENES,combo.name)
    if lay:
        # Small analytic surface ripple; fades to zero at the meniscus, retaining its exact seam.
        import re
        for filename in re.findall(r'"string filename" \["([^"]+\.ply)"\]',liquid_text):
            mesh=SCENES/filename;raw=mesh.read_bytes();end=raw.index(b'end_header\n')+len(b'end_header\n')
            count=int(re.search(rb'element vertex (\d+)',raw[:end]).group(1))
            vertices=np.frombuffer(raw[end:end+count*32],dtype='<f4').copy().reshape(count,8)
            x,y=vertices[:,0],vertices[:,1];rad=np.sqrt(x*x+y*y);R=vessel.r_in(level)
            fade=np.maximum(0,1-rad/R)**2
            phase=index*.39
            vertices[:,2]+=.025*fade*np.sin(2*x+phase)*np.cos(1.7*y-phase)
            dx=.05*fade*np.cos(2*x+phase)*np.cos(1.7*y-phase)
            dy=-.0425*fade*np.sin(2*x+phase)*np.sin(1.7*y-phase)
            vertices[:,3]-=dx;vertices[:,4]-=dy
            vertices[:,3:6]/=np.linalg.norm(vertices[:,3:6],axis=1)[:,None]
            mesh.write_bytes(raw[:end]+vertices.tobytes()+raw[end+count*32:])
    parts=[header(cfg,"frame.exr",spp,width,height),"WorldBegin",
           'LightSource "infinite" "rgb L" [.035 .035 .035]',
           backlight(sid),
           mat,revolved_body(vessel,lay,SCENES,True),liquid_text,
           fixtures(vessel,lay,sid=="D1",config)]
    if lay:
        rng=np.random.default_rng(913+index)
        # Submerged pumping: small entrained air bubbles, no fictitious top-down pour.
        for k in range(10):
            z=level-.08-rng.uniform(0,.45);x=rng.uniform(-2.5,3)
            if z<vessel.tb+.2:continue
            parts += [f'AttributeBegin\nMediumInterface "" "liq1"\nMaterial "dielectric" "float eta" [1.333]\nReverseOrientation\nTranslate {x} -1.8 {z}\nShape "sphere" "float radius" [{rng.uniform(.035,.085)}]\nAttributeEnd']
    path=SCENES/f"{sid}_{index:03d}.pbrt"
    path.write_text("\n".join(parts)+"\n",encoding="utf-8")
    gt=[] if level is None else [float(cfg.project(np.array([[0,-vessel.r_in(level),level]]))[0,1]/height)]
    return path,gt


def sensor(png,index,config):
    a=cv2.imread(str(png)).astype(np.float32)/255
    a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    a=cv2.GaussianBlur(a,(0,0),config["sensor"]["blur_px"])
    yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]
    field=1-.12*((xx/a.shape[1]-.50)**2+(yy/a.shape[0]-.45)**2)
    rng=np.random.default_rng(450+index)
    a=a*field*(1+.007*math.sin(index*.42))
    a+=rng.normal(0,config["sensor"]["noise_sigma"],a.shape)
    cv2.imwrite(str(png),np.clip(a*255,0,255).astype(np.uint8))


def render_sequences(config,preview=False,force=False,only=None,spp=None):
    SCENES.mkdir(parents=True,exist_ok=True);tex=maps(config)
    w,h=(720,540) if not preview else (480,360)
    n=3 if preview else config["render"]["frames"]
    spp=spp or (16 if preview else config["render"]["spp"])
    cases=[]
    for sid,spec in config["twins"].items():
        if only and sid!=only:continue
        target=OUT/(sid+"_preview" if preview else sid);frames=target/"frames";renders=target/"renders"
        for p in (frames,renders):p.mkdir(parents=True,exist_ok=True)
        rows=[];base=f"{sid}_empty"
        levels=[None]+list(np.linspace(*spec["level_z_range"],n))
        for j,z in enumerate(levels):
            path,gt=scene(config,tex,sid,z,j,w,h,spp)
            name=base if j==0 else f"frame_{j-1:03d}"
            png=frames/f"{name}.png";exr=renders/f"{name}.exr"
            signature=hashlib.sha256(path.read_bytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()
            sig=png.with_suffix(".sha256")
            if force or not png.exists() or not sig.exists() or sig.read_text()!=signature:
                import time
                t=time.monotonic()
                cmd=[str(PBRT_EXE),"--quiet","--nthreads",str(config["render"]["threads"]),"--seed",str(j),"--outfile",str(exr),path.name]
                result=subprocess.run(cmd,cwd=SCENES,capture_output=True,text=True)
                (target/"render_log.txt").write_text(result.stdout+result.stderr,encoding="utf-8")
                if result.returncode:raise RuntimeError(result.stderr+result.stdout)
                subprocess.run([str(IMGTOOL_EXE),"convert","--outfile",str(png),str(exr)],check=True,capture_output=True)
                sensor(png,j,config);sig.write_text(signature)
                print(f"{sid} {j}/{n}: {time.monotonic()-t:.1f}s",flush=True)
            if j:rows.append(dict(i=j-1,t=(j-1)/config["playback_fps"],level_z_cm=float(z),gt_rows_frac=gt))
        picks=[frames/f"{base}.png"]+[frames/f"frame_{i:03d}.png" for i in np.linspace(0,n-1,5).astype(int)]
        tiles=[cv2.resize(cv2.imread(str(p)),(360,270)) for p in picks]
        cv2.imwrite(str(target/"contact_sheet.png"),np.concatenate([np.concatenate(tiles[:3],1),np.concatenate(tiles[3:],1)],0))
        video=f"{sid}_filling.mp4";encode(frames,config["playback_fps"],target/video,crf=19)
        meta=dict(case=f"{sid}_reactor_twin",kind="video",baseline=base,label=spec["label"],
                  note="PBRT-v4 volumetric path tracing. Matched crop, rounded glass, deposits, shaft, elbow and optional ruler. Dimensions nominal; kinematic level ramp with bubbles, not CFD.",
                  image_dir=str(frames),video_path=str(target/video),video=video,fps=config["playback_fps"],
                  seconds=(n-1)/config["playback_fps"],n_frames=n,resolution=[w,h],spp=spp,
                  roi_frac=config["roi_frac"],frames=rows,reference_kind="rendered empty vessel",source_kind="synthetic")
        save_json(meta,target/"video_meta.json");cases.append(meta)
    if not preview:
        cases=[json.loads((OUT/s/"video_meta.json").read_text(encoding="utf-8")) for s in config["twins"] if (OUT/s/"video_meta.json").exists()]
        save_json(dict(cases=cases),OUT/"cases_twin.json")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preview",action="store_true")
    p.add_argument("--render",action="store_true");p.add_argument("--force",action="store_true")
    p.add_argument("--only",choices=["D1","D2"]);p.add_argument("--spp",type=int)
    a=p.parse_args();config=json.loads(CONFIG.read_text(encoding="utf-8"));OUT.mkdir(exist_ok=True)
    if a.preview or a.render:render_sequences(config,a.preview,a.force,a.only,a.spp)


if __name__=="__main__":main()

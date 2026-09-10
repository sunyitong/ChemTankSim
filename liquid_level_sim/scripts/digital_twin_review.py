"""Isolated material/lighting approval stills. Never writes a video or changes the web app.

Uses the existing PBRT geometry and inner-wall wet/dry interfaces. Output and scene
directories are separate from the delivered F4/F5 sequences. No image-generation AI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import digital_twin as twin
from common import PROJ_ROOT, PBRT_EXE, IMGTOOL_EXE, save_json
from scale_model import write_maps, _noise

OUT = PROJ_ROOT / "outputs" / "digital_twin_review"
SCENES = PROJ_ROOT / "scenes" / "digital_twin_review"
CONFIG = PROJ_ROOT / "configs" / "digital_twin_review.json"
ORIGINAL_MATERIALS = twin.materials
ORIGINAL_FIXTURES = twin.fixtures
ORIGINAL_MAPS = twin.maps


def write_gray(path, data):
    cv2.imwrite(str(path), (np.clip(data, 0, 1) * 255 + .5).astype(np.uint8))


def deposit_maps(config, review):
    """Separate fixed inner-wall flecks from porous mineral haze and sparse outside dust.

    The source references show asymmetric islands, clear gaps, vertical trails and
    denser material near the bowl. These are material-coordinate features, not
    image-space noise; wetting changes mineral scattering but does not erase silt.
    """
    texdir = OUT / "textures"
    texdir.mkdir(parents=True, exist_ok=True)
    # Reuse the existing ruler texture and map-file naming contract in this isolated directory.
    ORIGINAL_MAPS(config)
    rng = np.random.default_rng(review["texture_seed"])
    h, w = 1536, 3072
    z = (1 - np.arange(h, dtype=np.float32) / h)[:, None] * config["geometry"]["rim_z"]
    bottom = np.clip((12.5 - z) / 8, 0, 1)
    cloud = _noise(rng, (h, w), 22, 44, 3)
    small = _noise(rng, (h, w), 100, 220, 2)
    streak_field = _noise(rng, (h, w), 7, 170, 2)
    p = review["deposit"]
    haze = p["haze_strength"] * np.clip((cloud - .31) * 2.0, 0, 1)
    tau = .11 + haze * (1 + p["bottom_multiplier"] * bottom) * (.6 + .7 * small)
    tau += .045 * streak_field * bottom

    flecks = np.zeros((h, w), np.float32)
    for _ in range(p["fleck_count"]):
        x, y = int(rng.integers(w)), int(rng.integers(h))
        # Non-uniform density gives small ragged colonies with transparent gaps.
        acceptance = .04 + .48 * float(cloud[y, x]) + .63 * float(bottom[y, 0])
        if rng.random() > acceptance:
            continue
        rx = int(np.clip(rng.lognormal(-.1, .65), 1, 6))
        ry = max(1, int(rx * rng.uniform(.65, 2.0)))
        strength = float(rng.uniform(.24, .92))
        cv2.ellipse(flecks, (x, y), (rx, ry), float(rng.uniform(0, 180)), 0, 360, strength, -1)
        if rx > 1:
            cv2.circle(flecks, (x + rx, y - ry // 2), max(1, rx // 2), strength * .55, -1)
    flecks = cv2.GaussianBlur(flecks, (0, 0), .40)

    # Broken, gently tilted historical lines; do not confuse them with the new moving surface.
    theta = np.arange(w, dtype=np.float32)[None, :] / w * 2 * np.pi
    for height, amplitude, width, creep in p["rings"]:
        dz = z - height - .10 * np.sin(theta + .5) - .035 * np.sin(theta * 16)
        ring = np.exp(-.5 * (dz / width) ** 2)
        ring += .48 * np.exp(-np.maximum(dz,0)/creep) * (dz>0)
        tau += amplitude * ring * (.40 + .85 * np.clip((cloud-.12)*1.6,0,1)) * (.7+.5*small)

    # Local drainage tracks with bead-like ends, using small patches rather than a uniform veil.
    trails = np.zeros((h, w), np.float32)
    for _ in range(95):
        x, y = int(rng.integers(6, w - 6)), int(rng.integers(230, h - 100))
        length = int(rng.integers(18, 170))
        dx = int(rng.integers(-5, 6))
        value = float(rng.uniform(.025, .13))
        cv2.line(trails, (x, y), (x + dx, min(h - 1, y + length)), value, int(rng.integers(1, 4)))
        cv2.ellipse(trails, (x + dx, min(h - 1, y + length)), (3, 5), 0, 0, 360, value * 1.5, 1)
    tau += cv2.GaussianBlur(trails, (0, 0), .65)
    tau += .22 * flecks
    # Fine particulate haze concentrates in the lower-left and lower-middle sectors
    # seen in the references; avoid a uniform image-wide grain layer.
    u = np.arange(w, dtype=np.float32)[None, :] / w
    patch = np.zeros((h,w), np.float32)
    for centre,zc,spread in [(.86,6.0,.095),(.69,8.0,.07),(.21,6.5,.12)]:
        distance = np.abs((u-centre+.5)%1-.5)
        patch += np.exp(-.5*(distance/spread)**2) * np.exp(-.5*((z-zc)/2.0)**2)
    tau += .14 * patch * np.clip((small-.30)*1.8,0,1)
    files = write_maps(tau.astype(np.float32), texdir, "reactor", iron=.10)
    # Old compacted deposits remain milky underwater, unlike a fresh highly porous film.
    write_gray(texdir/files["scale_amount_wet"], 1-np.exp(-tau/p["wet_scattering_reduction"]))
    silt = np.clip(flecks * p["fleck_strength"] * (.75 + .65 * bottom) + .10 * patch * np.clip((small-.38)*2,0,1), 0, .98)
    write_gray(texdir / "inner_silt.png", silt)
    # Outside remains mostly clear; the heavy dirt belongs to the inner wall.
    write_gray(texdir / "dirt.png", .004 * small + .008 * flecks)
    write_gray(texdir / "inner_relief.png", np.clip(flecks * .45 + tau * .22, 0, 1))
    # Microscale unevenness modulates reflections without frosting the entire pane.
    write_gray(texdir / "outer_roughness.png", .003 + .013 * np.clip((cloud - .35) * 2, 0, 1))
    save_json({"inner_fleck_coverage_gt_0_2": float((silt > .2).mean()),
               "inner_haze_mean_tau": float(tau.mean()), "fixed_seed": review["texture_seed"],
               "comparison_references": ["15_525.png", "60_30.png"]}, OUT / "material_stats.json")
    return {**files, "kind": "scale"}


def material_factory(review):
    def materials(combo, tex, lay):
        text = ORIGINAL_MATERIALS(combo, tex, lay)
        p = review["deposit"]
        text = text.replace('"float roughness" [0.22]', f'"float roughness" [{p["dry_roughness"]}]')
        text = text.replace('"float roughness" [0.10]', f'"float roughness" [{p["wet_roughness"]}]')
        inner_names = ["wall_in_dry"] + [f"wall_in_wet_liq{i+1}" for i in range(len(lay or []))]
        for name in inner_names:
            text = text.replace(f'MakeNamedMaterial "{name}"', f'MakeNamedMaterial "{name}_mineral"')
        td = (OUT / "textures").as_posix()
        text += f'\nTexture "innerSilt" "float" "imagemap" "string filename" ["{td}/inner_silt.png"] "string encoding" "linear"'
        text += f'\nTexture "innerRelief" "float" "imagemap" "string filename" ["{td}/inner_relief.png"] "string encoding" "linear" "float scale" [.0015]'
        text += '\nMakeNamedMaterial "fixed_silt" "string type" "diffusetransmission" "rgb reflectance" [.065 .058 .045] "rgb transmittance" [.03 .027 .023] "texture displacement" "innerRelief"'
        for name in inner_names:
            text += f'\nMakeNamedMaterial "{name}" "string type" "mix" "string materials" ["{name}_mineral" "fixed_silt"] "texture amount" "innerSilt"'
        return text
    return materials


def quad(points, emission=None, material=None, uv=False):
    text = 'AttributeBegin\n'
    if emission is not None:
        text += f'AreaLightSource "diffuse" {emission} "bool twosided" [true]\n'
    text += (material or 'Material "diffuse" "rgb reflectance" [.7 .7 .7]') + '\n'
    text += 'Shape "trianglemesh" "point3 P" [' + ' '.join(str(v) for p in points for v in p) + '] '
    if uv:
        text += '"point2 uv" [0 0 1 0 1 1 0 1] '
    return text + '"integer indices" [0 1 2 0 2 3]\nAttributeEnd'


def board_factory(review):
    def backlight(sid):
        x = review["panel_width_cm"] / 2
        z0 = review["panel_center_z_cm"] - review["panel_height_cm"] / 2
        z1 = z0 + review["panel_height_cm"]
        y = review["panel_y_cm"]
        L = review["backlight_radiance"]
        if sid == "D1":
            emission = f'"rgb L" [{L} {L} {L}]'
        else:
            source = PROJ_ROOT / "outputs" / "print_patterns" / review["pattern"]
            board = cv2.imread(str(source))
            # Use the exact authored PDF companion image at its 21 x 29.7 cm A4 dimensions.
            # Mirror world-space x only, undoing the camera's world +x -> image-left mapping.
            target = OUT / "textures" / review["pattern"]
            cv2.imwrite(str(target), cv2.flip(board, 1))
            emission = f'"string filename" ["{target.as_posix()}"] "float scale" [{L}]'
        parts = [quad([(-x,y,z0),(x,y,z0),(x,y,z1),(-x,y,z1)], emission, uv=True)]
        # Opaque A4 light-box frame and shallow aluminium tray (outside the printed area).
        for x0,x1,za,zb in [(-x-.35,-x,z0-.35,z1+.35),(x,x+.35,z0-.35,z1+.35),(-x,x,z0-.35,z0),(-x,x,z1,z1+.35)]:
            parts.append(quad([(x0,y-.02,za),(x1,y-.02,za),(x1,y-.02,zb),(x0,y-.02,zb)],
                              material='Material "coateddiffuse" "rgb reflectance" [.13 .14 .15] "float roughness" [.22]'))
        return '\n'.join(parts)
    return backlight


def environment(review):
    envdir = PROJ_ROOT / "outputs" / "env"
    meta = json.loads((envdir / (review["environment"] + "_meta.json")).read_text())
    file = (envdir / meta["equiarea"]).as_posix()
    scale = review["environment_mean_radiance"] / meta["mean_luminance"]
    parts = [f'AttributeBegin\nRotate {review["environment_rotation_deg"]} 0 0 1\nLightSource "infinite" "string filename" ["{file}"] "float scale" [{scale}]\nAttributeEnd']
    # Reflected overhead softbox and a narrow cool side light, both outside the camera crop.
    key = '"rgb L" [' + ' '.join(map(str, review["key_radiance"])) + ']'
    edge = '"rgb L" [' + ' '.join(map(str, review["edge_radiance"])) + ']'
    parts.append(quad([(-9,-18,22),(7,-18,22),(7,-7,22),(-9,-7,22)], key))
    parts.append(quad([(-13,-13,4),(-13,-6,4),(-13,-6,19),(-13,-13,19)], edge))
    parts.append(quad([(-5,-55,17.4),(5,-55,17.4),(5,-55,19.5),(-5,-55,19.5)],
                      '"rgb L" [3.0 2.95 2.85]'))
    # A pale bench and dark camera-side card create grounded lower reflections and contrast.
    parts.append(quad([(-60,-65,-2),(60,-65,-2),(60,35,-2),(-60,35,-2)],
                      material='Material "coateddiffuse" "rgb reflectance" [.28 .29 .30] "float roughness" [.30]'))
    parts.append(quad([(9,-20,0),(13,-20,0),(13,-20,22),(9,-20,22)],
                      material='Material "diffuse" "rgb reflectance" [.035 .035 .035]'))
    return '\n'.join(parts)


def fixture_factory(review):
    def fixtures(vessel, lay, ruler, config):
        text = ORIGINAL_FIXTURES(vessel, lay, ruler, config)
        # Actual coating response reveals the curved pipe and lid under room reflections.
        text = text.replace('MakeNamedMaterial "black" "string type" "diffuse" "rgb reflectance" [.012 .012 .012]',
                            'MakeNamedMaterial "hardware_matte" "string type" "diffuse" "rgb reflectance" [.016 .017 .017]\nMakeNamedMaterial "hardware_coated" "string type" "coateddiffuse" "rgb reflectance" [.022 .023 .023] "float roughness" [.20] "bool remaproughness" [false]\nMakeNamedMaterial "black" "string type" "mix" "string materials" ["hardware_matte" "hardware_coated"] "float amount" [.25]')
        text = text.replace('MakeNamedMaterial "lid" "string type" "diffuse" "rgb reflectance" [.065 .065 .065]',
                            'MakeNamedMaterial "lid" "string type" "coateddiffuse" "rgb reflectance" [.035 .036 .038] "float roughness" [.12] "bool remaproughness" [false]')
        text = text.replace('Material "diffuse" "rgb reflectance" [0.04 0.04 0.04]',
                            'Material "coateddiffuse" "rgb reflectance" [.028 .029 .030] "float roughness" [.19] "bool remaproughness" [false]')
        return text
    return fixtures


def camera_finish(path, review, monochrome):
    image = cv2.imread(str(path)).astype(np.float32) / 255
    if monochrome:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (0,0), review["sensor"]["blur_px"])
    h,w = image.shape[:2]
    yy,xx = np.mgrid[:h,:w]
    vignette = 1 - .09 * ((xx/w-.5)**2 + (yy/h-.45)**2)
    if image.ndim == 3:
        vignette = vignette[...,None]
    image *= vignette
    image += np.random.default_rng(982).normal(0, review["sensor"]["noise_sigma"], image.shape)
    cv2.imwrite(str(path), (np.clip(image,0,1)*255+.5).astype(np.uint8))


def finish_scene(path, review):
    """Identical final scene treatment for the approved still and continuous sequence."""
    text = path.read_text()
    text = text.replace('LightSource "infinite" "rgb L" [.035 .035 .035]', environment(review))
    td = (OUT/"textures").as_posix()
    text = text.replace('MakeNamedMaterial "wall_clear"',
                        f'Texture "clearRoughness" "float" "imagemap" "string filename" ["{td}/outer_roughness.png"] "string encoding" "linear"\nMakeNamedMaterial "wall_clear"')
    text = text.replace('"float roughness" [.006]', '"texture roughness" "clearRoughness"')
    path.write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["D1","D2"])
    parser.add_argument("--spp", type=int)
    parser.add_argument("--width", type=int)
    args = parser.parse_args()
    config = json.loads(twin.CONFIG.read_text())
    review = json.loads(CONFIG.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    SCENES.mkdir(parents=True, exist_ok=True)
    twin.OUT, twin.SCENES = OUT, SCENES
    tex = deposit_maps(config, review)
    twin.materials = material_factory(review)
    twin.backlight = board_factory(review)
    twin.fixtures = fixture_factory(review)
    width = args.width or review["preview"]["width"]
    height = round(width * 3/4)
    spp = args.spp or review["preview"]["spp"]
    manifest = {"status":"awaiting user approval", "kind":"isolated stills, no animation", "settings":review, "stills":[]}
    for sid in ["D1", "D2"]:
        if args.only and args.only != sid:
            continue
        path,gt = twin.scene(config,tex,sid,review["preview"][sid+"_level_z_cm"],51,width,height,spp)
        text = finish_scene(path, review)
        png = OUT / f"{sid}_approval.png"
        exr = OUT / f"{sid}_approval.exr"
        start = time.monotonic()
        print(f"Static approval {sid}: {width}x{height}, {spp} spp", flush=True)
        cmd = [str(PBRT_EXE),"--quiet","--nthreads","16","--seed","51","--outfile",str(exr),path.name]
        result = subprocess.run(cmd,cwd=SCENES,capture_output=True,text=True)
        (OUT/f"{sid}_render_log.txt").write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr+result.stdout)
        subprocess.run([str(IMGTOOL_EXE),"convert","--outfile",str(png),str(exr)],check=True,capture_output=True)
        camera_finish(png,review,monochrome=sid=="D1")
        manifest["stills"].append({"id":sid,"image":png.name,"gt_rows_frac":gt,
                                  "scene_sha256":hashlib.sha256(text.encode()).hexdigest(),
                                  "resolution":[width,height],"spp":spp,"seconds":round(time.monotonic()-start,2)})
        print(f"Finished {png.name}: {time.monotonic()-start:.1f}s",flush=True)
    save_json(manifest, OUT/"approval_manifest.json")


if __name__ == "__main__":
    main()

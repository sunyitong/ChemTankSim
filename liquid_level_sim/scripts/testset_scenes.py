"""Scene generation for the Phase-1 test images: odd-shaped vessels, wall deposits, lens variants,
and layered immiscible liquids. Physical modelling follows the MVP (docs/RESEARCH.md §4): nested
dielectrics without air gaps, absorbing media, meniscus on the free surface, Lambertian light panel.

Deposits: the outer wall is a pbrt `mix` of frosted glass (textured roughness) and a translucent
`diffusetransmission` deposit; the mix `amount` texture is the deposit coverage, so partially
covered pixels still show the backlight pattern. The dry inner wall carries scale rings only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mvp_common import PROJ_ROOT, Setup, save_json
from mvp_scene import _f, header as _mvp_header, panel, pedestal  # noqa: F401
from testset_shapes import VESSELS, Vessel, free_surface, revolve

TS_OUT = PROJ_ROOT / "outputs" / "testset"
TS_SCENES = PROJ_ROOT / "scenes" / "testset"
TS_TEX = TS_OUT / "textures"
TS_PATTERNS = TS_OUT / "patterns"
TS_RENDERS = TS_OUT / "renders"
TS_IMAGES = TS_OUT / "images"
REL_TEX = "../../outputs/testset/textures"
REL_PAT = "../../outputs/testset/patterns"

VESSELS = dict(VESSELS)
VESSELS["cylinder"] = Vessel("cylinder", lambda z: 7.8, H=20.5, t=0.3, tb=0.5, label="Cylinder 15 x 20 cm (MVP)")

CAMERAS = {"normal": dict(fov=24.0, dist=60.0), "wide": dict(fov=44.0, dist=34.0), "tele": dict(fov=11.0, dist=130.0),
           "macro": dict(fov=5.5, dist=110.0)}          # macro: ~10 x 14 cm field at the vessel, aimed with cam_dz

# eta at ~550 nm; sigma_a per cm (R,G,B); meniscus rise h_m and capillary length l_c (cm)
LIQUIDS = {
    "water":         dict(eta=1.333, sigma_a=(0.0035, 0.0015, 0.0003), h_m=0.30, l_c=0.27, label="water"),
    "tea":           dict(eta=1.335, sigma_a=(0.05, 0.18, 0.45),       h_m=0.30, l_c=0.27, label="tea (tinted water)"),
    "sunflower_oil": dict(eta=1.472, sigma_a=(0.03, 0.07, 0.32),       h_m=0.22, l_c=0.19, label="sunflower oil"),
    "hexane":        dict(eta=1.375, sigma_a=(0.001, 0.001, 0.002),    h_m=0.18, l_c=0.17, label="n-hexane"),
    "dcm":           dict(eta=1.424, sigma_a=(0.001, 0.001, 0.002),    h_m=0.15, l_c=0.14, label="dichloromethane"),
}
ETA_GLASS = 1.50


@dataclass
class Combo:
    name: str
    vessel: str
    camera: str
    pattern: str
    dirt: str | None                      # None | light | medium | heavy
    dirt_seed: int
    layers: list                          # [(liquid, fill fraction of inner height)] bottom -> top
    cam_dz: float = 0.0                   # camera height offset from the vessel mid-height (cm)
    part: str = "A"
    note: str = ""


COMBOS = [
    Combo("A1_gourd_normal_checker", "gourd", "normal", "rg_checker", "light", 11, [("water", 0.45)]),
    Combo("A2_gourd_wide_mosaic", "gourd", "wide", "mosaic", "heavy", 12, [("water", 0.62)]),
    Combo("A3_erlenmeyer_tele_grid", "erlenmeyer", "tele", "grid", "medium", 13, [("water", 0.38)]),
    Combo("A4_florence_normal_checker", "florence", "normal", "rg_checker", "light", 14, [("water", 0.55)]),
    Combo("A5_reagent_wide_checker", "reagent", "wide", "rg_checker", "heavy", 15, [("water", 0.50)]),
    Combo("A6_erlenmeyer_normal_mosaic_tea", "erlenmeyer", "normal", "mosaic", "light", 16, [("tea", 0.42)], cam_dz=2.0),
    Combo("B1_cyl_water_oil_checker", "cylinder", "normal", "rg_checker", None, 0, [("water", 0.40), ("sunflower_oil", 0.25)], part="B",
          note="water below, sunflower oil above: dn = 0.14, oil slightly yellow"),
    Combo("B2_cyl_water_hexane_checker", "cylinder", "normal", "rg_checker", None, 0, [("water", 0.40), ("hexane", 0.25)], part="B",
          note="water below, n-hexane above: dn = 0.04, both colourless (hardest case)"),
    Combo("B3_cyl_dcm_water_checker", "cylinder", "normal", "rg_checker", None, 0, [("dcm", 0.30), ("water", 0.30)], part="B",
          note="dichloromethane below (denser), water above: dn = 0.09"),
    Combo("B4_cyl_water_hexane_white", "cylinder", "normal", "white", None, 0, [("water", 0.40), ("hexane", 0.25)], part="B",
          note="same as B2 against a plain white panel"),
]


# ----------------------------------------------------------------------------- helpers
def camera_setup(combo: Combo, vessel: Vessel, width: int, height: int) -> Setup:
    cam = CAMERAS[combo.camera]
    return Setup(r_in=vessel.r_in(vessel.H * 0.5) if vessel.kind == "revolved" else vessel.params["R"], wall=vessel.t,
                 bottom=vessel.tb, height=vessel.inner_height, cam_dist=cam["dist"], cam_z=vessel.H / 2 + combo.cam_dz,
                 fov=cam["fov"], width=width, height_px=height, panel_y=25.0, panel_w=60.0, panel_h=80.0, panel_scale=0.8,
                 pedestal_r=min(4.0, 0.7 * vessel.r_out(0.0)) if vessel.kind == "revolved" else 5.4, pedestal_h=8.0)


def header(cfg: Setup, out_image: str, spp: int, width: int, height: int) -> str:
    ex, ey, ez = cfg.eye
    lx, ly, lz = cfg.look
    return "\n".join([
        f"# test-set scene -> {out_image}",
        f"LookAt {_f(ex)} {_f(ey)} {_f(ez)}  {_f(lx)} {_f(ly)} {_f(lz)}  0 0 1",
        f'Camera "perspective" "float fov" [{_f(cfg.fov)}]',
        f'Sampler "zsobol" "integer pixelsamples" [{spp}]',
        # no path regularisation: it would roughen every slightly-rough dielectric to alpha >= 0.1 on
        # later bounces and turn a faint deposit haze into frosted glass
        'Integrator "volpath" "integer maxdepth" [48] "bool regularize" [false]',
        'PixelFilter "gaussian"',
        f'Film "rgb" "string filename" ["{out_image}"] "integer xresolution" [{width}] "integer yresolution" [{height}] "bool savefp16" [false]',
    ])


def materials(combo: Combo, tex: dict | None, lay: list | None = None) -> str:
    L = [f'MakeNamedMaterial "glass" "string type" "dielectric" "float eta" [{_f(ETA_GLASS)}]']
    if not tex or not (tex.get("kind") == "scale" or "dirt_amount" in tex):      # clean glass
        L += ['MakeNamedMaterial "wall_out" "string type" "dielectric" "float eta" [1.5]',
              'MakeNamedMaterial "wall_in_dry" "string type" "dielectric" "float eta" [1.5]']
        return "\n".join(L)
    if tex.get("kind") == "scale":
        # limescale on the INNER wall (scale_model.py): thin scattering layer, interaction probability
        # 1 - exp(-tau) as mix amount; submerged parts use the index-matched (wet) amount
        # Layer model per pixel: P(interact) = 1 - exp(-tau) picks the deposit; inside the deposit the
        # thin film scatters mostly FORWARD (rough dielectric transmission keeps a blurred view of the
        # backlight pattern) while thick crust scatters diffusely (chalky white); the share of diffuse
        # scattering is the thickness texture. Submerged deposit: smaller tau and weaker forward lobe.
        L += [
            f'Texture "sAmtDry" "float" "imagemap" "string filename" ["{REL_TEX}/{tex["scale_amount_dry"]}"] "string encoding" "linear"',
            f'Texture "sAmtWet" "float" "imagemap" "string filename" ["{REL_TEX}/{tex["scale_amount_wet"]}"] "string encoding" "linear"',
            f'Texture "sThick" "float" "imagemap" "string filename" ["{REL_TEX}/{tex["scale_thick"]}"] "string encoding" "linear"',
            f'Texture "sRefl" "spectrum" "imagemap" "string filename" ["{REL_TEX}/{tex["scale_refl"]}"]',
            f'Texture "sTrans" "spectrum" "imagemap" "string filename" ["{REL_TEX}/{tex["scale_trans"]}"]',
            'MakeNamedMaterial "wall_out" "string type" "dielectric" "float eta" [1.5]',
            'MakeNamedMaterial "dep_diffuse" "string type" "diffusetransmission" "texture reflectance" "sRefl" "texture transmittance" "sTrans"',
            f'MakeNamedMaterial "dep_fwd_dry" "string type" "dielectric" "float eta" [{_f(ETA_GLASS)}] "float roughness" [0.22] "bool remaproughness" false',
            'MakeNamedMaterial "deposit_dry" "string type" "mix" "string materials" ["dep_fwd_dry" "dep_diffuse"] "texture amount" "sThick"',
            'MakeNamedMaterial "wall_in_dry" "string type" "mix" "string materials" ["glass" "deposit_dry"] "texture amount" "sAmtDry"',
        ]
        for i, l in enumerate(lay or []):
            eta = _f(ETA_GLASS / l["eta"])
            L += [f'MakeNamedMaterial "glass_wet_liq{i + 1}" "string type" "dielectric" "float eta" [{eta}]',
                  f'MakeNamedMaterial "dep_fwd_wet{i + 1}" "string type" "dielectric" "float eta" [{eta}] "float roughness" [0.10] "bool remaproughness" false',
                  f'MakeNamedMaterial "deposit_wet{i + 1}" "string type" "mix" "string materials" ["dep_fwd_wet{i + 1}" "dep_diffuse"] "texture amount" "sThick"',
                  f'MakeNamedMaterial "wall_in_wet_liq{i + 1}" "string type" "mix" "string materials" ["glass_wet_liq{i + 1}" "deposit_wet{i + 1}"] "texture amount" "sAmtWet"']
        return "\n".join(L)
    L += [
        f'Texture "dirtA" "float" "imagemap" "string filename" ["{REL_TEX}/{tex["dirt_amount"]}"] "string encoding" "linear"',
        f'Texture "ringsA" "float" "imagemap" "string filename" ["{REL_TEX}/{tex["rings"]}"] "string encoding" "linear"',
        f'Texture "roughT" "float" "imagemap" "string filename" ["{REL_TEX}/{tex["rough"]}"] "string encoding" "linear" "float scale" [0.12]',
        f'Texture "dirtC" "spectrum" "imagemap" "string filename" ["{REL_TEX}/{tex["dirt_color"]}"]',
        'Texture "dirtR" "spectrum" "scale" "texture tex" "dirtC" "float scale" [0.35]',
        'Texture "dirtT" "spectrum" "scale" "texture tex" "dirtC" "float scale" [0.60]',
        'MakeNamedMaterial "deposit" "string type" "diffusetransmission" "texture reflectance" "dirtR" "texture transmittance" "dirtT"',
        # remaproughness=false: the texture sets the microfacet alpha directly (default sqrt remap turns
        # a faint 0.02 film into alpha 0.14, i.e. visibly frosted glass everywhere)
        f'MakeNamedMaterial "glass_rough" "string type" "dielectric" "float eta" [{_f(ETA_GLASS)}] "texture roughness" "roughT" "bool remaproughness" false',
        'MakeNamedMaterial "wall_out" "string type" "mix" "string materials" ["glass_rough" "deposit"] "texture amount" "dirtA"',
        'MakeNamedMaterial "wall_in_dry" "string type" "mix" "string materials" ["glass" "deposit"] "texture amount" "ringsA"',
    ]
    return "\n".join(L)


def layer_heights(vessel: Vessel, layers: list) -> list[dict]:
    """Bottom->top list of {name, z0, z1, eta, ...}; z1 of the top layer is the free surface."""
    out, z = [], vessel.tb if vessel.kind == "revolved" else 0.0
    for liq, frac in layers:
        z1 = z + frac * vessel.inner_height
        out.append(dict(name=liq, z0=z, z1=z1, **LIQUIDS[liq]))
        z = z1
    return out


def wall_intervals(vessel: Vessel, lay: list[dict], z_bottom: float, extra_breaks=()) -> list[tuple[float, float, str | None]]:
    """Split the inner wall at every liquid boundary (top liquid extends to its meniscus contact)."""
    breaks = {z_bottom, vessel.H, *extra_breaks}
    for l in lay:
        breaks.add(l["z0"]); breaks.add(l["z1"])
    if lay:
        breaks.add(min(vessel.H, lay[-1]["z1"] + lay[-1]["h_m"]))
    zs = sorted(b for b in breaks if z_bottom - 1e-9 <= b <= vessel.H + 1e-9)
    out = []
    for a, b in zip(zs[:-1], zs[1:]):
        if b - a < 1e-6:
            continue
        mid = 0.5 * (a + b); med = None
        for i, l in enumerate(lay):
            top = l["z1"] + (l["h_m"] if i == len(lay) - 1 else 0.0)
            if l["z0"] <= mid < top:
                med = f"liq{i + 1}"
        out.append((a, b, med))
    return out


def medium_lines(lay: list[dict]) -> list[str]:
    L = []
    for i, l in enumerate(lay):
        sa = " ".join(_f(v) for v in l["sigma_a"])
        L.append(f'MakeNamedMedium "liq{i + 1}" "string type" "homogeneous" "rgb sigma_a" [{sa}] "rgb sigma_s" [0 0 0] "float scale" [1]')
    return L


def interface_disc(z: float, r: float, n_below: float, n_above: float, med_below: str, med_above: str) -> str:
    """Horizontal liquid|liquid or liquid|air interface with eta >= 1 ('inside' = denser index)."""
    if n_below >= n_above:
        return "\n".join(["AttributeBegin", f'  MediumInterface "{med_below}" "{med_above}"',
                          f'  Material "dielectric" "float eta" [{_f(n_below / n_above)}]',
                          f'  Shape "disk" "float height" [{_f(z)}] "float radius" [{_f(r)}]', "AttributeEnd"])
    return "\n".join(["AttributeBegin", f'  MediumInterface "{med_above}" "{med_below}"',
                      f'  Material "dielectric" "float eta" [{_f(n_above / n_below)}]', "  ReverseOrientation",
                      f'  Shape "disk" "float height" [{_f(z)}] "float radius" [{_f(r)}]', "AttributeEnd"])


# ----------------------------------------------------------------------------- vessel bodies
def revolved_body(vessel: Vessel, lay: list[dict], mesh_dir: Path, scale: bool = False) -> str:
    H, tb = vessel.H, vessel.tb
    outer = mesh_dir / f"mesh_{vessel.name}_outer.ply"
    revolve(vessel.r_out, 0.0, H, outer, outward=True, v_scale=1.0 / H)
    L = ["AttributeBegin  # outer wall", '  NamedMaterial "wall_out"', f'  Shape "plymesh" "string filename" ["{outer.name}"]', "AttributeEnd",
         "AttributeBegin  # rim + outer bottom", '  NamedMaterial "glass"',
         f'  Shape "disk" "float height" [{_f(H)}] "float radius" [{_f(vessel.r_out(H))}] "float innerradius" [{_f(vessel.r_in(H))}]',
         "  AttributeBegin", "    ReverseOrientation", f'    Shape "disk" "float height" [0] "float radius" [{_f(vessel.r_out(0.0))}]', "  AttributeEnd",
         "AttributeEnd"]
    # inner bottom
    if lay:
        L += ["AttributeBegin  # inner bottom (glass|liquid)", f'  MediumInterface "" "liq1"',
              f'  Material "dielectric" "float eta" [{_f(ETA_GLASS / lay[0]["eta"])}]',
              f'  Shape "disk" "float height" [{_f(tb)}] "float radius" [{_f(vessel.r_in(tb))}]', "AttributeEnd"]
    else:
        L += ["AttributeBegin  # inner bottom (dry)", '  NamedMaterial "glass"',
              f'  Shape "disk" "float height" [{_f(tb)}] "float radius" [{_f(vessel.r_in(tb))}]', "AttributeEnd"]
    # inner wall segments
    for a, b, med in wall_intervals(vessel, lay, tb):
        mesh = mesh_dir / f"mesh_{vessel.name}_inner_{a:.3f}_{b:.3f}.ply"
        revolve(vessel.r_in, a, b, mesh, outward=False, v_scale=1.0 / H, n_z=max(8, int(96 * (b - a) / H) + 2))
        if med:
            idx = int(med[3:]) - 1
            mat = f'  NamedMaterial "wall_in_wet_{med}"' if scale else f'  Material "dielectric" "float eta" [{_f(ETA_GLASS / lay[idx]["eta"])}]'
            L += [f"AttributeBegin  # inner wall wetted by {lay[idx]['name']}", f'  MediumInterface "" "{med}"', mat,
                  f'  Shape "plymesh" "string filename" ["{mesh.name}"]', "AttributeEnd"]
        else:
            L += ["AttributeBegin  # dry inner wall", '  NamedMaterial "wall_in_dry"',
                  f'  Shape "plymesh" "string filename" ["{mesh.name}"]', "AttributeEnd"]
    return "\n".join(L)


def florence_body(vessel: Vessel, lay: list[dict], mesh_dir: Path, scale: bool = False) -> tuple[str, callable]:
    R, rn, t, H = vessel.params["R"], vessel.params["r_neck"], vessel.t, vessel.H
    zc0 = R; Ri = R - t; rni = rn - t
    zj_out = zc0 + math.sqrt(R * R - rn * rn); zj_in = zc0 + math.sqrt(Ri * Ri - rni * rni)
    z_bottom = zc0 - Ri

    def r_in(z):
        return math.sqrt(max(Ri * Ri - (z - zc0) ** 2, 0.0)) if z <= zj_in else rni

    L = ["AttributeBegin  # outer sphere + neck", '  NamedMaterial "wall_out"',
         f"  AttributeBegin\n    Translate 0 0 {_f(zc0)}\n    Shape \"sphere\" \"float radius\" [{_f(R)}] \"float zmax\" [{_f(zj_out - zc0)}]\n  AttributeEnd",
         f'  Shape "cylinder" "float radius" [{_f(rn)}] "float zmin" [{_f(zj_out)}] "float zmax" [{_f(H)}]', "AttributeEnd",
         "AttributeBegin  # rim", '  NamedMaterial "glass"',
         f'  Shape "disk" "float height" [{_f(H)}] "float radius" [{_f(rn)}] "float innerradius" [{_f(rni)}]', "AttributeEnd"]
    for a, b, med in wall_intervals(vessel, lay, z_bottom, extra_breaks=(zj_in,)):
        if med:
            idx = int(med[3:]) - 1
            mat = [f'  MediumInterface "" "{med}"',
                   f'  NamedMaterial "wall_in_wet_{med}"' if scale else f'  Material "dielectric" "float eta" [{_f(ETA_GLASS / lay[idx]["eta"])}]']
        else:
            mat = ['  NamedMaterial "wall_in_dry"']
        if b <= zj_in + 1e-6:
            shape = (f"  AttributeBegin\n    ReverseOrientation\n    Translate 0 0 {_f(zc0)}\n"
                     f"    Shape \"sphere\" \"float radius\" [{_f(Ri)}] \"float zmin\" [{_f(a - zc0)}] \"float zmax\" [{_f(b - zc0)}]\n  AttributeEnd")
        else:
            shape = (f"  AttributeBegin\n    ReverseOrientation\n"
                     f"    Shape \"cylinder\" \"float radius\" [{_f(rni)}] \"float zmin\" [{_f(a)}] \"float zmax\" [{_f(b)}]\n  AttributeEnd")
        L += ["AttributeBegin  # inner piece", *mat, shape, "AttributeEnd"]
    # cork ring support (the sphere rests on the ring's inner top edge) + pedestal below it
    ring_r = vessel.params["ring_r"]; z_r = zc0 - math.sqrt(R * R - ring_r * ring_r)
    L += ["AttributeBegin  # cork ring", '  Material "coateddiffuse" "rgb reflectance" [0.30 0.20 0.12] "float roughness" [0.5]',
          f'  Shape "cylinder" "float radius" [{_f(ring_r + 1.4)}] "float zmin" [{_f(z_r - 3.0)}] "float zmax" [{_f(z_r)}]',
          f"  AttributeBegin\n    ReverseOrientation\n    Shape \"cylinder\" \"float radius\" [{_f(ring_r)}] \"float zmin\" [{_f(z_r - 3.0)}] \"float zmax\" [{_f(z_r)}]\n  AttributeEnd",
          f'  Shape "disk" "float height" [{_f(z_r)}] "float radius" [{_f(ring_r + 1.4)}] "float innerradius" [{_f(ring_r)}]',
          f"  AttributeBegin\n    ReverseOrientation\n    Shape \"disk\" \"float height\" [{_f(z_r - 3.0)}] \"float radius\" [{_f(ring_r + 1.4)}] \"float innerradius\" [{_f(ring_r)}]\n  AttributeEnd",
          "AttributeEnd"]
    return "\n".join(L), r_in


def liquids_block(vessel: Vessel, lay: list[dict], r_in, mesh_dir: Path, tag: str) -> str:
    if not lay:
        return ""
    L = medium_lines(lay)
    for i in range(len(lay) - 1):                              # liquid|liquid interfaces
        lo, hi = lay[i], lay[i + 1]
        L.append(interface_disc(lo["z1"], r_in(lo["z1"]) - 1e-3, lo["eta"], hi["eta"], f"liq{i + 1}", f"liq{i + 2}"))
    top = lay[-1]; zt = top["z1"]; zc = min(vessel.H, zt + top["h_m"])
    mesh = mesh_dir / f"mesh_{tag}_surface.ply"
    free_surface(mesh, r_in(zt), r_in(zc), zc - zt, top["l_c"])
    L += ["AttributeBegin  # free surface + meniscus", f'  MediumInterface "liq{len(lay)}" ""',
          f'  Material "dielectric" "float eta" [{_f(top["eta"])}]', f"  Translate 0 0 {_f(zt)}",
          f'  Shape "plymesh" "string filename" ["{mesh.name}"]', "AttributeEnd"]
    return "\n".join(L)


# ----------------------------------------------------------------------------- scene assembly
def build_scene(combo: Combo, filled: bool, width: int, height: int, spp: int, tex: dict | None, out_image: str) -> tuple[str, dict]:
    vessel = VESSELS[combo.vessel]
    cfg = camera_setup(combo, vessel, width, height)
    lay = layer_heights(vessel, combo.layers) if filled else []
    pattern_rel = None if combo.pattern == "white" else f"{REL_PAT}/{combo.pattern}.png"
    parts = [header(cfg, out_image, spp, width, height), "", "WorldBegin", panel(cfg, combo.pattern, pattern_rel)]
    scale = bool(tex and tex.get("kind") == "scale")
    if tex and tex.get("ambient"):                          # lit room: makes translucent deposits look white, not dark
        a = tex["ambient"]
        parts.append(f'LightSource "infinite" "rgb L" [{_f(a)} {_f(a)} {_f(a * 1.05)}]')
    if vessel.kind == "revolved":
        parts.append(pedestal(cfg))
        parts.append(materials(combo, tex, lay))
        parts.append(revolved_body(vessel, lay, TS_SCENES, scale))
        r_in = vessel.r_in
    else:
        parts.append(materials(combo, tex, lay))
        body, r_in = florence_body(vessel, lay, TS_SCENES, scale)
        parts.append(body)
        cfg2 = Setup(**{**cfg.__dict__, "pedestal_gap": 3.05 - (vessel.params["R"] - math.sqrt(vessel.params["R"] ** 2 - vessel.params["ring_r"] ** 2))})
        parts.append(pedestal(cfg2))
    parts.append(liquids_block(vessel, lay, r_in, TS_SCENES, combo.name))
    # ground truth: image rows of each interface at the FRONT inner wall
    gt = {"camera": {"eye": cfg.eye, "look": cfg.look, "fov_short_axis_deg": cfg.fov}, "vessel": vessel.label,
          "inner_height_cm": vessel.inner_height, "layers": []}
    z_bottom = vessel.tb if vessel.kind == "revolved" else vessel.params["R"] - (vessel.params["R"] - vessel.t)
    rows = cfg.project(np.array([[0.0, -r_in(vessel.H - 1e-3), vessel.H], [0.0, -r_in(z_bottom + 1e-3), z_bottom]]))
    gt["rim_row_front"], gt["bottom_row_front"] = float(rows[0, 1]), float(rows[1, 1])
    for i, l in enumerate(lay):
        p = cfg.project(np.array([[0.0, -r_in(l["z1"]), l["z1"]], [0.0, r_in(l["z1"]), l["z1"]]]))
        gt["layers"].append({"liquid": l["name"], "label": l["label"], "eta": l["eta"], "z_top_cm": l["z1"], "z_bottom_cm": l["z0"],
                             "fill_of_inner_height": (l["z1"] - z_bottom) / vessel.inner_height,
                             "top_row_front": float(p[0, 1]), "top_row_back": float(p[1, 1]),
                             "kind": "free surface" if i == len(lay) - 1 else "liquid-liquid interface"})
    return "\n".join(parts) + "\n", gt


def baseline_name(combo: Combo) -> str:
    return f"base_{combo.vessel}_{combo.camera}_{combo.pattern}_{combo.dirt or 'clean'}{combo.dirt_seed}_dz{combo.cam_dz:g}"

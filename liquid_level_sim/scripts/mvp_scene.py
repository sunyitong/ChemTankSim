"""Emit pbrt-v4 scenes for the Phase-0 MVP (docs/RESEARCH.md §4).

Physical modelling notes
------------------------
* No fake air gap between liquid and glass. The inner glass wall is split at the liquid contact
  height: the wetted part is a glass|water dielectric (eta = n_glass / n_water), the dry part is
  glass|air (eta = n_glass). The inner bottom is glass|water, the free surface is water|air.
  Normals of every inner-cavity surface point INTO the cavity (ReverseOrientation on the inner
  cylinder), so "inside" is always the glass and eta > 1 everywhere.
* The liquid volume is a homogeneous absorbing medium ("water"), bounded by the wetted wall,
  the inner bottom and the free surface via MediumInterface.
* Free surface = flat disc + meniscus band, built as one watertight revolved triangle mesh
  (binary PLY) with analytic normals; the meniscus profile is the linearised capillary rise
  z(r) = h_m * exp(-(R - r) / l_c), rescaled to hit exactly 0 at the band's inner edge.
* Light panel: Lambertian emitter. Uniform white uses "rgb L"; patterns use "string filename"
  (spatially varying emission looked up through the mesh uv).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from mvp_common import (FILLS, PATTERNS, PATTERN_DIR, RENDER_DIR, SCENE_DIR, Setup, ensure_dirs,
                        job_name, save_json)


def _f(x: float) -> str:
    return f"{x:.6g}"


# ----------------------------------------------------------------------------- free surface mesh
def write_free_surface_ply(cfg: Setup, path: Path, n_ang: int = 720, n_flat: int = 6, n_men: int = 40) -> dict:
    """Revolved mesh of the liquid free surface at z = 0 (translate to the level in the scene)."""
    R, w, h, lc = cfg.r_in, cfg.meniscus_w, cfg.meniscus_h, cfg.capillary_len
    if cfg.meniscus:
        r_flat = np.linspace(0.0, R - w, n_flat + 1)[1:]                     # exclude r = 0 (centre vertex)
        s = np.linspace(0.0, 1.0, n_men + 1)[1:] ** 1.6                       # denser near the wall
        r_men = (R - w) + w * s
        r = np.concatenate([r_flat, r_men])
        k = math.exp(-w / lc)
        z = np.where(r > R - w, h * (np.exp(-(R - r) / lc) - k) / (1 - k), 0.0)
        dzdr = np.where(r > R - w, (h / lc) * np.exp(-(R - r) / lc) / (1 - k), 0.0)
        z[-1], r[-1] = h, R                                                  # exact contact line
    else:
        r = np.linspace(0.0, R, n_flat + 1)[1:]
        z = np.zeros_like(r)
        dzdr = np.zeros_like(r)

    t = np.linspace(0.0, 2 * math.pi, n_ang, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    K = len(r)
    V = np.zeros((1 + K * n_ang, 3), np.float32)
    N = np.zeros_like(V)
    V[0] = (0, 0, 0)
    N[0] = (0, 0, 1)
    for k in range(K):
        i0 = 1 + k * n_ang
        V[i0:i0 + n_ang, 0] = r[k] * ct
        V[i0:i0 + n_ang, 1] = r[k] * st
        V[i0:i0 + n_ang, 2] = z[k]
        nrm = np.stack([-dzdr[k] * ct, -dzdr[k] * st, np.ones(n_ang)], axis=1)   # up & toward the axis
        N[i0:i0 + n_ang] = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)

    faces = []
    a = np.arange(n_ang)
    b = (a + 1) % n_ang
    # centre fan (CCW seen from +z)
    faces.append(np.stack([np.zeros(n_ang, int), 1 + a, 1 + b], axis=1))
    for k in range(K - 1):
        i0, i1 = 1 + k * n_ang, 1 + (k + 1) * n_ang
        faces.append(np.stack([i0 + a, i1 + a, i1 + b], axis=1))
        faces.append(np.stack([i0 + a, i1 + b, i0 + b], axis=1))
    F = np.concatenate(faces, axis=0).astype(np.int32)

    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        f"element vertex {len(V)}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        f"element face {len(F)}",
        "property list uchar int vertex_indices",
        "end_header", ""])
    vdata = np.concatenate([V, N], axis=1).astype("<f4").tobytes()
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    farr = np.zeros(len(F), fdt)
    farr["n"] = 3
    farr["i"] = F
    blob = header.encode("ascii") + vdata + farr.tobytes()
    if not path.exists() or path.read_bytes() != blob:      # don't touch a file a running render may be reading
        path.write_bytes(blob)
    return {"vertices": int(len(V)), "faces": int(len(F)), "meniscus": cfg.meniscus,
            "contact_height": float(z[-1]) if cfg.meniscus else 0.0}


# ----------------------------------------------------------------------------- scene pieces
def header(cfg: Setup, out_image: str, spp: int, width: int, height: int) -> str:
    ex, ey, ez = cfg.eye
    lx, ly, lz = cfg.look
    return "\n".join([
        f"# MVP liquid-level scene  ->  {out_image}",
        f"LookAt {_f(ex)} {_f(ey)} {_f(ez)}  {_f(lx)} {_f(ly)} {_f(lz)}  0 0 1",
        f'Camera "perspective" "float fov" [{_f(cfg.fov)}]',
        f'Sampler "zsobol" "integer pixelsamples" [{spp}]',
        f'Integrator "volpath" "integer maxdepth" [{cfg.maxdepth}]',
        'PixelFilter "gaussian"',
        f'Film "rgb" "string filename" ["{out_image}"] "integer xresolution" [{width}] '
        f'"integer yresolution" [{height}] "bool savefp16" [false]',
    ])


def panel(cfg: Setup, pattern: str, pattern_rel: str | None) -> str:
    hw, hh = cfg.panel_w / 2, cfg.panel_h / 2
    y, zc = cfg.panel_y, cfg.panel_center_z
    # pbrt is left-handed: world +x is on the image LEFT for this camera, so u=0 at x=+hw.
    P = [(+hw, y, zc - hh), (-hw, y, zc - hh), (-hw, y, zc + hh), (+hw, y, zc + hh)]
    UV = [(0, 0), (1, 0), (1, 1), (0, 1)]
    Ps = " ".join(_f(c) for p in P for c in p)
    UVs = " ".join(_f(c) for uv in UV for c in uv)
    if pattern == "white":
        light = f'AreaLightSource "diffuse" "rgb L" [1 1 1] "float scale" [{_f(cfg.panel_scale)}] "bool twosided" true'
    else:
        light = (f'AreaLightSource "diffuse" "string filename" ["{pattern_rel}"] '
                 f'"float scale" [{_f(cfg.panel_scale)}] "bool twosided" true')
    return "\n".join([
        "AttributeBegin  # light panel (Lambertian light box)",
        f"  {light}",
        '  Material "diffuse" "rgb reflectance" [0 0 0]',
        f'  Shape "trianglemesh" "point3 P" [{Ps}] "point2 uv" [{UVs}] "integer indices" [0 1 2 0 2 3]',
        "AttributeEnd",
    ])


def pedestal(cfg: Setup) -> str:
    z1 = -cfg.pedestal_gap
    z0 = z1 - cfg.pedestal_h
    return "\n".join([
        "AttributeBegin  # matte black pedestal",
        '  Material "diffuse" "rgb reflectance" [0.04 0.04 0.04]',
        f'  Shape "cylinder" "float radius" [{_f(cfg.pedestal_r)}] "float zmin" [{_f(z0)}] "float zmax" [{_f(z1)}]',
        f'  Shape "disk" "float height" [{_f(z1)}] "float radius" [{_f(cfg.pedestal_r)}]',
        "AttributeEnd",
    ])


def container_and_liquid(cfg: Setup, fill: float, ply_rel: str) -> str:
    R, Ro, b, zr = cfg.r_in, cfg.r_out, cfg.bottom, cfg.z_rim
    ng, nw = cfg.eta_glass, cfg.eta_liquid
    L = []
    # ---- air-facing glass surfaces (inside = glass)
    L += ["AttributeBegin  # glass, air-facing surfaces",
          f'  Material "dielectric" "float eta" [{_f(ng)}]',
          f'  Shape "cylinder" "float radius" [{_f(Ro)}] "float zmin" [0] "float zmax" [{_f(zr)}]',
          f'  Shape "disk" "float height" [{_f(zr)}] "float radius" [{_f(Ro)}] "float innerradius" [{_f(R)}]',
          "  AttributeBegin", "    ReverseOrientation",
          f'    Shape "disk" "float height" [0] "float radius" [{_f(Ro)}]',
          "  AttributeEnd"]
    if fill <= 0:
        L += ["  AttributeBegin  # dry inner wall (normal into the cavity)", "    ReverseOrientation",
              f'    Shape "cylinder" "float radius" [{_f(R)}] "float zmin" [{_f(b)}] "float zmax" [{_f(zr)}]',
              "  AttributeEnd",
              f'  Shape "disk" "float height" [{_f(b)}] "float radius" [{_f(R)}]  # dry inner bottom',
              "AttributeEnd"]
        return "\n".join(L)

    zl, zc = cfg.z_level(fill), min(cfg.z_contact(fill), zr)
    L += ["  AttributeBegin  # dry inner wall above the contact line", "    ReverseOrientation",
          f'    Shape "cylinder" "float radius" [{_f(R)}] "float zmin" [{_f(zc)}] "float zmax" [{_f(zr)}]',
          "  AttributeEnd",
          "AttributeEnd"]
    # ---- liquid medium + wetted glass (glass|water) + free surface (water|air)
    sa = " ".join(_f(v) for v in cfg.sigma_a)
    ss = " ".join(_f(v) for v in cfg.sigma_s)
    L += [f'MakeNamedMedium "water" "string type" "homogeneous" "rgb sigma_a" [{sa}] "rgb sigma_s" [{ss}] "float scale" [1]',
          "AttributeBegin  # wetted inner glass: inside = glass, outside (normal side) = water",
          '  MediumInterface "" "water"',
          f'  Material "dielectric" "float eta" [{_f(ng / nw)}]',
          f'  Shape "disk" "float height" [{_f(b)}] "float radius" [{_f(R)}]',
          "  AttributeBegin", "    ReverseOrientation",
          f'    Shape "cylinder" "float radius" [{_f(R)}] "float zmin" [{_f(b)}] "float zmax" [{_f(zc)}]',
          "  AttributeEnd",
          "AttributeEnd",
          "AttributeBegin  # free surface (+ meniscus): inside = water, outside = air",
          '  MediumInterface "water" ""',
          f'  Material "dielectric" "float eta" [{_f(nw)}]',
          f"  Translate 0 0 {_f(zl)}",
          f'  Shape "plymesh" "string filename" ["{ply_rel}"]',
          "AttributeEnd"]
    return "\n".join(L)


def scene_text(cfg: Setup, pattern: str, fill: float, out_image: str, spp: int, width: int, height: int,
               pattern_rel: str | None, ply_rel: str) -> str:
    parts = [header(cfg, out_image, spp, width, height), "", "WorldBegin",
             panel(cfg, pattern, pattern_rel), pedestal(cfg), container_and_liquid(cfg, fill, ply_rel), ""]
    return "\n".join(parts)


def overview_text(cfg: Setup, out_image: str, spp: int, ply_rel: str) -> str:
    """Oblique 3/4 view of the whole setup (camera proxy included) for the report."""
    eye = (-55.0, -75.0, 40.0)
    look = (0.0, -15.0, 10.0)
    cam_proxy = "\n".join([
        "AttributeBegin  # camera proxy (grey box + lens) at the real camera position",
        '  Material "coateddiffuse" "rgb reflectance" [0.25 0.25 0.27] "float roughness" [0.3]',
        f"  Translate 0 {_f(-cfg.cam_dist)} {_f(cfg.cam_z)}",
        '  Shape "trianglemesh" "point3 P" [-2 -3 -1.5  2 -3 -1.5  2 3 -1.5  -2 3 -1.5  -2 -3 1.5  2 -3 1.5  2 3 1.5  -2 3 1.5] '
        '"integer indices" [0 1 2 0 2 3  4 6 5 4 7 6  0 4 5 0 5 1  1 5 6 1 6 2  2 6 7 2 7 3  3 7 4 3 4 0]',
        '  Shape "cylinder" "float radius" [1.2] "float zmin" [3] "float zmax" [5]',
        "AttributeEnd"])
    body = "\n".join([
        f"# MVP setup overview -> {out_image}",
        f"LookAt {_f(eye[0])} {_f(eye[1])} {_f(eye[2])}  {_f(look[0])} {_f(look[1])} {_f(look[2])}  0 0 1",
        'Camera "perspective" "float fov" [42]',
        f'Sampler "zsobol" "integer pixelsamples" [{spp}]',
        f'Integrator "volpath" "integer maxdepth" [{cfg.maxdepth}]',
        'PixelFilter "gaussian"',
        f'Film "rgb" "string filename" ["{out_image}"] "integer xresolution" [1200] "integer yresolution" [900]',
        "", "WorldBegin",
        'LightSource "infinite" "rgb L" [0.08 0.085 0.10]',
        panel(cfg, "rg_checker", "../../outputs/mvp/patterns/rg_checker.png"),
        pedestal(cfg),
        container_and_liquid(cfg, 0.5, ply_rel),
        cam_proxy, ""])
    return body


# ----------------------------------------------------------------------------- driver
VARIANT_PATTERNS = ("white", "rg_checker")
VARIANT_FILLS_NOMEN = (0.50, 0.52)
VARIANT_FILLS_SEED = (0.50,)


def write_scenes(cfg: Setup, patterns=PATTERNS, fills=FILLS, spp: int | None = None,
                 res: tuple[int, int] | None = None, tag: str = "", variants: bool = True) -> dict:
    """Main grid (patterns x fills) plus two control sets when `variants` is on:
      * "seed"        the same scene rendered with another RNG seed  -> Monte-Carlo noise floor
      * "nomeniscus"  flat free surface                               -> how much of the level
                                                                         visibility is the meniscus
    """
    ensure_dirs()
    spp = spp or cfg.spp
    width, height = res or (cfg.width, cfg.height_px)
    ply = SCENE_DIR / "free_surface.ply"
    mesh_info = write_free_surface_ply(cfg, ply)
    jobs = []

    def add(cfg_: Setup, pattern: str, fill: float, name: str, variant: str, ply_name: str, seed: int | None = None):
        pattern_rel = None if pattern == "white" else f"../../outputs/mvp/patterns/{pattern}.png"
        exr = f"{name}.exr"
        scene_path = SCENE_DIR / f"{name}.pbrt"
        if seed is None:                       # seed variants re-use the main scene file
            scene_path.write_text(scene_text(cfg_, pattern, fill, exr, spp, width, height, pattern_rel, ply_name),
                                  encoding="utf-8")
        near, far = cfg_.level_rows(fill)
        j = {"name": name, "pattern": pattern, "fill": fill, "variant": variant,
             "level_cm": fill * cfg_.height, "z_level": cfg_.z_level(fill), "meniscus": cfg_.meniscus,
             "gt_row_near": near, "gt_row_far": far,
             "scene": scene_path.name, "exr": exr, "png": f"{name}.png",
             "spp": spp, "width": width, "height": height}
        if seed is not None:
            j["seed"] = seed
        jobs.append(j)

    for pattern in patterns:
        for fill in fills:
            add(cfg, pattern, fill, job_name(pattern, fill) + tag, "main", ply.name)
    if variants:
        for pattern in (p for p in VARIANT_PATTERNS if p in patterns):
            for fill in (f for f in VARIANT_FILLS_SEED if f in fills):
                base = job_name(pattern, fill) + tag
                j = dict(next(x for x in jobs if x["name"] == base))
                j.update(name=base + "_seed1", variant="seed", seed=1, exr=base + "_seed1.exr", png=base + "_seed1.png")
                jobs.append(j)
        cfg_flat = replace(cfg, meniscus=False)
        ply_flat = SCENE_DIR / "free_surface_flat.ply"
        write_free_surface_ply(cfg_flat, ply_flat)
        for pattern in (p for p in VARIANT_PATTERNS if p in patterns):
            for fill in (f for f in VARIANT_FILLS_NOMEN if f in fills):
                add(cfg_flat, pattern, fill, job_name(pattern, fill) + "_nomen" + tag, "nomeniscus", ply_flat.name)
    ov = SCENE_DIR / f"overview{tag}.pbrt"
    ov.write_text(overview_text(cfg, f"overview{tag}.exr", max(64, spp // 4), ply.name), encoding="utf-8")
    manifest = {"setup": cfg.to_dict(), "mesh": mesh_info, "jobs": jobs,
                "overview": {"scene": ov.name, "exr": f"overview{tag}.exr", "png": f"overview{tag}.png"}}
    save_json(manifest, SCENE_DIR / f"manifest{tag}.json")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spp", type=int, default=None)
    ap.add_argument("--res", default=None, help="WxH, default 1080x1440")
    ap.add_argument("--no-meniscus", action="store_true")
    a = ap.parse_args()
    cfg = Setup(meniscus=not a.no_meniscus)
    res = tuple(int(v) for v in a.res.lower().split("x")) if a.res else None
    m = write_scenes(cfg, spp=a.spp, res=res)
    print(f"wrote {len(m['jobs'])} scenes to {SCENE_DIR}; mesh {m['mesh']}")
    print(f"lens focal {cfg.lens_focal():.2f} cm, predicted horizontal magnification x{cfg.lens_magnification():.2f} (flipped)")
    print("container box (rows/cols):", cfg.container_box())
    for j in m["jobs"][:len(FILLS)]:
        print(f"  fill {j['fill']:.2f}: level {j['level_cm']:.1f} cm -> gt rows near {j['gt_row_near']:.1f} far {j['gt_row_far']:.1f}")

"""Generate pbrt-v4 scenes of a transparent container partially filled with liquid.

For every sample we emit four scene files sharing the same camera:

  <sid>_beauty.pbrt     physically based render (dielectric glass, absorbing liquid, lights)
  <sid>_liquid.pbrt     GT pass: liquid body emits white, everything else is an invisible
                        "interface" surface, no lights, maxdepth 0  -> geometric liquid mask
  <sid>_surface.pbrt    GT pass: only the liquid top surface emits -> free-surface mask
  <sid>_container.pbrt  GT pass: only the glass container emits    -> container ROI mask

The GT passes see through the glass (it is replaced with pbrt's "interface" material),
so the masks describe the *true geometric* projection of the liquid, not the refracted
appearance. An analytic projection of the liquid free-surface rim (using pbrt's own
LookAt / perspective conventions) is stored alongside as a cross-check.

Units: centimetres. z is up; the container stands on the table plane z = 0.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from common import SCENES_DIR, PASSES, ensure_dirs, save_json

# --------------------------------------------------------------------------- presets
LIQUIDS = {
    # name: (eta, sigma_a per cm as RGB). Absorption gives the liquid its colour.
    "water":      (1.333, (0.010, 0.006, 0.003)),
    "tea":        (1.340, (0.060, 0.220, 0.600)),
    "blue_dye":   (1.335, (0.700, 0.250, 0.030)),
    "oil":        (1.470, (0.030, 0.080, 0.500)),
    "milk_dilute":(1.345, (0.020, 0.020, 0.020)),  # slightly turbid, uses sigma_s below
}
LIQUID_SIGMA_S = {"milk_dilute": (0.12, 0.12, 0.12)}

GLASS_ETA = {"soda_lime": 1.52, "borosilicate": 1.47, "acrylic": 1.49}


@dataclass
class Sample:
    sid: str
    container: str            # "cylinder" | "box"
    r_in: float               # cylinder inner radius, or box inner half-width (x)
    half_y: float             # box inner half-depth (y); ignored for cylinder
    wall: float               # wall thickness
    height: float             # inner height (bottom of cavity to rim)
    fill: float               # fill fraction of inner height
    liquid: str
    liquid_eta: float
    sigma_a: tuple
    sigma_s: tuple
    glass: str
    glass_eta: float
    eye: tuple
    look: tuple
    fov: float
    panel_L: float
    sky_L: float
    checker_scale: float
    width: int
    height_px: int
    spp: int
    mask_spp: int = 16
    gap: float = 0.02         # air gap between liquid and glass (avoids coincident surfaces)

    # derived
    liquid_height: float = field(init=False)
    gt_row_analytic: float = field(init=False)   # image row of the *near* edge of the free surface
    gt_row_far_analytic: float = field(init=False)

    def __post_init__(self):
        self.liquid_height = self.fill * self.height
        near, far = self.project_surface_rim()
        self.gt_row_analytic = near
        self.gt_row_far_analytic = far

    # ---------------------------------------------------------------- geometry helpers
    @property
    def z_bottom_inner(self) -> float:
        return self.wall

    @property
    def z_surface(self) -> float:
        return self.wall + self.gap + self.liquid_height

    @property
    def z_rim(self) -> float:
        return self.wall + self.height

    def surface_rim_points(self, n: int = 720) -> np.ndarray:
        z = self.z_surface
        if self.container == "cylinder":
            r = self.r_in - self.gap
            t = np.linspace(0, 2 * math.pi, n, endpoint=False)
            return np.stack([r * np.cos(t), r * np.sin(t), np.full(n, z)], axis=1)
        ax, ay = self.r_in - self.gap, self.half_y - self.gap
        # dense sampling of the rectangle boundary
        pts = []
        for (x0, y0, x1, y1) in [(-ax, -ay, ax, -ay), (ax, -ay, ax, ay), (ax, ay, -ax, ay), (-ax, ay, -ax, -ay)]:
            s = np.linspace(0, 1, n // 4, endpoint=False)
            pts.append(np.stack([x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, np.full(len(s), z)], axis=1))
        return np.concatenate(pts, axis=0)

    def project(self, pts: np.ndarray) -> np.ndarray:
        """World -> raster (px, py) replicating pbrt-v4 LookAt + perspective + screen window."""
        eye = np.array(self.eye, float)
        look = np.array(self.look, float)
        up = np.array([0.0, 0.0, 1.0])
        d = look - eye
        d /= np.linalg.norm(d)
        right = np.cross(up / np.linalg.norm(up), d)
        right /= np.linalg.norm(right)
        new_up = np.cross(d, right)
        rel = pts - eye
        xc = rel @ right
        yc = rel @ new_up
        zc = rel @ d
        inv_tan = 1.0 / math.tan(math.radians(self.fov) / 2)
        xs = xc / zc * inv_tan
        ys = yc / zc * inv_tan
        aspect = self.width / self.height_px
        if aspect > 1:
            xmin, xmax, ymin, ymax = -aspect, aspect, -1.0, 1.0
        else:
            xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0 / aspect, 1.0 / aspect
        px = (xs - xmin) / (xmax - xmin) * self.width
        py = (ymax - ys) / (ymax - ymin) * self.height_px
        return np.stack([px, py], axis=1)

    def project_surface_rim(self) -> tuple[float, float]:
        rp = self.project(self.surface_rim_points())
        return float(rp[:, 1].max()), float(rp[:, 1].min())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passes"] = {p: f"{self.sid}_{p}.pbrt" for p in PASSES}
        return d


# --------------------------------------------------------------------------- pbrt emitters
def _f(x: float) -> str:
    return f"{x:.6g}"


def cylinder(r: float, z0: float, z1: float) -> str:
    return f'Shape "cylinder" "float radius" [{_f(r)}] "float zmin" [{_f(z0)}] "float zmax" [{_f(z1)}]'


def disk(z: float, r: float, r_inner: float = 0.0, flip: bool = False) -> str:
    s = f'Shape "disk" "float height" [{_f(z)}] "float radius" [{_f(r)}] "float innerradius" [{_f(r_inner)}]'
    return f"AttributeBegin\n  ReverseOrientation\n  {s}\nAttributeEnd" if flip else s


def quad(p0, p1, p2, p3) -> str:
    """Two triangles for a quad given CCW (seen from the outside / normal side)."""
    P = " ".join(_f(c) for p in (p0, p1, p2, p3) for c in p)
    return f'Shape "trianglemesh" "point3 P" [{P}] "integer indices" [0 1 2 0 2 3]'


def box_faces(xmin, xmax, ymin, ymax, zmin, zmax, faces=("px", "nx", "py", "ny", "pz", "nz"), inward=False) -> str:
    """Axis-aligned box; normals outward (or inward if inward=True)."""
    c = lambda x, y, z: (x, y, z)  # noqa: E731
    F = {
        "pz": (c(xmin, ymin, zmax), c(xmax, ymin, zmax), c(xmax, ymax, zmax), c(xmin, ymax, zmax)),
        "nz": (c(xmin, ymin, zmin), c(xmin, ymax, zmin), c(xmax, ymax, zmin), c(xmax, ymin, zmin)),
        "py": (c(xmin, ymax, zmin), c(xmin, ymax, zmax), c(xmax, ymax, zmax), c(xmax, ymax, zmin)),
        "ny": (c(xmin, ymin, zmin), c(xmax, ymin, zmin), c(xmax, ymin, zmax), c(xmin, ymin, zmax)),
        "px": (c(xmax, ymin, zmin), c(xmax, ymax, zmin), c(xmax, ymax, zmax), c(xmax, ymin, zmax)),
        "nx": (c(xmin, ymin, zmin), c(xmin, ymin, zmax), c(xmin, ymax, zmax), c(xmin, ymax, zmin)),
    }
    out = []
    for f in faces:
        q = F[f]
        if inward:
            q = (q[0], q[3], q[2], q[1])
        out.append(quad(*q))
    return "\n".join(out)


def rect_frame(xo0, xo1, yo0, yo1, xi0, xi1, yi0, yi1, z) -> str:
    """Flat annular frame between outer and inner rectangles at height z, normal +z."""
    o = lambda x, y: (x, y, z)  # noqa: E731
    parts = [
        quad(o(xo0, yo0), o(xo1, yo0), o(xi1, yi0), o(xi0, yi0)),   # front strip (-y)
        quad(o(xi1, yi0), o(xo1, yo0), o(xo1, yo1), o(xi1, yi1)),   # right strip (+x)
        quad(o(xi0, yi1), o(xi1, yi1), o(xo1, yo1), o(xo0, yo1)),   # back strip (+y)
        quad(o(xo0, yo0), o(xi0, yi0), o(xi0, yi1), o(xo0, yo1)),   # left strip (-x)
    ]
    return "\n".join(parts)


def container_shapes(s: Sample) -> str:
    w, H = s.wall, s.height
    if s.container == "cylinder":
        r_in, r_out = s.r_in, s.r_in + w
        return "\n".join([
            cylinder(r_out, 0.0, H + w),                # outer wall
            cylinder(r_in, w, H + w),                   # inner wall
            disk(H + w, r_out, r_in),                   # rim (annulus)
            disk(0.0, r_out, 0.0, flip=True),           # outer bottom, normal -z
            disk(w, r_in, 0.0),                         # inner bottom, normal +z
        ])
    ax, ay = s.r_in, s.half_y
    return "\n".join([
        box_faces(-ax - w, ax + w, -ay - w, ay + w, 0.0, H + w, faces=("px", "nx", "py", "ny", "nz")),
        box_faces(-ax, ax, -ay, ay, w, H + w + 1e-3, faces=("px", "nx", "py", "ny", "nz"), inward=True),
        rect_frame(-ax - w, ax + w, -ay - w, ay + w, -ax, ax, -ay, ay, H + w),
    ])


def liquid_side_and_bottom(s: Sample) -> str:
    z0, z1 = s.wall + s.gap, s.z_surface
    if s.container == "cylinder":
        r = s.r_in - s.gap
        return "\n".join([cylinder(r, z0, z1), disk(z0, r, 0.0, flip=True)])
    ax, ay = s.r_in - s.gap, s.half_y - s.gap
    return box_faces(-ax, ax, -ay, ay, z0, z1, faces=("px", "nx", "py", "ny", "nz"))


def liquid_top(s: Sample) -> str:
    z1 = s.z_surface
    if s.container == "cylinder":
        return disk(z1, s.r_in - s.gap, 0.0)
    ax, ay = s.r_in - s.gap, s.half_y - s.gap
    return box_faces(-ax, ax, -ay, ay, s.wall + s.gap, z1, faces=("pz",))


EMISSIVE = 'AreaLightSource "diffuse" "rgb L" [1 1 1] "bool twosided" true\n  Material "diffuse" "rgb reflectance" [0 0 0]'
INTERFACE = 'Material "interface"'


def header(s: Sample, mode: str, out_name: str) -> str:
    ex, ey, ez = s.eye
    lx, ly, lz = s.look
    lines = [
        f"# liquid-level study sample {s.sid} pass={mode}",
        f"LookAt {_f(ex)} {_f(ey)} {_f(ez)}  {_f(lx)} {_f(ly)} {_f(lz)}  0 0 1",
        f'Camera "perspective" "float fov" [{_f(s.fov)}]',
    ]
    if mode == "beauty":
        lines += [
            f'Sampler "zsobol" "integer pixelsamples" [{s.spp}]',
            'Integrator "volpath" "integer maxdepth" [40] "bool regularize" [true]',
            'PixelFilter "gaussian"',
            f'Film "rgb" "string filename" ["{out_name}"] "integer xresolution" [{s.width}] '
            f'"integer yresolution" [{s.height_px}] "bool savefp16" [false]',
        ]
    else:
        lines += [
            f'Sampler "independent" "integer pixelsamples" [{s.mask_spp}]',
            'Integrator "path" "integer maxdepth" [0]',
            'PixelFilter "box"',
            f'Film "rgb" "string filename" ["{out_name}"] "integer xresolution" [{s.width}] '
            f'"integer yresolution" [{s.height_px}]',
        ]
    return "\n".join(lines)


def world(s: Sample, mode: str) -> str:
    L = ["WorldBegin"]
    beauty = mode == "beauty"

    # --- lights + environment (beauty only)
    if beauty:
        L.append(f'LightSource "infinite" "rgb L" [{_f(s.sky_L)} {_f(s.sky_L)} {_f(s.sky_L * 1.08)}]')
        L.append("AttributeBegin  # soft key light panel above/front")
        L.append(f'  AreaLightSource "diffuse" "rgb L" [{_f(s.panel_L)} {_f(s.panel_L)} {_f(s.panel_L)}]')
        L.append("  Translate -6 -14 30")
        L.append("  Rotate 155 1 0 0")
        L.append('  Shape "disk" "float radius" [9]')
        L.append("AttributeEnd")
        L.append("AttributeBegin  # dim fill light from the other side")
        L.append(f'  AreaLightSource "diffuse" "rgb L" [{_f(s.panel_L * 0.25)} {_f(s.panel_L * 0.25)} {_f(s.panel_L * 0.28)}]')
        L.append("  Translate 18 -6 20")
        L.append("  Rotate -125 0 1 0")
        L.append('  Shape "disk" "float radius" [6]')
        L.append("AttributeEnd")

    # --- table + backdrop
    L.append("AttributeBegin  # table")
    if beauty:
        cs = s.checker_scale
        L.append(f'  Texture "checks" "spectrum" "checkerboard" "float uscale" [{_f(cs)}] "float vscale" [{_f(cs)}] '
                 '"rgb tex1" [0.72 0.70 0.66] "rgb tex2" [0.18 0.18 0.20]')
        L.append('  Material "coateddiffuse" "texture reflectance" "checks" "float roughness" [0.15]')
    else:
        L.append("  " + INTERFACE)
    L.append("  " + quad((-60, -60, 0), (60, -60, 0), (60, 60, 0), (-60, 60, 0)))
    L.append("AttributeEnd")
    L.append("AttributeBegin  # backdrop wall")
    if beauty:
        L.append('  Texture "stripes" "spectrum" "checkerboard" "float uscale" [24] "float vscale" [1] '
                 '"rgb tex1" [0.55 0.57 0.60] "rgb tex2" [0.30 0.31 0.34]')
        L.append('  Material "diffuse" "texture reflectance" "stripes"')
    else:
        L.append("  " + INTERFACE)
    L.append("  " + quad((-60, 25, 0), (-60, 25, 60), (60, 25, 60), (60, 25, 0)))
    L.append("AttributeEnd")

    # --- glass container
    L.append("AttributeBegin  # container")
    if beauty:
        L.append(f'  Material "dielectric" "float eta" [{_f(s.glass_eta)}]')
    elif mode == "container":
        L.append("  " + EMISSIVE)
    else:
        L.append("  " + INTERFACE)
    L.append(container_shapes(s))
    L.append("AttributeEnd")

    # --- liquid
    if s.liquid_height > 1e-4:
        if beauty:
            sa = " ".join(_f(v) for v in s.sigma_a)
            ss = " ".join(_f(v) for v in s.sigma_s)
            L.append(f'MakeNamedMedium "liq" "string type" "homogeneous" "rgb sigma_a" [{sa}] "rgb sigma_s" [{ss}] "float scale" [1]')
        L.append("AttributeBegin  # liquid side + bottom")
        if beauty:
            L.append('  MediumInterface "liq" ""')
            L.append(f'  Material "dielectric" "float eta" [{_f(s.liquid_eta)}]')
        elif mode == "liquid":
            L.append("  " + EMISSIVE)
        else:
            L.append("  " + INTERFACE)
        L.append(liquid_side_and_bottom(s))
        L.append("AttributeEnd")
        L.append("AttributeBegin  # liquid free surface")
        if beauty:
            L.append('  MediumInterface "liq" ""')
            L.append(f'  Material "dielectric" "float eta" [{_f(s.liquid_eta)}]')
        elif mode in ("liquid", "surface"):
            L.append("  " + EMISSIVE)
        else:
            L.append("  " + INTERFACE)
        L.append(liquid_top(s))
        L.append("AttributeEnd")
    return "\n".join(L)


def write_sample(s: Sample, out_dir: Path) -> dict:
    files = {}
    for mode in PASSES:
        img_name = f"{s.sid}_{mode}.exr" if mode == "beauty" else f"{s.sid}_{mode}.png"
        text = header(s, mode, img_name) + "\n\n" + world(s, mode) + "\n"
        p = out_dir / f"{s.sid}_{mode}.pbrt"
        p.write_text(text, encoding="utf-8")
        files[mode] = p.name
    meta = s.to_dict()
    save_json(meta, out_dir / f"{s.sid}.json")
    return meta


# --------------------------------------------------------------------------- sampling
def random_sample(idx: int, rng: random.Random, containers: list[str], width: int, height_px: int,
                  spp: int, mask_spp: int, fill: float | None = None) -> Sample:
    container = containers[idx % len(containers)]
    liquid = rng.choice(list(LIQUIDS))
    eta, sigma_a = LIQUIDS[liquid]
    glass = rng.choice(list(GLASS_ETA))
    H = rng.uniform(8.0, 12.0)
    r_in = rng.uniform(2.6, 4.0)
    half_y = rng.uniform(1.8, 3.2) if container == "box" else r_in
    wall = rng.uniform(0.18, 0.32)
    f = fill if fill is not None else rng.uniform(0.08, 0.92)

    # camera: in front (-y), slightly above the rim, mild azimuth
    dist = rng.uniform(26.0, 40.0)
    az = math.radians(rng.uniform(-28.0, 28.0))
    ez = rng.uniform(H + wall + 1.5, H + wall + 10.0)
    eye = (dist * math.sin(az), -dist * math.cos(az), ez)
    look = (rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), (H + wall) * rng.uniform(0.42, 0.58))
    fov = rng.uniform(22.0, 32.0)
    return Sample(
        sid=f"s{idx:03d}", container=container, r_in=r_in, half_y=half_y, wall=wall, height=H, fill=f,
        liquid=liquid, liquid_eta=eta, sigma_a=sigma_a, sigma_s=LIQUID_SIGMA_S.get(liquid, (0.0, 0.0, 0.0)),
        glass=glass, glass_eta=GLASS_ETA[glass], eye=eye, look=look, fov=fov,
        panel_L=rng.uniform(10.0, 20.0), sky_L=rng.uniform(0.30, 0.65), checker_scale=rng.uniform(14.0, 30.0),
        width=width, height_px=height_px, spp=spp, mask_spp=mask_spp,
    )


def generate(n: int, seed: int, containers: list[str], width: int, height_px: int, spp: int, mask_spp: int,
             fill_sweep: bool = False) -> list[dict]:
    ensure_dirs()
    rng = random.Random(seed)
    metas = []
    for i in range(n):
        fill = None
        if fill_sweep:  # deterministic, evenly spaced fill levels for controlled experiments
            fill = 0.08 + 0.84 * i / max(n - 1, 1)
        s = random_sample(i, rng, containers, width, height_px, spp, mask_spp, fill)
        metas.append(write_sample(s, SCENES_DIR))
    save_json(metas, SCENES_DIR / "dataset.json")
    return metas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--containers", default="cylinder,box")
    ap.add_argument("--res", default="640x480")
    ap.add_argument("--spp", type=int, default=128)
    ap.add_argument("--mask-spp", type=int, default=16)
    ap.add_argument("--fill-sweep", action="store_true", help="evenly spaced fill fractions instead of random")
    a = ap.parse_args()
    w, h = (int(v) for v in a.res.lower().split("x"))
    metas = generate(a.n, a.seed, a.containers.split(","), w, h, a.spp, a.mask_spp, a.fill_sweep)
    print(f"wrote {len(metas)} samples x {len(PASSES)} passes to {SCENES_DIR}")
    for m in metas:
        print(f"  {m['sid']}: {m['container']:8s} fill={m['fill']:.3f} liquid={m['liquid']:12s} "
              f"gt_row(analytic)={m['gt_row_analytic']:.1f}")


if __name__ == "__main__":
    main()

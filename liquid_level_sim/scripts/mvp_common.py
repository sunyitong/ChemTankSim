"""Shared configuration for the Phase-0 MVP (see docs/RESEARCH.md §4).

Scene: a 15 cm (inner diameter) x 20 cm (inner height) glass cylinder standing on a small dark
pedestal, a horizontal pinhole camera in front of the side-wall midpoint, and a large Lambertian
light panel behind the cylinder that fills the whole camera frame. Units: centimetres, z up,
cylinder axis = z axis, outer bottom face of the container at z = 0.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from common import IMGTOOL_EXE, PBRT_EXE, PROJ_ROOT, check_pbrt, load_json, run, save_json  # noqa: F401

MVP_OUT = PROJ_ROOT / "outputs" / "mvp"
PATTERN_DIR = MVP_OUT / "patterns"
RENDER_DIR = MVP_OUT / "renders"
COMPARE_DIR = MVP_OUT / "compare"
SCENE_DIR = PROJ_ROOT / "scenes" / "mvp"
MANIFEST = MVP_OUT / "manifest.json"

PATTERNS = ("white", "rg_checker", "mosaic", "grid")
FILLS = (0.0, 0.25, 0.50, 0.52, 0.75)
SENS_PAIR = (0.50, 0.52)          # 4 mm level change used for the sensitivity test


def fill_tag(f: float) -> str:
    return f"f{int(round(f * 100)):03d}"


def job_name(pattern: str, fill: float) -> str:
    return f"{pattern}_{fill_tag(fill)}"


def ensure_dirs() -> None:
    for d in (MVP_OUT, PATTERN_DIR, RENDER_DIR, COMPARE_DIR, SCENE_DIR):
        d.mkdir(parents=True, exist_ok=True)


@dataclass
class Setup:
    # ---- container (cm)
    r_in: float = 7.5            # inner radius  (15 cm inner diameter)
    wall: float = 0.3            # side-wall thickness
    bottom: float = 0.5          # bottom-glass thickness
    height: float = 20.0         # inner height (inner bottom -> rim)
    eta_glass: float = 1.50
    # ---- liquid
    eta_liquid: float = 1.333
    sigma_a: tuple = (0.0035, 0.0015, 0.0003)   # pure water, per cm (R,G,B)
    sigma_s: tuple = (0.0, 0.0, 0.0)
    meniscus: bool = True
    meniscus_h: float = 0.30     # rise at the wall (cm)
    capillary_len: float = 0.27  # water capillary length (cm)
    meniscus_w: float = 1.35     # radial width of the meniscus band (= 5 capillary lengths)
    # ---- camera (horizontal, in front of the side-wall midpoint)
    cam_dist: float = 60.0       # from the cylinder axis, along -y
    cam_z: float | None = None   # default: mid-height of the container's outer extent
    fov: float = 24.0            # pbrt fov = angle across the SHORTER image axis (width, portrait)
    width: int = 1080
    height_px: int = 1440
    # ---- light panel (Lambertian emitter, fills the frame)
    panel_y: float = 25.0        # distance behind the axis
    panel_w: float = 50.0
    panel_h: float = 60.0
    panel_scale: float = 0.8     # radiance scale (white -> ~0.8 linear in the image)
    # ---- pedestal (matte black) so the panel stays visible below the container
    pedestal_r: float = 4.0
    pedestal_h: float = 8.0
    pedestal_gap: float = 0.05
    # ---- rendering
    spp: int = 1024
    maxdepth: int = 64

    def __post_init__(self):
        if self.cam_z is None:
            self.cam_z = self.z_rim / 2.0

    # ---------------------------------------------------------------- derived geometry
    @property
    def r_out(self) -> float:
        return self.r_in + self.wall

    @property
    def z_rim(self) -> float:
        return self.bottom + self.height

    @property
    def eye(self) -> tuple:
        return (0.0, -self.cam_dist, self.cam_z)

    @property
    def look(self) -> tuple:
        return (0.0, 0.0, self.cam_z)

    @property
    def panel_center_z(self) -> float:
        return self.cam_z

    def z_level(self, fill: float) -> float:
        return self.bottom + fill * self.height

    def z_contact(self, fill: float) -> float:
        """Height at which the liquid meets the inner wall (top of the meniscus)."""
        return self.z_level(fill) + (self.meniscus_h if self.meniscus else 0.0)

    # analytic thick-lens estimate (docs/RESEARCH.md §3.2), ignoring the thin glass wall
    def lens_focal(self) -> float:
        n = self.eta_liquid
        return n * self.r_in / (2.0 * (n - 1.0))

    def lens_magnification(self) -> float:
        f = self.lens_focal()
        return f / (self.panel_y - f)

    # ---------------------------------------------------------------- pbrt-consistent projection
    def project(self, pts: np.ndarray) -> np.ndarray:
        """World -> raster (px, py), replicating pbrt-v4 LookAt + perspective + screen window.
        NOTE pbrt is left-handed: with this camera, world +x appears on the image LEFT."""
        eye = np.array(self.eye, float)
        look = np.array(self.look, float)
        up = np.array([0.0, 0.0, 1.0])
        d = look - eye
        d /= np.linalg.norm(d)
        right = np.cross(up, d)
        right /= np.linalg.norm(right)
        new_up = np.cross(d, right)
        rel = np.atleast_2d(pts) - eye
        xc, yc, zc = rel @ right, rel @ new_up, rel @ d
        inv_tan = 1.0 / math.tan(math.radians(self.fov) / 2)
        xs, ys = xc / zc * inv_tan, yc / zc * inv_tan
        aspect = self.width / self.height_px
        if aspect > 1:
            xmin, xmax, ymin, ymax = -aspect, aspect, -1.0, 1.0
        else:
            xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0 / aspect, 1.0 / aspect
        px = (xs - xmin) / (xmax - xmin) * self.width
        py = (ymax - ys) / (ymax - ymin) * self.height_px
        return np.stack([px, py], axis=1)

    def level_rows(self, fill: float) -> tuple[float, float]:
        """Image rows of the free-surface edge at the front inner wall (near) and back wall (far)."""
        z = self.z_level(fill)
        p = self.project(np.array([[0.0, -self.r_in, z], [0.0, self.r_in, z]]))
        return float(p[0, 1]), float(p[1, 1])

    def container_box(self) -> tuple[int, int, int, int]:
        """(row_top, row_bottom, col_left, col_right) of the container silhouette."""
        # vertical extent: rim / base at the front outer wall; horizontal: silhouette tangents
        pts = np.array([[0.0, -self.r_out, self.z_rim], [0.0, -self.r_out, 0.0]])
        p = self.project(pts)
        top, bot = p[:, 1].min(), p[:, 1].max()
        ang = math.asin(self.r_out / self.cam_dist)
        half = math.tan(ang) / math.tan(math.radians(self.fov) / 2) * (self.width / 2)
        cx = self.width / 2
        # far-wall rim is higher in the image than the near rim; include it
        p_far = self.project(np.array([[0.0, self.r_out, self.z_rim]]))
        top = min(top, p_far[0, 1])
        return int(math.floor(top)), int(math.ceil(bot)), int(math.floor(cx - half)), int(math.ceil(cx + half))

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(r_out=self.r_out, z_rim=self.z_rim, eye=self.eye, look=self.look,
                 lens_focal_cm=self.lens_focal(), lens_magnification=self.lens_magnification(),
                 container_box=self.container_box())
        return d

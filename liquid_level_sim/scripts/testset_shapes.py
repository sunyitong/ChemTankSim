"""Container geometry for the test set: revolved glass vessels with a real wall thickness.

A vessel is described by its OUTER side profile r_out(z) (z from the outer bottom face at 0 to the
rim at H) and a wall thickness t. The inner side profile is the normal offset of the outer one,
r_in(z) = r_out(z) - t * sqrt(1 + r_out'(z)^2), which is exact for slanted walls as long as the
profile has no horizontal tangent (|dr/dz| is kept below ~1.5 in all profiles here). Flat-bottom
vessels have an inner bottom face at z = tb. The round-bottom flask is built from pbrt's analytic
spheres and cylinders instead (see testset_scenes.florence_pieces).

Meshes are written as binary PLY with analytic normals and (u, v) = (phi / 2pi, z / H) so a 2-D
dirt texture wraps around the vessel consistently on every piece of the wall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------- profiles
def gaussian_bulbs(base: float, bulbs: list[tuple[float, float, float]]) -> Callable[[float], float]:
    """r(z) = base + sum A * exp(-((z - zc) / w)^2): smooth, single-valued, finite slope."""
    def r(z):
        return base + sum(A * math.exp(-((z - zc) / w) ** 2) for A, zc, w in bulbs)
    return r


def smoothstep(a: float, b: float, x: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    t = (x - a) / (b - a)
    return t * t * (3 - 2 * t)


def erlenmeyer_profile(r_base=6.5, r_neck=2.1, z_shoulder=14.0, blend=1.6):
    """Cone from the base to the shoulder, smooth fillet into a cylindrical neck."""
    def r(z):
        cone = r_base + (r_neck - r_base) * min(z, z_shoulder) / z_shoulder
        s = smoothstep(z_shoulder - blend, z_shoulder + blend, z)
        return cone * (1 - s) + r_neck * s
    return r


def reagent_profile(r_body=5.0, r_neck=2.8, z_body=13.0, z_neck=15.5):
    """Straight body, cosine shoulder into a wide neck."""
    def r(z):
        if z <= z_body:
            return r_body
        if z >= z_neck:
            return r_neck
        t = (z - z_body) / (z_neck - z_body)
        return r_neck + (r_body - r_neck) * 0.5 * (1 + math.cos(math.pi * t))
    return r


@dataclass
class Vessel:
    name: str
    r_out: Callable[[float], float]
    H: float                       # rim height (outer bottom face at z = 0)
    t: float = 0.25                # side-wall thickness
    tb: float = 0.5                # bottom thickness
    label: str = ""
    kind: str = "revolved"         # "revolved" | "florence"
    params: dict = field(default_factory=dict)

    def slope(self, z: float, h: float = 1e-3) -> float:
        return (self.r_out(min(z + h, self.H)) - self.r_out(max(z - h, 0.0))) / (min(z + h, self.H) - max(z - h, 0.0))

    def r_in(self, z: float) -> float:
        return max(0.3, self.r_out(z) - self.t * math.sqrt(1 + self.slope(z) ** 2))

    @property
    def inner_height(self) -> float:
        return self.H - self.tb

    def z_level(self, fill: float) -> float:
        return self.tb + fill * self.inner_height

    def volume_ml(self, z_top: float, n: int = 400) -> float:
        zs = np.linspace(self.tb, z_top, n)
        rs = np.array([self.r_in(z) for z in zs])
        return float(np.trapezoid(math.pi * rs ** 2, zs))


VESSELS = {
    # gourd / double-bulb flask: wide lower bulb, waist, smaller upper bulb, short neck
    "gourd": Vessel("gourd", gaussian_bulbs(2.3, [(5.0, 4.6, 3.9), (2.9, 15.4, 2.7)]), H=21.0, t=0.25, tb=0.5,
                    label="Gourd flask"),
    "erlenmeyer": Vessel("erlenmeyer", erlenmeyer_profile(), H=18.5, t=0.22, tb=0.45, label="Erlenmeyer 1 L"),
    "reagent": Vessel("reagent", reagent_profile(), H=18.5, t=0.28, tb=0.6, label="Wide-mouth reagent bottle"),
    "florence": Vessel("florence", lambda z: 0.0, H=20.0, t=0.22, tb=0.0, label="Round-bottom flask",
                       kind="florence", params={"R": 6.3, "r_neck": 1.75, "ring_r": 4.0}),
}


# ----------------------------------------------------------------------------- mesh writers
def write_ply(path: Path, V: np.ndarray, N: np.ndarray, UV: np.ndarray, F: np.ndarray) -> None:
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {len(V)}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float u", "property float v",
        f"element face {len(F)}", "property list uchar int vertex_indices", "end_header", ""])
    vdata = np.concatenate([V, N, UV], axis=1).astype("<f4").tobytes()
    fdt = np.dtype([("n", "u1"), ("i", "<i4", (3,))])
    farr = np.zeros(len(F), fdt)
    farr["n"] = 3
    farr["i"] = F.astype(np.int32)
    blob = header.encode("ascii") + vdata + farr.tobytes()
    if not path.exists() or path.read_bytes() != blob:
        path.write_bytes(blob)


def revolve(profile: Callable[[float], float], z0: float, z1: float, path: Path, outward: bool,
            v_scale: float, n_z: int = 96, n_ang: int = 256) -> None:
    """Surface of revolution of r = profile(z), z in [z0, z1]. Normals point away from the axis when
    `outward` (outer wall) or toward it (inner wall, so 'inside' is the glass on both)."""
    zs = np.linspace(z0, z1, n_z)
    rs = np.array([profile(z) for z in zs])
    drdz = np.gradient(rs, zs)
    t = np.linspace(0, 2 * math.pi, n_ang, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    V = np.zeros((n_z * n_ang, 3), np.float32)
    N = np.zeros_like(V)
    UV = np.zeros((n_z * n_ang, 2), np.float32)
    for k in range(n_z):
        i0 = k * n_ang
        V[i0:i0 + n_ang, 0] = rs[k] * ct
        V[i0:i0 + n_ang, 1] = rs[k] * st
        V[i0:i0 + n_ang, 2] = zs[k]
        nrm = np.stack([ct, st, np.full(n_ang, -drdz[k])], axis=1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        N[i0:i0 + n_ang] = nrm if outward else -nrm
        UV[i0:i0 + n_ang, 0] = t / (2 * math.pi)
        UV[i0:i0 + n_ang, 1] = zs[k] * v_scale
    a = np.arange(n_ang)
    b = (a + 1) % n_ang
    faces = []
    for k in range(n_z - 1):
        i0, i1 = k * n_ang, (k + 1) * n_ang
        faces.append(np.stack([i0 + a, i1 + a, i1 + b], axis=1))
        faces.append(np.stack([i0 + a, i1 + b, i0 + b], axis=1))
    write_ply(path, V, N, UV, np.concatenate(faces))


def free_surface(path: Path, R_level: float, r_contact: float, h_m: float, l_c: float, path_flat: bool = False,
                 n_ang: int = 512, n_flat: int = 6, n_men: int = 32) -> None:
    """Liquid free surface at z = 0 (translate to the level): flat disc of radius R_level - w plus a
    meniscus band rising by h_m to the wall contact radius r_contact (wall may be slanted)."""
    w = min(1.35, 0.35 * R_level) if h_m > 0 else 0.0
    r_flat = np.linspace(0.0, R_level - w, n_flat + 1)[1:]
    if h_m > 0:
        s = np.linspace(0.0, 1.0, n_men + 1)[1:]
        k = math.exp(-w / l_c)
        r_men = (R_level - w) + (r_contact - (R_level - w)) * s
        z_men = h_m * (np.exp(-(1 - s) * w / l_c) - k) / (1 - k)
        r = np.concatenate([r_flat, r_men]); z = np.concatenate([np.zeros_like(r_flat), z_men])
    else:
        r, z = r_flat, np.zeros_like(r_flat)
    dz = np.gradient(z, r) if len(r) > 1 else np.zeros_like(r)
    t = np.linspace(0, 2 * math.pi, n_ang, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    K = len(r)
    V = np.zeros((1 + K * n_ang, 3), np.float32); N = np.zeros_like(V); UV = np.zeros((len(V), 2), np.float32)
    N[0] = (0, 0, 1)
    for k in range(K):
        i0 = 1 + k * n_ang
        V[i0:i0 + n_ang, 0] = r[k] * ct; V[i0:i0 + n_ang, 1] = r[k] * st; V[i0:i0 + n_ang, 2] = z[k]
        nrm = np.stack([-dz[k] * ct, -dz[k] * st, np.ones(n_ang)], axis=1)
        N[i0:i0 + n_ang] = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    a = np.arange(n_ang); b = (a + 1) % n_ang
    faces = [np.stack([np.zeros(n_ang, int), 1 + a, 1 + b], axis=1)]
    for k in range(K - 1):
        i0, i1 = 1 + k * n_ang, 1 + (k + 1) * n_ang
        faces.append(np.stack([i0 + a, i1 + a, i1 + b], axis=1))
        faces.append(np.stack([i0 + a, i1 + b, i0 + b], axis=1))
    write_ply(path, V, N, UV, np.concatenate(faces))


def flat_disc(path: Path, R: float, n_ang: int = 256) -> None:
    free_surface(path, R, R, 0.0, 1.0, n_ang=n_ang)


if __name__ == "__main__":
    for v in VESSELS.values():
        if v.kind != "revolved":
            continue
        zs = np.linspace(0, v.H, 9)
        print(f"{v.name:12s} H={v.H} r_out: " + " ".join(f"{v.r_out(z):.2f}" for z in zs))
        print(f"{'':12s} r_in : " + " ".join(f"{v.r_in(z):.2f}" for z in zs) + f"  max|slope|={max(abs(v.slope(z)) for z in np.linspace(0.05, v.H - 0.05, 400)):.2f}"
              f"  volume={v.volume_ml(v.H):.0f} ml")

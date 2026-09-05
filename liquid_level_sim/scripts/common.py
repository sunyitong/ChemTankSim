"""Shared paths and helpers for the liquid-level rendering study."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]          # liquid_level_sim/
MATTING_ROOT = PROJ_ROOT.parent                          # Matting/
PBRT_BUILD = MATTING_ROOT / "pbrt-v4" / "build"
PBRT_EXE = PBRT_BUILD / "pbrt.exe"
IMGTOOL_EXE = PBRT_BUILD / "imgtool.exe"

SCENES_DIR = PROJ_ROOT / "scenes" / "generated"
OUT_DIR = PROJ_ROOT / "outputs"
RENDER_DIR = OUT_DIR / "renders"
MASK_DIR = OUT_DIR / "masks"
PREVIEW_DIR = OUT_DIR / "previews"

# Four render passes per sample. "beauty" is the physically based image,
# the others are emission-only ground-truth passes (see gen_scenes.py).
PASSES = ("beauty", "liquid", "surface", "container")


def ensure_dirs() -> None:
    for d in (SCENES_DIR, RENDER_DIR, MASK_DIR, PREVIEW_DIR):
        d.mkdir(parents=True, exist_ok=True)


def check_pbrt() -> None:
    if not PBRT_EXE.exists():
        sys.exit(f"pbrt.exe not found at {PBRT_EXE}. Run Matting\\build_pbrt.bat first.")


def run(cmd: list[str], cwd: Path | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, check=True,
                          stdout=subprocess.PIPE if quiet else None,
                          stderr=subprocess.STDOUT if quiet else None, text=True)


def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p: Path) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

"""Environment-light description shared by the renderers (see env_map.py for the HDRI conversion)."""
from __future__ import annotations

import json
from pathlib import Path

from mvp_common import PROJ_ROOT

ENV_DIR = PROJ_ROOT / "outputs" / "env"
REL_ENV = "../../outputs/env"            # relative to scenes/testset/
TARGET_MEAN = 0.28                        # mean environment radiance relative to the panel's 0.8


def env_light(env_id: str | None = None, rotate: float = 0.0, scale_mult: float = 1.0) -> dict:
    """tex["env"] entry for build_scene: file (relative to the scene dir), scale, rotation about z (deg)."""
    metas = sorted(ENV_DIR.glob("*_meta*.json")) if env_id is None else []
    meta_path = ENV_DIR / f"{env_id}_meta.json" if env_id else (metas[-1] if metas else ENV_DIR / "env_meta.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    return {"file": f"{REL_ENV}/{meta['equiarea']}", "scale": TARGET_MEAN / meta["mean_luminance"] * scale_mult,
            "rotate": rotate, "id": meta["id"]}

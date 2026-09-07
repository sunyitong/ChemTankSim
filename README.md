# ChemTankSim — liquid-level detection from refraction of a backlit pattern

Synthetic-data study of reading the liquid level in a transparent vessel from how the liquid
refracts a structured light panel behind it, rendered with a physically based renderer
(pbrt-v4), plus a browser tool that compares an empty-vessel baseline with a filled photo.

**Live demo (GitHub Pages):** `https://sunyitong.github.io/ChemTankSim/` — the Level Diff Bench
web app with 25 bundled test cases. Gallery of renders: `https://sunyitong.github.io/ChemTankSim/gallery.html`

## What is here

| Path | Content |
|---|---|
| `liquid_level_sim/docs/RESEARCH.md` | Research plan, physics, experiment log, limescale morphology references (Chinese) |
| `liquid_level_sim/scripts/mvp_*.py` | Phase 0: 15 × 20 cm cylinder, four backlight patterns, five levels, diff / sensitivity / ray-trace analysis |
| `liquid_level_sim/scripts/testset_*.py` | Phase 1: gourd / Erlenmeyer / reagent / round-bottom vessels, wide–tele lenses, layered liquids, sensor-noise post-processing, evaluation |
| `liquid_level_sim/scripts/scale_model.py`, `scale_run.py` | Physically motivated limescale deposits (water-line ring, creep, film, tide marks, droplets, drips) |
| `liquid_level_sim/scripts/env_map.py`, `env_light.py`, `lab_run.py` | Laboratory HDRI environment light (Poly Haven CC0, equal-area map for pbrt-v4) and the K group: two more vessels with a steel fixture, limescale and a changed room light in the filled frame |
| `liquid_level_sim/scripts/real_cases.py` | First real footage: a phone photo pair (empty / filled tumbler) and a 43 s pouring video in front of a backlit checker panel, cropped to a portrait window, sampled and embedded as cases P1 / P2 (no ground truth) |
| `liquid_level_sim/scripts/inserts_run.py`, `pour_video.py`, `eval_video.py` | Fixtures that partly occlude the pattern (stirrer shaft, thermometer, dosing tube) and three 20 fps pouring sequences (cylinder; Erlenmeyer and reagent bottle with limescale, a dosing tube through the neck and the lab HDRI) with per-frame evaluation and a temporal tracker |
| `liquid_level_sim/scripts/level_detect.py`, `eval_all.py` | Level detector v3 (geometry first: per-row refraction-warp jumps, identity-plateau end, contact-line dip; deposits are photometric only and ignored) and its multi-resolution evaluation |
| `liquid_level_sim/webapp/level_diff_bench.html` | Single-file web app: ROI box, difference heat map / overlay, adaptive robust diff, v3 multi-level detection (same code as `level_detect.py`), 37 bundled cases (35 synthetic with ground truth, a real photo pair and a real pouring video), video mode for four sequences (real-time frame-by-frame detection, multi-hypothesis temporal tracker that holds rather than extrapolates, level-vs-time trace), single / multi-level mode preset per case |
| `liquid_level_sim/outputs/**/images`, `compare`, `eval` | Rendered PNG datasets, contact sheets, metrics, reports (EXR originals are not tracked) |
| `docs/` | GitHub Pages site (`index.html` = web app, `gallery.html` = renders) |

## Reproduce

1. Clone pbrt-v4 into `./pbrt-v4` (`git clone https://github.com/mmp/pbrt-v4 && cd pbrt-v4 && git checkout 5f7a606 && git submodule update --init --recursive`)
   and build it with `build_pbrt.bat` (MSVC 2022 + Ninja; CPU only).
2. `cd liquid_level_sim && uv venv --python 3.11 .venv && uv pip install numpy opencv-python matplotlib jinja2 ninja imageio`
3. Phase 0: `.venv\Scripts\python scripts\mvp_run.py --spp 512` → `outputs/mvp/report.html`
4. Phase 1 test set: `scripts\testset_run.py`, evaluation `scripts\testset_eval.py`, limescale samples `scripts\scale_run.py`
5. Detector evaluation: `scripts\eval_all.py --scales=1,0.85,0.75,0.65,0.55 outputs/testset/cases.json outputs/scale/cases.json` → `outputs/eval_all/eval_all.md`
6. Environment light: `scripts\env_map.py --id childrens_hospital`; fixtures, K group and video: `scripts\inserts_run.py --env childrens_hospital`, `scripts\lab_run.py`, `scripts\pour_video.py` (→ `outputs/pour/<seq>/*.mp4`), `scripts\eval_video.py outputs\pour\<seq>\video_meta.json`
7. Web app bundle: `scripts\webapp_bundle.py <out.js> outputs/testset/cases.json outputs/scale/cases.json outputs/inserts/cases.json outputs/lab/cases.json outputs/pour/cases_video.json` then `scripts\webapp_build.py <template> <out.js> <fragment>`

Details, parameters and findings: [liquid_level_sim/README.md](liquid_level_sim/README.md) and
[liquid_level_sim/docs/RESEARCH.md](liquid_level_sim/docs/RESEARCH.md).

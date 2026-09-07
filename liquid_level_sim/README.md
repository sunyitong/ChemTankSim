# 透明容器液位视觉识别 —— pbrt-v4 物理渲染合成数据

研究计划、假设与路线图见 **[docs/RESEARCH.md](docs/RESEARCH.md)**（总纲，按阶段追加实验记录）。
渲染器为本仓库 `../pbrt-v4/build/pbrt.exe`（CPU 版，见 `../build_pbrt.bat`）。

## 目录

```
liquid_level_sim/
├─ docs/RESEARCH.md        研究计划 / 物理原理 / 域随机化设计 / 路线图 / 实验记录
├─ .venv/                  uv 创建的 Python 3.11 环境（numpy, opencv, matplotlib, jinja2, ninja, imageio）
├─ scripts/
│  ├─ common.py            路径常量（pbrt.exe / imgtool.exe）
│  │  --- Phase 0 MVP：15×20 cm 圆柱 + 水平相机 + 发光板 ---
│  ├─ mvp_common.py        Setup 数据类（几何、相机、板、渲染参数）+ pbrt 一致的投影 / 真值行
│  ├─ mvp_patterns.py      发光板图案贴图：white / rg_checker / mosaic / grid
│  ├─ mvp_scene.py         pbrt 场景生成（嵌套介质无空气隙、弯月面 PLY 网格、贴图发光板、对照组）
│  ├─ mvp_render.py        批量渲染 + EXR→PNG
│  ├─ mvp_compare.py       差分 / 灵敏度 / 噪声底 / 弯月面对照 / 棋盘周期 → 图 + metrics.json + report.html
│  ├─ mvp_run.py           一键：patterns → scenes → render → compare
│  │  --- Phase 1 测试集：异形容器 + 壁面脏物 + 镜头变化 + 分层液体 ---
│  ├─ testset_shapes.py     旋转体容器剖面（葫芦瓶 / 锥形瓶 / 广口试剂瓶 / 圆柱）、法线偏置内壁、PLY 网格、液面+弯月面
│  ├─ testset_textures.py   程序化水垢/污渍贴图：覆盖率、颜色、粗糙度、内壁水线环
│  ├─ testset_scenes.py     组合定义 COMBOS（A1–A6 单液体、B1–B4 分层液体）、材质（mix 玻璃/沉积）、圆底烧瓶解析球面
│  ├─ testset_post.py       相机后处理：曝光抖动、亚像素平移、散粒+读出噪声
│  ├─ testset_run.py        一键：图案 → 贴图 → 场景 → 渲染 → 后处理 → 拼图 + manifest.json（含真值行）
│  ├─ level_detect.py       液位检测 v3（几何优先：逐行折射放大率跳变 + 恒等平台终点 + 接触线暗行；网页 JS 与之同源）
│  ├─ eval_all.py           全部案例 × 多工作分辩率评估 → outputs/eval_all/{eval_all.md, eval_all.json, profiles_*.png}
│  ├─ inserts_run.py        夹具部分遮挡样本（搅拌轴 / 温度计 / 加液管）→ outputs/inserts/ （网页 R 组）
│  ├─ pour_video.py         注液序列：20 fps × 4 s 液位上升 + 液柱/液滴 → outputs/pour/F1_cyl_pour.mp4 + video_meta.json（网页 F 组）
│  ├─ eval_video.py         注液序列逐帧评估 + 多假说时序跟踪 → outputs/pour/{eval_video.json, level_vs_time.png}
│  ├─ env_map.py / env_light.py  实验室 HDRI（Poly Haven CC0）→ pbrt 等面积环境贴图；scripts 通过 --env <id> 使用
│  ├─ lab_run.py            K 组：试剂瓶 / 葫芦瓶 + 金属棒 + 水垢 + 实验室 HDRI（对比帧环境光旋转、减弱）→ outputs/lab/
│  │  --- 旧管线（随机烧杯 + GT 掩膜通道，保留供 Phase 1 复用） ---
│  ├─ gen_scenes.py / render.py / analyze.py / build_report.py / run_all.py
├─ webapp/level_diff_bench.html   单文件网页应用（英文 UI）：上传空容器基准图 + 有液位对比图，框选 ROI，
│                                 差异热图 / 叠加模式，右侧 0–100% 刻度与液位估计；Adaptive 鲁棒差分（热图）；
│                                 液位检测 v3（几何优先，见 level_detect.py）：顶面 + 液-液界面，模式标签 WARP / IDENTITY-END / PHOTOMETRIC；
│                                 顶栏按钮内嵌 33 组案例（A 容器/镜头、B 分层液体、S/V 水垢、R 夹具遮挡、K 金属 + 水垢 + HDRI、F 注液视频；含真值与误差显示）；
│                                 视频模式：原速播放逐帧检测 + 多假说时序跟踪、进度条/逐帧步进、液位-时间轨迹；纯前端，双击直接打开
│                                 重建：webapp_bundle.py <cases_embed.js> <cases.json…> [cases_video.json] → webapp_build.py <模板> <cases_embed.js> <输出>
├─ scenes/mvp/             <pattern>_f<fill>[_nomen].pbrt, free_surface*.ply, manifest*.json
├─ scenes/generated/       旧管线场景
└─ outputs/
   ├─ mvp/patterns/        板图案 PNG（50 px/cm）
   ├─ mvp/renders/         <name>.exr（线性）/ .png（sRGB）/ .pfm（分析缓存）, overview.png
   ├─ mvp/compare/full/    对比图、metrics.json、report.html（冒烟测试在 compare/smoke/）
   ├─ mvp/report.html      Phase 0 报告（指向 compare/full）
   └─ renders/ masks/ previews/ report.html   旧管线输出
```

## 运行（Phase 0）

```bat
cd liquid_level_sim
.venv\Scripts\python scripts\mvp_run.py --smoke                 REM 270x360, 64 spp, ~1 分钟，检查几何/曝光
.venv\Scripts\python scripts\mvp_run.py --spp 512               REM 1080x1440, 26 张 + 总览，32 核约 1.5 小时
.venv\Scripts\python scripts\mvp_run.py --spp 1024 --force      REM 更高质量重渲
.venv\Scripts\python scripts\mvp_run.py --skip-render           REM 只重跑对比与报告
```

渲染批次包含：4 图案 × 5 液位（0, 0.25, 0.50, 0.52, 0.75）主图；`white`/`rg_checker` 在 fill 0.50 换随机种子的**噪声底对照**；
`white`/`rg_checker` 在 fill 0.50/0.52 的**平液面（无弯月面）对照**。

## 运行（Phase 1 测试集）

```bat
.venv\Scripts\python scripts\testset_run.py --smoke          REM 300x400, 24 spp, 约 1 分钟
.venv\Scripts\python scripts\testset_run.py                  REM 900x1200, 160 spp, 18 张约 15-20 分钟
.venv\Scripts\python scripts\testset_run.py --only A1,B2     REM 子集
.venv\Scripts\python scripts\testset_run.py --dirt           REM 启用壁面脏物贴图（默认关闭：洁净玻璃）
```

输出 `outputs/testset/images/*.png`（含噪声后处理，可直接喂给 `webapp/level_diff_bench.html`：`base_*` 为基准、`A*`/`B*` 为对比图），
`outputs/testset/renders/*.exr`（干净线性图），`outputs/testset/manifest.json`（每张图的真值：各液面/界面在前内壁处的像素行、cm 高度、填充率、相机参数、抖动量），
`outputs/testset/contact_sheet.png`。

## 运行（水垢参考样本）

```bat
.venv\Scripts\python scripts\scale_run.py --smoke      REM 检查
.venv\Scripts\python scripts\scale_run.py              REM 5 个场景 × (空/注水)，900x1200, 192 spp
```

`scripts/scale_model.py` 按 RESEARCH.md §12 的物理沉积模型生成内壁水垢光学厚度图（水线环 + 爬升指状带、
水线以下薄膜、潮痕、干涸水滴环、流痕、浸没段折射率匹配变透明），`testset_scenes.py` 的 scale 材质把它渲染为
"前向散射薄层 + 漫射厚层"的两级混合。输出 `outputs/scale/{images,renders,contact_sheet.png,manifest.json,cases.json}`。

## 场景模型要点（单位 cm，z 向上）

- 容器：内径 15、内高 20、壁厚 0.3、底厚 0.5，玻璃 η=1.50；黑色哑光圆台托起，使发光板在容器下方也可见。
- 液体：水 η=1.333，`homogeneous` 吸收介质；**液体与玻璃直接接触**（内壁在接触线处切成湿/干两段，
  湿段 `dielectric eta=η_g/η_w`，干段 `eta=η_g`；液面 `eta=η_w`；`MediumInterface` 包围水体）。
- 弯月面：h=0.3 cm、毛细长度 0.27 cm 的旋转面网格（二进制 PLY，带解析法线），与平液面和内壁精确接缝。
- 相机：针孔 `perspective`，位于 (0, −60, 10.25)，光轴水平，横向 fov 24°（1080×1440 竖构图）。
- 发光板：`AreaLightSource "diffuse"`，50×60 cm 平面位于轴后 25 cm，`"string filename"` 贴图发光 / `rgb L` 均匀白。
- 积分器 `volpath` maxdepth 64、`zsobol`、gaussian 滤波；EXR 线性输出，imgtool 转 sRGB PNG。

## 分析指标（`mvp_compare.py`）

| 指标 | 含义 |
|---|---|
| `dE_vs_empty_*` | 与空容器渲染的 CIE Lab ΔE（腔内 / 液面以下 / 液面以上），液体段改变背景的强度 |
| `sensitivity.dE_mean_band` | fill 0.50→0.52（+4 mm）在液位带内的 ΔE |
| `noise_floor` | 同场景两种子渲染之差，ΔE 的蒙特卡洛噪声底；`snr_band` = 灵敏度 / 噪声底 |
| `no_meniscus` | 平液面对照下的灵敏度与液位线对比度 |
| `line_contrast` | 单张图液位线相对上下邻域的亮度下降比例 |
| `transition_row` | 差分逐行曲线的过渡行，与解析真值行（前/后内壁）比较 |
| `checker_period` | 棋盘格 R−G 过零间距法测周期：板直视 / 透过空气段 / 透过液体段，与近轴柱面透镜预测比较 |

## 环境说明

- Python 环境用 [uv](https://docs.astral.sh/uv/) 重建：`uv venv --python 3.11 .venv && uv pip install numpy opencv-python matplotlib jinja2 ninja imageio`
  （`.venv_broken_othermachine/` 是另一台机器上建的失效环境，可删）。
- OpenCV 5 的 wheel 不含 OpenEXR，分析用 `imgtool convert` 把 EXR 转成 PFM 再用 numpy 读取。
- GPU：本机 RTX 5090 + CUDA 13.3，但未装 OptiX SDK；安装后 `build_pbrt.bat -DPBRT_OPTIX7_PATH="..."` 重编译，渲染加 `--gpu`。

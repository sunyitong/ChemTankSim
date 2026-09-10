# RealCapture 反应容器数字孪生

## 项目现状（2026-09-10 文件审阅）

项目研究通过透明容器对背光图案产生的折射变化识别液位。已有三部分：

1. `pbrt-v4` CPU 物理渲染，以及 Phase 0 圆柱验证、Phase 1 异形瓶／分层液体／水垢／夹具／环境光数据。
2. `scripts/level_detect.py` 的 v3 检测器和 `eval_video.py` 的多假说时序跟踪器。网页实现对应的几何检测和时序规则，不是神经网络。
3. `webapp/level_diff_bench.html` 与 `docs/index.html` 中已有 37 个案例，含 3 段合成注液视频和原有真实拍摄 P1/P2。已有记录指出白背光、遮挡、旧水线与漂移仍是困难条件。

`RealCapture/15% Pump 1` 包含 1,315 张有效 TIFF；`60% Pump 2` 包含 76 张。均为 Basler 的 720×540、8 位灰度图。忽略 macOS `._` 元数据旁车文件。两组只是外观参考，文件夹百分比没有作为泵速、真实液位或标签使用。首帧已经有液体。

## 本次模型

使用现有 pbrt-v4 的 `volpath` 积分器，复用已有旋转体、玻璃／水接触界面、介质吸收、弯月面、干湿水垢混合材质与 MP4 编码流程。新模型按照片比例配置约 15 cm 外径、弧形底部、黑色上盖和边夹、中央粗轴、右侧弯管、左侧半透明标尺（第一组）、颗粒污垢、旧水线与纵向流痕。壁厚和深度是可调整的近似参数；现有单一视角参考不构成完整计量标定。

图像全部来自三维光线追踪。没有用照片作为容器图片或视频背景进行贴图合成。水面缓慢改变高度，叠加在接触边缘衰减为零的小幅解析波动与水内气泡。它是用于视觉识别验证的运动学注液动画，不是计算泵流场的 CFD。

- **F4 / Digital twin 1**：均匀白背光；标尺与内部硬件遮挡。
- **F5 / Digital twin 2（新版）**：原 PDF 第 5 页的 P5 8 mm 彩色随机方格背板，按 A4 实际尺寸建模；图案在液面下发生折射。首版 ArUco 已被用户否定，新版不再采用。
- 720×540，25 fps。视频每帧独立物理渲染、每帧关键帧编码，便于原网页逐帧分析。
- `video_meta.json` 中的真值为液面在前侧内壁位置的几何投影。网页仍按原逻辑显示 ROI 高度百分比，不等于物理容量百分比。

遵循最终要求：保留原网页排版与算法，只向现有 Video 案例组加入 F4/F5；没有新增本次实拍案例、参数组件或说明区。检测结果和误差由原算法直接计算，未作修正。

## 完成验证

`verify_twin_bundle.py` 验证两段视频均为 101 个可解码的独立画面、720×540、25 fps，并逐项确认原有 37 个案例和资源不变。剔除三条案例／资源数据赋值后，网页全文与本次修改前的 Git 版本一致，包含全部 HTML、CSS、检测与跟踪 JavaScript。

首版视频的历史评估：原 `eval_video.py` 在 D1 上逐帧检测 101/101，90 分位绝对误差约 1.25 个 ROI 高度百分点；首版 D2 为 101/101，但后段高液位误检，90 分位绝对误差约 29.80 个百分点，最大约 34.84。新版结果存于 `outputs/digital_twin_approved` 中各目录的 `eval_video.json`，不能据此声称真实数据精度。

已确认重污垢新版的原算法离线结果（误差单位为 ROI 高度百分点；逐帧误差分位数只统计有检测结果的帧）：

| 视频 | 有检测结果 / 总帧数 | 逐帧 P90 绝对误差 | 逐帧最大绝对误差 |
|---|---:|---:|---:|
| F4 白背光 | 101 / 101 | 6.89 | 15.08 |
| F5 P5 彩色背光 | 82 / 101 | 36.57 | 68.42 |

后段高液位受厚水垢和模糊图案干扰，仍有误检或漏检。保留原检测与跟踪代码，未调参或替换结果。视频本身的冻结已经修复；新视频的本地浏览器测试覆盖解码时间、初中末帧像素变化、灰度／彩色资产、拖动、重播和快速案例切换，见 `browser_local_verification.json`。

浏览器验证更正：上一轮看到进度条前进就认为视频已播放，验证不充分。进一步检查发现 F5 的实际媒体时间一直为 0 秒，本地服务器不支持视频 Range 跳转；固定 40% 不能归因于算法跟踪。此问题已在 `c0ac5f1` 中通过 Blob 加载、跳转检查及取消处理修复，并通过本地和 GitHub Pages 的真实解码帧测试。详见 `DIGITAL_TWIN_REVIEW.md`。

## 复现

在项目根目录执行：

```powershell
# 新克隆仓库时，先从已跟踪 HDR 文件生成等面积 EXR 环境贴图：
& liquid_level_sim/.venv/Scripts/python.exe liquid_level_sim/scripts/env_map.py --id childrens_hospital
& liquid_level_sim/.venv/Scripts/python.exe liquid_level_sim/scripts/digital_twin_approved.py --spp 64
& liquid_level_sim/.venv/Scripts/python.exe liquid_level_sim/scripts/digital_twin_web.py --source liquid_level_sim/outputs/digital_twin_approved
& liquid_level_sim/.venv/Scripts/python.exe liquid_level_sim/scripts/verify_twin_bundle.py --source liquid_level_sim/outputs/digital_twin_approved
& liquid_level_sim/.venv/Scripts/python.exe -m http.server 8765 --bind 127.0.0.1 --directory docs
```

几何参数：`configs/digital_twin.json`；已确认的材质和灯光参数：`configs/digital_twin_review.json`（其中 status 是静帧制作阶段的记录，用户批准另存于 render_recipe）。单独渲染：`digital_twin_approved.py --only D2 --spp 64`；缓存允许中断后继续。依赖已有 Python 环境里的 numpy、OpenCV、imageio-ffmpeg；渲染使用仓库现有 `pbrt-v4/build/pbrt.exe` 与 `imgtool.exe`。

`outputs/digital_twin_approved/D1` 与 `D2` 保存新版视频、空容器参考、接触表与逐帧真值；`outputs/digital_twin/` 是首版历史输出。网页部署目录和本地网页各自带相对路径 `twin_assets`，新资源使用内容版本号避免读到旧缓存。请使用上面的本地 HTTP 服务或 GitHub Pages 查看；外部视频需要通过 fetch 读取，浏览器直接打开 file:// 页面可能受本地文件权限限制。生成脚本本身不推送 Git。

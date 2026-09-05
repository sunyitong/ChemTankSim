"""Build outputs/report.html from metrics.json + previews (self-contained, relative image paths)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Template  # noqa: E402

from common import OUT_DIR, PREVIEW_DIR, load_json  # noqa: E402

TEMPLATE = Template("""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Liquid-level rendering study — report</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;color:#222;background:#fafafa}
 h1{font-size:22px} h2{font-size:17px;margin-top:28px}
 table{border-collapse:collapse;font-size:13px} th,td{border:1px solid #ddd;padding:4px 8px;text-align:right}
 th{background:#eee} td.l,th.l{text-align:left}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(640px,1fr));gap:14px}
 .card{background:#fff;border:1px solid #ddd;padding:8px;border-radius:6px}
 .card img{width:100%;image-rendering:auto}
 .meta{font-size:12px;color:#555;margin-top:4px}
 .bad{color:#b00} .ok{color:#080}
 code{background:#eee;padding:1px 4px;border-radius:3px}
</style></head><body>
<h1>透明容器液位渲染验证 — pbrt-v4 合成数据报告</h1>
<p class="meta">generated {{ now }} · {{ s.n }} samples · renderer: pbrt-v4 (CPU, volpath) · legend: <span style="color:#0a0">green</span>=GT level line (free-surface near edge), <span style="color:#c00">red</span>=baseline estimate, cyan=liquid GT contour, magenta=free surface, orange=container ROI</p>

<h2>Summary</h2>
<table>
<tr><th class="l">metric</th><th>value</th><th class="l">meaning</th></tr>
<tr><td class="l">evaluated</td><td>{{ s.n_evaluated }}/{{ s.n }}</td><td class="l">samples with a visible free surface</td></tr>
{% if s.mae_px is defined %}
<tr><td class="l">MAE (px)</td><td>{{ '%.2f' % s.mae_px }}</td><td class="l">baseline level-line error in pixels</td></tr>
<tr><td class="l">median |err| (px)</td><td>{{ '%.2f' % s.median_abs_px }}</td><td class="l"></td></tr>
<tr><td class="l">RMSE (px)</td><td>{{ '%.2f' % s.rmse_px }}</td><td class="l"></td></tr>
<tr><td class="l">bias (px)</td><td>{{ '%+.2f' % s.bias_px }}</td><td class="l">positive = estimate below GT (too low a level)</td></tr>
<tr><td class="l">MAE (mm)</td><td>{{ '%.2f' % s.mae_mm }}</td><td class="l">using cm/px scale from container extent</td></tr>
<tr><td class="l">MAE fill fraction (2D)</td><td>{{ '%.3f' % s.mae_frac }}</td><td class="l">naive rim-to-base linear mapping</td></tr>
<tr><td class="l">within 3 px</td><td>{{ '%.0f%%' % (100*s.within_3px) }}</td><td class="l"></td></tr>
<tr><td class="l">within 2% fill</td><td>{{ '%.0f%%' % (100*s.within_2pct) }}</td><td class="l"></td></tr>
{% endif %}
{% if s.gt_check_mean_abs_px is defined %}
<tr><td class="l">GT self-check: |rendered − analytic| mean / max (px)</td><td>{{ '%.2f / %.2f' % (s.gt_check_mean_abs_px, s.gt_check_max_abs_px) }}</td><td class="l">free-surface near edge from the GT render vs. analytic camera projection; should be ≲1 px</td></tr>
{% endif %}
{% for k,v in s.items() if k.startswith('detector_') %}
<tr><td class="l">{{ k }}</td><td>{{ '%.3f' % v }}</td><td class="l">per baseline detector (primary = step)</td></tr>
{% endfor %}
{% for k,v in s.items() if k.startswith('mae_px_') %}
<tr><td class="l">MAE (px) — {{ k[7:] }}</td><td>{{ '%.2f' % v }}</td><td class="l">per container / liquid (primary detector)</td></tr>
{% endfor %}
</table>

{% if has_plot %}<h2>Error vs. fill level</h2><img src="error_plot.png" style="max-width:900px">{% endif %}

<h2>Samples</h2>
<table>
<tr><th class="l">sid</th><th class="l">container</th><th class="l">liquid</th><th>fill</th><th>h (cm)</th><th>fov</th><th>GT row</th><th>analytic</th><th>est row (step)</th><th>err px (step)</th><th>err px (gradient)</th><th>err mm</th><th>err fill</th></tr>
{% for r in rows %}
<tr><td class="l">{{ r.sid }}</td><td class="l">{{ r.container }}</td><td class="l">{{ r.liquid }}</td>
<td>{{ '%.3f' % r.fill }}</td><td>{{ '%.2f' % r.liquid_height_cm }}</td><td>{{ '%.1f' % r.fov }}</td>
<td>{{ r.gt_row_near is not none and '%.1f' % r.gt_row_near or '—' }}</td>
<td>{{ '%.1f' % r.gt_row_analytic }}</td>
<td>{{ r.est_row is not none and '%.1f' % r.est_row or '—' }}</td>
<td class="{{ 'bad' if r.err_px is not none and r.err_px|abs > 3 else 'ok' }}">{{ r.err_px is not none and '%+.2f' % r.err_px or '—' }}</td>
<td class="{{ 'bad' if r.err_px_gradient is not none and r.err_px_gradient|abs > 3 else 'ok' }}">{{ r.err_px_gradient is not none and '%+.2f' % r.err_px_gradient or '—' }}</td>
<td>{{ r.err_mm is not none and '%+.2f' % r.err_mm or '—' }}</td>
<td>{{ r.err_frac_2d is not none and '%+.3f' % r.err_frac_2d or '—' }}</td></tr>
{% endfor %}
</table>

<h2>Previews (beauty | overlay | liquid GT | surface+container GT)</h2>
<div class="grid">
{% for r in rows %}
<div class="card"><img src="previews/{{ r.sid }}_strip.png" loading="lazy">
<div class="meta">{{ r.sid }} · {{ r.container }} · {{ r.liquid }} · fill {{ '%.3f' % r.fill }} · eye z {{ '%.1f' % r.eye_z }} cm{% if r.err_px is not none %} · err {{ '%+.2f' % r.err_px }} px{% endif %}</div></div>
{% endfor %}
</div>

<h2>Files</h2>
<p class="meta">scenes: <code>scenes/generated/&lt;sid&gt;_{beauty,liquid,surface,container}.pbrt</code> + <code>&lt;sid&gt;.json</code> ·
renders: <code>outputs/renders/&lt;sid&gt;_beauty.{exr,png}</code> · GT masks: <code>outputs/masks/</code> · metrics: <code>outputs/metrics.{json,csv}</code></p>
</body></html>
""")


def plot_errors(rows: list[dict], out: Path) -> bool:
    ok = [r for r in rows if r.get("err_px") is not None]
    if not ok:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for c, mk in (("cylinder", "o"), ("box", "s")):
        rr = [r for r in ok if r["container"] == c]
        if rr:
            axes[0].scatter([r["fill"] for r in rr], [r["err_px"] for r in rr], marker=mk, label=c, alpha=0.8)
            axes[1].scatter([r["fill"] for r in rr], [r["gt_analytic_vs_render_px"] for r in rr], marker=mk, label=c, alpha=0.8)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_xlabel("fill fraction (GT)"); axes[0].set_ylabel("baseline error (px)"); axes[0].set_title("Baseline level-line error")
    axes[0].legend()
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("fill fraction (GT)"); axes[1].set_ylabel("rendered − analytic (px)"); axes[1].set_title("GT self-check")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return True


def main():
    m = load_json(OUT_DIR / "metrics.json")
    rows, s = m["samples"], m["summary"]
    has_plot = plot_errors(rows, OUT_DIR / "error_plot.png")
    html = TEMPLATE.render(rows=rows, s=s, has_plot=has_plot, now=dt.datetime.now().strftime("%Y-%m-%d %H:%M"))
    (OUT_DIR / "report.html").write_text(html, encoding="utf-8")
    print("wrote", OUT_DIR / "report.html")


if __name__ == "__main__":
    main()

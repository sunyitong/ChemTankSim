"""Backlit calibration patterns for A4 transparency film (to be taped onto an A4 white light box).

Eight print-ready pages, 300 dpi, exact A4 (210 x 297 mm), 6 mm clear border for the printer, a footer with the
pattern id, the cell size and a 100 mm scale bar (check it with a ruler after printing: it must measure 100 mm,
i.e. the file was printed at 100 %, not "fit to page").

Why these patterns (see docs/RESEARCH.md):
  * black / clear cells give the highest contrast on film: black ink blocks the light box, clear film passes it;
  * a red/green checker matches the synthetic panel of the study and the first real tests;
  * APERIODIC patterns (random-colour mosaic, random-width bars) are what the detector likes best: a periodic
    checker magnified by the liquid correlates with itself at several magnifications (aliasing, false identity
    rows, false second interfaces) and cannot show the left-right flip of a real image; random cells can;
  * vertical bars have the same content in every row, so a vertical misalignment between the empty baseline
    and the filled frame (camera drift, the glass moving) does not change what a row sees — only the liquid's
    horizontal magnification does;
  * cell sizes: the detector wants 6-12 cells across the part of the panel seen THROUGH the vessel. Seen through
    a liquid the panel appears magnified 1.3-4x (narrow vessel, near panel -> large M), so a 7 cm glass shows
    only 2-5 cm of panel: use 4 mm cells; a 15 cm vessel or a distant panel: 8-12 mm cells.

  python scripts/print_patterns.py   -> outputs/print_patterns/{P1..P8 *.png, ChemTankSim_backlit_patterns_A4.pdf,
                                        contact_sheet.png, README.md}
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from mvp_common import PROJ_ROOT

OUT = PROJ_ROOT / "outputs" / "print_patterns"
DPI = 300
PX_MM = DPI / 25.4
A4_W, A4_H = 210.0, 297.0                        # mm, portrait
MARGIN = 6.0                                     # mm clear border (printer margin)
FOOTER = 14.0                                    # mm strip at the bottom: label + scale bar
W, H = round(A4_W * PX_MM), round(A4_H * PX_MM)  # 2480 x 3508

# inks that are dense on transparency film (RGB values as printed on clear film over a white light box)
BLACK, CLEAR = (0, 0, 0), (255, 255, 255)
RED, GREEN, BLUE = (230, 30, 30), (30, 190, 30), (30, 60, 230)
CYAN, MAGENTA, YELLOW = (0, 190, 220), (220, 0, 180), (250, 220, 0)


def mm(v: float) -> int:
    return int(round(v * PX_MM))


def page() -> np.ndarray:
    return np.full((H, W, 3), 255, np.uint8)


def pattern_box():
    """Pixel box of the pattern area: inside the margin, above the footer."""
    x0, y0 = mm(MARGIN), mm(MARGIN)
    x1, y1 = W - mm(MARGIN), H - mm(MARGIN + FOOTER)
    return x0, y0, x1, y1


def fill_cells(img, box, cell_mm: float, colour_fn):
    """Tile the box with square cells of `cell_mm`, anchored at the box centre; colour_fn(i, j) -> RGB."""
    x0, y0, x1, y1 = box
    cell = cell_mm * PX_MM
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    xs = np.floor((np.arange(x0, x1) - cx) / cell).astype(int)
    ys = np.floor((np.arange(y0, y1) - cy) / cell).astype(int)
    ui, uj = np.unique(xs), np.unique(ys)
    lut = {(i, j): colour_fn(i, j) for j in uj for i in ui}
    cols = np.array([[lut[(i, j)] for i in ui] for j in uj], np.uint8)
    img[y0:y1, x0:x1] = cols[np.searchsorted(uj, ys)[:, None], np.searchsorted(ui, xs)[None, :]]


def checker(img, box, cell_mm, c1, c2):
    fill_cells(img, box, cell_mm, lambda i, j: c1 if (i + j) % 2 == 0 else c2)


def mosaic(img, box, cell_mm, seed):
    """Aperiodic colour mosaic: each cell draws from a palette weighted towards the dense inks; no two
    neighbours share a colour (so every cell edge is a real edge)."""
    palette = [BLACK, RED, GREEN, BLUE, CLEAR, MAGENTA, CYAN, YELLOW]
    weights = np.array([0.22, 0.16, 0.16, 0.16, 0.12, 0.06, 0.06, 0.06])
    rng = np.random.default_rng(seed)
    cache = {}

    def colour(i, j):
        for _ in range(20):
            k = int(rng.choice(len(palette), p=weights))
            if cache.get((i - 1, j)) != k and cache.get((i, j - 1)) != k:
                break
        cache[(i, j)] = k
        return palette[k]
    fill_cells(img, box, cell_mm, colour)


def bars(img, box, palette, w_min_mm, w_max_mm, seed):
    """Aperiodic vertical bars of random width; every row identical. Adjacent bars never share a colour."""
    x0, y0, x1, y1 = box
    rng = np.random.default_rng(seed)
    x, last = x0, None
    while x < x1:
        wpx = mm(rng.uniform(w_min_mm, w_max_mm))
        k = int(rng.integers(len(palette)))
        while len(palette) > 1 and k == last:
            k = int(rng.integers(len(palette)))
        img[y0:y1, x:min(x1, x + wpx)] = palette[k]
        last, x = k, x + wpx


def footer(img, pid: str, title: str, cell_txt: str):
    """Label, 100 mm scale bar with 10 mm ticks, and corner marks for a quick perspective check."""
    x0, y0, x1, y1 = pattern_box()
    yb = H - mm(MARGIN) - mm(2.5)                 # scale bar baseline
    xs = x0 + mm(4)
    cv2.rectangle(img, (xs, yb - mm(1.2)), (xs + mm(100), yb), BLACK, -1)
    for k in range(11):
        xt = xs + mm(10 * k)
        cv2.rectangle(img, (xt - mm(0.25), yb - mm(3.5 if k % 5 == 0 else 2.5)), (xt + mm(0.25), yb), BLACK, -1)
    cv2.putText(img, "0", (xs - mm(1), yb - mm(4.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, "50 mm", (xs + mm(50) - mm(6), yb - mm(4.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, "100 mm", (xs + mm(100) - mm(5), yb - mm(4.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK, 2, cv2.LINE_AA)
    lines = [f"ChemTankSim backlit pattern {pid}: {title}, {cell_txt}", "A4 at 300 dpi. Print at 100 % (no fit-to-page); the bar must measure 100 mm."]
    xl, avail = xs + mm(112), x1 - (xs + mm(112))
    for k, txt in enumerate(lines):                              # two lines, scaled to fit the remaining width
        sc = 0.8
        while cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)[0][0] > avail and sc > 0.4:
            sc -= 0.05
        cv2.putText(img, txt, (xl, yb - mm(5.5 if k == 0 else 1.2)), cv2.FONT_HERSHEY_SIMPLEX, sc, BLACK, 2, cv2.LINE_AA)
    # corner marks just inside the pattern area (L shapes), for a perspective / scale check in the photo
    L, t = mm(6), mm(0.8)
    for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.rectangle(img, (cx, cy), (cx + sx * L, cy + sy * t), CLEAR, -1); cv2.rectangle(img, (cx, cy), (cx + sx * t, cy + sy * L), CLEAR, -1)
        cv2.rectangle(img, (cx + sx * t, cy + sy * t), (cx + sx * L, cy + sy * 2 * t), BLACK, -1); cv2.rectangle(img, (cx + sx * t, cy + sy * t), (cx + sx * 2 * t, cy + sy * L), BLACK, -1)


PATTERNS = [
    ("P1", "checker_bw_4mm", "black / clear checkerboard", "4 mm cells", lambda im, b: checker(im, b, 4.0, BLACK, CLEAR),
     "Highest contrast on film. Narrow vessels and strong magnification (a 7 cm glass shows 2-5 cm of panel)."),
    ("P2", "checker_bw_8mm", "black / clear checkerboard", "8 mm cells", lambda im, b: checker(im, b, 8.0, BLACK, CLEAR),
     "Highest contrast. Wide vessels (>= 12 cm) or a panel far behind the vessel."),
    ("P3", "checker_rg_6mm", "red / green checkerboard", "6 mm cells", lambda im, b: checker(im, b, 6.0, RED, GREEN),
     "Same pattern family as the synthetic study and the first real tests (colour cameras). Periodic: prefer P4-P8 for the detector."),
    ("P4", "mosaic_4mm", "aperiodic colour mosaic", "4 mm cells", lambda im, b: mosaic(im, b, 4.0, 4),
     "Recommended for the level detector: no aliasing, flip visible. Narrow vessels / high magnification."),
    ("P5", "mosaic_8mm", "aperiodic colour mosaic", "8 mm cells", lambda im, b: mosaic(im, b, 8.0, 8),
     "Recommended general-purpose pattern for 8-15 cm vessels at 40-80 cm camera distance."),
    ("P6", "mosaic_12mm", "aperiodic colour mosaic", "12 mm cells", lambda im, b: mosaic(im, b, 12.0, 12),
     "Wide vessels, distant camera, or low-resolution video (cells must stay >= 6 px after magnification)."),
    ("P7", "bars_bw_random", "aperiodic black / clear bars", "2-9 mm widths", lambda im, b: bars(im, b, [BLACK, CLEAR], 2.0, 9.0, 7),
     "Every row identical: immune to vertical misalignment (camera drift, vessel shifting, refocus). Max contrast."),
    ("P8", "bars_rgbk_random", "aperiodic colour bars", "3-10 mm widths", lambda im, b: bars(im, b, [BLACK, RED, GREEN, BLUE, CLEAR], 3.0, 10.0, 8),
     "Row-invariant like P7, with colour for the RGB correlation; flip and magnification unambiguous."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages, thumbs, readme = [], [], []
    for pid, name, title, cell_txt, draw, note in PATTERNS:
        img = page(); draw(img, pattern_box()); footer(img, pid, title, cell_txt)
        cv2.imwrite(str(OUT / f"{pid}_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        pages.append(Image.fromarray(img))
        th = cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        cv2.putText(th, pid, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        thumbs.append(cv2.copyMakeBorder(th, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(90, 90, 90)))
        readme.append(f"| {pid} | `{pid}_{name}.png` | {title} | {cell_txt} | {note} |")
    pages[0].save(OUT / "ChemTankSim_backlit_patterns_A4.pdf", "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    rows = [np.concatenate(thumbs[i:i + 4], axis=1) for i in range(0, len(thumbs), 4)]
    cv2.imwrite(str(OUT / "contact_sheet.png"), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    (OUT / "README.md").write_text(
        "# Backlit calibration patterns (A4 transparency film)\n\n"
        "Print `ChemTankSim_backlit_patterns_A4.pdf` (one pattern per page) at **100 % scale** on inkjet or laser transparency film, "
        "then tape the film onto the A4 white light box. Check the 100 mm scale bar with a ruler after printing. "
        "Pigment inkjet on inkjet film gives the densest black; a laser print on laser film is fine too. "
        "Mount the film flat against the diffuser (no air gap) with the pattern area centred behind the vessel; keep the room dark.\n\n"
        "| id | file | pattern | size | when to use |\n|---|---|---|---|---|\n" + "\n".join(readme) + "\n\n"
        "Choosing a size: the detector wants 6-12 cells across the part of the panel seen through the vessel. Through the liquid the panel "
        "appears magnified 1.3-4x, so a 7 cm glass shows only 2-5 cm of panel (use 4 mm cells or the bars); a 15 cm vessel or a panel far "
        "behind the vessel needs 8-12 mm cells. In the image, cells must stay at least ~6 px wide after the working resolution (the web app "
        "analyses at half resolution of a 675 px wide frame).\n\n"
        "Why aperiodic: a periodic checker magnified by the liquid correlates with itself at several magnifications, which produced false "
        "identity rows and false second interfaces in the real footage of 2026-09-07; random-colour cells and random-width bars do not, "
        "and they also make the left-right flip of a real image visible. Bars (P7, P8) are additionally immune to a vertical misalignment "
        "between the empty baseline and the filled frame, because every row carries the same content.\n", encoding="utf-8")
    print(f"{len(PATTERNS)} patterns -> {OUT} ({W}x{H} px, {DPI} dpi); pdf {(OUT / 'ChemTankSim_backlit_patterns_A4.pdf').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

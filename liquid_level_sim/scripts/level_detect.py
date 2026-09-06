"""Level detection v3 — geometry first, photometry second.

Principle
---------
A liquid changes the GEOMETRY of the backlight pattern seen through the vessel (horizontal
magnification / flip, different for every liquid), whereas deposits, dirt, exposure drift and
noise change only the PHOTOMETRY (contrast, haze, brightness) and leave pattern edges in place.
So the primary signal is the per-row horizontal warp between the clean baseline and the compare
frame; levels are the rows where that warp JUMPS. Within one liquid the warp varies smoothly with
the vessel radius; at the free surface and at every liquid|liquid interface it jumps. Rows where
the pattern cannot be matched (dense deposit, total-internal-reflection band at a surface, white
panel) carry no warp estimate: they are treated as gaps, never as evidence. A jump is measured
ACROSS gaps by fitting local lines to the nearest reliable rows on each side, so a deposit ring
(same warp on both sides) scores zero while a surface band (different warp on the two sides)
scores high and is placed at the upper edge of the band, which is the geometric level.

Pipeline (mirrored by the JavaScript in level_diff_bench.html)
  1. robust pair: 2x2 downsample, 3x3 blur, per-channel affine exposure match, integer shift search
  2. per row: best horizontal magnification M and flip mapping the baseline row onto the compare row
     (normalised cross-correlation over the central 70 % of the ROI columns, all channels)
  3. reliable rows: NCC >= NCC_MIN and the baseline row itself has pattern contrast
  4. jump statistic across gaps: weighted local linear fits on the nearest w reliable rows above and
     below r; J = difference of the two fits at r; z = |J| / robust residual scale
  5. peaks with z >= Z_MIN and |J| >= J_MIN, separated by >= 2w, top and bottom margins excluded;
     level row = upper edge of the gap the peak sits in, else the mid-crossing
  6. fallback when the baseline carries no pattern (white panel): spikes of the photometric row
     difference (meniscus / interface lines), flagged as low confidence
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

NCC_MIN = 0.35        # rows below this correlation carry no usable warp estimate
STRUCT_MIN = 0.045    # baseline row luminance std/mean below this = no pattern in that row
Z_MIN = 4.0           # jump significance (robust sigma units)
J_MIN = 0.08          # minimum |delta log M| (8 % magnification change)
MAX_LEVELS = 3
CENTRAL = 0.70        # fraction of ROI width used for the warp correlation (excludes wall edges)
SIGMA_MAX = 0.12      # plateau residual scale (log M) allowed for a secondary (liquid|liquid) boundary
Z_MIN_SECONDARY = 15.0  # significance required for a secondary boundary
BOTTOM_ZONE = 0.20      # fraction of the ROI height above the bottom where rays cross the vessel bottom
W_FLOOR = 0.02          # correlation weight of columns that did not change (geometry is measured where the image changed)
PHI_MIN = 0.05          # rows whose changed-column fraction is below this are identity (M = 1) by definition
PHI_MEAS = 0.20         # minimum changed-column fraction for a measurable warp (fewer columns are over-fitted)
NCC_SURE = 0.55         # correlation from which a warp estimate is trusted as evidence (identity or liquid)
LOGM_LIQ = 0.30         # |log M| from which a trusted warp counts as liquid (a lens), unless it is flipped
RHO_DIP = 0.92          # row luminance (relative to the air region) below which the contact line / TIR band begins
LINE_MIN = 0.06         # luminance anomaly (bright or dark, vs the local median) a liquid|liquid contact line must show


@dataclass
class Detection:
    mode: str                              # "warp" | "photometric"
    levels: list = field(default_factory=list)   # [{row, jump, z}] top -> bottom, ROI-local half-res rows
    logM: np.ndarray | None = None
    ncc: np.ndarray | None = None
    good: np.ndarray | None = None
    zscore: np.ndarray | None = None
    dE: np.ndarray | None = None           # photometric row profile (thresholded)
    frac_pattern: float = 0.0
    shift: tuple = (0, 0)
    ident: np.ndarray | None = None
    phi: np.ndarray | None = None
    f: np.ndarray | None = None
    cands: list = field(default_factory=list)   # jump candidates before the surface merge / secondary gates
    cls: np.ndarray | None = None               # per-row class: 1 identity, 2 liquid, 0 unknown
    rho: np.ndarray | None = None               # compare/baseline row luminance ratio
    ncc1: np.ndarray | None = None              # correlation of the identity hypothesis (M = 1)
    amb: np.ndarray | None = None               # fraction of warp hypotheses within 0.10 of the best NCC
    surface: int | None = None                  # row after the identity plateau (free-surface upper bound)


# ----------------------------------------------------------------------------- 1. robust pair
def down2(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return 0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2])


def robust_pair(b: np.ndarray, c: np.ndarray, R: int = 3):
    B, C = down2(b).astype(np.float32), down2(c).astype(np.float32)
    B = cv2.blur(B, (3, 3)); C = cv2.blur(C, (3, 3))
    for ch in range(3):
        mb, sb = B[..., ch].mean(), B[..., ch].std(); mc, sc = C[..., ch].mean(), C[..., ch].std()
        a = float(np.clip(sb / sc, 0.6, 1.6)) if sc > 1e-4 else 1.0
        C[..., ch] = C[..., ch] * a + (mb - a * mc)
    Lb = B @ np.array([0.299, 0.587, 0.114], np.float32); Lc = C @ np.array([0.299, 0.587, 0.114], np.float32)
    h, w = Lb.shape; best, bdx, bdy = np.inf, 0, 0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            ys, xs = slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx))
            cost = np.abs(Lb[ys, xs][::2, ::2] - Lc[ys.start + dy:ys.stop + dy, xs.start + dx:xs.stop + dx][::2, ::2]).mean()
            if cost < best - 1e-9:
                best, bdx, bdy = cost, dx, dy
    Ca = np.zeros_like(C)
    ys, xs = slice(max(0, -bdy), min(h, h - bdy)), slice(max(0, -bdx), min(w, w - bdx))
    Ca[ys, xs] = C[ys.start + bdy:ys.stop + bdy, xs.start + bdx:xs.stop + bdx]
    return B, Ca, (bdx, bdy)


def photometric_profile(B, Ca):
    """Thresholded colour-difference row mean (heat map / fallback)."""
    d = np.linalg.norm(B - Ca, axis=2) / math.sqrt(3)
    rm = d.mean(axis=1); quiet = np.argsort(rm)[:max(3, round(len(rm) * 0.25))]
    q = d[quiet]; T = max(0.008, float(q.mean() + 3 * q.std()))
    d = np.where(d > T, np.minimum(d, 0.6), 0.0)
    return d.mean(axis=1), d, T


# ----------------------------------------------------------------------------- 2. per-row warp
def change_weights(B, C, thresh: float, central=CENTRAL):
    """Per-pixel correlation weights for the warp fit and the per-row changed fraction phi.
    Geometry can only be measured where the image changed: columns whose appearance differs from
    the baseline (liquid-refracted region, deposit) count fully, unchanged columns (background
    outside an odd-shaped vessel, air region) are reduced to W_FLOOR. Without this a row inside a
    dark total-internal-reflection band still correlates with the baseline at M = 1 through the
    unchanged background columns and masquerades as an identity row. A row whose changed fraction
    is below PHI_MIN has, by definition, the identity warp (nothing moved), see detect()."""
    h, w, _ = B.shape
    d = np.linalg.norm(B - C, axis=2) / math.sqrt(3)
    k = max(3, int(w * 0.05) | 1)
    d = cv2.blur(d, (k, 1))
    ch = np.clip((d - thresh) / (4 * thresh + 1e-6), 0, 1).astype(np.float32)
    cx = (w - 1) / 2; c0, c1 = int(round(cx - central * w / 2)), int(round(cx + central * w / 2))
    phi = (d[:, c0:c1] > thresh).mean(axis=1)                              # fraction of changed columns
    return (W_FLOOR + (1 - W_FLOOR) * ch).astype(np.float32), phi


def warp_profile(B, C, weights=None, n_scale=49, m_lo=0.5, m_hi=8.0, central=CENTRAL, cols=None):
    """Per-row rigid horizontal warp (scale about the ROI centre, optional flip) maximising the
    weighted NCC between the compare row (columns `cols`) and the resampled baseline row.
    Returns logM, best NCC, flip and ncc1 = the NCC of the identity hypothesis (M = 1, no flip);
    with 49 log-spaced scales from 0.5 to 8 the scale M = 1 is evaluated exactly."""
    h, w, _ = B.shape
    cx = (w - 1) / 2
    if cols is None:
        c0, c1 = int(round(cx - central * w / 2)), int(round(cx + central * w / 2))
        xs_c = np.arange(c0, c1)
    else:
        xs_c = np.asarray(cols)
    scales = np.exp(np.linspace(math.log(m_lo), math.log(m_hi), n_scale))
    logM = np.zeros(h, np.float32); best = np.full(h, -1.0, np.float32); flip = np.ones(h, np.float32); ncc1 = np.zeros(h, np.float32)
    land = []                                                             # NCC of every hypothesis, for the ambiguity measure
    W = np.ones((h, len(xs_c)), np.float32) if weights is None else weights[:, xs_c]
    W3 = np.repeat(W, 3, axis=1)                                          # per channel
    Cc = C[:, xs_c, :].reshape(h, -1)
    Cm = (Cc * W3).sum(axis=1, keepdims=True) / W3.sum(axis=1, keepdims=True)
    Cc = (Cc - Cm) * np.sqrt(W3); nC = np.linalg.norm(Cc, axis=1) + 1e-6
    for sgn in (1.0, -1.0):
        for M in scales:
            xs = cx + sgn * (xs_c - cx) / M
            if ((xs >= 0) & (xs <= w - 1)).sum() < 0.6 * len(xs_c):
                continue
            xi = np.clip(xs, 0, w - 1); i0 = np.floor(xi).astype(int); t = (xi - i0)[None, :, None]; i1 = np.minimum(i0 + 1, w - 1)
            Bs = (B[:, i0, :] * (1 - t) + B[:, i1, :] * t).reshape(h, -1)
            Bm = (Bs * W3).sum(axis=1, keepdims=True) / W3.sum(axis=1, keepdims=True)
            Bs = (Bs - Bm) * np.sqrt(W3)
            ncc = (Cc * Bs).sum(axis=1) / (nC * (np.linalg.norm(Bs, axis=1) + 1e-6))
            if sgn > 0 and abs(M - 1.0) < 1e-6:
                ncc1 = ncc.astype(np.float32)
            land.append(ncc)
            up = ncc > best
            best[up] = ncc[up]; logM[up] = math.log(M); flip[up] = sgn
    # ambiguity: fraction of all hypotheses that match (nearly) as well as the best one. A row whose
    # compare content has no horizontal structure (one magnified cell across the columns, dark band)
    # is matched equally well by most warps: its estimate is meaningless whatever the best NCC is.
    L = np.stack(land, axis=1)
    amb = (L >= best[:, None] - 0.10).mean(axis=1).astype(np.float32)
    return logM, best, flip, ncc1, amb


def row_structure(B, central=CENTRAL):
    """Luminance std/mean of each baseline row over the central columns: does the panel carry pattern?"""
    h, w, _ = B.shape; cx = (w - 1) / 2
    c0, c1 = int(round(cx - central * w / 2)), int(round(cx + central * w / 2))
    lum = B[:, c0:c1, :] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return lum.std(axis=1) / (lum.mean(axis=1) + 1e-6)


# ----------------------------------------------------------------------------- 3/4. jumps across gaps
def median_filter1d(a, w):
    w = max(3, w | 1); pad = w // 2; ap = np.pad(a, pad, mode="edge")
    return np.array([np.median(ap[i:i + w]) for i in range(len(a))], np.float32)


def _wfit(idx, r, f, wgt):
    xx = idx - r; yy = f[idx]; ww = wgt[idx] + 1e-3
    A = np.stack([np.ones_like(xx, dtype=float), xx], axis=1) * ww[:, None]
    coef, *_ = np.linalg.lstsq(A, yy * ww, rcond=None)
    res = yy - (coef[0] + coef[1] * xx)
    return float(coef[0]), float(1.4826 * np.median(np.abs(res - np.median(res)))), float(ww.sum())


def gap_jumps(f: np.ndarray, good: np.ndarray, wgt: np.ndarray, w: int, sigma_floor: float):
    """Jump statistic at every row using the nearest `w` reliable rows above and below (skipping gaps).
    Also returns the upper-plateau value VL(r) used by locate()."""
    n = len(f); J = np.zeros(n); Z = np.zeros(n); VL = np.zeros(n); S = np.full(n, np.inf)
    gidx = np.nonzero(good)[0]
    if len(gidx) < 2 * max(4, w // 3):
        return J, Z, VL, S
    pos = np.searchsorted(gidx, np.arange(n))          # number of good rows strictly above r
    mn = max(4, w // 3)
    for r in range(n):
        above, below = gidx[max(0, pos[r] - w):pos[r]], gidx[pos[r]:min(len(gidx), pos[r] + w)]
        if len(above) < mn or len(below) < mn:
            continue
        if r - above[-1] > 3 * w or below[0] - r > 3 * w:  # gap too wide to bridge
            continue
        vl, sl, nl = _wfit(above, r, f, wgt); vr, sr, nr = _wfit(below, r, f, wgt)
        J[r] = vr - vl; VL[r] = vl
        sig = max(0.5 * (sl + sr), sigma_floor); S[r] = sig
        Z[r] = abs(J[r]) / (sig * math.sqrt(2.0 / max(min(nl, nr), 1.0)))
    return J, Z, VL, S


def band_step(dE, top, bot):
    """Photometric step inside an unreliable band [top, bot): rows of deposit haze above the liquid
    differ little from the baseline, rows of liquid / total internal reflection differ a lot. Returns
    the first row of the strongly-differing part, or None when the band has no such step."""
    if bot - top < 3:
        return None
    best, bk = -np.inf, None
    for k in range(top + 1, bot):
        d = dE[k:bot].mean() - dE[top:k].mean()
        if d > best:
            best, bk = d, k
    if bk is not None and dE[bk:bot].mean() >= 2.0 * dE[top:bk].mean() + 0.02:
        return int(bk)
    return None


def band_top(agree, r, w):
    """Walk up from row r through the transition band until a RUN of three consecutive rows satisfying
    `agree` (rows firmly on the plateau above) is met; returns the first row of the band, or None if
    no such run exists within 3w rows (faint reflections inside a total-internal-reflection band can
    mimic the plateau for single rows, hence the run)."""
    j = r
    while j > 2:
        if agree(j) and agree(j - 1) and agree(j - 2):
            return j + 1 if j < r else r
        if r - j >= 3 * w:
            return None
        j -= 1
    return None


def locate(f, good, ncc, dE, r, w, vl, J):
    """Level row for a jump peak: the upper edge of the transition band below the plateau above
    (band_top), refined by the photometric step inside the band (band_step); a clean step between two
    reliable rows is placed at its sub-row mid-crossing."""
    top = band_top(lambda j: good[j] and ncc[j] >= 0.55 and abs(f[j] - vl) <= abs(J) / 2, r, w)
    if top is None:                                        # never reached the upper plateau: not a boundary
        return None
    bk = band_step(dE, top, r + 1)
    if bk is not None:
        return float(bk)
    if top == r:
        return float(r)
    j = top - 1
    if good[j] and good[j + 1] and f[j + 1] != f[j]:       # clean step: sub-row mid-crossing
        mid = vl + J / 2
        t = (mid - f[j]) / (f[j + 1] - f[j])
        return j + float(np.clip(t, 0, 1))
    return float(top)


def identity_end(ident, known, w, edge_top, edge_bot):
    """End of the identity plateau. Above the liquid the compare frame reproduces the baseline geometry
    (M = 1, no flip) — through clean glass (nothing changed) as well as through deposit haze (pattern
    still in place). Below the surface the pattern is refracted (liquid) or reflected (total internal
    reflection band) and never matches the identity again. Rows whose warp cannot be measured (dense
    deposit, dark band, too few changed columns) are unknown and skipped, never counted as evidence.
    Returns the first known row r preceded by K known rows that are >= 60 % identity and followed by
    (including r) K known rows that are >= 80 % non-identity, or None."""
    n = len(ident); kidx = np.nonzero(known)[0]
    K = max(4, w // 2)
    for i in range(K, len(kidx) - K + 1):
        r = kidx[i]
        if r < edge_top or r >= n - edge_bot:
            continue
        if not ident[r] and ident[kidx[i - K:i]].mean() >= 0.6 and (~ident[kidx[i:i + K]]).mean() >= 0.8:
            return int(r)
    return None


def box_smooth(a: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k) | 1); pad = k // 2
    ap = np.pad(a.astype(np.float64), pad, mode="edge")
    return np.convolve(ap, np.ones(k) / k, mode="valid")


def step_from_profile(prof: np.ndarray) -> float:
    """Legacy photometric primary of the web app (largest mean(below) - mean(above) split of the
    difference profile, sub-row refined at the mid crossing) — kept for comparison in the evaluation."""
    h = len(prof); ks = max(3, round(h * 0.012) | 1); sm = box_smooth(prof, ks)
    cs = np.concatenate([[0.0], np.cumsum(sm)])
    min_seg = max(2, round(h * 0.03))
    i = np.arange(min_seg, h - min_seg + 1)
    score = (cs[h] - cs[i]) / (h - i) - cs[i] / i
    bi = int(i[np.argmax(score)])
    above, below = cs[bi] / bi, (cs[h] - cs[bi]) / (h - bi); mid = 0.5 * (above + below)
    for j in range(max(1, bi - ks), min(h - 1, bi + ks) + 1):
        if (sm[j - 1] - mid) * (sm[j] - mid) <= 0 and sm[j] != sm[j - 1]:
            return float(j - 1 + (mid - sm[j - 1]) / (sm[j] - sm[j - 1]))
    return float(bi)


def spikes(prof: np.ndarray, w: int, edge_top: int, edge_bot: int, z_min: float = 6.0, max_levels: int = MAX_LEVELS):
    """Photometric fallback (baseline without pattern): rows where the difference profile spikes
    (meniscus / interface line), in robust sigma units of the whole profile."""
    n = len(prof); sm = np.convolve(prof, np.ones(3) / 3, mode="same")
    base = np.median(sm); mad = 1.4826 * np.median(np.abs(sm - base)) + 1e-4
    z = (sm - base) / mad; z[:edge_top] = 0; z[n - edge_bot:] = 0
    peaks = []
    for r in np.argsort(-z):
        if z[r] < z_min or len(peaks) >= max_levels:
            break
        if all(abs(r - p) >= 2 * w for p in peaks):
            peaks.append(int(r))
    return sorted(peaks), z


def detect(b_roi: np.ndarray, c_roi: np.ndarray, max_levels: int = MAX_LEVELS, horizon_row: float | None = None) -> Detection:
    """b_roi / c_roi: float RGB (0..1) crops of the ROI at working resolution. Rows in the result
    are ROI-local half-resolution rows (multiply by 2 for working rows). `horizon_row`: ROI-local
    half-resolution row of the camera's optical axis (image centre for a level camera); it tells
    whether the free surface is seen from above or from below (None: assume from below)."""
    B, Ca, shift = robust_pair(b_roi, c_roi)
    n = B.shape[0]
    dE, _, thresh = photometric_profile(B, Ca)
    w = max(6, round(n * 0.06)); edge_top, edge_bot = max(w, round(n * 0.04)), max(w, round(n * 0.12))
    struct = row_structure(B) > STRUCT_MIN
    det = Detection(mode="warp", dE=dE, shift=shift)
    if struct.mean() < 0.3:                                # panel without pattern: photometric fallback,
        det.mode = "photometric"; det.frac_pattern = float(struct.mean())   # strongest line only (deposit
        pk, z = spikes(dE, w, edge_top, edge_bot, max_levels=1)              # rings are indistinguishable)
        det.zscore = z; det.levels = [{"row": float(p), "jump": float("nan"), "z": float(z[p])} for p in pk]
        return det
    weights, phi = change_weights(B, Ca, thresh)
    logM, ncc, flip, ncc1, amb = warp_profile(B, Ca, weights=weights)
    det.amb = amb
    unchanged = phi < PHI_MIN                              # nothing moved in this row: identity warp by definition
    logM[unchanged] = 0.0; flip[unchanged] = 1.0; ncc[unchanged] = 1.0; ncc1[unchanged] = 1.0
    # a warp is measurable only where enough columns changed (a handful of droplet columns would be
    # over-fitted by the 98 candidate warps) and the baseline row carries pattern
    measurable = struct & (unchanged | (phi >= PHI_MEAS))
    good = (ncc > NCC_MIN) & measurable
    det.logM, det.ncc, det.good, det.frac_pattern, det.phi, det.ncc1 = logM, ncc, good, float(good.mean()), phi, ncc1
    # Row classes. IDENTITY: the identity hypothesis explains the row (nearly) as well as any warp —
    # clean glass, deposit haze, film. LIQUID: a trusted warp that is a real lens (|log M| >= LOGM_LIQ)
    # or a mirror image. Everything else (dark band, dense deposit, wobbling deposit refraction, too
    # few changed columns) is UNKNOWN and never used as evidence for the surface.
    identity = struct & (unchanged | ((ncc1 >= NCC_SURE) & (ncc1 >= ncc - 0.15)))   # M = 1 has no free parameter: no over-fit
    liquid = measurable & ~identity & (ncc >= NCC_SURE) & ((np.abs(logM) >= LOGM_LIQ) | (flip < 0))
    det.cls = identity.astype(np.int8) + 2 * liquid.astype(np.int8)
    lum = np.array([0.299, 0.587, 0.114], np.float32); cx = (B.shape[1] - 1) / 2
    c0, c1 = int(round(cx - CENTRAL * B.shape[1] / 2)), int(round(cx + CENTRAL * B.shape[1] / 2))
    rho = (Ca[:, c0:c1] @ lum).mean(axis=1) / ((B[:, c0:c1] @ lum).mean(axis=1) + 1e-3); det.rho = rho
    idx = np.arange(n)
    accepted = []                                          # [{row, jump, z, z2, sigma, quality, ncc_below}]
    if good.sum() >= 2 * max(4, w // 3):
        lm = np.interp(idx, idx[good], logM[good]).astype(np.float32)
        f = median_filter1d(lm, max(3, round(n * 0.03))); det.f = f
        wgt = np.clip((ncc - NCC_MIN) / (1 - NCC_MIN), 0.05, 1.0) ** 2      # weakly correlated rows barely count
        J, Z, VL, S = gap_jumps(f, good, wgt, w, sigma_floor=0.02)
        J2, Z2, _, _ = gap_jumps(f, good, wgt, 2 * w, sigma_floor=0.02)     # coarser scale
        Z[:edge_top] = 0; Z[n - edge_bot:] = 0
        det.zscore = Z
        # A boundary is a step: the same jump is measured with w-row and 2w-row plateaus. Curvature
        # grows with the window and a noise blip shrinks with it — both are rejected.
        consistent = (Z >= Z_MIN) & (np.abs(J) >= J_MIN) & (Z2 >= 0.6 * Z_MIN) & (np.abs(J2 - J) <= 0.4 * np.abs(J) + 0.03)
        for r in np.argsort(-Z):
            if Z[r] < Z_MIN or len(accepted) >= max_levels + 2:
                break
            if not consistent[r]:
                continue
            row = locate(f, good, ncc, dE, int(r), w, VL[r], J[r])
            if row is None:
                continue
            if all(abs(row - a["row"]) >= 2 * w for a in accepted):      # separation on LOCATED rows
                lo, hi = max(0, int(row) - 2 * w), min(n, int(row) + 2 * w)
                q_up = good[lo:int(row)]; q_dn = good[int(row):hi]
                quality = min(q_up.mean() if len(q_up) else 0, q_dn.mean() if len(q_dn) else 0)
                ncc_dn = float(ncc[int(row):hi][q_dn].mean()) if q_dn.any() else 0.0
                # contact-line anomaly: two liquids meet the wall with a meniscus that redirects light, so a
                # liquid|liquid interface always shows a bright or dark line; a magnification regime change
                # inside ONE liquid (focal caustic of a spherical / conical vessel) is photometrically smooth.
                ri = int(round(row)); hl = max(3, round(n * 0.015)); near = rho[max(0, ri - hl):ri + hl + 1]
                ctx = np.concatenate([rho[max(0, ri - 2 * w):max(0, ri - hl)], rho[ri + hl + 1:min(n, ri + 2 * w)]])
                line = float(np.max(np.abs(near - np.median(ctx)))) if len(ctx) and len(near) else 0.0
                accepted.append({"row": row, "jump": float(J[r]), "z": float(Z[r]), "z2": float(Z2[r]), "sigma": float(S[r]),
                                 "quality": float(quality), "ncc_below": ncc_dn, "line": line})
        accepted.sort(key=lambda d: d["row"])
    det.cands = [dict(a) for a in accepted]
    # ---- free surface as the end of the identity plateau ------------------------------------
    # Above the liquid the compare frame reproduces the baseline geometry (clean glass, deposit haze,
    # film); below it never does again. The plateau ends at the last identity row that is preceded by
    # a mostly-identity stretch and followed by a long stretch with (almost) no identity rows; trailing
    # identity rows (gaps <= 2) still belong to the plateau. Unknown rows are transparent. The
    # identity end is used as the primary level when the jump detector found nothing within 2w below
    # it (odd vessel shapes where the liquid plateau cannot be measured); otherwise the jump's located
    # row (upper edge of the transition band, refined by the band's photometric step) is kept.
    long = max(2 * w, round(n * 0.15)); top = None
    for r in range(edge_top, n - edge_bot):
        if identity[r] and identity[max(0, r - long):r + 1].mean() >= 0.6 and identity[r + 1:r + 1 + long].mean() < 0.15:
            j = r
            while j + 1 < n - edge_bot and identity[j + 1:j + 4].any():
                j += 1
            top = j + 1; break
    det.ident = identity; det.surface = top
    if top is not None:
        accepted = [a for a in accepted if a["row"] >= top - w]
        first = accepted[0]["row"] if accepted else None
        # The contact line (meniscus at the front wall) is dark in transmission from any viewpoint,
        # and so is the total-internal-reflection band that starts at the level when the surface is
        # seen from below. Seen from above, the rows between the identity end and the level show the
        # far rim's view of the surface (mirrored panel / dark room): the first darkening below the
        # identity end is therefore the front-rim level. Reference = luminance of the air region.
        ref_rows = [r for r in range(max(0, top - long), top) if identity[r]]
        ref = float(np.median(rho[ref_rows])) if ref_rows else 1.0
        rs = np.convolve(rho, np.ones(3) / 3, mode="same")
        stop = int(min(n - 2, top + 3 * w, (first + w) if first is not None else n))
        dip = next((r for r in range(top, stop) if rs[r] < RHO_DIP * ref and rs[r + 1] < RHO_DIP * ref), None)
        if first is not None and first <= top + 2 * w:
            # The first jump belongs to the surface. A jump located above the identity end means the
            # trailing identity rows were band artefacts (mirrored periodic pattern): take the jump.
            # Otherwise the level is the contact-line dip, else the identity end moved down to the
            # photometric step inside the band when dense deposit haze sits directly above the liquid.
            if first < top:
                row_s = float(first)
            elif dip is not None:
                row_s = float(dip)
            else:
                bk = band_step(dE, top, int(first) + 1)
                row_s = float(bk if bk is not None else top)
            accepted[0] = {**accepted[0], "row": row_s}
        else:
            # The plateau ended long before the first jump (or there is none). If the rows in between
            # still show the pattern in place weakly (ncc1 >= 0.3: haze, film, thin deposit) the
            # plateau really continues and the jump is the surface; if they do not (dark, reflected or
            # refracted: a band the jump detector could not bridge) the identity end is the surface.
            between = ncc1[top:min(n, int(first))] if first is not None else np.zeros(0)
            if not len(between) or (between >= 0.3).mean() < 0.5:
                row_s = float(dip) if dip is not None else float(top)
                accepted.insert(0, {"row": row_s, "jump": float("nan"), "z": float("inf"), "z2": float("inf"), "sigma": 0.0, "quality": 1.0, "ncc_below": 1.0, "line": 1.0})
                det.mode = "warp/identity-end"
    # Secondary boundaries (liquid|liquid interfaces) are reported only where the magnification is
    # measured stably on both sides and the jump is a step: its significance must not shrink when the
    # plateau windows are doubled (curvature / caustic artefacts of a vessel whose radius varies do).
    # The lowest BOTTOM_ZONE of the ROI is where rays pass through the vessel bottom (a real warp
    # change that is not a liquid boundary): no secondary boundary is reported there.
    det.levels = [a for i, a in enumerate(accepted)
                  if i == 0 or (a["sigma"] <= SIGMA_MAX and a["z"] >= Z_MIN_SECONDARY and a["z2"] >= 0.6 * a["z"]
                                and a["quality"] >= 0.5 and a["ncc_below"] >= 0.55 and a["line"] >= LINE_MIN
                                and a["row"] < n * (1 - BOTTOM_ZONE))][:max_levels]
    for a in det.levels:
        for k in ("sigma", "quality", "ncc_below", "z2", "line"):
            a.pop(k, None)
    return det

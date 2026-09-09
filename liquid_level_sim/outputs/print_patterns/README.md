# Backlit calibration patterns (A4 transparency film)

Print `ChemTankSim_backlit_patterns_A4.pdf` (one pattern per page) at **100 % scale** on inkjet or laser transparency film, then tape the film onto the A4 white light box. Check the 100 mm scale bar with a ruler after printing. Pigment inkjet on inkjet film gives the densest black; a laser print on laser film is fine too. Mount the film flat against the diffuser (no air gap) with the pattern area centred behind the vessel; keep the room dark.

| id | file | pattern | size | when to use |
|---|---|---|---|---|
| P1 | `P1_checker_bw_4mm.png` | black / clear checkerboard | 4 mm cells | Highest contrast on film. Narrow vessels and strong magnification (a 7 cm glass shows 2-5 cm of panel). |
| P2 | `P2_checker_bw_8mm.png` | black / clear checkerboard | 8 mm cells | Highest contrast. Wide vessels (>= 12 cm) or a panel far behind the vessel. |
| P3 | `P3_checker_rg_6mm.png` | red / green checkerboard | 6 mm cells | Same pattern family as the synthetic study and the first real tests (colour cameras). Periodic: prefer P4-P8 for the detector. |
| P4 | `P4_mosaic_4mm.png` | aperiodic colour mosaic | 4 mm cells | Recommended for the level detector: no aliasing, flip visible. Narrow vessels / high magnification. |
| P5 | `P5_mosaic_8mm.png` | aperiodic colour mosaic | 8 mm cells | Recommended general-purpose pattern for 8-15 cm vessels at 40-80 cm camera distance. |
| P6 | `P6_mosaic_12mm.png` | aperiodic colour mosaic | 12 mm cells | Wide vessels, distant camera, or low-resolution video (cells must stay >= 6 px after magnification). |
| P7 | `P7_bars_bw_random.png` | aperiodic black / clear bars | 2-9 mm widths | Every row identical: immune to vertical misalignment (camera drift, vessel shifting, refocus). Max contrast. |
| P8 | `P8_bars_rgbk_random.png` | aperiodic colour bars | 3-10 mm widths | Row-invariant like P7, with colour for the RGB correlation; flip and magnification unambiguous. |

Choosing a size: the detector wants 6-12 cells across the part of the panel seen through the vessel. Through the liquid the panel appears magnified 1.3-4x, so a 7 cm glass shows only 2-5 cm of panel (use 4 mm cells or the bars); a 15 cm vessel or a panel far behind the vessel needs 8-12 mm cells. In the image, cells must stay at least ~6 px wide after the working resolution (the web app analyses at half resolution of a 675 px wide frame).

Why aperiodic: a periodic checker magnified by the liquid correlates with itself at several magnifications, which produced false identity rows and false second interfaces in the real footage of 2026-09-07; random-colour cells and random-width bars do not, and they also make the left-right flip of a real image visible. Bars (P7, P8) are additionally immune to a vertical misalignment between the empty baseline and the filled frame, because every row carries the same content.

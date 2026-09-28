# Paper draft

Working skeleton for the merger-timing paper. MNRAS template for now; moving
to AASTeX or A&A later is mostly a class-file and bibliography-style change.

## Build

```bash
sbatch build.sh          # latexmk -pdf main.tex via the texlive module
```

`mnras.cls` comes from the `texlive/20240312-xnyd` module; `build.sh` falls back
to fetching it from CTAN if a future TeX installation lacks it. Output:
`main.pdf` (gitignored — rebuild rather than commit).

## Layout

| file | what |
|---|---|
| `main.tex` | preamble, front matter and the `\input` list |
| `sections/` | one file per section (`0_abstract` ... `8_conclusions`, appendices `A_`, `B_`), bullet-point talking points under each heading |
| `refs.bib` | starter bibliography; entries marked TODO need checking on ADS |
| `figures/` | curated copies of existing plots (tracked in git on purpose) |
| `build.sh` | SLURM build script |

In the text, **red `[TODO: ...]`** marks things to write or decide, and
**orange `[STALE: ...]`** marks a figure or number that predates a fix and must
be regenerated before it can be used. Grey boxes are figures still to be made.
Citations are written out in the bullets for now, not yet `\cite`d, so the
bibliography section is empty in the PDF.

## Figures

`fig_*.pdf` are made by `scripts/make_figures.py` from the current datasets
(run from the repo root through SLURM:
`sbatch paper/scripts/run_py.sh paper/scripts/make_figures.py [name ...]`);
it prints the numbers each caption quotes. The `.png` files are copies of
older plots from the repo root, kept under their original names.

| figure | made by | inputs |
|---|---|---|
| `fig_radio_gallery.pdf` | `make_figures.py radio_gallery` | `dataset_nh4_512.h5`, `injected_nh4_simflux.h5` |
| `fig_radio_mf.pdf` | `make_figures.py radio_mf` | `injected_nh4_simflux.h5` (25 LoTSS targets) |
| `fig_mf_tsc.pdf` | `make_figures.py mf_tsc` | `dataset_nh4_128.h5` |
| `fig_lx_mass.pdf` | `make_figures.py lx_mass` | TNG catalogue, LoVoCCS target list, `lovoccs_wl_masses.csv`, `xray_real_placed_lx.h5` |
| `fig_xray_gallery.pdf` | `make_figures.py xray_gallery` | `xray_real_placed_lx.h5` |
| `fig_relic.pdf` | `detect_relics_radio.py --dataset dataset_nh4_512.h5`, then `make_figures.py relic` | `relic_catalog_radio_nh4.h5` |
| `fig_aug.pdf` | `submit_cnn_aug_oof_inner.sh`, then `make_figures.py aug` | `cnn_aug_oof_128_inner/` |
| `lotss_A119.png`, `lotss_A2443.png` | `download_lotss_image.py` | raw cutouts, placeholder for a proper gallery |
| `camels_mass_tsc.png` | CAMELS label scripts | negative result |

### Still to make (placeholders in `main.tex`)

- emitting-gas phase diagram, before/after the density cut
- label distribution and pseudo-TSC vs merger-catalogue TSC
- scalar baselines vs CNN bar chart
- true vs predicted merger time with residuals
- X-ray accuracy vs exposure
- simulated P150–M500 relation coloured by merger time
- radio vs X-ray predictions for the real clusters
- a uniform radio + X-ray gallery of real targets on the model grid

Numbers quoted in the bullets come from `PAPER_CHECKLIST.md` and the commit
log; several will move when the snapshot-91 data is added.

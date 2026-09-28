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
| `main.tex` | sections with bullet-point talking points under each heading |
| `refs.bib` | starter bibliography; entries marked TODO need checking on ADS |
| `figures/` | curated copies of existing plots (tracked in git on purpose) |
| `build.sh` | SLURM build script |

In the text, **red `[TODO: ...]`** marks things to write or decide, and
**orange `[STALE: ...]`** marks a figure or number that predates a fix and must
be regenerated before it can be used. Grey boxes are figures still to be made.
Citations are written out in the bullets for now, not yet `\cite`d, so the
bibliography section is empty in the PDF.

## Figures

Every figure in `figures/` is a copy of an existing plot in the repo root, kept
under its original name so its provenance is traceable.

| figure | made by | status |
|---|---|---|
| `forward_examples.png` | `plot_forward_examples.py` | stale: before noise-correlation fix and field injection |
| `regen_nh4_arc_s0.png` | `forward_model_lotss.py` | stale: analytic-noise tier, 17 targets |
| `minkowski.png` | `minkowski_functionals.py` | stale: pre-recentring, pre-density-cut |
| `sim_obs_dist.png` | `compare_sim_obs_distributions.py` | L_X panels valid; mass panel predates WL masses |
| `xray_real_vs_mock.png` | `plot_xray_real_vs_mock.py` | stale: before background fixes |
| `lotss_A119.png`, `lotss_A2443.png` | `download_lotss_image.py` | raw cutouts, placeholder for a proper gallery |
| `relic_v3_validation.png`, `relic_radio_validation.png` | `plot_relic_validation_v3.py`, `detect_relics_radio.py` | negative result; predates audit |
| `aug_comparison_128.png` | `train_cnn_aug.py` | negative result; single-split protocol |
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

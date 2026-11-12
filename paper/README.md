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
| `sections/` | manuscript sections (`0_abstract` ... `8_conclusions`, appendices `A_`, `B_`) |
| `refs.bib` | manuscript bibliography |
| `figures/` | curated copies of existing plots (tracked in git on purpose) |
| `build.sh` | SLURM build script |
| `todos.md` | current experiment, stale-result and prose checklist |
| `todos_archive_20261112.md` | historical checklist; pending statuses are superseded |

In the text, **red `[TODO: ...]`** marks things to write or decide, and
**orange `[STALE: ...]`** marks a figure or number that predates a fix and must
be regenerated before it can be used. **Blue `Suggestion:`** annotations
contain proposed prose or editorial decisions for review; they can be hidden
with `\showsuggestionsfalse` in `main.tex`. Grey boxes are figures still to be made.
The manuscript uses citations from `refs.bib`; remaining reference TODOs
are tracked in the source and current checklist.

## Figures

Nine figures were regenerated for the current (synchrotron) radio
prescription by `scripts/regen_stale.py` (review rounds 48--55); their exact
provenance (source run, file hash, replaced hash, task records) is in
`staged/PROMOTED_PACKAGE.json`. Each task writes `<task>.json` (every quoted
number, estimand, checks, input fingerprints) into its run directory, and all
except W8 also write `<task>_arrays.npz` (W8 keeps its target, prediction and
error-bar vectors in `w8.json`). The remaining figures were made by `scripts/make_figures.py`
(historical recipe; its radio input paths predate the regenerated model, so
do not use it for the nine figures below). Run everything from the repo root
through SLURM. The `.png` files are copies of older plots from the repo root,
kept under their original names.

Regeneration chain: `run_phase_syn.sh` (job A, `extract_phase_syn.py`, raw
cutouts -> `phase_cells_syn_144_8941185.npz` + manifest),
`run_cells_1400.sh` (job B, 1.4 GHz cells -> `dataset_syn_nh4_1400_128.h5` +
manifest), then `run_regen_final.sh` (all nine tasks, run 8942495) and
`run_regen_w8.sh` (W8 layout refresh, run 8945384).

| figure | made by | inputs |
|---|---|---|
| `fig_baselines.pdf` | `regen_stale.py` w1 (run 8942495) | `dataset_syn_nh4_144_128{,_shape}.h5`, `cnn_preds/pooled_{,shape_}syn_inner_s*.npz`, TNG catalogue |
| `fig_mf_tsc.pdf` | `regen_stale.py` w2 (run 8942495) | `dataset_syn_nh4_144_128.h5`, `dataset_syn_nh4_1400_128.h5` (job B) |
| `fig_phase.pdf` | `regen_stale.py` w3 (run 8942495) | job A product, `dataset_syn_nh4_{144,1400}_128.h5`, `Radio_Cells_syn_nh4_{144,1400}` |
| `fig_radio_gallery.pdf`, `fig_radio_mf.pdf` | `regen_stale.py` w5 (run 8942495) | `injected_syn_phys_b3000.h5`, `dataset_syn_nh4_144_512.h5` (26 LoTSS targets) |
| `fig_tsc_pred.pdf` | `regen_stale.py` w6 (run 8942495) | `cnn_preds/pooled_syn_inner_s*.npz`, `cnn_preds/syn_phys_b1000_joint_s*_preds.npz` |
| `fig_pm_relation.pdf` | `regen_stale.py` w7 (run 8942495) | `dataset_syn_nh4_144_512.h5`, `injected_syn_phys_b3000.h5` |
| `fig_real_preds.pdf` | `regen_stale.py` w8 (run 8945384) | `cnn_preds/syn_phys_b3000_sfimg_s*`, `cnn_preds/fix_x_s*`, `lovoccs_wl_masses.csv` |
| `fig_relic.pdf` | `regen_stale.py` w9 (run 8942495; redetection agrees element-wise with `relic_catalog_radio_syn_nh4_144.h5`) | `dataset_syn_nh4_144_512.h5`, `TSC_Cutimages/TSC_eachhalo_snap99.hdf5`, `dataset.h5` (legacy labels) |
| `fig_lx_mass.pdf` | `make_figures.py lx_mass` | TNG catalogue, LoVoCCS target list, `lovoccs_wl_masses.csv`, `xray_placed_144.h5` |
| `fig_xray_gallery.pdf` | `make_figures.py xray_gallery` | `xray_placed_144.h5` |
| `fig_aug.pdf` | `submit_cnn_aug_oof_inner.sh`, then `make_figures.py aug` | `cnn_aug_oof_128_inner/` (historical, non-independent) |
| `fig_obs_gallery.pdf` | `make_figures.py obs_gallery` | `injected_nh4_144_simflux.h5`, `xray_placed_144.h5` (obs groups) |
| `fig_labels.pdf` | `make_figures.py labels` | datasets, label files |
| `fig_xray_depth.pdf` | `submit_fig_data.sh`, `submit_xray_depth.sh`, then `make_figures.py xray_depth` | `cnn_preds/xdepth_*` |
| `camels_mass_tsc.png` | CAMELS label scripts | negative result |


## Numerical update (2026-11-05 results; status refreshed 2026-11-12)

The abstract, Results, realistic-performance table, Discussion and Conclusions
now use the reviewed regenerated-radio CNN results from
`logs/compare_syn_8428907.out`, accepted with
the corrected interpretations and validator changes in
`CODEX_CLAUDE_REVIEW.md`, rounds 26–29.
The radio realism summary uses
`logs/syn_inject_8296721.out`. These logs and the review record are local
Oscar evidence, not files included in the manuscript push.
These are **pseudo-TSC** results. The separate catalogue-TSC comparison,
target diagnostics and MSE follow-up were reviewed on 2026-11-06; choosing
how to incorporate them remains a manuscript decision. Their reviewed
findings and source paths are summarized in [todos.md](todos.md).

The X-ray reference (single-observation R² 0.406) is retained. The control
matched to the 3 Mpc **radio** experiment (R² 0.403) is reported separately;
the box label does not describe a different X-ray crop. Existing X-ray-only,
optical, loss/head and contrastive experiments keep their separate protocols.

An earlier source-only update replaced the regenerated CNN scores but left
the ancillary radio diagnostics and their figures marked stale. Those 17
annotations were then replaced with regenerated evidence (review rounds
48--56): nine figures and their captions/prose come from the mixed package in
`staged/PROMOTED_PACKAGE.json` (W1--W7 and W9 from consolidated run 8942495,
W8 from its layout refresh), with both run manifests retained unchanged. The
paper was rebuilt with `sbatch build.sh` and every page inspected; no `[STALE`
text remains.

See [todos.md](todos.md) for the remaining author decisions.

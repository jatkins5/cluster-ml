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

`fig_*.pdf` were made by `scripts/make_figures.py` from the datasets listed below
(run from the repo root through SLURM:
`sbatch paper/scripts/run_py.sh paper/scripts/make_figures.py [name ...]`);
it prints the numbers each caption quotes. Audit the script's input paths
before regeneration: this table records the existing assets, not an updated
recipe for the regenerated radio model. The `.png` files are copies of
older plots from the repo root, kept under their original names.

| figure | made by | inputs |
|---|---|---|
| `fig_radio_gallery.pdf` | `make_figures.py radio_gallery` | `dataset_nh4_144_512.h5`, `injected_nh4_144_simflux.h5` |
| `fig_radio_mf.pdf` | `make_figures.py radio_mf` | `injected_nh4_144_simflux.h5` (25 LoTSS targets) |
| `fig_mf_tsc.pdf` | `make_figures.py mf_tsc` | `dataset_nh4_144_128.h5` |
| `fig_lx_mass.pdf` | `make_figures.py lx_mass` | TNG catalogue, LoVoCCS target list, `lovoccs_wl_masses.csv`, `xray_placed_144.h5` |
| `fig_xray_gallery.pdf` | `make_figures.py xray_gallery` | `xray_placed_144.h5` |
| `fig_relic.pdf` | `detect_relics_radio.py --dataset dataset_nh4_144_512.h5`, then `make_figures.py relic` | `relic_catalog_radio_nh4_144.h5` |
| `fig_aug.pdf` | `submit_cnn_aug_oof_inner.sh`, then `make_figures.py aug` | `cnn_aug_oof_128_inner/` |
| `fig_obs_gallery.pdf` | `make_figures.py obs_gallery` | `injected_nh4_144_simflux.h5`, `xray_placed_144.h5` (obs groups) |
| `fig_phase.pdf` | `extract_phase.py`, then `make_figures.py phase` | snapshot-99 cutouts -> `phase_cells_144.npz` |
| `fig_labels.pdf`, `fig_baselines.pdf`, `fig_tsc_pred.pdf` | `make_figures.py labels baselines tsc_pred` | datasets, label files, saved CNN predictions |
| `fig_pm_relation.pdf` | `make_figures.py pm_relation` | `dataset_nh4_144_512.h5`, `injected_nh4_144_simflux.h5` |
| `fig_xray_depth.pdf` | `submit_fig_data.sh`, `submit_xray_depth.sh`, then `make_figures.py xray_depth` | `cnn_preds/xdepth_*` |
| `fig_real_preds.pdf` | `make_figures.py real_preds` | `cnn_preds/s9991_{sfimg,x}_*` |
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

No figures or PDF were regenerated in this source update. Orange annotations
identify retained figures and diagnostics from the previous radio prescription;
their numerical captions describe those older plots. The figure-input table
above is therefore a record of the existing assets, not a claim that they use
the regenerated model. Scalar/mass-bin, power–mass, gas-phase, intrinsic
Minkowski-functional, relic/peak and residual-bias summaries still need updated
evidence before their historical numbers can be replaced. Do not silently
substitute current scores into an unchanged plot's caption.

See [todos.md](todos.md) for the update inventory and prose decisions. The
standard build remains `sbatch build.sh`; this update did not submit that job
or change the running experiments.

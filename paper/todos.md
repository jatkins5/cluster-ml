# Manuscript work remaining

Updated 2026-11-12 after checking the manuscript, saved comparison logs,
follow-up outputs, and completed result reviews. This is the active list;
the [previous checklist](todos_archive_20261112.md) is archived because many
of its prose fixes and experiment requests have been completed.

This update publishes the regenerated-radio numerical edits and existing
blue suggestions. It does not regenerate figures or adopt every suggestion.
The paper still uses **pseudo-TSC** as its primary endpoint. Source logs and
prediction archives named below are on Oscar and are not included in this
manuscript push; the linked relic review bundle is already tracked.

## 1. Existing outputs that still need analysis or review

- [ ] **Radio–X-ray offset: finish the conditional interpretation.** Inspect
  `runs/followup_20261103T180830Z/offset/conditional_bias.csv`,
  `observed_age_sensitivity.csv`, `observed_offsets.csv`, and
  `offset_diagnostic.pdf`. Report support and uncertainty by mass, placement
  redshift, exposure and hypothetical true-age bin; retain unsupported bins
  rather than treating them as zero bias. The saved mixed-snapshot simulated
  mean is +0.010 Gyr [−0.049, +0.069], whereas the observed median is +0.493
  Gyr. Those are different summaries/cohorts and cannot by themselves show
  that the observed offset is unexplained. Observed true ages are unknown;
  the conditional comparison uses five matched models, whereas the headline
  real predictions average 25. Deliverable: a supported Discussion paragraph
  and, if useful, one conditional comparison plot. No initial training rerun
  is needed to inspect these outputs.
- [ ] **Relics: complete the visual review before fitting any age relation.**
  Use the six PDFs and tables in the
  [review bundle](../runs/followup_20261103T180830Z/README.md). Record coherent
  fronts, segmentation failures, plausible counterpart/event assignments,
  competing events, and confidence/reasons. Five of six halos have candidate
  opposite-side structures, not validated relic–event pairs. Snapshot 91
  precedes the selected event in all six halos; the two snapshots alone
  cannot establish association. Decide after review whether closer snapshots,
  trajectories and shock context justify a larger study. The 1.4 GHz pilot
  is not a replacement for the old 144 MHz projected-peak population test.

## 2. Completed analyses awaiting manuscript decisions

These do not belong in a queue of experiments still waiting to be scored.

- [x] **Regenerated-radio matrix, matched mass comparison and matched 3 Mpc
  joint comparison:** reviewed and incorporated into Results/the performance
  table; headline changes also appear in the summary sections. See the
  numerical inventory below. The existing Discussion request to run matched
  comparisons is obsolete; its adjacent suggestion supplies replacement text.
- [x] **Compressed-source morphology sensitivity:** already run with the
  regenerated radio inputs and represented in the table. At 3 Mpc,
  single-observation ensemble R² is 0.196 for the physical source profile
  and 0.209 for the compressed profile. This is a sensitivity comparison,
  not evidence that compressed morphology is a physically better model.
  Source: `logs/compare_syn_8428907.out`.
- [ ] **Catalogue-TSC comparison: choose whether/how to report it.** Production
  results and their interpretation were reviewed on 2026-11-06. Against the
  same capped catalogue endpoint, pseudo-trained → catalogue-trained R² is
  0.302 → 0.310 for X-ray and 0.170 → 0.185 for radio. Neither paired R²
  interval resolves a gain; the X-ray rank correlation decreases by 0.027
  [−0.053, −0.002]. These results do not establish equivalence or
  noninferiority, and comparing scores against different targets cannot
  establish label quality. Source: `logs/score_labels_8457071.out`.
- [ ] **Catalogue-TSC diagnostic and MSE follow-up: decide on a brief
  robustness paragraph or appendix.** Both are reviewed, not awaiting
  analysis. MSE changes catalogue-TSC X-ray R² from 0.310 to 0.334, with
  ΔR² +0.024 [−0.000, +0.049] (lower bound slightly negative), while broad
  age-dependent bias persists. Retain Huber on this evidence; no further
  loss sweep is required. The label-transfer diagnostic is not a ceiling
  on image information or a demonstrated superior calibration. Sources:
  `logs/tsc_diag_8460924.out`, `logs/score_lbl_mse_8463662.out`.
- [ ] **Spatial/centering pilot: decide whether to retain a historical pilot
  result or repeat with current radio inputs.** The existing pilot is
  already scored: masked/origin/registered R² = 0.3828/0.3936/0.3796, with
  all three paired R² intervals including zero. It uses the earlier radio
  inputs and does not establish that positions contain no useful signal.
  Source: `runs/spatial_20261103T021803Z/results.txt` and `provenance.json`.
  A current-prescription claim needs a matched rerun, not rewording alone.

## 3. Stale measurements and figures

The following need current-model evidence **if retained as current results**;
removing or explicitly retaining a result as historical is also an option.
Update the measurement, figure and caption together. The figure recipes in
[README.md](README.md) describe existing assets and still point to older
inputs; audit those paths before rebuilding.

| Item | Required update | Manuscript/figure affected |
|---|---|---|
| Clean ensemble and scalar/mass controls | Recompute the five-seed clean ensemble score; rerun scalar baselines and combined models using current features and matched folds; refresh mass-tercile scores, partial correlations and luminosity–prediction regressions. Existing CNN predictions may be reused where protocols match. | Results radio-only; `fig_baselines.pdf`; dependent claims in Abstract, Discussion, Conclusions |
| Intrinsic morphology | Re-extract current-map Minkowski features and rerun their predictive/threshold summaries, including any frequency comparison retained. This is separate from the completed compressed-source sensitivity experiment. | Results radio-only; `fig_mf_tsc.pdf` |
| Gas phase and frequency | Remeasure dense/cold fractions, brightest-cell contributions, radio-weighted Mach summaries, and the 144 MHz/1.4 GHz scatter and flux redistribution. Clarify cell versus projected pixel. | Simulations radio subsection; `fig_phase.pdf`; 99% statements in Discussion/Conclusions |
| Aperture retention | Recompute the emission fractions retained in 1 and 3 Mpc boxes from current maps. | Forward modelling (old 4% and 65%) |
| Mock images and realism curves | Regenerate the radio gallery and threshold-dependent morphology curves. Scalar agreement is already updated to 0.093/0.159/0.635, but the plotted curves are old. | `fig_radio_gallery.pdf`, `fig_radio_mf.pdf` |
| Prediction plots and age-dependent bias | Plot reviewed current predictions and recompute the bias summaries using the stated snapshot, target and ensemble. Do not substitute catalogue-TSC diagnostic values into the main pseudo-TSC result. | Results; `fig_tsc_pred.pdf`; old +0.35 and −1.2 to −1.4 Gyr statements |
| Radio power–mass relation | Refit slopes, scatter and residual correlation using current radio weights and the stated flux calibration. | Results/Discussion; `fig_pm_relation.pdf` |
| Observed predictions | Regenerate the plot from current radio predictions and recompute the separate radio/X-ray medians. The paired median offset and rank statistics are already updated; median differences do not determine separate medians. | LoVoCCS Results; `fig_real_preds.pdf`; old separate medians 1.10/0.83 Gyr |
| Projected peak/relic null test | If retained, verify the current catalogue's provenance and recompute peak statistics and plot for the regenerated 144 MHz maps. The small 3-D pilot does not replace this population diagnostic. | Discussion and negative-results appendix; `fig_relic.pdf` |

Do not classify every figure as stale merely because radio was regenerated.
The X-ray-only, optical, label-distribution and CAMELS analyses have separate
inputs/protocols. `fig_labels.pdf` needs revision only if its clock/label
comparison is changed; `fig_aug.pdf` remains a historical, non-independent
augmentation experiment and cannot substantiate a current augmentation gain.

## 4. New work or deferred experiments (no completed result to analyze)

- [ ] **Earlier-merger history diagnostic (B1/B2).** The separate analysis
  workflow has prepared `diagnose_history.py`, fixtures and a CPU launcher.
  The latest retained handoff is implementation review (round 44), with
  the B2 time-basis amendment and bootstrap-coverage qualification still
  under discussion. No production result is recorded there yet. Review
  production results when available; fixture results are not science results.
  This is a separate ongoing follow-up, not a prerequisite for publishing
  the already reviewed manuscript numbers.
- [ ] **Diffusion augmentation with independent test clusters.** Deferred.
  A valid new comparison must exclude downstream test clusters from generator
  training and selection as well as regressor training. The old comparison
  cannot establish the effect of augmentation. Decide whether the cost is
  justified before rerunning; retaining a qualified historical negative
  result is reasonable.
- [ ] **Other limitations are optional follow-ups, not automatic reruns:**
  density-threshold sensitivity, a decomposition of observing effects,
  probability calibration/coverage, alternative contrastive training, and
  transfer between simulations with consistent labels. No completed outputs
  for these tests were identified in this audit. Prioritize only those needed
  to support a retained claim.

## 5. Prose and provenance cleanup

- [ ] Adopt/revise the blue suggestions about matched comparisons, training
  versus evaluation, fixed noisy masses, and aperture labels. The matched
  X-ray control is 0.403; the retained 0.406 reference has different folds.
  “3 Mpc” describes the radio box, not a new X-ray crop.
- [ ] Replace old “trying to run” TODOs with the statuses above. The offset
  needs interpretation, the relics need visual/event review, and the matched
  comparisons already have reviewed results.
- [ ] Qualify the clean joint and density-cut gains: paired-seed p = 0.064
  and 0.197, respectively. An interval spanning zero is not equivalence.
  Describe shape normalization as retaining predictive signal, not retaining
  a measured percentage of physical merger-history information.
- [ ] Remove dependence on stale scalar/mass-bin, power–mass and gas-phase
  evidence from summary claims until those diagnostics are updated, or mark
  the historical scope explicitly where the claim is made.
- [ ] Check the revised radio derivation against the implemented lookup
  table. The old request to regenerate all radio CNN inputs is completed;
  physical validation and stale ancillary measurements remain separate tasks.
  Keep the density-cut explanation in its current locations for now, as
  requested; check the brightest-cell/pixel terminology.
- [ ] If adding catalogue-TSC training results, distinguish their continuous
  `T_coll` clock, cap and missing-event convention from the historical
  snapshot-derived catalogue comparison in Simulations/`fig_labels.pdf`.
  Catalogue completeness/event selection and the original score producer
  remain limitations; the partial reconstruction did not validate all halos.
- [ ] Review the existing ERGO-ML comparison suggestion and consolidate the
  repeated cross-study qualification. Complete the de Gasperin citation TODO
  if the radio power–mass comparison is retained.
- [ ] After choosing which results to keep, update figure inputs/captions,
  remove resolved annotations, compile the paper and inspect the PDF. This
  source-only push has not regenerated assets or compiled a new PDF.

Resolved older requests include the 352/351 snapshot description and grouping
by parent halo, the explicit pseudo-TSC definition, the expanded X-ray mock
methods, both TNG attribution citations, and the radio subsection structure.
The archive preserves their original wording; its unchecked boxes are not a
second active task list.

## Numerical changes included in this push

Source: `logs/compare_syn_8428907.out`, reviewed 2026-11-05. Seed scatter and
conditional parent-bootstrap intervals are different uncertainty summaries.

| Quantity | Current value |
|---|---|
| Clean radio / X-ray / joint mean R² | 0.498 / 0.511 / 0.535 |
| Shape-normalized radio | 0.413 ± 0.020; paired Δ −0.085 ± 0.043, p=0.012 |
| Clean joint minus radio | +0.037 ± 0.032, p=0.064 |
| Density cut, no cut → cut | 0.488 → 0.498, p=0.197 |
| Realistic radio, 1 / 3 Mpc single-observation five-model ensemble | 0.073 / 0.196 |
| Matched mass only → radio + mass | 0.176 → 0.305; Δ +0.129 [0.068, 0.190] |
| Retained X-ray reference → joint with 1 Mpc radio (descriptive) | 0.406 → 0.426; Δ +0.020 [−0.032, 0.074] |
| Matched X-ray control → joint with 3 Mpc radio | 0.403 → 0.401; Δ −0.002 [−0.057, 0.057] |
| Observed radio/X-ray Spearman | 0.67, p=0.006 |
| Partial rank correlation given redshift / WL mass | 0.74, p=0.003 / 0.58, p=0.078 (mass n=11) |
| Median paired observed radio minus X-ray offset | 0.49 Gyr; 14/15 positive |

The realistic table's pooled scores, seed scatter, one-model scores and
prediction–mass correlations also use this log. The radio-realism summary
is 0.093 versus 0.159 for the compressed source and 0.635 for background only,
from `logs/syn_inject_8296721.out`.

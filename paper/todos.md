# Manuscript work remaining

Updated 2026-11-12 after checking the manuscript, saved comparison logs,
follow-up outputs, and completed result reviews. This is the active list;
the [previous checklist](todos_archive_20261112.md) is archived because many
of its prose fixes and experiment requests have been completed.

This update publishes the regenerated-radio numerical edits and existing
blue suggestions; the stale figures and diagnostics were subsequently
regenerated and the PDF rebuilt (section 3). Blue suggestions are not adopted
on the author's behalf.
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

## 3. Stale measurements and figures (regenerated)

All 17 `\stale{}` annotations were replaced with regenerated evidence
(review rounds 48--55): `scripts/regen_stale.py`, consolidated run
`staged/regen_final_8942495` (W1--W7, W9) and W8 refresh
`staged/regen_W8_8945948` (earlier refresh 8945384 retained as superseded); provenance in `staged/PROMOTED_PACKAGE.json`;
annotation-to-evidence checklist in `CODEX_CLAUDE_REVIEW.md` (round 56).
Replacement estimands that differ from the historical ones are stated in
the text: the half-sample scalar+CNN comparison replaces the leaky stacked
0.541; the Minkowski lowest threshold selects the emitting footprint; the
aperture fractions are medians over projections; the frequency scatter and
redistribution use explicitly defined replacement statistics; the
power--mass fits use the log of projection-averaged linear power.

- [ ] Author review of the replaced text and the blue suggestions that now
  reference regenerated results (Results, Discussion, Conclusions, Abstract).

The X-ray-only, optical, label-distribution and CAMELS analyses have separate
inputs/protocols. `fig_labels.pdf` needs revision only if its clock/label
comparison is changed; `fig_aug.pdf` remains a historical, non-independent
augmentation experiment and cannot substantiate a current augmentation gain.

## 4. New work or deferred experiments (no completed result to analyze)

- [x] **Earlier-merger history diagnostic (B1/B2).** Completed and closed
  (production job 8936281; review rounds 42--47). B1 passed; the held-out
  B2 criterion failed, so no activity-target training. The proposed text is
  the blue suggestion in `B_negative_results.tex`; deferred follow-ups are
  not reopened.
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
- [x] Summary claims that depended on stale scalar/mass-bin, power–mass and
  gas-phase evidence now cite the regenerated diagnostics.
- [ ] Check the revised radio derivation against the implemented lookup
  table. The old request to regenerate all radio CNN inputs is completed,
  and the ancillary measurements have been regenerated (section 3);
  physical validation remains a separate task.
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
- [x] Update figure inputs/captions, remove resolved stale annotations,
  compile the paper and inspect the PDF (done with the regeneration package;
  section 3).

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

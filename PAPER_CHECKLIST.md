# Pre-paper checklist

Written 2026-09-09 after the audit in `audit_results.py` and
`audit_mass_confound.py` (SLURM jobs 6121694, 6121703). Ordered by what
blocks what: everything in **A** changes numbers we would otherwise print,
so it comes before the writing. **B** is new analysis the paper needs, **C**
is data hygiene, **D** is figures and text.

Claimed spine of the paper, for reference while working through this:

1. TSC is recoverable from clean TNG-Cluster radio maps (pooled CNN, 5-fold OOF).
2. The gas carrying the H&B weight is cold stripped ISM, not ICM; a density
   cut improves both distributional realism and predictivity.
3. A FullReal forward model (field injection + beam-correlated noise) closes
   8/10 image statistics against LoTSS.
4. The signal survives observational degradation and yields in-range,
   seed-stable predictions on real LoTSS clusters.

Items A1 and B1 are the two that currently undercut 1 and 4.

---

## A. Fix before quoting any number — DONE 2026-09-09

Outcome (jobs 6122014 / 6122015, `summarize_a3.py`). The honest numbers, to
be used everywhere from here:

| configuration | old (best-epoch on test fold) | inner-split selection | final epoch |
|---|---|---|---|
| pooled CNN, pseudo-TSC, n_H<1e-4, 128px | 0.567 ± 0.018 | **0.487 ± 0.033** | 0.514 ± 0.026 |
| field-injected transfer, mock OOF | 0.426 | **0.374 / 0.360** | 0.357 / 0.366 |

So the headline baseline loses **0.053–0.080** and the transfer result
**0.052–0.066**. The transfer drop is smaller than the +0.32 the logs
implied because the optimizer fix (A2) also stabilised the training curve,
so the two changes partly cancel — the model really is better, it was just
being scored dishonestly.

`inner` and `final` are statistically indistinguishable on the baseline
(paired −0.027 ± 0.018 SE, p=0.22, though 0/5 seeds favour inner — it pays
15% of its training data for the privilege). Reporting `inner` as the
headline anyway: with `final`, the epoch count is itself a hyperparameter
that was historically chosen by looking at test performance, and inner
selection makes it per-fold and data-driven. Quote `final` alongside it.

**The mass confound is untouched by the protocol fix** — ρ(OOF pred, M500)
is still −0.86 to −0.89 and ρ(real pred, total SNR) −0.77 to −0.91 across
all four transfer runs. B1/B2 are now the binding items.

- [x] **A1. Remove best-epoch-on-the-test-fold selection.**
  `train_cnn.py`, `train_cnn_pooled.py`, `train_cnn_mock.py` all keep the
  epoch with the best validation score *on the fold they then report*, with
  no inner split. Measured inflation (best − final-epoch fold R², from the
  logs): **+0.043** on the pooled/shallow baselines (60 folds), +0.039 on
  the density-cut grid (75 folds), **+0.32** on the injected-transfer runs
  (10 folds). Paired comparisons survive — both arms carry the same bias —
  but absolute numbers do not.
  *Do:* add an inner validation split for checkpoint selection, or fix the
  epoch count and let the cosine schedule land. Re-run headline configs.
- [x] **A2. Fix `train_cnn_mock.py`'s optimizer.** It uses Adam lr 1e-3 with
  no schedule, so per-epoch val R² swings between −0.5 and +0.5 and "0.426"
  is substantially the pick of the best swing. Match `train_cnn_pooled.py`
  (AdamW 3e-4 + cosine).
- [x] **A3. Re-run and re-record after A1/A2.** Pooled CNN + pseudo-TSC +
  `dataset_nh4_128.h5` (5 seeds); injected transfer (`injected_nh4.h5`,
  2+ seeds). Expect 0.567 → ~0.52 and 0.426 → lower.
- [x] **A4. Save OOF predictions from `train_cnn_pooled.py`.** It currently
  prints R² and discards `oof_preds`; B1 and the scatter figure both need
  them on disk (`np.savez` keyed by `halo_id`, as `train_cnn_mock.py` does).

## B. Analysis the paper is missing

- [ ] **B1. Mass / brightness control.** The headline result is only ~0.1
  above a one-number baseline, and the transfer model is mostly a mass
  regressor. Evidence:

  | baseline (clean sims, 5-fold OOF ridge) | OOF R² | Spearman vs TSC |
  |---|---|---|
  | log total linear weight | **0.432** | −0.635 |
  | log M500 | 0.314 | −0.559 |
  | mean arcsinh pixel | 0.204 | −0.381 |
  | all five scalars | **0.455** | — |
  | MF-XGBoost (existing) | 0.454 | — |
  | pooled CNN (pre-A1) | 0.567 | — |

  Transfer runs: ρ(pred, M500) = **−0.87**, ρ(pred, TSC) = 0.63,
  partial ρ(pred, TSC | M) = 0.30–0.39, R² inside mass terciles 0.08–0.24.
  *Do:* report R² within mass bins and partial correlations; put the scalar
  baselines in the main results table.
- [ ] **B2. Decide the mass treatment for the transfer model, then re-run.**
  The forward model discards sim brightness and re-anchors flux to
  P150 ∝ M500^3.55 (Cuciti+2023), so mock SNR *is* mass by construction.
  Cleanest fix: feed M500 as an explicit scalar input so the image has to
  supply the residual (LoVoCCS has WL masses for the real side). Alternative:
  a disturbance-dependent anchor — Cuciti Fig. 3 shows halos above the P–M
  relation are the X-ray-disturbed ones, and ~half of clusters at these
  masses have only upper limits, so a real *relaxed* cluster reads to our
  model as "faint → low mass → old" rather than "relaxed".
- [ ] **B3. Reconcile with Lee's relic-separation relation.** Their group now
  publishes TSC = 0.52 d_drr/R500c − 0.24, r = 0.83 in TNG-Cluster
  (arXiv:2510.21632); our README records four independent negative attempts
  to reproduce it. Most likely explanation is selection — double-relic
  systems and major mergers only, vs our any-mass-ratio label on all
  clusters — but the paper has to say so explicitly rather than leave a
  contradiction with a paper on the same simulation.
- [ ] **B4. Confirm the density cut with ckong13.** Lee+2024 applies no
  gas-phase cut, so `--max-nh 1e-4` is a departure from the published
  TNG-Cluster relic model, not just our post-processing. Needs her sign-off
  and its own methods subsection.

## C. Data hygiene

- [ ] **C1. Name-key bug drops 5 usable LoTSS targets.**
  `forward_model_lotss.load_obs` keys the target CSV by name with *spaces*
  removed (`MKW3s`) but the FITS files use underscores (`lotss_MKW_3s.fits`),
  so MKW 3s, RXC J0034.2−0204, RXC J0034.6−0208, RXC J1217.6+0339 and
  RX J0820.9+0751 are skipped as "no redshift in target list" — the CSV has
  their redshifts. Strip spaces *and* underscores on both sides.
- [ ] **C2. Re-download the three low-z targets larger.** A2052, A2063,
  A2147 (all z=0.035) fail the 1 Mpc crop because their cutouts are
  800 px × 1.5″ = 20′ and 1 Mpc subtends ~24′ there.
  `download_lotss_image.py --size-arcmin 40`. (A1750 is genuinely blank.)
- [ ] **C3. Rebuild the `obs/` groups** in `mock_dataset_*.h5` and
  `injected_*.h5` after C1/C2 — inference sample 17 → ~25.
- [ ] **C4. Regenerate the forward-model tables.** The `regen_*_mfs.npz`
  files (correlated noise, GroupPos, cuts) exist but the README KS tables
  still show pre-fix numbers. Nothing predating `--correlated-noise` should
  be published.
- [ ] **C5. Flag or re-run the stale downstream work.** CAMELS, diffusion,
  X-ray and dual-encoder results all predate GroupPos re-centring and the
  density cut. Decide per-result: re-run, or state the caveat.

## D. Writing, figures, corrections

- [ ] **D1. README number corrections.** Every CNN number in the Comparison
  table is on the retired protocol. The nh4 pooled baseline is now
  **0.487 ± 0.033** (inner) / 0.514 ± 0.026 (final) — see the A table — not
  0.567, and certainly not the single lucky 0.564 still printed. The
  re-centring comparison (0.536 → 0.549) and the density-cut grid
  (0.546 → 0.567) are *paired* results, so their deltas survive, but their
  absolute values do not: either re-run those grids under `--select inner`
  or quote only the deltas. Sweep for remaining single-seed numbers.
- [ ] **D2. Citation fix.** The injection precedent is **Bruno et al. 2023,
  A&A 672, A41** (Botteon is a co-author); memory and two commit messages
  say "Botteon 2023". Cite Cuciti's **BCES Y|X fit for 0.06 < z < 0.4**
  specifically — A = 1.1, B = 3.55, σ_raw = 0.35 — since the orthogonal
  slope for the same sample is 4.79 and we use the Y|X constants.
- [ ] **D3. Figures.**
  - label distribution + pseudo-TSC vs merger-catalog TSC (Spearman 0.961,
    supports using either)
  - example gallery: clean sim / SemiReal / FullReal / real LoTSS
  - domain-gap statistics table (8/10 KS tests non-rejecting)
  - OOF scatter, true vs predicted, with per-bin residuals
  - **real-cluster predictions vs total SNR and vs mass** — own the
    confound in the paper rather than let a referee find it
- [ ] **D4. Do not present A399/A401 as validation.** They come out youngest
  in every run, but ρ(pred, total cutout SNR) = −0.83 across all runs vs
  −0.25 with L_X: the ordering is a brightness ordering. Cross-run agreement
  is high (mean Spearman 0.90 over 12 runs), which makes it stable, not
  correct.

---

## Settled by the audit — no action needed

- **Prediction "compression" on real data is ordinary shrinkage.** Real
  pred sd 0.67 vs OOF mock pred sd 0.69, slope(pred|label) 0.38. The
  analytic-noise runs (real sd 1.3–1.7) were the anomaly, not the injected
  ones. This retires the open worry in `field-injection-realism`.
- **The two labels agree.** pseudo-TSC vs merger-catalog `tsc_gyr`:
  Spearman 0.961 (Pearson 0.932); the 34 clusters capped at 4.0 Gyr have
  median merger-TSC 4.98 Gyr, so the cap is honest.
- **Real-cluster rankings are stable across configurations.** Mean off-
  diagonal Spearman 0.90 over 12 independently trained runs.

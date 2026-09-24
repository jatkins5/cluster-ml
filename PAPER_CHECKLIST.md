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

- [x] **B1. Mass / brightness control.** DONE 2026-09-09
  (`analyze_mass_control.py`, job 6125088). **The clean-sim CNN survives the
  control; the answer differs from the transfer model's.**

  | predictor of pseudo-TSC (5-fold OOF, same splits) | R² |
  |---|---|
  | log M500 alone | 0.314 |
  | log total linear weight alone | 0.432 |
  | all 5 scalars (ridge) | 0.455 |
  | pooled CNN, per seed | 0.487 ± 0.033 |
  | pooled CNN, 5-seed ensemble | **0.528** |
  | scalars + CNN | 0.538 (**+0.083** over scalars) |
  | mass + CNN | 0.529 (+0.215 over mass) |

  Within mass terciles, where a pure mass model scores zero:

  | tercile | n | CNN R² | scalars R² | mass R² | CNN ρ |
  |---|---|---|---|---|---|
  | low | 118 | 0.332 | 0.287 | 0.040 | 0.586 |
  | mid | 117 | 0.322 | −0.021 | −0.021 | 0.507 |
  | high | 117 | **0.222** | **−0.278** | −0.107 | 0.448 |

  Partial Spearman(CNN, TSC | log M500) = **+0.475** (raw 0.675), and the
  CNN residual still tracks the TSC residual at ρ = 0.468 after removing all
  five scalars. The high-mass tercile is the cleanest evidence: the scalars
  go *negative* there (−0.278) while the CNN holds 0.222.

  The honest caveat, to state in the paper: the scalars explain **72%** of
  the CNN's own output (R² 0.723; 0.62 from total flux alone). The CNN is
  substantially a brightness model — it is just not *only* one.

  Note this is the **clean-sim** model, where "total weight" is the
  simulation's own predicted power, a physically meaningful quantity. The
  ρ = −0.87 mass-reading result is the **transfer** model, where brightness
  was replaced by the Cuciti mass relation — which is B2, and why the two
  give different answers.

  *Remaining:* repeat the within-mass and partial-correlation analysis on
  the transfer model after B2 is decided; put the scalar-baseline table in
  the paper's main results.

  Original evidence that motivated this item:

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
- [x] **B2. Mass treatment — DONE 2026-09-22** (option 1, mass conditioning;
  option 4 already done). `train_cnn_mock.py --mass`, jobs 6622280/6623225,
  5 seeds x 4 conditions on `injected_nh4.h5` (25 obs, 208 backgrounds).

  | condition | mock OOF R² | ρ(pred,M500) | partial at fixed mass | R² in mass terciles |
  |---|---|---|---|---|
  | mass only | 0.279 | −0.99 | **−0.093** | −0.06 / −0.01 / −0.04 |
  | image only | 0.338 | −0.89 | 0.312 | 0.06 / 0.11 / 0.17 |
  | image + mass | 0.358 | −0.90 | 0.342 | 0.09 / 0.15 / 0.18 |
  | image + blurred mass | 0.367 | −0.90 | 0.355 | 0.09 / 0.17 / 0.18 |

  **The headline: the image is worth +0.079 ± 0.012 R² over mass alone
  (p=0.0024, 5/5 seeds), and +0.088 ± 0.013 when the mass carries the real
  sample's measured error.** The mass-only control scores ~0 inside every
  mass tercile and −0.09 on the partial correlation, exactly as a function
  of mass must, which is what validates the test.

  Two secondary results. Conditioning does **not** improve raw accuracy over
  image-only (+0.020 ± 0.016, p=0.29, 3/5) — it changes what can be *claimed*,
  not what is achieved. And blurring the mass by 25% costs nothing
  (+0.009, p=0.23), so the claim survives being made with observed masses
  rather than true ones, which is the form the paper needs.

  On real data, giving the model mass breaks the brightness ordering:
  ρ(pred, M500) goes from −1.00 (mass-only, by construction) to −0.36.
  A1650 is the clearest case — mass alone calls it 0.68 Gyr because it is
  massive; the image pulls it to 2.01.

  **A401 is now weaker as a validation, not stronger.** It is the most
  massive target (log M500c 14.92) and the mass-only model already assigns
  it the lowest TSC of all 17. Its partner A399 has no weak-lensing mass and
  drops out of the conditioned sample entirely.

- [x] **B2 option 3 — DONE 2026-09-23.** Dropped the observed flux anchor;
  each mock's total flux is now the emission model's own predicted power
  (`--flux-mode sim`), with one global constant setting the absolute scale.
  Jobs 6651354 / 6651423, 3 seeds.

  | condition | anchored | sim power | ρ(pred,M) anch → sim | partial anch → sim |
  |---|---|---|---|---|
  | image only | 0.338 | **0.412** | −0.89 → **−0.61** | 0.31 → **0.46** |
  | image + blurred mass | 0.367 | **0.437** | −0.90 → −0.67 | 0.36 → 0.46 |
  | mass only | 0.279 | 0.286 | −0.99 → −0.99 | −0.09 → −0.09 |

  **The increment over mass-only nearly doubles: +0.151 ± 0.006 (p=0.0016,
  3/3) against +0.088 ± 0.013 anchored.** The mass-only baseline is
  unchanged, as it must be — it never sees an image. Realism is comparable
  (separations within ±1 except `max`), and real predictions stay in range
  (median 1.70, range 0.59–2.76). **This is the configuration the paper
  should use.**

  **It also turns the observed relation into a test the simulation passes
  and fails in specific ways:**

  | | simulation | Cuciti+2023 |
  |---|---|---|
  | slope | **3.30 ± 0.33** | 3.55 |
  | scatter about the relation | **1.65 dex** | 0.35 dex |

  The slope agrees within 1σ — a genuine success for the DSA model with our
  density cut. The scatter is **5× too large**, the same pathology as the
  concentrated weights. The normalisation is *not* a prediction (the global
  offset was fitted to the median), so only slope and scatter are testable.

  **And the scatter is a merger signal, in the direction the observations
  report.** Residual vs pseudo-TSC Spearman −0.365 (p=1.5e-12), monotonic
  across terciles: **+0.74 / +0.37 / −0.64 dex** from recent to relaxed.
  Cuciti's Fig. 3 finds the same sign for X-ray disturbance. The old design
  was filling exactly this quantity with N(0, 0.35 dex) of pure noise —
  it was destroying a real signal, not just adding a confound.

  *The caveat to state:* total flux now carries genuine merger information
  (corr(log P150, TSC) = −0.615), so part of this model's skill is "the
  simulation says recent mergers are brighter" rather than morphology. That
  is a testable physical prediction rather than an imposed relation, which
  is the improvement — but it is not a morphology claim, and the 1.4 dex
  swing across TSC terciles is far larger than observations support.
- [ ] **Weak-lensing peak check — DONE, and it is underpowered.**
  (`analyze_wl_peaks.py`, jobs 6651381 / 6651407.) The peak catalogue covers
  57 clusters but only **9** overlap our LOFAR targets, and that is the
  binding limit — using all 25 image-only predictions instead of the 17
  mass-conditioned ones gives the same 9. At S/N>3 every correlation is null
  and two have the wrong sign. At S/N>4, peak count gives ρ = −0.756,
  p = 0.030 in the expected direction, but it is one of ~16 tests, driven
  entirely by A401 and A1307 being the only clusters with a second peak, and
  it flips sign at the neighbouring threshold. At n=9 only |ρ| ≳ 0.65 is
  detectable. **Report as inconclusive; it neither supports nor refutes.**
  The idea is still right — weak lensing is a genuinely independent
  observable — it just needs a peak catalogue covering more of the LOFAR
  footprint.

- [x] **X-ray re-check — DONE 2026-09-23: it was a bug.** `build_xray_dataset.py`
  applied bare `np.arcsinh` to count rates of median 0.0034/pixel, where it
  is the identity; 92.8% of image variance sat in the central 0.1 r500 and
  the CNN never saw the outskirts. With `arcsinh(x / global median)`:
  X-ray alone **0.320 → 0.511 ± 0.026** (p=0.0003, 5/5), matching radio's
  0.487; radio + X-ray stacked **0.523 → 0.577 (+0.054)** against +0.018
  with the old stretch. Mass adds nothing on top. **Joint model run —
  confirmed:** `train_cnn_pooled.py --xray-dataset`, 5 seeds, same protocol:
  joint **0.537 ± 0.016** (ensemble 0.575) vs radio 0.487 / X-ray 0.511;
  joint − radio +0.050 ± 0.008 (p=0.004, 5/5), joint − X-ray +0.026
  (p=0.016, 5/5). The stacking estimate (0.577) was not optimistic. The gain
  is in dating older mergers (TSC > 2 Gyr RMSE 1.25 → 1.13), not recent
  ones. *Remaining:* the X-ray/dual README numbers are void; the transfer
  (real-data) version needs X-ray mocks, which do not exist.

- [x] **X-ray at realistic depth — DONE 2026-09-23; redshift placement 2026-09-24 (X-ray 0.470, joint 0.500, both n.s. vs z=0.05)** (`build_xray_realistic.py`).
  The mocks were Chandra ACIS-I at 2 Ms with no sky backgrounds; real
  archival Chandra for our targets is a median 50 ks (19/26 observed). Thinned
  the photon counts to real depths and added soxs sky backgrounds:
  1 ks 0.391, 3 ks 0.402, 10 ks 0.440, 30 ks 0.446, **archive-drawn (median
  50 ks) 0.483**, 100 ks 0.498, 2 Ms 0.484. No loss at real depths; below
  ~10 ks the within-mass skill goes first. *Caveats:* mocks fixed at
  z=0.05 and TNG is ~3.6x over-luminous, so a real z~0.1 cluster at 50 ks
  behaves like ~3–5 ks here (~0.40). **The X-ray FoV is ±500 kpc (the ACIS-I
  square), not ±1 r500** as the README states. **Joint degraded model done:** radio injected + X-ray at archive depth,
  0.513 ± 0.001 vs radio 0.412 (+0.101, p=0.003) and X-ray 0.483 (+0.030,
  n.s.), 3 seeds. **Real Chandra downloaded:** 55 obs / 19 targets (25
  ACIS-I, 30 ACIS-S), 820 MB. *Remaining:* process real Chandra into model
  inputs (needs a redshift/brightness decision and exposure maps); redshift
  placement of the mocks.

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

### B2 decision brief — what to do about the flux anchor

**The mechanism.** `forward_model_lotss.make_mock` throws away the
simulation's own predicted radio power and re-anchors every mock's total
flux to the Cuciti+2023 relation, `total_flux_jy(log_m500, z, rng)`:

    log10(P150 / 10^24.5) = 1.1 + 3.55 · log10(M500 / 10^14.9) + N(0, 0.35 dex)

So mock brightness is, by construction, a steep function of halo mass plus
random noise. In TNG-Cluster mass and TSC are correlated (ρ = −0.56: massive
halos assemble late), so a model that reads brightness gets most of the way
to the label without looking at morphology at all. That is exactly what we
measure — ρ(OOF pred, M500) = −0.87, R² inside mass terciles 0.08–0.24 — and
on real data the prediction ordering tracks total map SNR at ρ = −0.83.

**The part that is a physics error, not just a statistical nuisance.** The
0.35 dex scatter is applied as *random* noise. Observationally it is not
random: Cuciti's Fig. 3 shows clusters above the P–M relation are the
X-ray-disturbed ones, and roughly half the clusters in this mass range have
no detected halo at all, only upper limits. The scatter about the relation
*is* the merger signal. We are randomising the very quantity that carries
the information we then ask the model to recover, and a genuinely relaxed
real cluster reads to our model as "faint → low mass → old" rather than
"relaxed".

**Options.**

| | what it does | buys | costs | effort |
|---|---|---|---|---|
| **1. Condition on mass** | feed log M500 to the head alongside the image embedding | the claim becomes "R² at fixed mass", which is what a referee asks for; mirrors how Cuciti works with P–M residuals | needs masses for the real clusters, and their 20–30% errors open a new sim/obs gap (train with matched noise on the mass input) | ~½ day + data |
| **2. Disturbance-dependent scatter** | make the offset from the P–M relation depend on dynamical state; give relaxed clusters upper-limit flux | fixes the physics error directly | circular: if we impose "disturbed → brighter" from an observed relation, the model learns our assumption, not the simulation's physics | ~1 day + re-validation |
| **3. Use the sim's own power** | drop the anchor, keep the DSA model's relative power, set only the overall normalisation | most defensible: total flux becomes a *prediction* carrying merger state, and the Cuciti relation becomes a validation test instead of an input | sim power is uncalibrated across ~55 decades and pathologically concentrated; sim L_X is 3.6× tilted, so the absolute scale is suspect; the brightness distribution may stop matching LoTSS, undoing part of the 8/10 KS agreement | several days, re-validates everything |
| **4. Flux-normalised ablation** | divide each image by its own total flux so brightness carries nothing | a clean "morphology alone" number — an honest lower bound on the image claim | discards information that is physically real | ~hours |

**Option 4 result — DONE 2026-09-09** (`build_shape_dataset.py`,
`compare_shape.py`, jobs 6125878 / 6125963 / 6126951). Every projection
rescaled to the dataset median total linear weight — the maps spanned
6.89 dex before, exactly one value after. Same 5 seeds, same folds, paired:

| | full flux | shape only |
|---|---|---|
| per-seed OOF R² | 0.487 ± 0.033 | **0.418 ± 0.027** |
| 5-seed ensemble | 0.528 | 0.473 |
| within mass terciles | 0.332 / 0.322 / 0.222 | 0.246 / 0.258 / 0.140 |
| partial ρ(pred, TSC \| M500) | 0.475 | **0.460** |
| increment over the 5 scalars | +0.083 | +0.052 |

Paired shape − full = **−0.069 ± 0.023 SE, p=0.040, 5/5 seeds negative**.

So morphology alone is worth **0.418**, comfortably above mass alone
(0.314), and brightness adds ~0.07 on top. Two readings that matter more
than the headline number:

1. **The mass-independent part of the signal is morphological.** Removing
   flux barely moves the partial correlation at fixed mass (0.475 → 0.460).
   Flux was mostly carrying the mass-correlated part of the label, which is
   exactly what B1's tercile table implied.
2. **Brightness and morphology are not separable in this data.** The
   shape-only model's output is *still* 59% predictable from total flux
   (ρ = −0.84 with it) even though every map it saw had identical total
   flux. Bright clusters genuinely look different, so the CNN reconstructs
   the brightness we deleted. This bounds what any ablation of this kind
   can establish, and is worth a sentence in the paper: "morphology-only"
   and "brightness-only" are not orthogonal decompositions here.

**Recommendation.** Do **4** immediately regardless of what else we pick —
it is cheap and it bounds the morphology claim, which is the number the
paper actually lives or dies on. Take **1** as the primary model for this
paper. Treat **3** as the more interesting physics question and a candidate
follow-up (or a section if time allows). Avoid **2** as the primary: the
circularity is hard to write around.

1 and 4 together give a clean decomposition — morphology-only, mass-only,
and joint — which is a better results table than any single number.

**Blocking dependency for option 1:** masses for the real clusters. LoVoCCS
II weak-lensing masses (Fu et al. 2024, 58 clusters) are the right source
and the table has still not been obtained — flagged as pending since the
sim-vs-obs distributional work. Fallbacks: Planck PSZ2 M_SZ (most of these
are Abell clusters), or an L_X–M scaling, since all 17 targets have L_X in
the target list. Ask the PI.

## C. Data hygiene

- [x] **C1. DONE** — Name-key bug dropped 5 usable LoTSS targets.
  `forward_model_lotss.load_obs` keys the target CSV by name with *spaces*
  removed (`MKW3s`) but the FITS files use underscores (`lotss_MKW_3s.fits`),
  so MKW 3s, RXC J0034.2−0204, RXC J0034.6−0208, RXC J1217.6+0339 and
  RX J0820.9+0751 are skipped as "no redshift in target list" — the CSV has
  their redshifts. Strip spaces *and* underscores on both sides.
- [x] **C2. DONE** (and the background fields needed the same fix — see C3).
  Re-download the three low-z targets larger. A2052, A2063,
  A2147 (all z=0.035) fail the 1 Mpc crop because their cutouts are
  800 px × 1.5″ = 20′ and 1 Mpc subtends ~24′ there.
  `download_lotss_image.py --size-arcmin 40`. (A1750 is genuinely blank.)
- [x] **C3. DONE** — rebuilt `injected_nh4.h5`: **25 obs** (17 with a
  weak-lensing mass), all 352 clusters injected. The first rebuild silently
  dropped to 235/352 because the 158 background fields were also 20′ and
  could not supply a 1 Mpc box at z=0.035 — the same bug as C2, one level
  down. Re-downloaded the fields at 40′ (208 kept, `lotss_fields_40/`).
  `mock_dataset_*.h5` (analytic-noise tier) has *not* been rebuilt.
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

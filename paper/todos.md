# Simulations section review TODOs

Review date: 2026-10-28. Reviewed revision: `8323cf7`.

These findings concern the new prose and retained outline bullets in
[sections/2_simulations.tex](sections/2_simulations.tex). They were checked
against the implementation, saved dataset metadata and summaries, original
mock-generation scripts, and cited papers. The items below are unresolved;
recording them does not change the manuscript or analysis.

## 1. Describe the merger labels actually used

- [ ] Replace the new paragraph's description of the target as the nearest
  major merger. The training target is **pseudo-TSC**: the linearly
  interpolated window size at which the merger score first reaches 0.5,
  capped at 4 Gyr if no crossing occurs. At earlier snapshots, use the
  past-only score; at snapshot 99 the past-only and all-event scores agree.
  The implementation assigns 0.1 Gyr when the first sampled window already
  exceeds the threshold. See [build_dataset.py](../build_dataset.py).
- [ ] Distinguish pseudo-TSC from the catalogue comparison clock. The latter
  measures time since the **most recent past first-pericentre passage**,
  without an additional mass-ratio cut. Selecting the nearest event in
  either time direction would allow future collisions at earlier snapshots.
  Say “any catalogued mass ratio” rather than implying completeness for
  arbitrarily small mergers.
- [ ] Retain the verified Spearman correlation of **0.961** between
  pseudo-TSC and the catalogue clock for 346 snapshot-99 clusters. Clarify
  that the median mass ratio **0.093**, approximately 1:11, describes the
  most recent events that set that clock, not every event in the catalogue.
  See [the catalogue-label audit](../logs/val_proxy_4684631.out).
- [ ] Remove or explicitly retain as historical the major-merger-only
  accuracy comparison, which used evaluation-fold checkpoint selection.
  It cannot establish the accompanying claim that radio morphology traces
  recent disturbances of any size. A scientific comparison requires a
  rerun under the current evaluation protocol.

## 2. Reconcile the radio derivation with the implemented model

- [ ] Resolve the discrepancy between the new analytical prescription and
  the weights used to build the existing maps. The draft's constants are
  internally consistent under its stated assumptions:
  `C_tau,0 = 4.8534e-5`, `I_* = 18.106`, and
  `P_* = 9.126e24 W Hz^-1`. However,
  [build_radio_cells.py](../build_radio_cells.py) uses a coefficient of
  `5.2e23 W Hz^-1` and the stored `normalized_psi_table.npz` spectral factor,
  whose definition differs from the draft's Phi.
- [ ] Treat this as a change to relative shock weights as well as absolute
  normalization. At downstream temperature 7 keV, comparing the complete
  expressions gives draft/code luminosity ratios of **3.50 at Mach 2,
  10.57 at Mach 3, and 13.52 at Mach 4**. A single global rescaling cannot
  reconcile them. Either document the implemented lookup-table prescription
  or regenerate and reassess the affected datasets and results before
  presenting the revised prescription as the method used.

Evidence: the stored table and its generator,
`Radio_Data/generate_phi_func.ipynb`; scalar numerical checks of the draft;
[Lee et al. (2024), equation 9](https://arxiv.org/html/2311.06340v2#S2.SS3).
The numerical checks compare the emission expressions before the separate
map-unit rescaling.

## 3. Correct the sample description and independence language

- [ ] State that the dataset contains **352 snapshot-99 clusters and 351
  matched snapshot-91 progenitors**, with three orthogonal projections each.
  The current prose implies that all 352 are available at both epochs.
- [ ] Replace the independence TODO with a statement about grouping:
  “All projections and both snapshots associated with a given present-day
  cluster are assigned to the same cross-validation fold.” The weak
  correlation between labels at different epochs does not establish
  statistical independence of the images or clusters.
- [ ] Identify the quoted correlation as a **pseudo-TSC** correlation.
  Verified values are an epoch separation of **1.336 Gyr**, a median
  snapshot-91 minus snapshot-99 mass difference of **−0.029 dex**, and a
  matched pseudo-TSC Spearman correlation of **0.324**. Describe these as
  evidence of temporal variation, without calling the samples independent.

Evidence: `dataset_nh4_144_128.h5`, `dataset_nh4_144_s91_128.h5`,
[merge_snapshots.py](../merge_snapshots.py), and
[make_fold_manifest.py](../make_fold_manifest.py).

## 4. Update the context of the older radio bullets

- [ ] Replace or label as historical the **29.5%** brightest-cell fraction.
  The current 144 MHz phase-summary data give **25.0% before the density
  cut and 5.1% afterward**, consistent with the figure caption.
- [ ] Label the Mach-number statistics explicitly. Values near **26**
  describe a radio-weighted median; values near **62** describe a
  radio-weighted mean. They are different summaries, not interchangeable
  estimates. The current figure's weighted-mean values are **61.7 to 4.8**.
- [ ] Explain that the **0.035 dex** scatter in the 144 MHz versus 1.4 GHz
  flux comparison is measured **after dividing the 144 MHz weights by
  12.5**. It is residual variation after a global rescaling, not the
  absolute change in physical luminosity between frequencies. The roughly
  5% redistribution of normalized map flux is supported by the same check.
- [ ] Keep the density cut justified by the modelling assumptions and
  observational realism. The current clean-map comparison is correctly
  reported as **no demonstrated accuracy gain**: `R^2 = 0.500 -> 0.485`,
  paired-seed `p = 0.164`. Do not restore the older significance claim.

Evidence: `phase_cells_144.npz`,
[radio-theory diagnostics](../logs/radio_theory_6759835.out),
[frequency comparison](../logs/nu144_build_6951638.out), and
[current-protocol comparison](../logs/paperfig_7255643.out).

## 5. Complete and qualify the X-ray mock description

- [ ] State the observed image band: **0.1–2.0 keV**. The source plasma
  model and the catalogue luminosity band are separate quantities.
- [ ] Give the full emitting-gas selection: **T > 3e5 K**, zero
  star-formation rate, and **GFM_CoolingRate < 0**, within a cube of side
  `2 R500c` centred on the snapshot's `GroupPos`. The APEC source model
  uses the simulated temperatures, metallicities, and emission measures.
- [ ] Specify the initial absorption model: `wabs` with
  **N_H = 1e20 cm^-2**. Describe target-dependent absorption corrections
  in the forward-modelling section.
- [ ] Explain the initial background choices. The 2 Ms SOXS ACIS-I Cycle-22
  mocks include the instrumental particle background; Galactic foreground
  and point-source backgrounds are disabled at this stage and added during
  subsequent forward modelling.
- [ ] Distinguish the source-selection volume, stored image grid, and
  exposed detector footprint. The approximately **±500 kpc** detector
  coverage applies to the snapshot-99 placement at **z_obs = 0.05**.
  At **z_obs = 0.0994**, the native snapshot-91 detector footprint covers
  a larger physical area. The common 128-pixel grid has approximately
  **18.9 kpc per pixel**; detector coverage does not fill that entire grid.
- [ ] Describe count recovery as approximate where the original exposure
  varies across chip gaps and dither edges. Binomial overlap allocation
  preserves integer counts during snapshot-91 regridding, but does not
  eliminate that underlying approximation.
- [ ] Retain the scaled stretch `arcsinh(x/a)` and explain that `a` is a
  global scale shared across the corresponding dataset and both snapshots.
  Keep target placement, realistic depth, luminosity calibration, and
  background adjustments in the forward-modelling section.

Evidence: the original snapshot-99 and snapshot-91 X-ray generation scripts,
[build_xray_dataset.py](../build_xray_dataset.py),
[xray_counts_regrid.py](../xray_counts_regrid.py), and
[build_xray_realistic.py](../build_xray_realistic.py).

## 6. Fix citations and remaining prose issues

- [ ] Replace `Nelson2015`, the original Illustris public-data release,
  with the TNG release reference, **Nelson et al. (2019)**. Retain
  Nelson et al. (2024) for TNG-Cluster. This follows the
  [TNG-Cluster citation guidance](https://www.tng-project.org/data/cluster/).
  The stated cosmological parameters and the distinct M200c and M500c mass
  ranges are consistent with the simulation papers.
- [ ] Deduplicate the `Nelson2024` and `Lee2024` keys in
  [refs.bib](refs.bib), retaining the complete, verified entries.
- [ ] Change “diffuse shock acceleration” to “diffusive shock acceleration.”
- [ ] Consolidate the consecutive “Radio emission model” and “Synchrotron
  emission from shock-accelerated electrons” subsection headings, or give
  them an intentional hierarchy.
- [ ] Complete the sentence ending “the adopted spectral and pitch-angle.”

# Radio/X-ray spatial pilot

Prepared 2026-11-02 following the request to run a small background pilot.
This tests prediction of pseudo-TSC on held-out simulated observations.
It does not calibrate physical collision dates or establish observational transfer.

## Fixed experiment

Inputs: injected_fix_phys_b3000_s9991.h5 and xray_fix_b3000_s9991.h5.
Join by (group_id, snap) and require identical labels, unique row IDs,
redshifts, realization counts and projection counts. The X-ray file without
b3000 uses different placement redshifts and must not be substituted.
Both snapshots of a parent stay together in folds_e0.npz, including its
fixed inner-selection sets.

Three arms share exactly the same JointViewScalarCNN architecture,
initialization seeds, images, quality flags, fitting data and training:

1. masked: geometric slots zeroed after fitting-only normalization.
2. origin: radio centroid distance and emission-weighted mean radius about
   the nominal cutout centre.
3. registered: the same two measurements about the measured X-ray peak.

The availability flags enter every arm. Thus masked is an image-plus-flags
control, not an exact reproduction of an older image-only run. Origin
controls for benefits of supplying radio geometry without the X-ray
reference. Registered-minus-origin includes reference measurement errors;
it is not a general test of all cross-band geometry.

Seeds 42–46; five outer folds; 60 epochs; batch size 32; Huber delta 0.5 Gyr;
AdamW/cosine schedule as in the existing trainer. Pooled-view training;
checkpoint selection on individual observations in the fixed inner set.
Evaluate five-seed ensembles on snapshot-99 individual views/realizations,
without averaging them together. Training also uses snapshot 91.
Scalar imputation/normalization use fitting data only. Images retain stored
global arcsinh stretches, without extra per-fold standardization, matching
train_cnn_mock. The per-view fusion layer is identical across arms.

Primary: registered minus masked. Diagnostics: origin minus masked and
registered minus origin. Paired parent-halo bootstrap percentile intervals
(4000 draws) condition on fitted runs; also report individual-seed differences.
A gain of at least 0.02 in ensemble R² with MAE increasing by no more than
0.02 Gyr is a practical follow-up criterion, not a significance threshold.
All comparisons are exploratory.

## Measurements and coordinate conventions

Features are measured separately on each noisy observation, without labels,
pristine source maps, true R500, or other views.

- Undo radio arcsinh to recover S/rms. Select 8-connected islands above
  3 rms with at least four pixels; weight by positive excess over 3 rms.
  Distances use the 3 Mpc grid in kpc. This is a summary, not relic
  identification or a clean source estimate.
- Undo X-ray arcsinh up to its common multiplicative constant. Smooth
  with Gaussian sigma 2 pixels; find the peak within 2 arcmin of the
  nominal centre, capped at 80% of the usable aperture. Require it to
  exceed the median plus three robust scatter estimates in the 60–90%
  aperture annulus. This is a quality flag, not a calibrated significance.
- Measure radio-centroid distance and emission-weighted mean pixel radius
  about the reference. Symmetric opposite relics can have zero centroid
  distance but nonzero mean radius. Unavailable measurements are missing
  in both geometry arms; keep their flags and do not drop halos. Stop before
  training if fewer than 5% of observations have usable geometry.

Radio histogram row/column axes are (x,y), (y,z), (x,z). Both original
snapshot generators call pyXSIM with LOS z,x,y, with no custom north vector.
pyXSIM screen horizontal/vertical axes are (x,y), (y,z), (z,x); SOXS writes
FITS arrays vertical/horizontal. Therefore transpose the first two X-ray
projections, leaving the third unchanged. This is derived from source
conventions, not selected using labels or image correlations.
Apply it to peak coordinates and X-ray CNN images, so paired augmentation
uses common physical axes. The image pixel scales remain distinct and
are explicitly used in distances; the CNN uses separate encoders.

Sources:

- project_cells.py and build_xray_dataset.py.
- TNGCluster_Xray_Snap99/snap99_{x,y,z}/X-ray_chandra_Snap0_{x,y,z}.py.
- /oscar/data/idellant/Chuiyang/TNGCluster_Xray_Snap91/snap91_{x,y,z}/X-ray_chandra_Snap_{x,y,z}.py.
- [pyXSIM projection functions](https://github.com/jzuhone/pyxsim/blob/main/pyxsim/lib/sky_functions.pyx).
- Installed soxs/events/images.py: transposed output; negative RA pixel scale.

The old cross-band image correlations were about 0.02 and did not provide
strong empirical confirmation of alignment. This pilot uses source-defined
conventions. The original generator environment's software-version history
is incompletely preserved; real registration needs validation before use.

## Execution and scope

Real predictions are deferred: existing X-ray HDF5 files lack the original
catalogue-to-peak offset. process_chandra.py now preserves catalogue RA/Dec
and peak offsets for future rebuilds, without changing image processing.
This job does not rebuild observations or alter original datasets.
A positive simulation result would motivate that follow-up and sensitivity
checks for centring, thresholds, aperture and background structure.

submit_spatial_pilot.sh requests one GPU, four CPUs, 48 GB, at most six hours.
All 15 runs execute sequentially in one job, with no array or monitor.
Freeze the listed source files in a unique run directory before submission.
Large inputs are staged on node-local scratch; source/input hashes and
paired metadata are recorded. The job extracts features, trains and scores,
then removes its own scratch directory.

Outputs: provenance.json, features.npz, feature_quality.json, prediction
files saved after each fold, results.json, results.txt, and COMPLETE only
after scoring succeeds. Incomplete predictions cannot be scored.
The manuscript is untouched.

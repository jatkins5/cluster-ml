#!/usr/bin/env python3
"""E1/E2 IDEALISED optical pilot, version 2 (review round 8).

Inputs are idealised galaxy structure with a simulated hot-gas centre and the
true r500 (optical_catalog.py): FoF membership, intrinsic total photometry.
The centre also sets the member apertures, the BCG choice, the gap and the
richness, so this is not optical information alone, and any increment over
noisy mass does not isolate morphology from the information carried by the
true radius. An information-content test, not an observational forecast.

Rows: one per (parent halo, snapshot, projection). Outer folds: the shared
manifest folds_e0.npz by parent halo; training uses both snapshots of the
outer-training halos, scoring the held-out snapshot-99 projections.
No redshift control: this pilot is UNPLACED (snapshot epoch is not an
observable); the placed X-ray comparison must use placement redshift.

Arms (same rows, folds, model settings):
  A  controls            noisy mass (keyed per halo/snapshot), richness
  B  + structure         gap12, off_bcg_x, off_bcg_gal, asym
  E  controls + colour   (colour-only arm, no structure)
  C  + structure + colour
  R  radial-mixing null  as C, g-r permuted ACROSS radii within each
                         cluster/projection and 0.5-mag bin: preserves the
                         colour-magnitude distribution, destroys only the
                         radial colour association (verified below on a
                         constructed example before use)
Colour features: blue fraction, red-sequence scatter, inner-minus-outer
colour residual, against a red sequence fitted on the applicable FIT data
only (outer-training for the final model, inner-fit for tuning).

Models: ridge, alpha chosen by GROUPED inner CV over parent halos (5 folds
of the outer-training halos; imputation, scaling and the red sequence refit
on each inner-fit split), then refit on all outer-training rows; histogram
gradient boosting with fixed settings (< 10000 rows, so no internal
validation split). Outputs saved with provenance (optical_pilot_v2.npz).
"""
import hashlib
import json
import subprocess

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STRUCT = ["gap12", "off_bcg_x", "off_bcg_gal", "asym"]
CONTROL = ["mass_noisy", "richness"]
ALPHAS = np.logspace(-3, 3, 13)
FEAT_CSV, MEM_NPY, MANIFEST = ("optical_features_idealised.csv",
                               "optical_members_idealised.npy", "folds_e0.npz")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
PROV = dict(sources={f: sha(f) for f in ("train_optical_pilot.py", "optical_catalog.py")},
            inputs={f: sha(f) for f in (FEAT_CSV, MEM_NPY, MANIFEST)},
            head=subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip())

# ---------------------------------------------------------------- inputs
df = pd.read_csv(FEAT_CSV)
mem = np.load(MEM_NPY)                      # group, snap, proj, M_r, g-r, R/r500
mf = np.load(MANIFEST)
keys = list(zip(df.group_id, df.snap, df.proj))
assert len(set(keys)) == len(keys), "duplicate (parent, snapshot, projection) rows"
assert np.isfinite(mem[:, 3:]).all(), "non-finite member photometry/radii"
fold_of = dict(zip(mf["group_id"], mf["fold"]))

lab, logm = {}, {}
with h5py.File("dataset_nh4_144_128.h5") as f:
    for h, t in zip(f["meta/halo_id"][:], f["labels/pseudo_tsc"][:]):
        lab[(int(h), 99)] = float(t)
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    for h, m in zip(f["haloID"][:], f["mhalo_500c"][:]):
        logm[(int(h), 99)] = float(m)
with h5py.File("dataset_nh4_144_s91_128.h5") as f:
    for h, t, m in zip(f["meta/halo_id"][:], f["labels/pseudo_tsc"][:], f["meta/log_m500"][:]):
        lab[(int(h), 91)] = float(t)
        logm[(int(h), 91)] = float(m)
missing = [(g, s) for g, s in zip(df.group_id, df.snap) if (g, s) not in lab]
if missing:
    raise SystemExit(f"{len(missing)} feature rows have no label, e.g. {missing[:3]}")
missing = [g for g in df.group_id.unique() if g not in fold_of]
if missing:
    raise SystemExit(f"{len(missing)} parent halos not in {MANIFEST}")
wl = pd.read_csv("lovoccs_wl_masses.csv")
err_dex = (wl.loc[wl.has_lotss.astype(bool), "frac_err"] / np.log(10)).to_numpy()


def noisy_mass(g, s, seed=0):
    r = np.random.default_rng([seed, int(g), int(s), 0])
    return logm[(g, s)] + r.normal(0.0, r.choice(err_dex))


df["y"] = [lab[(g, s)] for g, s in zip(df.group_id, df.snap)]
df["mass_noisy"] = [noisy_mass(g, s) for g, s in zip(df.group_id, df.snap)]
df["fold"] = [fold_of[g] for g in df.group_id]
for c in CONTROL + STRUCT:
    assert np.isfinite(df[c]).all() or c in STRUCT, f"non-finite control {c}"
by_key = {}
for row in mem:
    by_key.setdefault((int(row[0]), int(row[1]), int(row[2])), []).append(row[3:])
by_key = {k: np.array(v) for k, v in by_key.items()}


# ---------------------------------------------------------------- colour
def radial_mix(m, rs):
    """Permute g-r ACROSS radii within each 0.5-mag bin of this cluster/
    projection: the colour-magnitude distribution is preserved, the radial
    colour association is destroyed."""
    m = m.copy()
    mb = np.floor(m[:, 0] / 0.5)
    for b in np.unique(mb):
        idx = np.where(mb == b)[0]
        m[idx, 1] = m[rs.permutation(idx), 1]
    return m


def fit_red_sequence(fit_keys):
    M = np.concatenate([by_key[k] for k in fit_keys if k in by_key])
    x, y = M[:, 0], M[:, 1]
    keep = np.ones(len(x), bool)
    for _ in range(5):
        b, a = np.polyfit(x[keep], y[keep], 1)
        res = y - (a + b * x)
        sd = res[keep].std()
        keep = np.abs(res) < 2 * sd
    return float(a), float(b), float(sd)


def colour_feats(all_keys, rs_fit, mix_seed=None):
    a, b, sd = rs_fit
    rs = None if mix_seed is None else np.random.default_rng(mix_seed)
    out = np.full((len(all_keys), 3), np.nan)
    for n, k in enumerate(all_keys):
        m = by_key.get(k)
        if m is None or len(m) < 3:
            continue
        if rs is not None:
            m = radial_mix(m, rs)
        res = m[:, 1] - (a + b * m[:, 0])
        red = res > -2 * sd
        out[n, 0] = np.mean(res < -0.1)
        out[n, 1] = res[red].std() if red.sum() >= 3 else np.nan
        inn, outr = m[:, 2] < 0.5, m[:, 2] >= 0.5
        if inn.sum() and outr.sum():
            out[n, 2] = res[inn].mean() - res[outr].mean()
    return out


# constructed check (review R8): the control must change the radial statistic
# and preserve the colour-magnitude distribution
rs_chk = np.random.default_rng(0)
Mr = np.repeat([-21.2, -20.7, -20.2, -19.7], 10)
R = np.tile(np.r_[np.full(5, 0.2), np.full(5, 0.8)], 4)
col = 0.7 + 0.1 * (R < 0.5)                       # redder inside
toy = np.column_stack([Mr, col, R])
mixed = radial_mix(toy, rs_chk)
trend = lambda m: m[m[:, 2] < 0.5, 1].mean() - m[m[:, 2] >= 0.5, 1].mean()
assert abs(trend(toy) - 0.1) < 1e-9
assert np.allclose(np.sort(mixed[:, 1]), np.sort(toy[:, 1]))
for b in np.unique(np.floor(Mr / 0.5)):
    s_ = np.floor(Mr / 0.5) == b
    assert np.allclose(np.sort(mixed[s_, 1]), np.sort(toy[s_, 1]))
# the statistic must actually change (a no-op shuffle would pass the above)
assert abs(trend(mixed)) < 0.5 * abs(trend(toy)), f"radial trend not removed: {trend(mixed):+.3f}"
print(f"radial-mixing control check: toy radial trend {trend(toy):+.3f} -> "
      f"{trend(mixed):+.3f}; colour-magnitude distribution preserved")


# ---------------------------------------------------------------- models
def ridge_pipe(alpha):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), Ridge(alpha=alpha))


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=20,
                                         l2_regularization=1.0, random_state=0)


def arm_matrix(arm, rows, colour):
    X = df.loc[rows, CONTROL].to_numpy(float)
    if arm in ("B", "C", "R"):
        X = np.column_stack([X, df.loc[rows, STRUCT].to_numpy(float)])
    if arm in ("E", "C", "R"):
        X = np.column_stack([X, colour[rows]])
    return X


ARMS = {"A": "controls", "B": "+ structure", "E": "controls + colour",
        "C": "+ structure + colour", "R": "radial-mixing null of C"}
y_all = df.y.to_numpy()
g_all = df.group_id.to_numpy()
pred = {(m, a): np.full(len(df), np.nan) for m in ("ridge", "hgb") for a in ARMS}
chosen_alpha, rs_params = {}, {}
for k in range(int(mf["folds"])):
    tr = np.where(df.fold != k)[0]
    te = np.where((df.fold == k) & (df.snap == 99))[0]
    tr_keys = [keys[i] for i in tr]
    rs_fit = fit_red_sequence(tr_keys)
    rs_params[k] = rs_fit
    col = colour_feats(keys, rs_fit)
    col_mix = colour_feats(keys, rs_fit, mix_seed=1000 + k)
    for arm in ARMS:
        colour = col_mix if arm == "R" else col
        # grouped inner CV for ridge alpha; red sequence refit on inner-fit
        gkf = GroupKFold(n_splits=5)
        cv_err = np.zeros(len(ALPHAS))
        for itr, iva in gkf.split(tr, groups=g_all[tr]):
            fit_rows, val_rows = tr[itr], tr[iva]
            icol = colour_feats(keys, fit_red_sequence([keys[i] for i in fit_rows]),
                                mix_seed=None if arm != "R" else 2000 + k)
            Xf, Xv = arm_matrix(arm, fit_rows, icol), arm_matrix(arm, val_rows, icol)
            for j, al in enumerate(ALPHAS):
                p_ = ridge_pipe(al).fit(Xf, y_all[fit_rows]).predict(Xv)
                cv_err[j] += np.sum((y_all[val_rows] - p_) ** 2)
        al = float(ALPHAS[np.argmin(cv_err)])
        chosen_alpha[(k, arm)] = al
        Xtr, Xte = arm_matrix(arm, tr, colour), arm_matrix(arm, te, colour)
        pred[("ridge", arm)][te] = ridge_pipe(al).fit(Xtr, y_all[tr]).predict(Xte)
        pred[("hgb", arm)][te] = hgb().fit(Xtr, y_all[tr]).predict(Xte)

# ---------------------------------------------------------------- report
sc = (df.snap == 99).to_numpy()
y, g = y_all[sc], g_all[sc]
ug = np.unique(g)
idx = {h: np.where(g == h)[0] for h in ug}
rng = np.random.default_rng(0)
print(f"\nscored: {sc.sum()} snapshot-99 single observations, {len(ug)} parent halos")
PAIRS = [("B", "A"), ("E", "A"), ("C", "B"), ("C", "R")]
for mname in ("ridge", "hgb"):
    P = {a: pred[(mname, a)][sc] for a in ARMS}
    assert all(np.isfinite(v).all() for v in P.values())
    print(f"\n== {mname}" + (f"  (alphas chosen: {sorted(set(v for (kk, a), v in chosen_alpha.items()))})"
                            if mname == "ridge" else ""))
    for a, lab_ in ARMS.items():
        print(f"  {a} {lab_:<26} R2 {r2_score(y, P[a]):.3f}  MAE {mean_absolute_error(y, P[a]):.3f} Gyr")
    boots = {pr: [] for pr in PAIRS}
    for _ in range(4000):
        s_ = np.concatenate([idx[h] for h in rng.choice(ug, len(ug))])
        for (b_, a_) in PAIRS:
            boots[(b_, a_)].append(r2_score(y[s_], P[b_][s_]) - r2_score(y[s_], P[a_][s_]))
    for (b_, a_), v in boots.items():
        d = r2_score(y, P[b_]) - r2_score(y, P[a_])
        print(f"  {b_} - {a_}: R2 {d:+.3f} [{np.percentile(v, 2.5):+.3f}, "
              f"{np.percentile(v, 97.5):+.3f}] (paired halo bootstrap; conditional on the fits)")

np.savez("optical_pilot_v2.npz",
         keys=np.array(keys), fold=df.fold.to_numpy(), y=y_all,
         mass_noisy=df.mass_noisy.to_numpy(),
         **{f"pred_{m}_{a}": pred[(m, a)] for m in ("ridge", "hgb") for a in ARMS},
         chosen_alpha=json.dumps({f"{k}_{a}": v for (k, a), v in chosen_alpha.items()}),
         red_sequence=json.dumps({str(k): v for k, v in rs_params.items()}),
         provenance=json.dumps(PROV), arms=json.dumps(ARMS))
print("\nsaved optical_pilot_v2.npz (predictions, keys, folds, noisy masses, alphas, "
      "red-sequence parameters, provenance)")
print("IDEALISED: FoF members, true r500, intrinsic total photometry, simulated hot-gas "
      "centre (which also sets apertures, BCG, gap and richness).")

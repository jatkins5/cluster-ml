#!/usr/bin/env python3
"""E1/E2 IDEALISED optical pilot (review rounds 4-7): does galaxy structure,
then colour, add information about merger time beyond noisy mass and
richness? FoF membership, true r500, intrinsic total photometry -- an
information-content test, not an observational forecast.

Rows: one per (parent halo, snapshot, projection) from optical_catalog.py.
Folds: the shared persisted manifest (folds_e0.npz) by parent halo; training
uses both snapshots of the training halos, scoring uses the held-out
snapshot-99 projections (single observations, 3 per halo).
Arms (same rows, folds, model settings):
  A  controls      noisy mass (keyed draw per halo/snapshot), richness
  B  + structure   gap12, off_bcg_x, off_bcg_gal, asym
  C  + colour      blue fraction, red-sequence scatter, colour radial trend
  D  + shuffled    as C, colours permuted among members of the same
                   cluster/projection at fixed magnitude and radius bin
The red sequence (g-r vs M_r, iteratively 2-sigma clipped) is fitted on the
TRAINING fold's members only. Models: ridge (median imputation + missing
indicators, fitted in-fold) and histogram gradient boosting (native NaN).
Reported: R2 and MAE per arm, paired halo-bootstrap differences B-A, C-B,
C-D (resampling parent halos; conditional on the fitted models).
"""
import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng0 = np.random.default_rng(0)
STRUCT = ["gap12", "off_bcg_x", "off_bcg_gal", "asym"]
COLOUR = ["blue_frac", "rs_scatter", "colour_radial"]

df = pd.read_csv("optical_features_idealised.csv")
mem = np.load("optical_members_idealised.npy")       # group, snap, proj, M_r, g-r, R
mf = np.load("folds_e0.npz")
fold_of = dict(zip(mf["group_id"], mf["fold"]))

# labels and true masses
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
wl = pd.read_csv("lovoccs_wl_masses.csv")
err_dex = (wl.loc[wl.has_lotss.astype(bool), "frac_err"] / np.log(10)).to_numpy()


def noisy_mass(g, s, seed=0):
    r = np.random.default_rng([seed, int(g), int(s), 0])
    sig = r.choice(err_dex)
    return logm[(g, s)] + r.normal(0.0, sig)


df = df[[(g, s) in lab for g, s in zip(df.group_id, df.snap)]].copy()
df["y"] = [lab[(g, s)] for g, s in zip(df.group_id, df.snap)]
df["mass_noisy"] = [noisy_mass(g, s) for g, s in zip(df.group_id, df.snap)]
df["fold"] = [fold_of[g] for g in df.group_id]
print(f"{len(df)} rows ({df.group_id.nunique()} parent halos); snapshots "
      f"{df.snap.value_counts().to_dict()}; structural missing fractions "
      + ", ".join(f"{c} {df[c].isna().mean():.1%}" for c in STRUCT))

key = list(zip(df.group_id, df.snap, df.proj))
by_key = {}
for row in mem:
    by_key.setdefault((int(row[0]), int(row[1]), int(row[2])), []).append(row[3:])
by_key = {k: np.array(v) for k, v in by_key.items()}


def shuffled(members, rs):
    """Permute g-r among members in the same (M_r 0.5-mag bin, R bin)."""
    m = members.copy()
    mb = np.floor(m[:, 0] / 0.5)
    rb = (m[:, 2] >= 0.5).astype(int)
    for cell in set(zip(mb, rb)):
        idx = np.where((mb == cell[0]) & (rb == cell[1]))[0]
        m[idx, 1] = m[rs.permutation(idx), 1]
    return m


def fit_red_sequence(train_keys):
    M = np.concatenate([by_key[k] for k in train_keys if k in by_key])
    x, y = M[:, 0], M[:, 1]
    keep = np.ones(len(x), bool)
    for _ in range(5):
        b, a = np.polyfit(x[keep], y[keep], 1)
        res = y - (a + b * x)
        sd = res[keep].std()
        keep = res > -2 * sd                    # clip the blue cloud
        keep &= res < 2 * sd
    return a, b, sd


def colour_feats(keys, rs_fit, shuffle_seed=None):
    a, b, sd = rs_fit
    rs = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    out = np.full((len(keys), 3), np.nan)
    for n, k in enumerate(keys):
        m = by_key.get(k)
        if m is None or len(m) < 3:
            continue
        if rs is not None:
            m = shuffled(m, rs)
        res = m[:, 1] - (a + b * m[:, 0])
        red = res > -2 * sd
        out[n, 0] = np.mean(res < -0.1)
        out[n, 1] = res[red].std() if red.sum() >= 3 else np.nan
        inn, outr = m[:, 2] < 0.5, m[:, 2] >= 0.5
        if inn.sum() and outr.sum():
            out[n, 2] = res[inn].mean() - res[outr].mean()
    return out


def design(rows, cols, colour=None):
    X = rows[cols].to_numpy(float)
    return X if colour is None else np.column_stack([X, colour])


def models():
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    hgb = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                max_leaf_nodes=15, min_samples_leaf=20,
                                                l2_regularization=1.0, random_state=0)
    return {"ridge": ridge, "hgb": hgb}


def impute_fit(Xtr):
    med = np.nanmedian(Xtr, axis=0)
    def f(X):
        miss = np.isnan(X)
        return np.column_stack([np.where(miss, med, X), miss.astype(float)])
    return f


ARMS = ["A controls", "B + structure", "C + colour", "D + shuffled colour"]
pred = {(mname, arm): np.full(len(df), np.nan) for mname in models() for arm in ARMS}
for k in range(int(mf["folds"])):
    tr = (df.fold != k).to_numpy()
    te = ((df.fold == k) & (df.snap == 99)).to_numpy()
    tr_keys = [key[i] for i in np.where(tr)[0]]
    rsf = fit_red_sequence(tr_keys)
    col_all = colour_feats(key, rsf)
    col_shuf = colour_feats(key, rsf, shuffle_seed=100 + k)
    Xs = {"A controls": design(df, ["mass_noisy", "richness"]),
          "B + structure": design(df, ["mass_noisy", "richness"] + STRUCT)}
    Xs["C + colour"] = np.column_stack([Xs["B + structure"], col_all])
    Xs["D + shuffled colour"] = np.column_stack([Xs["B + structure"], col_shuf])
    y = df.y.to_numpy()
    for mname, mk in models().items():
        for arm, X in Xs.items():
            if mname == "ridge":
                imp = impute_fit(X[tr])
                m = mk().fit(imp(X[tr]), y[tr])
                pred[(mname, arm)][te] = m.predict(imp(X[te]))
            else:
                m = mk().fit(X[tr], y[tr])
                pred[(mname, arm)][te] = m.predict(X[te])

sc = (df.snap == 99).to_numpy()
y = df.y.to_numpy()[sc]
g = df.group_id.to_numpy()[sc]
ug = np.unique(g)
idx = {h: np.where(g == h)[0] for h in ug}
print(f"\nscored: {sc.sum()} snapshot-99 single observations from {len(ug)} parent halos")
for mname in models():
    print(f"\n== {mname}")
    P = {arm: pred[(mname, arm)][sc] for arm in ARMS}
    for arm in ARMS:
        print(f"  {arm:<22} R2 {r2_score(y, P[arm]):.3f}  MAE {mean_absolute_error(y, P[arm]):.3f} Gyr")
    boots = {pair: [] for pair in (("B + structure", "A controls"), ("C + colour", "B + structure"),
                                   ("C + colour", "D + shuffled colour"))}
    for _ in range(4000):
        s = np.concatenate([idx[h] for h in rng0.choice(ug, len(ug))])
        for (b_, a_) in boots:
            boots[(b_, a_)].append(r2_score(y[s], P[b_][s]) - r2_score(y[s], P[a_][s]))
    for (b_, a_), v in boots.items():
        d = r2_score(y, P[b_]) - r2_score(y, P[a_])
        print(f"  {b_[:1]} - {a_[:1]}: R2 {d:+.3f} [{np.percentile(v, 2.5):+.3f}, "
              f"{np.percentile(v, 97.5):+.3f}] (paired halo bootstrap)")
print("\nIDEALISED: FoF membership, true r500, intrinsic total photometry, gas-centroid "
      "'X-ray' centre. Not an observational forecast.")

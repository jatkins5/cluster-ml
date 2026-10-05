"""Snapshot 91 inventory, before building anything from it (read-only on
Chuiyang's directories).

  1. labels: at snap 99 label_score_all == label_score_pre (no future yet);
     at 91 'all' also counts mergers still to come, so the past-only score is
     the one comparable with the z=0 training label. Pseudo-TSC from each.
  2. halo map: snap91 FOF -> z=0 FOF, and the units of its R500 / GroupPos
  3. X-ray: exact redshift, image grid, event-file columns, and whether the
     particle background is in the events (counts above 9.5 keV)
"""
import glob
import pickle

import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import Planck15
from astropy.io import fits
from scipy import stats

C = "/oscar/data/idellant/Chuiyang"
H = Planck15.h

# ---------------------------------------------------------------- labels
with open("feats_labels_dict_tngcluster.pkl", "rb") as f:
    pkl = pickle.load(f)
with h5py.File("dataset.h5") as f:
    tau = f["labels/tau_gyr"][:]
    hid99 = f["meta/halo_id"][:]
    ptsc99 = f["labels/pseudo_tsc"][:]


def pseudo(scores):
    """first tau where the score crosses 0.5, as build_dataset.py does"""
    above = np.where(np.asarray(scores) >= 0.5)[0]
    return tau[above[0]] if len(above) else tau[-1]


KEYS = sorted([k for k in pkl[hid99[0]][99] if k.startswith("label_score_all_tau")],
              key=lambda k: float(k.split("tau")[1]))
assert np.allclose([float(k.split("tau")[1]) for k in KEYS], tau)


TAUS = [k.split("_tau")[1] for k in KEYS]


def scores(e, kind):
    """kind is the key prefix after 'label_score_' ('' for prepost's plain one)"""
    pre = "label_score" + (f"_{kind}" if kind else "")
    return np.array([e.get(f"{pre}_tau{t}", np.nan) for t in TAUS], float)


def prefixes(e):
    return sorted({k.split("_tau")[0] for k in e if k.startswith("label_score")})


print("ours,  snap 99:", prefixes(pkl[hid99[0]][99]), " snap 91:",
      prefixes(pkl[hid99[0]][91]))
with open(f"{C}/TNGCluster_Xray/feats_labels_dict_TNGCluster_prepost.pkl", "rb") as f:
    pp = pickle.load(f)
k0 = list(pp)[0]
print(f"prepost: {len(pp)} halos, key type {type(k0).__name__}, first {k0}; "
      f"value type {type(pp[k0]).__name__}")
if isinstance(pp[k0], dict):
    print("  snapshots:", sorted(pp[k0])[:3], "...", sorted(pp[k0])[-3:])
    for s in (99, 91):
        if s in pp[k0]:
            print(f"  snap {s}:", prefixes(pp[k0][s]))
            print("    other keys:", [k for k in pp[k0][s] if not k.startswith("label_score")][:15])
print("  prepost key sample:", [k for k in pp[k0][91] if k.startswith("label")][:3])
for s_ in (99, 91):
    for mine, theirs in (("all", ""), ("pre", "premerger")):
        d = [np.nanmax(np.abs(scores(pp[h][s_], theirs) - scores(pkl[h][s_], mine)))
             for h in hid99 if h in pp and s_ in pp[h] and s_ in pkl.get(h, {})]
        print(f"  snap {s_}: ours '{mine}' vs prepost '{theirs or 'label_score'}': "
              f"max |diff| median {np.nanmedian(d):.3g}, n={len(d)}")
e = pkl[hid99[0]][91]
print("using snap-91 label keys:", prefixes(e))
p91pre, p91all, ok, miss = [], [], [], []
for h in hid99:
    if 91 not in pkl.get(h, {}):
        continue
    s_pre, s_all = scores(pkl[h][91], "pre"), scores(pkl[h][91], "all")
    if np.isnan(s_pre).any() or np.isnan(s_all).any():
        miss.append(h)
        continue
    p91pre.append(pseudo(s_pre)); p91all.append(pseudo(s_all)); ok.append(h)
p91pre, p91all = np.array(p91pre), np.array(p91all)
p99 = np.array([ptsc99[list(hid99).index(h)] for h in ok])
print(f"{len(ok)} halos with complete snap-91 labels; {len(miss)} incomplete, e.g. "
      f"{[(h, prefixes(pkl[h][91])) for h in miss[:2]]}")
for name, v in [("pre", p91pre), ("all", p91all)]:
    print(f"  pseudo-TSC(91, {name}): median {np.median(v):.2f}, at cap "
          f"{(v >= tau[-1]).mean():.0%}; Spearman vs z=0 {stats.spearmanr(v, p99)[0]:+.2f}")
print(f"  z=0 pseudo-TSC: median {np.median(p99):.2f}, at cap {(p99 >= tau[-1]).mean():.0%}")
print(f"  pre != all at 91: {(p91pre != p91all).mean():.0%} of halos")

# ---------------------------------------------------------------- halo map
m = pd.read_csv(f"{C}/TNGCluster_Cutout/fof_halo_to_sub91.csv")
print(f"\nhalo map: {len(m)} rows, z=0 ids in dataset: "
      f"{np.isin(m.FOF_Halo_ID_At99, hid99).sum()}")
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    r99 = dict(zip(f["haloID"][:], f["r500c"][:] * 1000))
z91 = 0.1
a = 1 / (1 + z91)
rr = np.array([row.Group_R_Crit500 * a / H / r99[row.FOF_Halo_ID_At99]
               for row in m.itertuples() if row.FOF_Halo_ID_At99 in r99])
print(f"  R500(91) as ckpc/h -> physical kpc, over R500(99): median "
      f"{np.median(rr):.2f} (5-95% {np.percentile(rr, 5):.2f}-{np.percentile(rr, 95):.2f})")
cut = glob.glob(f"{C}/TNGCluster_Cutout/snap91/cutout_*.hdf5")
print(f"  cutouts: {len(cut)}")
with h5py.File(sorted(cut)[0]) as f:
    print("  cutout gas fields:", sorted(f["PartType0"].keys()))

# ---------------------------------------------------------------- X-ray
X = f"{C}/TNGCluster_Xray_Snap91"
for ax in "xyz":
    print(f"  snap91_{ax}: {len(glob.glob(f'{X}/snap91_{ax}/halo_*_img_{ax}.fits'))} images")
p = f"{X}/snap91_z/halo_{m.FOF_Halo_ID_At91[1]}_evt_z.fits"
if not glob.glob(p):
    p = sorted(glob.glob(f"{X}/snap91_z/halo_*_evt_z.fits"))[1]
with fits.open(p, memmap=True) as h:
    ev = h["EVENTS"]
    print(f"\nevents {p.split('/')[-1]}: {ev.header['NAXIS2']} rows, columns "
          f"{ev.columns.names}")
    for k in ("EXPOSURE", "TCRVL2", "TCDLT2", "TLMAX2", "INSTRUME", "REDSHIFT"):
        print(f"  {k} = {ev.header.get(k)}")
    en = ev.data["ENERGY"] / 1000.0
    print(f"  energies: {np.percentile(en, [1, 50, 99])} keV; above 9.5 keV "
          f"{(en > 9.5).sum()} (particle background present if > 0)")
img = p.replace("_evt_", "_img_")
with fits.open(img) as h:
    print(f"  image {h[0].data.shape}, CDELT {h[0].header.get('CDELT1')}")

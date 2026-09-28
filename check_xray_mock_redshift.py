"""At what distance were the X-ray mocks actually observed: z = 0.05 or 0.02?

The generation scripts on disk set `redshift = 0.05` for snapshot 99;
Chuiyang recalls 0.02. The two differ by (D_L(0.05)/D_L(0.02))^2 ~ 7 in
photon count, so the images themselves can decide. For each cluster, predict
the 2 Ms, 0.1-2 keV ACIS-I (Cycle 22) count from the catalogue 0.5-2 keV
luminosity at each candidate redshift -- APEC kT = 5 keV, Z = 0.3, the mock's
absorption -- and compare with the counts actually in the mock images.

The ACIS-I field covers only the inner ~+-500 kpc (at z = 0.05), so the
measured/predicted ratio should sit somewhat below 1 at the true redshift;
at the wrong one it is off by the factor of ~7 either way.
"""
import os

import h5py
import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15
from astropy.io import fits
from soxs import ApecGenerator

NH_MOCK = 0.01
KT, ABUND = 5.0, 0.3
T_S = 2.0e6

arf = os.path.expanduser("~/.cache/soxs/acisi_aimpt_cy22.arf")
with fits.open(arf) as h:
    d = h["SPECRESP"].data
    arf_e = 0.5 * (d["ENERG_LO"] + d["ENERG_HI"])
    arf_a = d["SPECRESP"]


def counts_per_energy_flux(z):
    """0.1-2 keV counts/s per (erg/s/cm^2) of rest-frame-ish 0.5-2 keV flux."""
    spec = ApecGenerator(0.05, 12.0, 4000).get_spectrum(KT, ABUND, z, 1.0)
    spec.apply_foreground_absorption(NH_MOCK, model="wabs")
    e = spec.emid.value
    ph = spec.flux.value * spec.de.value          # photons/s/cm^2 per bin
    cnt = (ph * np.interp(e, arf_e, arf_a, left=0, right=0))[(e >= 0.1) & (e <= 2.0)].sum()
    spec0 = ApecGenerator(0.05, 12.0, 4000).get_spectrum(KT, ABUND, z, 1.0)
    e0 = spec0.emid.value
    ef = (spec0.flux.value * spec0.de.value * e0 * 1.602e-9)[(e0 >= 0.5) & (e0 <= 2.0)].sum()
    return cnt / ef


with h5py.File("xray_counts_2Ms_128.h5") as f:
    counts = f["counts"][:].sum(axis=(2, 3)).astype(float)   # (N, 3)
    hid = f["meta/halo_id"][:]
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    lx = dict(zip(f["haloID"][:], f["xray_0.5-2.0kev"][:]))
logL = np.array([lx[h] for h in hid])
meas = counts.mean(axis=1)

print(f"{len(hid)} clusters; measured 2 Ms counts in the mock images: median "
      f"{np.median(meas):.3g}")
for z in (0.02, 0.05):
    dl = Planck15.luminosity_distance(z).to(u.cm).value
    k = counts_per_energy_flux(z)
    pred = 10 ** logL / (4 * np.pi * dl ** 2) * k * T_S
    r = meas / pred
    print(f"  assumed z={z}: predicted median {np.median(pred):.3g}  "
          f"measured/predicted median {np.median(r):.3f} "
          f"(10-90% {np.percentile(r, 10):.3f}-{np.percentile(r, 90):.3f})")
print("expected at the true redshift: somewhat below 1 (the chip sees only "
      "the inner part of the cluster); the wrong one is off by ~7x")

# Per-cluster: a mock simulated at z = 0.02 would stand out as a ~7x outlier
# in measured/predicted at the z = 0.05 hypothesis.
dl = Planck15.luminosity_distance(0.05).to(u.cm).value
r05 = meas / (10 ** logL / (4 * np.pi * dl ** 2) * counts_per_energy_flux(0.05) * T_S)
print(f"\nclusters with measured/predicted(z=0.05) > 3: {(r05 > 3).sum()}, "
      f"< 0.33: {(r05 < 0.33).sum()}  (a z=0.02 mock would be ~7)")
for h in (19429412, 19313671, 19102051, 19170433, 17950429):
    i = int(np.where(hid == h)[0][0])
    print(f"  halo {h}: ratio {r05[i]:.2f}")
order = np.argsort(r05)
print("  lowest 3:", [(int(hid[i]), round(float(r05[i]), 2)) for i in order[:3]],
      " highest 3:", [(int(hid[i]), round(float(r05[i]), 2)) for i in order[-3:]])

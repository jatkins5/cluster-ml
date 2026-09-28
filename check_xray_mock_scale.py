"""Second, independent test of the mock X-ray redshift: angular scale.

The photon-count test (check_xray_mock_redshift.py) depends on the
catalogue L_X definition. This one does not: it compares the radial
surface-brightness profile of the mock image, in arcsec, with the projected
emission-measure profile of the same cluster's hot gas, in kpc, and finds
the kpc-per-arcsec scale that lines the two up. Planck15 gives ~0.98 kpc/"
at z = 0.05 and ~0.41 kpc/" at z = 0.02 -- a factor 2.4, easily resolved.

Gas selection mirrors the X-ray script (T > 3e5 K, no star formation, inside
the 2 r500 cube around GroupPos); emission measure ~ m * rho per cell.
Read-only on Chuiyang's cutouts and images.
"""
import glob
import os
import re

import h5py
import numpy as np
from astropy import constants as c
from astropy import units as u
from astropy.cosmology import Planck15
from astropy.io import fits

from build_dataset import load_group_centers

H0 = Planck15.h
CUT = "/oscar/data/idellant/Chuiyang/TNGCluster_Cutout/snap99"
GC = "/oscar/data/idellant/Chuiyang/groupcat_classification/groupcat_099"
XR = "TNGCluster_Xray_Snap99/snap99_z"

gpos, _ = load_group_centers(GC)
with h5py.File("Radio_Data/TNG-Cluster_Catalog.hdf5") as f:
    r500 = dict(zip(f["haloID"][:], f["r500c"][:] * 1000.0))   # kpc

files = sorted(glob.glob(os.path.join(CUT, "cutout_sub*_FOF*.hdf5")),
               key=os.path.getsize)[:40:8]                     # 5 small ones
redges = np.logspace(np.log10(15), np.log10(420), 18)          # arcsec
for p in files:
    hid = int(re.search(r"_FOF(\d+)\.hdf5", p).group(1))
    img = os.path.join(XR, f"halo_{hid}_img.fits")
    if not os.path.exists(img) or hid not in r500:
        continue
    with h5py.File(p, "r") as f:
        g = f["PartType0"]
        pos = g["Coordinates"][:] / H0                          # kpc (a=1)
        m = g["Masses"][:].astype(float)
        rho = g["Density"][:].astype(float)
        ue = g["InternalEnergy"][:].astype(float)
        xe = g["ElectronAbundance"][:].astype(float)
        sfr = g["StarFormationRate"][:]
    mu = 4.0 / (1 + 3 * 0.76 + 4 * 0.76 * xe) * c.m_p.cgs.value
    T = (2 / 3) * ue / c.k_B.cgs.value * 1e10 * mu
    d = pos - gpos[hid]
    R5 = r500[hid]
    sel = (T > 3e5) & (sfr == 0) & np.all(np.abs(d) < R5, axis=1)
    Rk = np.hypot(d[sel, 0], d[sel, 1])
    em = (m * rho)[sel]
    with fits.open(img) as h:
        im = h[0].data.astype(float)
    n = im.shape[0]
    cy, cx = np.unravel_index(np.argmax(np.where(im > 0, im, 0)), im.shape)
    yy, xx = np.mgrid[:n, :n]
    ra = np.hypot(yy - cy, xx - cx) * 0.492                     # arcsec
    # image profile S(theta)
    S_img = np.array([im[(ra >= a) & (ra < b)].mean()
                      for a, b in zip(redges[:-1], redges[1:])])
    th = np.sqrt(redges[:-1] * redges[1:])
    best = None
    for scale in np.linspace(0.2, 1.6, 141):                    # kpc per arcsec
        kedges = redges * scale
        S_gas = np.array([em[(Rk >= a) & (Rk < b)].sum() / (np.pi * (b * b - a * a))
                          for a, b in zip(kedges[:-1], kedges[1:])])
        ok = (S_img > 0) & (S_gas > 0)
        if ok.sum() < 8:
            continue
        diff = np.log(S_img[ok]) - np.log(S_gas[ok])
        cost = np.var(diff)                                     # shape only
        if best is None or cost < best[1]:
            best = (scale, cost)
    kz = {z: Planck15.kpc_proper_per_arcmin(z).value / 60 for z in (0.02, 0.05)}
    print(f"halo {hid:>9}  r500 {R5:6.0f} kpc  best-fit {best[0]:.2f} kpc/arcsec"
          f"   (z=0.02 -> {kz[0.02]:.2f}, z=0.05 -> {kz[0.05]:.2f})")

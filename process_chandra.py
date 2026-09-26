#!/usr/bin/env python3
"""Turn the downloaded Chandra ACIS-I data into model inputs matching the mocks.

The redshift-placed training mocks (xray_real_placed.h5) are, per 128-grid
block of fixed *physical* size (38 x 0.492" at z=0.05), photon counts in
0.1-2 keV divided by the exposure in ks for the on-axis Cycle-22 ACIS-I
response, zeroed outside a common physical aperture, and stretched with
arcsinh(x / a) for one stored constant a. This reproduces each step on the
real data:

  1. CIAO fluximage per observation -> exposure map at 1.2 keV (cm^2 s),
     the energy the mocks' exposure maps used. CIAO lives in ./ciao_env and
     is called through a subprocess; this script runs in the project venv.
  2. Events in 0.1-2 keV binned directly onto the physical grid at the
     target's own redshift -- no image resampling -- centred on the smoothed
     X-ray peak near the catalogue position (the mocks are halo-centred, and
     the mock X-ray peak sits a median 0.01 r500 from the image centre).
  3. Exposure converted to "Cycle-22-equivalent seconds": exposure map /
     A_eff(1.2 keV) of the Cycle-22 response the mocks were simulated with.
     That corrects vignetting and the ACIS contamination build-up (early
     observations had much more soft effective area), at 1.2 keV only --
     a band-average approximation. Observations of one target are summed.
  4. Blocks with less than 30% of the target's median exposure are zeroed,
     mimicking the unexposed sky in the mocks; then the common aperture and
     the training stretch constant.

Only ACIS-I observations are used; ACIS-S targets need a footprint model
first.
"""
import argparse
import glob
import os
import re
import subprocess

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

from build_xray_realistic import (BLOCK, GRID, Z_MOCK, aperture_mask, cosmo,
                                  sky_area_factor)

CIAO = "/oscar/data/idellant/cluster-ml/ciao_env"
EMIN, EMAX = 0.1, 2.0
key = lambda s: re.sub(r"[\s_]+", "", str(s)).upper()


def ciao(cmd):
    """Run a CIAO command inside the CIAO environment."""
    full = ("module load miniforge3/25.3.0-3-a6hh >/dev/null 2>&1; "
            'source "$(conda info --base)/etc/profile.d/conda.sh"; '
            f"conda activate {CIAO}; {cmd}")
    r = subprocess.run(["bash", "-lc", full], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def aeff_cy22(energy_kev=1.2):
    arf = os.path.expanduser("~/.cache/soxs/acisi_aimpt_cy22.arf")
    with fits.open(arf) as h:
        d = h["SPECRESP"].data
        e = 0.5 * (d["ENERG_LO"] + d["ENERG_HI"])
        return float(np.interp(energy_kev, e, d["SPECRESP"]))


def block_kpc():
    c = cosmo()
    return (BLOCK * 0.492 / 206265.0
            * c.angular_diameter_distance(Z_MOCK).to("kpc").value)


def expmap_for(obsdir, obsid):
    """Exposure map (cm^2 s) at 1.2 keV via fluximage; cached per obsid."""
    evt = glob.glob(os.path.join(obsdir, "*evt2.fits*"))[0]
    out = os.path.join(obsdir, "fi")
    done = glob.glob(out + "*.expmap")
    if not done:
        asol = glob.glob(os.path.join(obsdir, "*asol1.fits*"))
        bpix = glob.glob(os.path.join(obsdir, "*bpix1.fits*"))
        ciao(f"punlearn fluximage; fluximage infile='{evt}' outroot='{out}' "
             f"bands='{EMIN}:{EMAX}:1.2' binsize=1 "
             f"asolfile='{asol[0] if asol else ''}' "
             f"badpixfile='{bpix[0] if bpix else ''}' maskfile=none "
             f"cleanup=yes clobber=yes verbose=0")
        done = glob.glob(out + "*.expmap")
    return evt, done[0]


def particle_band_ratio():
    """Counts in 0.1-2 keV over counts in 9.5-12 keV for the ACIS-I particle
    background, from the same soxs spectrum the mocks' particle background
    was drawn from. ACIS PI channels are 14.6 eV wide."""
    pha = os.path.expanduser("~/.cache/soxs/chandra_acisi_cy22_particle_bkgnd.pha")
    with fits.open(pha) as h:
        d = h[1].data
        chan = d["CHANNEL"].astype(float)
        cts = (d["COUNTS"] if "COUNTS" in d.names
               else d["RATE"]).astype(float)
    e = (chan + 0.5) * 0.0146
    soft = cts[(e >= EMIN) & (e <= EMAX)].sum()
    hard = cts[(e >= 9.5) & (e <= 12.0)].sum()
    return soft / hard


def mock_particle_rate(pbkg_h5="xray_partbkg_2Ms_128.h5"):
    """The mocks' particle background, counts per ks per grid block."""
    from build_xray_realistic import T_MOCK_KS
    with h5py.File(pbkg_h5, "r") as f:
        c = f["counts"][:].astype(float)
    yy, xx = np.mgrid[:GRID, :GRID]
    rr = np.hypot(yy - GRID / 2, xx - GRID / 2) * BLOCK * 0.492 / 60.0
    return float(np.median(c[:, (rr > 2) & (rr < 6)])) / T_MOCK_KS


def hard_band_rate(evt, emap):
    """Particle background in the real data: 9.5-12 keV events (the mirrors
    focus almost nothing there) per arcmin^2 of exposed chip per ks of
    livetime."""
    with fits.open(evt) as h:
        d, hd = h["EVENTS"].data, h["EVENTS"].header
        live = float(hd.get("LIVETIME", hd.get("EXPOSURE"))) / 1e3
        e = d["energy"] / 1000.0
        n_hard = int(((e >= 9.5) & (e <= 12.0)).sum())
    with fits.open(emap) as h:
        em = h[0].data
        pix_arcmin2 = (abs(h[0].header["CDELT1"]) * 60.0) ** 2
    area = float((em > 0.1 * np.nanmax(em)).sum()) * pix_arcmin2
    return n_hard / area / live, live


def events_radec(evt):
    """RA/Dec and energy (keV) of every event, from the sky x/y columns."""
    with fits.open(evt) as h:
        d, hd = h["EVENTS"].data, h["EVENTS"].header
        cols = [c.name.lower() for c in h["EVENTS"].columns]
        ix, iy = cols.index("x") + 1, cols.index("y") + 1
        w = WCS(naxis=2)
        w.wcs.ctype = [hd[f"TCTYP{ix}"], hd[f"TCTYP{iy}"]]
        w.wcs.crval = [hd[f"TCRVL{ix}"], hd[f"TCRVL{iy}"]]
        w.wcs.crpix = [hd[f"TCRPX{ix}"], hd[f"TCRPX{iy}"]]
        w.wcs.cdelt = [hd[f"TCDLT{ix}"], hd[f"TCDLT{iy}"]]
        e = d["energy"] / 1000.0
        keep = (e >= EMIN) & (e <= EMAX)
        ra, dec = w.wcs_pix2world(d["x"][keep], d["y"][keep], 1)
    return ra, dec


def to_grid_offsets(ra, dec, ra0, dec0, z):
    """(RA, Dec) -> block coordinates on the physical grid centred on ra0/dec0."""
    c = cosmo()
    kpc_per_arcsec = c.angular_diameter_distance(z).to("kpc").value / 206265.0
    dx = -(ra - ra0) * np.cos(np.radians(dec0)) * 3600.0 * kpc_per_arcsec
    dy = (dec - dec0) * 3600.0 * kpc_per_arcsec
    b = block_kpc()
    return dx / b + GRID / 2, dy / b + GRID / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="chandra")
    ap.add_argument("--train-h5", default="xray_real_placed.h5",
                    help="source of the stretch constant and the mock "
                         "brightness distribution to compare against")
    ap.add_argument("--targets-csv", default="LoVoCCS_target_list - lovoccs.csv")
    ap.add_argument("--output", default="xray_obs_acisi.h5")
    ap.add_argument("--min-exp-frac", type=float, default=0.3)
    ap.add_argument("--no-particle-correct", dest="particle_correct",
                    action="store_false",
                    help="the original normalisation, kept for comparison")
    ap.add_argument("--no-fill-gaps", dest="fill_gaps", action="store_false")
    args = ap.parse_args()

    obs = pd.read_csv(os.path.join(args.data_dir, "observations.csv"))
    obs = obs[obs["detector"].astype(str).str.strip() == "ACIS-I"]
    tl = pd.read_csv(args.targets_csv)
    tl["k"] = tl["name"].map(key)
    pos = {r.k: (float(r["ra(deg)"]), float(r["dec(deg)"]),
                 float(r["redshift"])) for _, r in tl.iterrows()}
    with h5py.File(args.train_h5, "r") as f:
        a = float(f.attrs["stretch_scale"])
        mock_img = f["mock/images"][:]
        mock_z = f["mock/z"][:]
    A = aeff_cy22()
    print(f"stretch constant a={a:.4g} from {args.train_h5}; "
          f"A_eff(1.2 keV, cy22) = {A:.1f} cm^2; block = {block_kpc():.1f} kpc")
    # Particle background, handled separately from focused photons: it is
    # not collected by the mirrors, so dividing it by an effective-area-scaled
    # exposure (1.5-2.6x the livetime for these epochs) pushed it below
    # anything in the mocks. Subtract the real particle level, normalise only
    # the focused photons, then add back the mocks' particle level.
    R_soft_hard = particle_band_ratio() if args.particle_correct else None
    p_mock = mock_particle_rate() if args.particle_correct else 0.0
    if args.particle_correct:
        print(f"particle background: 0.1-2 / 9.5-12 keV = {R_soft_hard:.3f}; "
              f"mock level {p_mock:.4f} counts/ks/block")

    out_img, names, keys, zs, texp, cover = [], [], [], [], [], []
    for tgt, grp in obs.groupby(obs["target"]):
        k = key(tgt)
        ra0, dec0, z = pos[k]
        counts = np.zeros((GRID, GRID))
        t_eq = np.zeros((GRID, GRID))
        p_real = np.zeros((GRID, GRID))          # expected particle counts
        all_ra, all_dec = [], []
        for _, o in grp.iterrows():
            d = os.path.join(args.data_dir, k, str(int(o["obsid"])))
            try:
                evt, emap = expmap_for(d, int(o["obsid"]))
            except Exception as e:
                print(f"  {tgt} {o['obsid']}: fluximage failed ({str(e)[:200]})")
                continue
            ra, dec = events_radec(evt)
            all_ra.append(ra)
            all_dec.append(dec)
            with fits.open(emap) as h:
                em, ew = h[0].data.astype(np.float64), WCS(h[0].header)
            # exposure sampled on a 4x4 sub-grid per block, in sky coords
            sub = (np.arange(GRID * 4) + 0.5) / 4.0
            gx, gy = np.meshgrid(sub, sub)
            c = cosmo()
            kpa = c.angular_diameter_distance(z).to("kpc").value / 206265.0
            b = block_kpc()
            dra = -(gx - GRID / 2) * b / kpa / 3600.0 / np.cos(np.radians(dec0))
            ddec = (gy - GRID / 2) * b / kpa / 3600.0
            px, py = ew.wcs_world2pix(ra0 + dra, dec0 + ddec, 0)
            vals = ndimage.map_coordinates(em, [py.ravel(), px.ravel()],
                                           order=0, cval=0.0).reshape(gx.shape)
            t_eq += vals.reshape(GRID, 4, GRID, 4).mean(axis=(1, 3)) / A
            if args.particle_correct:
                hr, live = hard_band_rate(evt, emap)
                onchip = (vals > 0).reshape(GRID, 4, GRID, 4).mean(axis=(1, 3))
                # block area on the sky at the target's redshift
                blk_sky = (block_kpc() / kpa / 60.0) ** 2
                p_real += hr * R_soft_hard * blk_sky * live * onchip
                print(f"    {o['obsid']}: particle {hr * R_soft_hard:.3f} "
                      f"counts/ks/arcmin^2 in 0.1-2 keV (9.5-12 keV scaled)")
        if not all_ra:
            continue
        ra = np.concatenate(all_ra)
        dec = np.concatenate(all_dec)
        # Recentre on the smoothed X-ray peak within 2 arcmin of the target.
        gx, gy = to_grid_offsets(ra, dec, ra0, dec0, z)
        H, _, _ = np.histogram2d(gy, gx, bins=GRID, range=[[0, GRID], [0, GRID]])
        sm = ndimage.gaussian_filter(H, 2.0)
        c = cosmo()
        near = int(round(120.0 * c.angular_diameter_distance(z).to("kpc").value
                         / 206265.0 / block_kpc()))
        yy, xx = np.mgrid[:GRID, :GRID]
        win = np.hypot(yy - GRID / 2, xx - GRID / 2) <= max(near, 2)
        py, px = np.unravel_index(np.argmax(np.where(win, sm, -1)), sm.shape)
        shift_y, shift_x = py + 0.5 - GRID / 2, px + 0.5 - GRID / 2
        gx, gy = gx - shift_x, gy - shift_y
        t_eq = ndimage.shift(t_eq, (-shift_y, -shift_x), order=0, cval=0.0)
        p_real = ndimage.shift(p_real, (-shift_y, -shift_x), order=0, cval=0.0)
        counts, _, _ = np.histogram2d(gy, gx, bins=GRID,
                                      range=[[0, GRID], [0, GRID]])
        t_ks = t_eq / 1000.0
        good = t_ks > args.min_exp_frac * np.median(t_ks[t_ks > 0])
        m = aperture_mask(z) & good
        if args.particle_correct:
            # focused photons normalised by the effective-area exposure,
            # particle background at the mocks' level per ks
            # the mocks' particle level per block, scaled to the sky area a
            # block covers at this redshift (the grid is fixed in kpc)
            rate = ((counts - p_real) / np.where(good, t_ks, 1.0)
                    + p_mock * sky_area_factor(z))
        else:
            rate = counts / np.where(good, t_ks, 1.0)
        rate = np.where(m, rate, 0.0)
        if args.fill_gaps:
            # Chip gaps and chip edges inside the aperture: fill from the
            # neighbours (normalised convolution) instead of leaving zero
            # lines the mocks never contain.
            ap = aperture_mask(z)
            hole = ap & ~good
            if hole.any():
                w = ndimage.gaussian_filter(good.astype(float), 1.5)
                v = ndimage.gaussian_filter(np.where(good, rate, 0.0), 1.5)
                fill = np.where(w > 1e-3, v / np.maximum(w, 1e-3), p_mock)
                rate = np.where(hole, fill, rate)
                m = ap
        out_img.append(np.arcsinh(rate / a).astype(np.float32))
        names.append(tgt)
        keys.append(k)
        zs.append(z)
        texp.append(float(np.median(t_ks[good])))
        cover.append(float(m.sum() / aperture_mask(z).sum()))
        print(f"  {tgt:<18} z={z:.4f}  {len(grp)} obs  "
              f"exposure {texp[-1]:.1f} ks  counts in aperture "
              f"{counts[m].sum():.3g}  aperture exposed {cover[-1]:.0%}  "
              f"peak shift {np.hypot(shift_x, shift_y) * block_kpc():.0f} kpc")

    imgs = np.stack(out_img)
    # How bright are the real clusters compared with mocks at similar z?
    lin_real = np.sinh(imgs.astype(np.float64)) * a
    lin_mock = np.sinh(mock_img.astype(np.float64)) * a
    print("\nmean count rate in the aperture (counts/ks/block): real vs mocks "
          "at |dz| < 0.01")
    ratios = []
    for i, z in enumerate(zs):
        sel = np.abs(mock_z - z) < 0.01
        if not sel.any():
            continue
        mm = lin_mock[sel].reshape(-1, GRID, GRID)
        msk = aperture_mask(z)
        rm = np.median(mm[:, msk].mean(axis=1))
        rr = lin_real[i][msk].mean()
        ratios.append(rr / rm)
        print(f"  {names[i]:<18} real {rr:.3g}  median mock {rm:.3g}  "
              f"ratio {rr / rm:.2f}")
    print(f"median real / mock: {np.median(ratios):.2f}  (the L_X comparison "
          f"with the catalogue gave ~1/3.6 = 0.28)")

    with h5py.File(args.output, "w") as g:
        o = g.create_group("obs")
        o.create_dataset("images", data=imgs)
        o.create_dataset("name", data=np.array(names, dtype=h5py.string_dtype()))
        o.create_dataset("key", data=np.array(keys, dtype=h5py.string_dtype()))
        o.create_dataset("z", data=np.array(zs))
        o.create_dataset("exposure_ks", data=np.array(texp))
        o.create_dataset("aperture_exposed", data=np.array(cover))
        g.attrs["stretch_scale"] = a
        g.attrs["train_h5"] = args.train_h5
    print(f"\nwrote {args.output}: {imgs.shape}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prepare standardized physical-frame images from MeerKAT FITS files for ML input.

Regrids MeerKAT radio images to a common 256x256 pixel grid covering 2 Mpc x 2 Mpc
(7.8 kpc/pixel), centered on each cluster. Optionally smooths to larger physical beam
sizes (15, 25, 50 kpc) using Gaussian convolution.

Output is suitable for training/inference with a CNN model predicting cluster merger
time-since-collision from radio morphology.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates


# Output image parameters
NPIX = 256
PHYS_SIZE_KPC = 2000.0  # 2 Mpc
PIXEL_SCALE_KPC = PHYS_SIZE_KPC / NPIX  # 7.8125 kpc/pixel

# Smoothing beam targets (kpc FWHM)
BEAM_TARGETS_KPC = [15, 25, 50]

# Planck 2018 cosmology
COSMO = FlatLambdaCDM(H0=67.4, Om0=0.315)

# Cluster data from LoVoCCS catalog
CLUSTERS = [
    {'name': 'A85',   'ra': 10.46,  'dec': -9.30,  'z': 0.0555},
    {'name': 'A133',  'ra': 15.68,  'dec': -21.87, 'z': 0.0569},
    {'name': 'A3667', 'ra': 303.13, 'dec': -56.83, 'z': 0.0556},
]


def load_meerkat_image(name):
    """Load a MeerKAT FITS image and return 2D data, WCS, header, beam size."""
    fits_path = f'meerkat_images/meerkat_{name}.fits'
    hdul = fits.open(fits_path)
    header = hdul[0].header
    data = np.squeeze(hdul[0].data).astype(np.float64)

    full_wcs = WCS(header)
    wcs = full_wcs.celestial if full_wcs.naxis > 2 else full_wcs

    # Beam from CLEANBMJ/CLEANBMN (degrees), fallback to BMAJ/BMIN
    bmaj = header.get('CLEANBMJ', header.get('BMAJ', None))
    bmin = header.get('CLEANBMN', header.get('BMIN', None))
    if bmaj is not None and bmin is not None:
        beam_arcsec = np.sqrt(bmaj * bmin) * 3600  # geometric mean, arcsec
    else:
        beam_arcsec = 8.0
        print(f"  WARNING: No beam info found, using {beam_arcsec}\" default")

    hdul.close()
    return data, wcs, header, beam_arcsec


def regrid_to_physical(data, wcs, ra_center, dec_center, z):
    """
    Regrid a sky-plane image to a 256x256 physical-frame grid.

    For each output pixel, computes the physical offset from cluster center,
    converts to RA/Dec via tangent-plane projection, then maps to input pixel
    coordinates for interpolation.
    """
    d_a = COSMO.angular_diameter_distance(z).to('kpc').value
    cos_dec = np.cos(np.radians(dec_center))

    # Physical offsets from center for each output pixel
    offsets_kpc = (np.arange(NPIX) - (NPIX - 1) / 2.0) * PIXEL_SCALE_KPC

    # 2D grids: dx varies along columns (East), dy varies along rows (North)
    dx_kpc, dy_kpc = np.meshgrid(offsets_kpc, offsets_kpc)

    # Physical offset -> angular offset -> RA/Dec
    # East (positive dx) = decreasing RA
    ra_grid = ra_center - np.degrees(dx_kpc / d_a) / cos_dec
    dec_grid = dec_center + np.degrees(dy_kpc / d_a)

    # RA/Dec -> input pixel coordinates
    input_x, input_y = wcs.world_to_pixel_values(ra_grid, dec_grid)

    # Replace NaN/inf with 0 for interpolation
    data_clean = np.where(np.isfinite(data), data, 0.0)

    # Interpolate (map_coordinates expects [row, col] = [y, x])
    coords = np.array([input_y.ravel(), input_x.ravel()])
    output = map_coordinates(data_clean, coords, order=1, mode='constant', cval=0.0)

    return output.reshape(NPIX, NPIX), d_a


def smooth_to_beam(image, native_beam_kpc, target_beam_kpc):
    """Smooth image to a target beam, subtracting native beam in quadrature."""
    fwhm_to_sigma = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    sigma_native = native_beam_kpc * fwhm_to_sigma
    sigma_target = target_beam_kpc * fwhm_to_sigma

    if sigma_target <= sigma_native:
        print(f"    Target beam ({target_beam_kpc:.1f} kpc) <= native "
              f"({native_beam_kpc:.1f} kpc), skipping")
        return image.copy()

    sigma_convolve_kpc = np.sqrt(sigma_target**2 - sigma_native**2)
    sigma_convolve_pix = sigma_convolve_kpc / PIXEL_SCALE_KPC

    print(f"    Smoothing: {native_beam_kpc:.1f} -> {target_beam_kpc:.1f} kpc "
          f"(convolve sigma = {sigma_convolve_pix:.2f} pix)")

    return gaussian_filter(image, sigma=sigma_convolve_pix)


def make_output_header(ra_center, dec_center, z, d_a, beam_kpc=None):
    """Create a FITS header for the output physical-frame image."""
    pixel_scale_deg = np.degrees(PIXEL_SCALE_KPC / d_a)

    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = NPIX
    header['NAXIS2'] = NPIX
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CRPIX1'] = (NPIX + 1) / 2.0
    header['CRPIX2'] = (NPIX + 1) / 2.0
    header['CRVAL1'] = ra_center
    header['CRVAL2'] = dec_center
    header['CDELT1'] = -pixel_scale_deg
    header['CDELT2'] = pixel_scale_deg
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['BUNIT'] = 'JY/BEAM'
    header['REDSHIFT'] = (z, 'Cluster redshift')
    header['DA_KPC'] = (d_a, 'Angular diameter distance [kpc]')
    header['PIXSCALE'] = (PIXEL_SCALE_KPC, 'Pixel scale [kpc]')
    header['PHYSSIZE'] = (PHYS_SIZE_KPC, 'Image physical size [kpc]')
    if beam_kpc is not None:
        header['BEAM_KPC'] = (beam_kpc, 'Beam FWHM [kpc]')
    header['HISTORY'] = 'Regridded to physical frame by prepare_ml_images.py'

    return header


def make_comparison_plot(name, native, smoothed_dict, output_dir):
    """4-panel comparison: native + 3 smoothed versions."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    panels = [('Native', native)] + [(f'{k} kpc', v) for k, v in smoothed_dict.items()]
    extent = [-PHYS_SIZE_KPC / 2, PHYS_SIZE_KPC / 2,
              -PHYS_SIZE_KPC / 2, PHYS_SIZE_KPC / 2]

    for ax, (label, img) in zip(axes, panels):
        positive = img[img > 0]
        vmax = np.percentile(positive, 99.5) if len(positive) > 0 else 1.0
        vmin = -0.1 * vmax

        ax.imshow(img, origin='lower', cmap='inferno',
                  vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(f'{name} - {label}', fontsize=11)
        ax.set_xlabel('kpc')
        ax.set_ylabel('kpc')

        # 300 kpc scale bar
        ax.plot([-400, -100], [-900, -900], 'w-', linewidth=2)
        ax.text(-250, -850, '300 kpc', color='white', ha='center', fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{name}_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def make_summary_plot(cluster_images, output_dir):
    """All clusters side by side at native resolution."""
    n = len(cluster_images)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    extent = [-PHYS_SIZE_KPC / 2, PHYS_SIZE_KPC / 2,
              -PHYS_SIZE_KPC / 2, PHYS_SIZE_KPC / 2]

    for ax, (name, img) in zip(axes, cluster_images):
        positive = img[img > 0]
        vmax = np.percentile(positive, 99.5) if len(positive) > 0 else 1.0
        vmin = -0.1 * vmax

        ax.imshow(img, origin='lower', cmap='inferno',
                  vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(f'{name} (native)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Physical offset (kpc)')
        ax.set_ylabel('Physical offset (kpc)')

    plt.suptitle('MeerKAT clusters — 2 Mpc x 2 Mpc physical frame (256x256)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


def main():
    output_dir = 'ml_images'
    os.makedirs(output_dir, exist_ok=True)

    print("Preparing ML images: MeerKAT -> physical frame (256x256, 2 Mpc)")
    print("=" * 70)

    summary_images = []

    for cluster in CLUSTERS:
        name = cluster['name']
        ra, dec, z = cluster['ra'], cluster['dec'], cluster['z']

        print(f"\n{'=' * 70}")
        print(f"Processing {name} (z={z}, RA={ra}, Dec={dec})")
        print(f"{'=' * 70}")

        d_a = COSMO.angular_diameter_distance(z).to('kpc').value
        print(f"  D_A = {d_a:.1f} kpc")
        print(f"  Pixel scale: {PIXEL_SCALE_KPC:.2f} kpc "
              f"= {np.degrees(PIXEL_SCALE_KPC / d_a) * 3600:.2f}\"")

        # Load input
        data, wcs, orig_header, beam_arcsec = load_meerkat_image(name)
        beam_kpc = np.radians(beam_arcsec / 3600.0) * d_a
        print(f"  Input: {data.shape}, beam: {beam_arcsec:.1f}\" = {beam_kpc:.1f} kpc")

        # Regrid
        print(f"  Regridding to {NPIX}x{NPIX} ({PHYS_SIZE_KPC:.0f} kpc)...")
        native_image, _ = regrid_to_physical(data, wcs, ra, dec, z)
        print(f"  Output range: [{native_image.min():.6f}, {native_image.max():.6f}] Jy/beam")

        # Save native
        hdr = make_output_header(ra, dec, z, d_a, beam_kpc=beam_kpc)
        native_path = os.path.join(output_dir, f'{name}_native.fits')
        fits.writeto(native_path, native_image, hdr, overwrite=True)
        print(f"  Saved: {native_path}")

        summary_images.append((name, native_image))

        # Smooth to target beams
        smoothed = {}
        for target_kpc in BEAM_TARGETS_KPC:
            sm = smooth_to_beam(native_image, beam_kpc, target_kpc)
            smoothed[target_kpc] = sm

            hdr_sm = make_output_header(ra, dec, z, d_a, beam_kpc=target_kpc)
            sm_path = os.path.join(output_dir, f'{name}_beam{target_kpc}kpc.fits')
            fits.writeto(sm_path, sm, hdr_sm, overwrite=True)
            print(f"  Saved: {sm_path}")

        # Comparison plot
        make_comparison_plot(name, native_image, smoothed, output_dir)

    # Summary plot
    make_summary_plot(summary_images, output_dir)

    print(f"\n{'=' * 70}")
    print(f"Done! All outputs in: {output_dir}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

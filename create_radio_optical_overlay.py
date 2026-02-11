#!/usr/bin/env python3
"""
Create composite images with VLASS radio contours overlaid on Legacy Survey optical RGB.
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import AsinhStretch, ImageNormalize
from astropy.stats import mad_std


def create_rgb_from_legacy(legacy_hdu):
    """
    Convert Legacy Survey grz bands to RGB image.

    Parameters:
    -----------
    legacy_hdu : HDU object
        FITS HDU with grz bands (shape: 3, ny, nx)

    Returns:
    --------
    tuple : (rgb_array, wcs) - RGB array normalized to [0,1] and 2D WCS
    """
    data = legacy_hdu[0].data
    header = legacy_hdu[0].header

    # Data shape is (3, ny, nx) for g, r, z bands
    g_band = data[0]
    r_band = data[1]
    z_band = data[2]

    # Map to RGB: z->R, r->G, g->B (standard optical mapping)
    # Use asinh stretch for better dynamic range
    stretch = AsinhStretch(a=0.1)

    def normalize_band(band):
        """Normalize a band to [0, 1] with asinh stretch."""
        # Handle NaN values
        band = np.nan_to_num(band, nan=0.0)
        # Clip negative values
        band = np.clip(band, 0, None)
        # Normalize to [0, 1] based on percentiles
        vmin = np.percentile(band[band > 0], 1) if np.any(band > 0) else 0
        vmax = np.percentile(band, 99.5)
        if vmax <= vmin:
            vmax = vmin + 1
        band_norm = (band - vmin) / (vmax - vmin)
        band_norm = np.clip(band_norm, 0, 1)
        # Apply asinh stretch
        band_stretched = stretch(band_norm)
        return band_stretched

    r_img = normalize_band(z_band)
    g_img = normalize_band(r_band)
    b_img = normalize_band(g_band)

    # Stack into RGB
    rgb = np.stack([r_img, g_img, b_img], axis=-1)

    # Get 2D WCS (drop the band axis)
    wcs_full = WCS(header)
    if wcs_full.naxis > 2:
        wcs = wcs_full.celestial
    else:
        wcs = wcs_full

    return rgb, wcs


def calculate_contour_levels(data, sigma_levels=[3, 5, 10, 20, 50, 100]):
    """
    Calculate contour levels based on MAD-estimated RMS.

    Parameters:
    -----------
    data : ndarray
        Radio image data
    sigma_levels : list
        Sigma multiples for contour levels

    Returns:
    --------
    ndarray : Contour levels in data units
    """
    # Use MAD to estimate RMS (robust to sources)
    rms = mad_std(data, ignore_nan=True)
    levels = np.array(sigma_levels) * rms
    return levels, rms


def create_overlay_image(vlass_fits, legacy_fits, output_path, contour_color='cyan',
                         sigma_levels=[3, 5, 10, 20, 50, 100]):
    """
    Create overlay image with radio contours on optical RGB.

    Parameters:
    -----------
    vlass_fits : str
        Path to VLASS FITS file
    legacy_fits : str
        Path to Legacy Survey FITS file
    output_path : str
        Output PNG path
    contour_color : str
        Color for radio contours
    sigma_levels : list
        Sigma multiples for contour levels

    Returns:
    --------
    bool : True if successful
    """
    # Load VLASS data
    with fits.open(vlass_fits) as vlass_hdu:
        vlass_data = np.squeeze(vlass_hdu[0].data)
        vlass_header = vlass_hdu[0].header
        vlass_wcs_full = WCS(vlass_header)
        vlass_wcs = vlass_wcs_full.celestial if vlass_wcs_full.naxis > 2 else vlass_wcs_full

    # Load Legacy data
    with fits.open(legacy_fits) as legacy_hdu:
        rgb, optical_wcs = create_rgb_from_legacy(legacy_hdu)

    # Calculate contour levels
    levels, rms = calculate_contour_levels(vlass_data, sigma_levels)
    print(f"  Radio RMS: {rms*1e6:.1f} uJy/beam")
    print(f"  Contour levels: {', '.join([f'{l*1e3:.2f}' for l in levels])} mJy/beam")

    # Create figure with optical WCS
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=optical_wcs)

    # Display optical RGB
    ax.imshow(rgb, origin='lower', interpolation='nearest')

    # Set axis limits to optical image extent (clips contours to this region)
    ny, nx = rgb.shape[:2]
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)

    # Overlay radio contours using WCS transform
    # This handles the coordinate transformation automatically
    contours = ax.contour(
        vlass_data,
        transform=ax.get_transform(vlass_wcs),
        levels=levels,
        colors=contour_color,
        linewidths=0.8,
        alpha=0.9
    )

    # Add contour labels for higher sigma levels
    if len(levels) > 2:
        try:
            ax.clabel(contours, levels[2:], inline=True, fontsize=8, fmt='%.1e')
        except Exception:
            pass  # Skip labels if no contours at those levels

    # Labels and title
    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)

    # Extract cluster name from filename
    name = os.path.basename(output_path).replace('overlay_', '').replace('.png', '')
    ax.set_title(f'{name}\nVLASS 3 GHz (cyan) + Legacy Survey grz', fontsize=12)

    # Add scale bar info
    ax.text(0.02, 0.02, f'RMS: {rms*1e6:.1f} μJy/beam',
            transform=ax.transAxes, fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Grid
    ax.coords.grid(color='white', ls='--', alpha=0.3)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)

    print(f"  Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Create radio-optical overlay images for LoVoCCS targets'
    )
    parser.add_argument('--all', action='store_true',
                        help='Process all targets with both VLASS and Legacy images')
    parser.add_argument('--clusters', nargs='+', default=None,
                        help='Specific cluster names to process (e.g., A780 A1644)')
    parser.add_argument('--output-dir', default='overlay_images',
                        help='Output directory for images (default: overlay_images)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip targets that already have overlay images')
    parser.add_argument('--contour-color', default='cyan',
                        help='Color for radio contours (default: cyan)')
    parser.add_argument('--vlass-dir', default='vlass_images',
                        help='Directory containing VLASS FITS files')
    parser.add_argument('--legacy-dir', default='legacy_images',
                        help='Directory containing Legacy Survey FITS files')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find matching pairs of VLASS and Legacy files
    vlass_files = {f.replace('vlass_', '').replace('.fits', ''): f
                   for f in os.listdir(args.vlass_dir) if f.endswith('.fits')}
    legacy_files = {f.replace('legacy_', '').replace('.fits', ''): f
                    for f in os.listdir(args.legacy_dir) if f.endswith('.fits')}

    # Find common targets
    common = set(vlass_files.keys()) & set(legacy_files.keys())
    print(f"Found {len(vlass_files)} VLASS files, {len(legacy_files)} Legacy files")
    print(f"Common targets: {len(common)}")
    print()

    if len(common) == 0:
        print("No matching pairs found. Run download_legacy_optical.py first.")
        return

    # Determine which targets to process
    if args.clusters:
        targets = [c for c in args.clusters if c in common]
        missing = [c for c in args.clusters if c not in common]
        if missing:
            print(f"Warning: Missing data for: {', '.join(missing)}")
        if len(targets) == 0:
            print("Error: None of the specified clusters have both VLASS and Legacy data")
            return
        print(f"Processing {len(targets)} specified cluster(s)")
    elif args.all:
        targets = sorted(common)
        print(f"Processing all {len(targets)} targets")
    else:
        # Default: sample of 3
        targets = sorted(common)[:3]
        print("Processing sample of 3 targets (use --all for all targets)")

    print()

    # Track results
    successful = []
    failed = []
    skipped = []

    for i, name in enumerate(targets, 1):
        vlass_path = os.path.join(args.vlass_dir, vlass_files[name])
        legacy_path = os.path.join(args.legacy_dir, legacy_files[name])
        output_path = os.path.join(args.output_dir, f"overlay_{name}.png")

        print(f"\n[{i}/{len(targets)}] {name}")
        print("=" * 70)

        # Skip if already exists
        if args.skip_existing and os.path.exists(output_path):
            print(f"  Skipping - already exists: {output_path}")
            skipped.append(name)
            continue

        try:
            result = create_overlay_image(
                vlass_fits=vlass_path,
                legacy_fits=legacy_path,
                output_path=output_path,
                contour_color=args.contour_color
            )
            if result:
                successful.append(name)
            else:
                failed.append(name)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(targets)}")
    print(f"Skipped: {len(skipped)}/{len(targets)}")
    print(f"Failed: {len(failed)}/{len(targets)}")
    if failed:
        print(f"\nFailed targets: {', '.join(failed)}")
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()

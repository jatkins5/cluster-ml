#!/usr/bin/env python3
"""
Download and display GMRT TGSS ADR1 (150 MHz) image cutouts for LoVoCCS targets.

This script downloads radio images from the TGSS ADR1 survey (GMRT 150 MHz)
via NASA SkyView for galaxy clusters that have GMRT matches.

Requires: lovoccs_gmrt_matches.csv (from query_gmrt_observations.py)
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch
from astroquery.skyview import SkyView
import numpy as np
import warnings
warnings.filterwarnings('ignore')

SURVEY_NAME = 'TGSS ADR1'


def download_gmrt_cutout(ra, dec, size=0.25, name="target", output_dir="gmrt_images",
                         pixels=500):
    """
    Download a GMRT TGSS 150 MHz image cutout via SkyView.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    size : float
        Image size in degrees (default 0.25 deg = 15 arcmin)
    name : str
        Target name for the filename
    output_dir : str
        Output directory
    pixels : int
        Image size in pixels

    Returns
    -------
    astropy.io.fits.HDUList or None
    """
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    print(f"Downloading TGSS cutout for {name}")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print(f"  Size: {size:.2f} deg ({size*60:.1f} arcmin), {pixels} px")

    try:
        hdu_list = SkyView.get_images(
            position=coord,
            survey=[SURVEY_NAME],
            radius=size * u.degree,
            pixels=pixels,
        )

        if not hdu_list or len(hdu_list) == 0:
            print("  No TGSS data returned")
            return None

        hdu = hdu_list[0]

        os.makedirs(output_dir, exist_ok=True)
        safe_name = name.replace(' ', '_')
        filename = os.path.join(output_dir, f"gmrt_{safe_name}.fits")
        hdu.writeto(filename, overwrite=True)

        data = np.squeeze(hdu[0].data)
        valid = data[~np.isnan(data)]
        if len(valid) > 0:
            print(f"  Max flux: {np.max(valid):.4e} Jy/beam, "
                  f"Mean: {np.mean(valid):.4e} Jy/beam")

        print(f"  Saved: {filename}")
        return hdu

    except Exception as e:
        print(f"  Error downloading: {e}")
        return None


def display_gmrt_image(hdu, name="target", output_dir="gmrt_images"):
    """
    Create a PNG visualization of a GMRT TGSS image.

    Parameters
    ----------
    hdu : astropy.io.fits.HDUList
        FITS HDU list containing the image
    name : str
        Target name for the title
    output_dir : str
        Output directory
    """
    data = np.squeeze(hdu[0].data)
    header = hdu[0].header

    wcs = WCS(header)
    if wcs.naxis > 2:
        wcs = wcs.celestial

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=wcs)

    try:
        norm = ImageNormalize(data, stretch=AsinhStretch())
    except Exception:
        norm = None

    im = ax.imshow(data, origin='lower', cmap='viridis', norm=norm,
                   interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux Density (Jy/beam)', fontsize=12)

    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)
    ax.set_title(f'GMRT TGSS 150 MHz: {name}', fontsize=14, fontweight='bold')
    ax.grid(color='white', ls='--', alpha=0.3)

    valid = data[~np.isnan(data)]
    if len(valid) > 0:
        stats_text = (f'Max: {np.max(valid):.2e} Jy/beam\n'
                      f'Mean: {np.mean(valid):.2e} Jy/beam')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    safe_name = name.replace(' ', '_')
    output_file = os.path.join(output_dir, f"gmrt_{safe_name}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Download GMRT TGSS 150 MHz images for LoVoCCS clusters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Download a few representative targets
  %(prog)s --all                     # Download all clusters with GMRT matches
  %(prog)s --clusters A85 A780       # Download specific clusters
  %(prog)s --png-only                # Regenerate PNGs from existing FITS
  %(prog)s --size 0.4                # Use larger cutout (0.4 deg = 24 arcmin)
        """
    )
    parser.add_argument('--all', action='store_true',
                        help='Process all clusters with GMRT matches')
    parser.add_argument('--clusters', nargs='+', default=None,
                        help='Specific cluster names to process')
    parser.add_argument('--output-dir', default='gmrt_images',
                        help='Output directory (default: gmrt_images)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip clusters that already have FITS+PNG')
    parser.add_argument('--png-only', action='store_true',
                        help='Only generate PNGs from existing FITS files')
    parser.add_argument('--size', type=float, default=0.25,
                        help='Image size in degrees (default: 0.25 = 15 arcmin)')
    parser.add_argument('--pixels', type=int, default=500,
                        help='Image size in pixels (default: 500)')

    args = parser.parse_args()

    # Load GMRT match results
    match_file = "lovoccs_gmrt_matches.csv"
    try:
        results = pd.read_csv(match_file)
    except FileNotFoundError:
        print(f"Error: {match_file} not found!")
        print("Please run query_gmrt_observations.py first.")
        return

    matched = results[results['has_gmrt_match'] == True].copy()
    matched = matched.sort_values('closest_flux_mJy', ascending=False)

    if len(matched) == 0:
        print("No clusters with GMRT matches found!")
        return

    print("=" * 80)
    print("LoVoCCS - GMRT TGSS 150 MHz Image Download")
    print("=" * 80)
    print(f"Clusters with GMRT matches: {len(matched)}")
    print(f"Image size: {args.size:.2f} deg ({args.size*60:.1f} arcmin), "
          f"{args.pixels} px")
    print(f"Output directory: {args.output_dir}")
    print()

    # Select targets
    if args.clusters:
        targets = matched[matched['name'].isin(args.clusters)]
        if len(targets) == 0:
            print(f"None of {args.clusters} found in GMRT matches.")
            print(f"Available: {list(matched['name'])}")
            return
    elif args.all:
        targets = matched
    else:
        # Default: pick 3 representative targets (bright, medium, faint)
        indices = [0, len(matched) // 2, len(matched) - 1]
        indices = sorted(set(i for i in indices if i < len(matched)))
        targets = matched.iloc[indices]
        print("Selected representative targets (use --all for all):")

    print(f"Processing {len(targets)} clusters\n")

    # Show target list
    for _, row in targets.iterrows():
        flux_str = (f"{row['closest_flux_mJy']:.1f} mJy"
                    if pd.notna(row['closest_flux_mJy']) else "N/A")
        print(f"  {row['name']:25s}  {row['n_gmrt_sources']:2.0f} sources, "
              f"flux: {flux_str}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    success = 0
    failed = 0

    for i, (_, row) in enumerate(targets.iterrows(), 1):
        name = row['name']
        safe_name = name.replace(' ', '_')
        fits_path = os.path.join(args.output_dir, f"gmrt_{safe_name}.fits")
        png_path = os.path.join(args.output_dir, f"gmrt_{safe_name}.png")

        print(f"[{i}/{len(targets)}] {name}")

        if args.skip_existing and os.path.exists(fits_path) and os.path.exists(png_path):
            print(f"  Skipping (already exists)")
            success += 1
            continue

        if args.png_only:
            if os.path.exists(fits_path):
                hdu = fits.open(fits_path)
                display_gmrt_image(hdu, name=name, output_dir=args.output_dir)
                success += 1
            else:
                print(f"  No FITS file found: {fits_path}")
                failed += 1
            continue

        hdu = download_gmrt_cutout(
            ra=row['ra'],
            dec=row['dec'],
            size=args.size,
            name=name,
            output_dir=args.output_dir,
            pixels=args.pixels,
        )

        if hdu is not None:
            display_gmrt_image(hdu, name=name, output_dir=args.output_dir)
            success += 1
        else:
            failed += 1

        print()

    print("=" * 80)
    print(f"Done: {success} succeeded, {failed} failed")
    print("=" * 80)


if __name__ == "__main__":
    main()

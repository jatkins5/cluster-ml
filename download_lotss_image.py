#!/usr/bin/env python3
"""
Download and display LoTSS DR3 (144 MHz) image cutouts for LoVoCCS targets.

This script downloads radio images from the LoTSS DR3 cutout service
for galaxy clusters that have LoTSS matches.

Cutout API:
  6" resolution:  https://lofar-surveys.org/dr3-cutout.fits?pos=RA+DEC&size=SIZE_ARCMIN
  20" resolution: https://lofar-surveys.org/dr3-low-cutout.fits?pos=RA+DEC&size=SIZE_ARCMIN

Requires: lovoccs_lotss_matches.csv (from match_lovoccs_lotss.py)
"""

import os
import argparse
import time

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore')

LOTSS_CUTOUT_URL = "https://lofar-surveys.org/dr3-cutout.fits"
LOTSS_LOW_CUTOUT_URL = "https://lofar-surveys.org/dr3-low-cutout.fits"


def download_lotss_cutout(ra, dec, size_arcmin=20, name="target",
                          output_dir="lotss_images", low_res=False):
    """
    Download a LoTSS DR3 144 MHz image cutout.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    size_arcmin : float
        Cutout size in arcminutes (default 20)
    name : str
        Target name for the filename
    output_dir : str
        Output directory
    low_res : bool
        If True, use 20" resolution endpoint instead of 6"

    Returns
    -------
    astropy.io.fits.HDUList or None
    """
    base_url = LOTSS_LOW_CUTOUT_URL if low_res else LOTSS_CUTOUT_URL
    res_label = '20"' if low_res else '6"'
    url = f"{base_url}?pos={ra}+{dec}&size={size_arcmin}"

    print(f"Downloading LoTSS DR3 cutout for {name} ({res_label} resolution)")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print(f"  Size: {size_arcmin:.1f} arcmin")

    safe_name = name.replace(' ', '_')
    os.makedirs(output_dir, exist_ok=True)
    fits_path = os.path.join(output_dir, f"lotss_{safe_name}.fits")

    try:
        start_time = time.time()
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()

        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0

        with open(fits_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        elapsed = time.time() - start_time
                        rate = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                        print(f"\r  Progress: {pct:.1f}% ({downloaded/(1024*1024):.1f} / "
                              f"{total_size/(1024*1024):.1f} MB, {rate:.1f} MB/s)",
                              end="", flush=True)
                    else:
                        print(f"\r  Downloaded: {downloaded/(1024*1024):.1f} MB",
                              end="", flush=True)

        elapsed = time.time() - start_time
        print(f"\n  Download complete ({elapsed:.1f}s, "
              f"{downloaded/(1024*1024):.1f} MB)")

        hdu = fits.open(fits_path)
        data = np.squeeze(hdu[0].data)
        if data is None:
            print("  ERROR: No data in FITS file")
            return None

        valid = data[np.isfinite(data)]
        if len(valid) > 0:
            print(f"  Max flux: {np.max(valid):.4e} Jy/beam, "
                  f"Mean: {np.mean(valid):.4e} Jy/beam")

        print(f"  Saved: {fits_path}")
        return hdu

    except Exception as e:
        print(f"\n  Error downloading: {e}")
        if os.path.exists(fits_path):
            os.remove(fits_path)
        return None


def display_lotss_image(hdu, name="target", output_dir="lotss_images"):
    """
    Create a PNG visualization of a LoTSS image.

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

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='lightgray')

    im = ax.imshow(data, origin='lower', cmap=cmap, norm=norm,
                   interpolation='nearest')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux Density (Jy/beam)', fontsize=12)

    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)
    ax.set_title(f'LoTSS 144 MHz (DR3): {name}', fontsize=14, fontweight='bold')
    ax.grid(color='white', ls='--', alpha=0.3)

    valid = data[np.isfinite(data)]
    if len(valid) > 0:
        stats_text = (f'Max: {np.max(valid):.2e} Jy/beam\n'
                      f'Mean: {np.mean(valid):.2e} Jy/beam')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    safe_name = name.replace(' ', '_')
    output_file = os.path.join(output_dir, f"lotss_{safe_name}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Download LoTSS DR3 144 MHz images for LoVoCCS clusters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Download a few representative targets
  %(prog)s --all                     # Download all clusters with LoTSS matches
  %(prog)s --clusters A401 A399      # Download specific clusters
  %(prog)s --png-only                # Regenerate PNGs from existing FITS
  %(prog)s --size 30                 # Use larger cutout (30 arcmin)
  %(prog)s --low-res                 # Use 20" resolution instead of 6"
        """
    )
    parser.add_argument('--all', action='store_true',
                        help='Process all clusters with LoTSS matches')
    parser.add_argument('--clusters', nargs='+', default=None,
                        help='Specific cluster names to process')
    parser.add_argument('--output-dir', default='lotss_images',
                        help='Output directory (default: lotss_images)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip clusters that already have FITS+PNG')
    parser.add_argument('--png-only', action='store_true',
                        help='Only generate PNGs from existing FITS files')
    parser.add_argument('--size', type=float, default=20,
                        help='Cutout size in arcminutes (default: 20)')
    parser.add_argument('--force', action='store_true',
                        help='Re-download even if FITS file exists')
    parser.add_argument('--low-res', action='store_true',
                        help='Use 20" resolution endpoint instead of 6"')

    args = parser.parse_args()

    # Load LoTSS match results
    match_file = "lovoccs_lotss_matches.csv"
    try:
        results = pd.read_csv(match_file)
    except FileNotFoundError:
        print(f"Error: {match_file} not found!")
        print("Please run match_lovoccs_lotss.py first.")
        return

    matched = results[results['has_lotss_match'] == True].copy()
    matched = matched.sort_values('closest_total_flux_mJy', ascending=False)

    if len(matched) == 0:
        print("No clusters with LoTSS matches found!")
        return

    print("=" * 80)
    print("LoVoCCS - LoTSS DR3 144 MHz Image Download")
    print("=" * 80)
    res_label = '20"' if args.low_res else '6"'
    print(f"Clusters with LoTSS matches: {len(matched)}")
    print(f"Cutout size: {args.size:.1f} arcmin, resolution: {res_label}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Select targets
    if args.clusters:
        targets = matched[matched['name'].isin(args.clusters)]
        if len(targets) == 0:
            print(f"None of {args.clusters} found in LoTSS matches.")
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
        flux_str = (f"{row['closest_total_flux_mJy']:.1f} mJy"
                    if pd.notna(row.get('closest_total_flux_mJy')) else "N/A")
        sep_str = (f"{row['closest_sep_arcmin']:.1f}'"
                   if pd.notna(row.get('closest_sep_arcmin')) else "N/A")
        print(f"  {row['name']:25s}  {row['n_lotss_sources']:3.0f} sources, "
              f"flux: {flux_str}, sep: {sep_str}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    success = 0
    failed = 0

    for i, (_, row) in enumerate(targets.iterrows(), 1):
        name = row['name']
        safe_name = name.replace(' ', '_')
        fits_path = os.path.join(args.output_dir, f"lotss_{safe_name}.fits")
        png_path = os.path.join(args.output_dir, f"lotss_{safe_name}.png")

        print(f"[{i}/{len(targets)}] {name}")

        if args.skip_existing and os.path.exists(fits_path) and os.path.exists(png_path):
            print(f"  Skipping (already exists)")
            success += 1
            continue

        if args.png_only:
            if os.path.exists(fits_path):
                hdu = fits.open(fits_path)
                display_lotss_image(hdu, name=name, output_dir=args.output_dir)
                success += 1
            else:
                print(f"  No FITS file found: {fits_path}")
                failed += 1
            continue

        # Skip download if FITS exists and --force not specified
        if os.path.exists(fits_path) and not args.force:
            print(f"  FITS already exists: {fits_path}")
            print(f"  Use --force to re-download")
            if not os.path.exists(png_path):
                try:
                    hdu = fits.open(fits_path)
                    display_lotss_image(hdu, name=name, output_dir=args.output_dir)
                except Exception as e:
                    print(f"  Error generating PNG: {e}")
            success += 1
            continue

        hdu = download_lotss_cutout(
            ra=row['ra'],
            dec=row['dec'],
            size_arcmin=args.size,
            name=name,
            output_dir=args.output_dir,
            low_res=args.low_res,
        )

        if hdu is not None:
            display_lotss_image(hdu, name=name, output_dir=args.output_dir)
            success += 1
        else:
            failed += 1

        print()

    print("=" * 80)
    print(f"Done: {success} succeeded, {failed} failed")
    print("=" * 80)


if __name__ == "__main__":
    main()

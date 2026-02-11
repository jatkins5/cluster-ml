#!/usr/bin/env python3
"""
Download optical cutouts from the DESI Legacy Imaging Surveys for LoVoCCS targets.
"""

import argparse
import os
import requests
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS


def load_target_coordinates(catalog_path="vlass_coverage_results.csv"):
    """
    Load target coordinates from the catalog file.

    Parameters:
    -----------
    catalog_path : str
        Path to catalog CSV file

    Returns:
    --------
    dict : {name: (ra, dec)} mapping
    """
    df = pd.read_csv(catalog_path)
    coords = {}
    for _, row in df.iterrows():
        coords[row['name']] = (row['ra'], row['dec'])
    return coords


def download_legacy_cutout(ra, dec, size_arcmin=8.0, name="target", output_dir=".",
                           layer="ls-dr10", pixscale=1.0):
    """
    Download a Legacy Survey optical cutout.

    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    size_arcmin : float
        Image size in arcminutes (default 8.0, max ~8.5 for 512 pixels at 1"/pix)
    name : str
        Target name for the filename
    output_dir : str
        Directory to save output files
    layer : str
        Legacy Survey layer (default: ls-dr10)
    pixscale : float
        Pixel scale in arcsec/pixel (default: 1.0 to match VLASS)

    Returns:
    --------
    str : Path to downloaded FITS file, or None if failed
    """
    # Calculate size in pixels (max 512)
    size_pix = int(size_arcmin * 60 / pixscale)
    if size_pix > 512:
        print(f"  Warning: Requested size {size_arcmin}' exceeds max (512 pix at {pixscale}\"/pix)")
        print(f"  Adjusting to 512 pixels ({512 * pixscale / 60:.1f}')")
        size_pix = 512

    # Build URL
    url = (
        f"https://www.legacysurvey.org/viewer/fits-cutout"
        f"?ra={ra}&dec={dec}&pixscale={pixscale}&layer={layer}"
        f"&size={size_pix}&bands=grz"
    )

    print(f"Downloading Legacy Survey cutout for {name}")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print(f"  Size: {size_pix} pixels ({size_pix * pixscale / 60:.1f}')")
    print(f"  Layer: {layer}")

    try:
        response = requests.get(url, timeout=60)

        if response.status_code == 404:
            print(f"  No Legacy Survey coverage at this position")
            return None
        elif response.status_code != 200:
            print(f"  HTTP error: {response.status_code}")
            return None

        # Save the FITS file
        filename = os.path.join(output_dir, f"legacy_{name.replace(' ', '_')}.fits")
        with open(filename, 'wb') as f:
            f.write(response.content)

        # Verify the file is valid FITS
        try:
            with fits.open(filename) as hdu:
                data = hdu[0].data
                if data is None:
                    print(f"  Error: Empty FITS file")
                    os.remove(filename)
                    return None
                print(f"  Downloaded: {filename}")
                print(f"  Shape: {data.shape} (bands: g, r, z)")
                return filename
        except Exception as e:
            print(f"  Error: Invalid FITS file - {e}")
            if os.path.exists(filename):
                os.remove(filename)
            return None

    except requests.exceptions.Timeout:
        print(f"  Error: Request timed out")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Download Legacy Survey optical cutouts for LoVoCCS targets'
    )
    parser.add_argument('--all', action='store_true',
                        help='Download all targets with VLASS images')
    parser.add_argument('--clusters', nargs='+', default=None,
                        help='Specific cluster names to download (e.g., A780 A1644)')
    parser.add_argument('--output-dir', default='legacy_images',
                        help='Output directory for images (default: legacy_images)')
    parser.add_argument('--size', type=float, default=8.0,
                        help='Image size in arcminutes (default: 8.0, max ~8.5)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip targets that already have FITS files')
    parser.add_argument('--layer', default='ls-dr10',
                        help='Legacy Survey layer (default: ls-dr10)')
    parser.add_argument('--pixscale', type=float, default=1.0,
                        help='Pixel scale in arcsec/pixel (default: 1.0)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load target coordinates from catalog
    target_coords = load_target_coordinates()
    print(f"Loaded coordinates for {len(target_coords)} targets from catalog")

    # Find all VLASS FITS files
    vlass_dir = "vlass_images"
    vlass_files = sorted([f for f in os.listdir(vlass_dir) if f.endswith('.fits')])

    print(f"Found {len(vlass_files)} VLASS FITS files")
    print()

    # Extract cluster names from filenames
    clusters = {}
    for f in vlass_files:
        # vlass_A780.fits -> A780
        name = f.replace('vlass_', '').replace('.fits', '')
        if name in target_coords:
            clusters[name] = target_coords[name]  # Store (ra, dec) tuple
        else:
            print(f"Warning: {name} not found in catalog, skipping")

    # Determine which targets to process
    if args.clusters:
        targets = {k: v for k, v in clusters.items() if k in args.clusters}
        if len(targets) == 0:
            print(f"Error: None of the specified clusters found")
            print(f"Available clusters: {', '.join(list(clusters.keys())[:20])}...")
            return
        print(f"Processing {len(targets)} specified cluster(s)")
    elif args.all:
        targets = clusters
        print(f"Processing all {len(targets)} targets")
    else:
        # Default: sample of 3
        names = list(clusters.keys())
        sample = [names[0], names[len(names)//2], names[-1]]
        targets = {k: clusters[k] for k in sample}
        print("Processing sample of 3 targets (use --all for all targets)")

    print()

    # Track results
    successful = []
    failed = []
    skipped = []

    for i, (name, coords) in enumerate(targets.items(), 1):
        ra, dec = coords
        output_file = os.path.join(args.output_dir, f"legacy_{name}.fits")

        print(f"\n[{i}/{len(targets)}] {name}")
        print("=" * 70)

        # Skip if already exists
        if args.skip_existing and os.path.exists(output_file):
            print(f"  Skipping - already exists: {output_file}")
            skipped.append(name)
            continue

        # Download optical cutout using catalog coordinates
        result = download_legacy_cutout(
            ra=ra,
            dec=dec,
            size_arcmin=args.size,
            name=name,
            output_dir=args.output_dir,
            layer=args.layer,
            pixscale=args.pixscale
        )

        if result:
            successful.append(name)
        else:
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

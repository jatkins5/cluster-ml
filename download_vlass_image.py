#!/usr/bin/env python3
"""
Download and display VLASS image cutouts for LoVoCCS targets.
"""

import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch
from astroquery.cadc import Cadc
import numpy as np


def validate_vlass_fits(fits_path, target_ra, target_dec):
    """
    Check if a FITS file actually contains the target position.

    Parameters:
    -----------
    fits_path : str
        Path to the FITS file
    target_ra : float
        Target Right Ascension in degrees
    target_dec : float
        Target Declination in degrees

    Returns:
    --------
    bool
        True if target is within image bounds, False otherwise
    """
    try:
        with fits.open(fits_path) as hdu:
            wcs_full = WCS(hdu[0].header)
            wcs_2d = wcs_full.celestial if wcs_full.naxis > 2 else wcs_full
            data = np.squeeze(hdu[0].data)
            target_pix = wcs_2d.world_to_pixel_values(target_ra, target_dec)
            ny, nx = data.shape
            return (0 <= target_pix[0] < nx) and (0 <= target_pix[1] < ny)
    except Exception as e:
        print(f"  Error validating {fits_path}: {e}")
        return False


def download_vlass_cutout(ra, dec, size=0.5, name="target", output_dir="."):
    """
    Download a VLASS image cutout using CADC.

    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    size : float
        Image size in degrees (default 0.5 deg = 30 arcmin)
    name : str
        Target name for the filename
    output_dir : str
        Directory to save output files

    Returns:
    --------
    HDU object with the image data
    """
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    print(f"Downloading VLASS cutout for {name}")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print(f"  Size: {size:.2f} degrees ({size*60:.1f} arcmin)")
    print()

    try:
        cadc = Cadc()

        # Query for VLASS images at this position
        print("Querying CADC for VLASS data...")
        result = cadc.query_region(
            coord,
            collection='VLASS',
            radius=size*u.degree
        )

        if result is None or len(result) == 0:
            print("No VLASS data found at this position")
            return None

        print(f"Found {len(result)} VLASS observations")
        print(f"Available columns: {result.colnames[:10]}...")

        # Get the first observation
        obs = result[0]
        print(f"\nUsing observation: {obs.get('observationID', 'N/A')}")

        # Get images
        print("Downloading images...")
        cutout_hdu_list = cadc.get_images(
            coord,
            radius=size*u.degree,
            collection='VLASS'
        )

        if cutout_hdu_list and len(cutout_hdu_list) > 0:
            print(f"Downloaded {len(cutout_hdu_list)} image(s)")

            # Find the image that actually contains our target
            best_hdu = None
            best_flux = -1

            for i, hdu in enumerate(cutout_hdu_list):
                data = np.squeeze(hdu[0].data)
                wcs_full = WCS(hdu[0].header)
                wcs_2d = wcs_full.celestial if wcs_full.naxis > 2 else wcs_full

                # Check if target is in this image
                target_pix = wcs_2d.world_to_pixel_values(ra, dec)
                ny, nx = data.shape
                in_image = (0 <= target_pix[0] < nx) and (0 <= target_pix[1] < ny)

                if in_image:
                    max_flux = np.nanmax(data)
                    obs_id = hdu[0].header.get('OBJECT', f'image_{i}')
                    print(f"  Image {i+1} ({obs_id}): contains target, max flux = {max_flux:.6f} Jy/beam")

                    # Keep the image with highest max flux
                    if max_flux > best_flux:
                        best_flux = max_flux
                        best_hdu = hdu

            if best_hdu is not None:
                # Save the FITS file
                filename = os.path.join(output_dir, f"vlass_{name.replace(' ', '_')}.fits")
                best_hdu.writeto(filename, overwrite=True)
                print(f"\nSaved FITS file: {filename} (max flux: {best_flux:.6f} Jy/beam)")
                return best_hdu
            else:
                print("ERROR: No image contains the target position - skipping")
                return None
        else:
            print("No image data returned")
            return None

    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def display_vlass_image(hdu, name="target", output_dir="."):
    """
    Display a VLASS image with proper WCS coordinates.

    Parameters:
    -----------
    hdu : HDU object
        FITS HDU containing the image data
    name : str
        Target name for the title
    output_dir : str
        Directory to save output files
    """
    # Extract data and WCS
    data = hdu[0].data
    header = hdu[0].header

    # Remove extra dimensions (e.g., frequency and Stokes)
    data = np.squeeze(data)  # Remove extra dimensions

    # Create a 2D WCS from the full WCS
    # VLASS images often have 4 dimensions (RA, Dec, Frequency, Stokes)
    # We need to slice to just the spatial dimensions
    full_wcs = WCS(header)
    if full_wcs.naxis > 2:
        # Drop the extra axes (typically frequency and Stokes)
        wcs = full_wcs.celestial
    else:
        wcs = full_wcs

    # Create figure with WCS projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=wcs)

    # Normalize the image with asinh stretch for better dynamic range
    norm = ImageNormalize(data, stretch=AsinhStretch())

    # Display the image
    im = ax.imshow(data, origin='lower', cmap='viridis', norm=norm)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux (Jy/beam)', fontsize=12)

    # Labels
    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)
    ax.set_title(f'VLASS Image: {name}', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(color='white', ls='--', alpha=0.3)

    # Save figure
    output_file = os.path.join(output_dir, f"vlass_{name.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved image: {output_file}")

    plt.close(fig)  # Close figure to free memory

def main():
    parser = argparse.ArgumentParser(
        description='Download and visualize VLASS image cutouts for LoVoCCS targets'
    )
    parser.add_argument('--all', action='store_true',
                       help='Download all targets with VLASS coverage')
    parser.add_argument('--clusters', nargs='+', default=None,
                       help='Specific cluster names to download (e.g., A780 A1644)')
    parser.add_argument('--output-dir', default='vlass_images',
                       help='Output directory for images (default: vlass_images)')
    parser.add_argument('--size', type=float, default=0.3,
                       help='Image size in degrees (default: 0.3 = 18 arcmin)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip targets that already have PNG files')
    parser.add_argument('--png-only', action='store_true',
                       help='Only generate PNG from existing FITS (no download)')
    parser.add_argument('--validate', action='store_true',
                       help='Check existing FITS files for target coverage (no download)')
    parser.add_argument('--force', action='store_true',
                       help='Re-download even if FITS file exists (for fixing bad files)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read the results file to find targets with VLASS coverage
    results = pd.read_csv("vlass_coverage_results.csv")

    # Filter for targets in VLASS
    in_vlass = results[results['in_vlass'] == True].sort_values('n_sources', ascending=False)

    print("LoVoCCS targets in VLASS (sorted by number of catalog sources):")
    print("=" * 70)
    for idx, row in in_vlass.head(10).iterrows():
        print(f"  {row['name']:30s} - {row['n_sources']:2d} catalog sources")
    print(f"  ... and {len(in_vlass) - 10} more")
    print(f"\nTotal: {len(in_vlass)} targets with VLASS coverage")
    print()

    # Determine which targets to process
    if args.clusters:
        # Specific clusters requested
        targets = in_vlass[in_vlass['name'].isin(args.clusters)]
        if len(targets) == 0:
            print(f"Error: None of the specified clusters found in VLASS matches")
            print(f"Available clusters: {', '.join(in_vlass['name'].tolist()[:20])}...")
            return
        print(f"Processing {len(targets)} specified cluster(s)")
    elif args.all:
        # All targets
        targets = in_vlass
        print(f"Processing all {len(targets)} targets with VLASS coverage")
    else:
        # Default: sample of 3 (high, medium, low)
        target_high = in_vlass.iloc[0]
        target_med = in_vlass.iloc[len(in_vlass)//2]
        low_sources = in_vlass[in_vlass['n_sources'] <= 3]
        if len(low_sources) > 0:
            target_low = low_sources.iloc[len(low_sources)//2]
        else:
            target_low = in_vlass.iloc[-3]
        targets = pd.DataFrame([target_high, target_med, target_low])
        print("Processing sample of 3 targets (use --all for all targets)")

    print()

    # Track results
    successful = []
    failed = []
    invalid = []  # For validation mode

    # Validation mode: check existing FITS files
    if args.validate:
        print("VALIDATION MODE: Checking existing FITS files for target coverage")
        print("=" * 70)
        for i, (idx, target) in enumerate(targets.iterrows(), 1):
            name = target['name']
            fits_file = os.path.join(args.output_dir, f"vlass_{name.replace(' ', '_')}.fits")

            if not os.path.exists(fits_file):
                print(f"  [{i}/{len(targets)}] {name}: NO FITS FILE")
                failed.append(name)
                continue

            is_valid = validate_vlass_fits(fits_file, target['ra'], target['dec'])
            if is_valid:
                print(f"  [{i}/{len(targets)}] {name}: VALID (target in image)")
                successful.append(name)
            else:
                print(f"  [{i}/{len(targets)}] {name}: INVALID (target NOT in image)")
                invalid.append(name)

        # Validation summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Valid files: {len(successful)}")
        print(f"Invalid files (target not covered): {len(invalid)}")
        print(f"Missing files: {len(failed)}")
        if invalid:
            print(f"\nInvalid files that need re-download:")
            for name in invalid:
                print(f"  - {name}")
            print(f"\nTo fix, run: python download_vlass_image.py --clusters {' '.join(invalid)} --force")
        return

    for i, (idx, target) in enumerate(targets.iterrows(), 1):
        name = target['name']
        png_file = os.path.join(args.output_dir, f"vlass_{name.replace(' ', '_')}.png")
        fits_file = os.path.join(args.output_dir, f"vlass_{name.replace(' ', '_')}.fits")

        print(f"\n[{i}/{len(targets)}] {name} ({target['n_sources']} catalog sources)")
        print("=" * 70)

        # Skip if already exists (unless --force is specified)
        if args.skip_existing and os.path.exists(png_file) and not args.force:
            print(f"  Skipping - PNG already exists: {png_file}")
            successful.append(name)
            continue

        # PNG-only mode: just generate from existing FITS
        if args.png_only:
            if os.path.exists(fits_file):
                print(f"  Generating PNG from existing FITS: {fits_file}")
                try:
                    hdu = fits.open(fits_file)
                    display_vlass_image(hdu, name=name, output_dir=args.output_dir)
                    successful.append(name)
                except Exception as e:
                    print(f"  Error generating PNG: {e}")
                    failed.append(name)
            else:
                print(f"  No FITS file found: {fits_file}")
                failed.append(name)
            continue

        # Skip if FITS exists and --force not specified
        if os.path.exists(fits_file) and not args.force:
            print(f"  FITS already exists: {fits_file}")
            print(f"  Use --force to re-download")
            # Still generate PNG if needed
            if not os.path.exists(png_file):
                try:
                    hdu = fits.open(fits_file)
                    display_vlass_image(hdu, name=name, output_dir=args.output_dir)
                except Exception as e:
                    print(f"  Error generating PNG: {e}")
            successful.append(name)
            continue

        # Download and display
        try:
            hdu = download_vlass_cutout(
                ra=target['ra'],
                dec=target['dec'],
                size=args.size,
                name=name,
                output_dir=args.output_dir
            )

            if hdu:
                display_vlass_image(hdu, name=name, output_dir=args.output_dir)
                successful.append(name)
            else:
                failed.append(name)
        except Exception as e:
            print(f"  Error: {e}")
            failed.append(name)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(targets)}")
    print(f"Failed: {len(failed)}/{len(targets)}")
    if failed:
        print(f"\nFailed targets: {', '.join(failed)}")
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()

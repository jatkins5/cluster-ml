#!/usr/bin/env python3
"""
Download and visualize MeerKAT MGCLS radio images for LoVoCCS targets.

The MeerKAT Galaxy Cluster Legacy Survey (MGCLS) DR1 (Knowles et al. 2022)
observed 115 galaxy clusters with MeerKAT L-band (~1.28 GHz, ~8" resolution).
18 of these overlap with LoVoCCS targets.

MGCLS DR1 data is publicly hosted on an S3-compatible object store at:
  https://archive-gw-1.kat.ac.za/public/repository/10.48479/7epd-w356/data/

Basic products are 16-plane FITS cubes:
  Plane 0: Stokes I continuum at ~1283 MHz reference frequency
  Plane 1: Spectral index
  Planes 2-15: 14 frequency channel images

Single-plane ("1pln") continuum-only images are also available (~100 MB
vs ~2 GB for full cubes).
"""

import argparse
import gzip
import os
import re
import shutil
import time
import xml.etree.ElementTree as ET

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch
import numpy as np
import requests


# MGCLS DR1 S3 archive
MGCLS_S3_HOST = "https://archive-gw-1.kat.ac.za"
MGCLS_BUCKET = "public"
MGCLS_DATA_PREFIX = "repository/10.48479/7epd-w356/data"
MGCLS_BASIC_PREFIX = f"{MGCLS_DATA_PREFIX}/basic_products"
MGCLS_ENHANCED_PREFIX = f"{MGCLS_DATA_PREFIX}/enhanced_products"


def list_mgcls_files(prefix, name_filter=None):
    """
    List files in the MGCLS S3 bucket under a given prefix.

    Uses the S3 ListObjectsV2 API on the public SARAO archive.

    Args:
        prefix: S3 key prefix to list under
        name_filter: Optional string to filter filenames (case-sensitive)

    Returns:
        list of dict with keys: key, size, filename
    """
    url = f"{MGCLS_S3_HOST}/{MGCLS_BUCKET}/"
    params = {
        'list-type': '2',
        'prefix': prefix + '/',
        'max-keys': '1000',
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Error listing bucket: {e}")
        return []

    # Parse S3 XML response
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        print(f"  Error parsing S3 response")
        return []

    files = []
    for contents in root.findall('s3:Contents', ns):
        key = contents.find('s3:Key', ns).text
        size = int(contents.find('s3:Size', ns).text)
        filename = key.split('/')[-1]
        if name_filter and name_filter not in filename:
            continue
        files.append({'key': key, 'size': size, 'filename': filename})

    return files


def discover_mgcls_files(mgcls_name, product_type="basic"):
    """
    Find the download URL for an MGCLS cluster image.

    Uses S3 bucket listing to discover files matching the cluster name.
    Prefers single-plane continuum images (1pln, ~100 MB) over full
    16-plane cubes (~2 GB) for faster downloads.

    MGCLS filenames are inconsistent:
      - Abell clusters: Abell_{N}.fits.gz or Abell_{N}.fits
      - J-name clusters: {name}.Fix.fits.gz or {name}.fits.gz
      - Single-plane: {name}_1pln.fits.gz or {name}.1pln.fits.gz

    Args:
        mgcls_name: MGCLS cluster name (e.g., "Abell 85", "J0431.4-6126")
        product_type: "basic" or "enhanced"

    Returns:
        tuple (url, is_compressed, is_cube) or (None, None, None)
    """
    prefix = MGCLS_BASIC_PREFIX if product_type == "basic" else MGCLS_ENHANCED_PREFIX
    safe_name = mgcls_name.replace(" ", "_")

    # List all files with this cluster name prefix
    files = list_mgcls_files(prefix, name_filter=safe_name)

    if not files:
        print(f"  No files found matching '{safe_name}' in {product_type} products")
        return None, None, None

    # Filter to FITS files that match this exact cluster name
    # (avoid e.g. "Abell_13" matching "Abell_133")
    fits_files = []
    for f in files:
        fname = f['filename']
        if '.fits' not in fname:
            continue
        # Check that the cluster name is followed by a separator, not more digits
        # e.g., "Abell_13.fits" or "Abell_13_1pln.fits" but not "Abell_133.fits"
        rest = fname[len(safe_name):]
        if rest and rest[0].isalnum() and rest[0] != 'B':
            # Starts with alphanumeric (except 'B' for observation variants like Abell_2811B)
            # — this is a different cluster name
            continue
        fits_files.append(f)

    if not fits_files:
        print(f"  No FITS files found for '{safe_name}'")
        return None, None, None

    # Categorize files
    single_plane = []  # 1pln files (preferred - much smaller)
    stokes_i_cubes = []  # Full Stokes I cubes
    other_fits = []

    for f in fits_files:
        fname = f['filename']
        # Skip polarization products (Q, U, V) and other non-Stokes-I
        if any(x in fname for x in ['QPoln', 'UPoln', 'VPoln', 'QPol.', 'UPol.', 'VPol.',
                                      '_Q.', '_U.', '_V.', 'IPoln', 'IPol.',
                                      'SNTab', '.Peel.']):
            continue

        if '1pln' in fname:
            single_plane.append(f)
        elif fname.startswith(safe_name):
            stokes_i_cubes.append(f)
        else:
            other_fits.append(f)

    # Prefer single-plane continuum (much smaller download)
    chosen = None
    is_cube = False
    if single_plane:
        chosen = single_plane[0]
        is_cube = False
    elif stokes_i_cubes:
        chosen = stokes_i_cubes[0]
        is_cube = True
    elif other_fits:
        chosen = other_fits[0]
        is_cube = True

    if chosen is None:
        print(f"  No suitable Stokes I file found for '{safe_name}'")
        return None, None, None

    url = f"{MGCLS_S3_HOST}/{MGCLS_BUCKET}/{chosen['key']}"
    is_compressed = chosen['filename'].endswith('.gz')
    size_mb = chosen['size'] / (1024 * 1024)
    print(f"  Found: {chosen['filename']} ({size_mb:.1f} MB)"
          f" [{'single-plane' if not is_cube else 'cube'},"
          f" {'gzipped' if is_compressed else 'uncompressed'}]")

    return url, is_compressed, is_cube


def download_mgcls_image(mgcls_name, ra, dec, lovoccs_name, output_dir=".",
                         product_type="basic"):
    """
    Download an MGCLS FITS image.

    Downloads the single-plane continuum image when available (preferred,
    ~100 MB). Falls back to the full 16-plane cube (~2 GB) and extracts
    plane 0 (Stokes I continuum).

    Handles both .fits and .fits.gz (gzipped) files.

    Args:
        mgcls_name: MGCLS cluster name
        ra: Target RA in degrees
        dec: Target Dec in degrees
        lovoccs_name: LoVoCCS cluster name (for output filename)
        output_dir: Output directory
        product_type: "basic" or "enhanced"

    Returns:
        astropy.io.fits.HDUList or None
    """
    safe_name = lovoccs_name.replace(" ", "_")
    fits_path = os.path.join(output_dir, f"meerkat_{safe_name}.fits")

    print(f"Downloading MGCLS {product_type} image for {lovoccs_name}")
    print(f"  MGCLS name: {mgcls_name}")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print()

    # Discover download URL
    print("  Searching for file in MGCLS archive...")
    url, is_compressed, is_cube = discover_mgcls_files(mgcls_name, product_type)

    if url is None:
        print(f"  ERROR: Could not find MGCLS file for {mgcls_name}")
        return None

    # Determine temp download path
    if is_compressed:
        dl_path = fits_path + '.gz'
    elif is_cube:
        dl_path = fits_path + '.cube'
    else:
        dl_path = fits_path

    # Stream download with progress
    try:
        print("  Downloading...")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()

        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()

        with open(dl_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        elapsed = time.time() - start_time
                        rate = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                        print(f"\r  Progress: {pct:.1f}% ({downloaded/(1024*1024):.1f} / "
                              f"{total_size/(1024*1024):.1f} MB, {rate:.1f} MB/s)", end="",
                              flush=True)
                    else:
                        print(f"\r  Downloaded: {downloaded/(1024*1024):.1f} MB", end="",
                              flush=True)

        elapsed = time.time() - start_time
        print(f"\n  Download complete ({elapsed:.1f}s)")

    except Exception as e:
        print(f"\n  ERROR downloading: {e}")
        if os.path.exists(dl_path):
            os.remove(dl_path)
        return None

    # Decompress if gzipped
    if is_compressed:
        print("  Decompressing .fits.gz...")
        decomp_path = dl_path[:-3]  # Remove .gz
        try:
            with gzip.open(dl_path, 'rb') as f_in, open(decomp_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(dl_path)
            dl_path = decomp_path
            print(f"  Decompressed to {os.path.getsize(dl_path)/(1024*1024):.1f} MB")
        except Exception as e:
            print(f"  ERROR decompressing: {e}")
            if os.path.exists(dl_path):
                os.remove(dl_path)
            return None

    # Process FITS: extract continuum plane if needed
    try:
        with fits.open(dl_path) as hdul:
            header = hdul[0].header.copy()
            data = hdul[0].data

            if data is None:
                print("  ERROR: No data in FITS file")
                return None

            ndim = data.ndim
            print(f"  Data shape: {data.shape} ({ndim}D)")

            if ndim == 2:
                # Already single-plane — just use as-is
                continuum = data
            elif ndim >= 3:
                # Extract plane 0 (Stokes I continuum)
                print("  Extracting Stokes I continuum (plane 0)...")
                if ndim == 4:
                    continuum = data[0, 0, :, :]
                else:
                    continuum = data[0, :, :]

                # Update header to 2D
                header['NAXIS'] = 2
                for key in ['NAXIS3', 'NAXIS4', 'CRPIX3', 'CRPIX4',
                            'CDELT3', 'CDELT4', 'CRVAL3', 'CRVAL4',
                            'CTYPE3', 'CTYPE4', 'CUNIT3', 'CUNIT4',
                            'PC3_1', 'PC3_2', 'PC3_3', 'PC3_4',
                            'PC4_1', 'PC4_2', 'PC4_3', 'PC4_4',
                            'PC1_3', 'PC1_4', 'PC2_3', 'PC2_4']:
                    if key in header:
                        del header[key]
                header['HISTORY'] = 'Extracted plane 0 (Stokes I continuum) from MGCLS cube'
            else:
                print(f"  ERROR: Unexpected data shape: {data.shape}")
                return None

        # Write final 2D continuum FITS
        hdu = fits.PrimaryHDU(data=continuum, header=header)
        hdu_list = fits.HDUList([hdu])
        hdu_list.writeto(fits_path, overwrite=True)

        # Clean up temp file if different from output
        if dl_path != fits_path and os.path.exists(dl_path):
            os.remove(dl_path)

        print(f"  Saved: {fits_path}")
        print(f"  Image size: {continuum.shape[1]} x {continuum.shape[0]} pixels")
        valid_data = continuum[np.isfinite(continuum)]
        if len(valid_data) > 0:
            print(f"  Max flux: {np.nanmax(valid_data):.6f} Jy/beam")

        return hdu_list

    except Exception as e:
        print(f"  ERROR processing FITS: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temp files
        for p in [dl_path, fits_path]:
            if p != fits_path and os.path.exists(p):
                os.remove(p)
        return None


def validate_meerkat_fits(fits_path, target_ra, target_dec):
    """
    Check if a FITS file actually contains the target position.

    Args:
        fits_path: Path to the FITS file
        target_ra: Target Right Ascension in degrees
        target_dec: Target Declination in degrees

    Returns:
        bool: True if target is within image bounds
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


def display_meerkat_image(hdu, name="target", output_dir="."):
    """
    Create a PNG visualization of a MeerKAT image.

    Uses AsinhStretch normalization, viridis colormap, WCS projection
    with RA/Dec axes and white dashed grid, matching VLASS display style.

    Args:
        hdu: FITS HDUList containing the image data
        name: Target name for the title
        output_dir: Directory to save output PNG
    """
    data = hdu[0].data
    header = hdu[0].header

    data = np.squeeze(data)

    full_wcs = WCS(header)
    if full_wcs.naxis > 2:
        wcs = full_wcs.celestial
    else:
        wcs = full_wcs

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=wcs)

    norm = ImageNormalize(data, stretch=AsinhStretch())

    im = ax.imshow(data, origin='lower', cmap='viridis', norm=norm)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux (Jy/beam)', fontsize=12)

    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)
    ax.set_title(f'MeerKAT 1.28 GHz (MGCLS): {name}', fontsize=14, fontweight='bold')

    ax.grid(color='white', ls='--', alpha=0.3)

    output_file = os.path.join(output_dir, f"meerkat_{name.replace(' ', '_')}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved image: {output_file}")

    plt.close(fig)


def load_mgcls_targets(clusters=None):
    """
    Load LoVoCCS targets that are in the MGCLS survey.

    Reads meerkat_verification_results.csv, filters for in_mgcls == 1,
    and cross-references with mgcls_clusters.csv for MGCLS names and
    precise coordinates.

    Args:
        clusters: Optional list of LoVoCCS cluster names to filter

    Returns:
        pandas.DataFrame with columns: name, ra, dec, mgcls_name
    """
    results = pd.read_csv("meerkat_verification_results.csv")
    mgcls_targets = results[results['in_mgcls'] == 1].copy()

    if len(mgcls_targets) == 0:
        print("ERROR: No clusters with in_mgcls=1 found in meerkat_verification_results.csv")
        return pd.DataFrame()

    mgcls_cat = pd.read_csv("mgcls_clusters.csv")
    mgcls_lookup = {row['name']: row for _, row in mgcls_cat.iterrows()}

    targets = []
    for _, row in mgcls_targets.iterrows():
        lovoccs_name = row['cluster_name']
        mgcls_name = row.get('mgcls_name', '')
        ra = row['ra']
        dec = row['dec']

        # Use MGCLS catalog coordinates if available (more precise)
        if pd.notna(mgcls_name) and mgcls_name in mgcls_lookup:
            mgcls_row = mgcls_lookup[mgcls_name]
            ra = mgcls_row['ra']
            dec = mgcls_row['dec']

        if pd.isna(ra) or pd.isna(dec):
            print(f"  WARNING: Skipping {lovoccs_name} - no coordinates")
            continue

        targets.append({
            'name': lovoccs_name,
            'ra': ra,
            'dec': dec,
            'mgcls_name': mgcls_name if pd.notna(mgcls_name) else lovoccs_name,
        })

    if clusters:
        targets = [t for t in targets if t['name'] in clusters]

    return pd.DataFrame(targets)


def main():
    parser = argparse.ArgumentParser(
        description='Download and visualize MeerKAT MGCLS radio images for LoVoCCS targets'
    )
    parser.add_argument('--all', action='store_true',
                       help='Download all 18 MGCLS targets')
    parser.add_argument('--clusters', nargs='+', default=None,
                       help='Specific cluster names to download (LoVoCCS names, e.g., A85 A3667)')
    parser.add_argument('--output-dir', default='meerkat_images',
                       help='Output directory for images (default: meerkat_images)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip targets that already have PNG files')
    parser.add_argument('--png-only', action='store_true',
                       help='Only generate PNG from existing FITS (no download)')
    parser.add_argument('--force', action='store_true',
                       help='Re-download even if FITS file exists')
    parser.add_argument('--validate', action='store_true',
                       help='Check existing FITS files for target coverage (no download)')
    parser.add_argument('--product-type', choices=['basic', 'enhanced'],
                       default='basic',
                       help='MGCLS product type: basic (default) or enhanced')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load targets
    all_targets = load_mgcls_targets()
    if len(all_targets) == 0:
        print("No MGCLS targets found. Check that meerkat_verification_results.csv exists.")
        return

    print("LoVoCCS targets in MGCLS DR1:")
    print("=" * 70)
    for _, row in all_targets.iterrows():
        print(f"  {row['name']:30s} -> MGCLS: {row['mgcls_name']}")
    print(f"\nTotal: {len(all_targets)} targets")
    print()

    # Determine which targets to process
    if args.clusters:
        targets = load_mgcls_targets(clusters=args.clusters)
        if len(targets) == 0:
            print(f"Error: None of the specified clusters found in MGCLS matches")
            print(f"Available clusters: {', '.join(all_targets['name'].tolist())}")
            return
        print(f"Processing {len(targets)} specified cluster(s)")
    elif args.all:
        targets = all_targets
        print(f"Processing all {len(targets)} MGCLS targets")
    else:
        default_names = ['A85', 'A3667', 'A133']
        targets = all_targets[all_targets['name'].isin(default_names)]
        if len(targets) == 0:
            targets = all_targets.head(3)
        print(f"Processing sample of {len(targets)} targets (use --all for all targets)")

    print()

    # Track results
    successful = []
    failed = []
    invalid = []

    # Validation mode
    if args.validate:
        print("VALIDATION MODE: Checking existing FITS files for target coverage")
        print("=" * 70)
        for i, (idx, target) in enumerate(targets.iterrows(), 1):
            name = target['name']
            fits_file = os.path.join(args.output_dir,
                                     f"meerkat_{name.replace(' ', '_')}.fits")

            if not os.path.exists(fits_file):
                print(f"  [{i}/{len(targets)}] {name}: NO FITS FILE")
                failed.append(name)
                continue

            is_valid = validate_meerkat_fits(fits_file, target['ra'], target['dec'])
            if is_valid:
                print(f"  [{i}/{len(targets)}] {name}: VALID (target in image)")
                successful.append(name)
            else:
                print(f"  [{i}/{len(targets)}] {name}: INVALID (target NOT in image)")
                invalid.append(name)

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
            print(f"\nTo fix, run: python download_meerkat_image.py "
                  f"--clusters {' '.join(invalid)} --force")
        return

    # Download/display loop
    for i, (idx, target) in enumerate(targets.iterrows(), 1):
        name = target['name']
        safe_name = name.replace(' ', '_')
        png_file = os.path.join(args.output_dir, f"meerkat_{safe_name}.png")
        fits_file = os.path.join(args.output_dir, f"meerkat_{safe_name}.fits")

        print(f"\n[{i}/{len(targets)}] {name} (MGCLS: {target['mgcls_name']})")
        print("=" * 70)

        # Skip if already exists
        if args.skip_existing and os.path.exists(png_file) and not args.force:
            print(f"  Skipping - PNG already exists: {png_file}")
            successful.append(name)
            continue

        # PNG-only mode
        if args.png_only:
            if os.path.exists(fits_file):
                print(f"  Generating PNG from existing FITS: {fits_file}")
                try:
                    hdu = fits.open(fits_file)
                    display_meerkat_image(hdu, name=name, output_dir=args.output_dir)
                    successful.append(name)
                except Exception as e:
                    print(f"  Error generating PNG: {e}")
                    failed.append(name)
            else:
                print(f"  No FITS file found: {fits_file}")
                failed.append(name)
            continue

        # Skip download if FITS exists and --force not specified
        if os.path.exists(fits_file) and not args.force:
            print(f"  FITS already exists: {fits_file}")
            print(f"  Use --force to re-download")
            if not os.path.exists(png_file):
                try:
                    hdu = fits.open(fits_file)
                    display_meerkat_image(hdu, name=name, output_dir=args.output_dir)
                except Exception as e:
                    print(f"  Error generating PNG: {e}")
            successful.append(name)
            continue

        # Download and display
        try:
            hdu = download_mgcls_image(
                mgcls_name=target['mgcls_name'],
                ra=target['ra'],
                dec=target['dec'],
                lovoccs_name=name,
                output_dir=args.output_dir,
                product_type=args.product_type,
            )

            if hdu:
                display_meerkat_image(hdu, name=name, output_dir=args.output_dir)
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
    print(f"Failed: {len(failed)}/{len(targets)}")
    if failed:
        print(f"\nFailed targets: {', '.join(failed)}")
        print(f"\nTo retry: python download_meerkat_image.py "
              f"--clusters {' '.join(failed)} --force")
    print(f"\nOutput directory: {args.output_dir}")


if __name__ == "__main__":
    main()

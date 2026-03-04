#!/usr/bin/env python3
"""
Script to match LoVoCCS galaxy cluster targets with LoTSS radio sources.

This script performs cross-matching between the LoVoCCS target list and the
LOFAR Two-metre Sky Survey (LoTSS) DR3 catalog (Shimwell et al. 2026) to
identify radio counterparts.

DR3 covers 88% of the northern sky (13.7M sources at 144 MHz, 6" resolution).
The PyBDSF source catalog is downloaded locally and matched in bulk using
astropy's search_around_sky.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table
from astropy import units as u

# LoTSS DR3 PyBDSF source catalog (v1.0, ~13.7M sources)
LOTSS_DR3_URL = "https://lofar-surveys.org/public/DR3/catalogues/LoTSS_DR3_v1.0.srl.fits"
LOTSS_DR3_FILE = "LoTSS_DR3_v1.0.srl.fits"


def parse_lovoccs_csv(filename):
    """Parse the LoVoCCS CSV file to extract target info."""
    # Read CSV, skipping the header rows
    df = pd.read_csv(filename, skiprows=1)

    # Extract relevant columns (name, ra, dec)
    targets = []
    for idx, row in df.iterrows():
        if idx >= 107:  # Stop at summary rows
            break
        try:
            # Column indices based on the CSV structure
            target_id = row.iloc[0]
            name = str(row.iloc[3]).strip()
            ra = float(row.iloc[5])
            dec = float(row.iloc[7])

            if not np.isnan(ra) and not np.isnan(dec) and name and name != 'nan':
                targets.append({
                    'id': target_id,
                    'name': name,
                    'ra': ra,
                    'dec': dec
                })
        except (ValueError, IndexError):
            continue

    return targets


def download_lotss_dr3_catalog():
    """
    Download the LoTSS DR3 PyBDSF source catalog if not already present.

    Streams the download with progress display. Cleans up partial files
    on failure.
    """
    if os.path.exists(LOTSS_DR3_FILE):
        size_mb = os.path.getsize(LOTSS_DR3_FILE) / (1024 * 1024)
        print(f"LoTSS DR3 catalog already exists: {LOTSS_DR3_FILE} ({size_mb:.0f} MB)")
        return True

    print(f"Downloading LoTSS DR3 catalog...")
    print(f"  URL: {LOTSS_DR3_URL}")

    partial_file = LOTSS_DR3_FILE + ".partial"
    try:
        resp = requests.get(LOTSS_DR3_URL, stream=True, timeout=60)
        resp.raise_for_status()

        total_size = int(resp.headers.get('content-length', 0))
        downloaded = 0
        start_time = time.time()

        with open(partial_file, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    elapsed = time.time() - start_time
                    rate = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        print(f"\r  Progress: {pct:.1f}% ({downloaded/(1024*1024):.0f} / "
                              f"{total_size/(1024*1024):.0f} MB, {rate:.1f} MB/s)",
                              end="", flush=True)
                    else:
                        print(f"\r  Downloaded: {downloaded/(1024*1024):.0f} MB ({rate:.1f} MB/s)",
                              end="", flush=True)

        elapsed = time.time() - start_time
        print(f"\n  Download complete ({elapsed:.0f}s, {downloaded/(1024*1024):.0f} MB)")

        os.rename(partial_file, LOTSS_DR3_FILE)
        return True

    except Exception as e:
        print(f"\n  ERROR downloading catalog: {e}")
        if os.path.exists(partial_file):
            os.remove(partial_file)
        return False


def load_lotss_dr3_catalog():
    """
    Load the LoTSS DR3 catalog and build a SkyCoord array.

    Returns:
        tuple: (astropy Table, SkyCoord array) or (None, None) on failure
    """
    print(f"Loading LoTSS DR3 catalog from {LOTSS_DR3_FILE}...")
    catalog = Table.read(LOTSS_DR3_FILE)
    print(f"  Loaded {len(catalog)} sources")

    # Verify expected columns exist
    expected_cols = ['RA', 'DEC', 'Peak_flux', 'Total_flux', 'Maj', 'Min', 'S_Code']
    missing = [c for c in expected_cols if c not in catalog.colnames]
    if missing:
        print(f"  WARNING: Missing expected columns: {missing}")
        print(f"  Available columns: {catalog.colnames}")
        return None, None

    # Check flux units -- PyBDSF default is Jy; report what we find
    for col in ['Peak_flux', 'Total_flux']:
        if catalog[col].unit is not None:
            print(f"  {col} unit: {catalog[col].unit}")
        else:
            print(f"  {col} unit: not set (assuming Jy)")

    # Build SkyCoord array
    print("  Building coordinate index...")
    coords = SkyCoord(ra=catalog['RA'], dec=catalog['DEC'], unit='deg', frame='icrs')
    print("  Done")

    return catalog, coords


def match_targets_lotss_dr3(targets, catalog, catalog_coords, radius_arcmin=10.0):
    """
    Bulk cross-match all LoVoCCS targets against the DR3 catalog.

    Uses search_around_sky for efficient spatial matching of all targets
    at once against the full 13.7M source catalog.

    Parameters:
        targets: list of dicts with 'ra', 'dec', 'name', 'id'
        catalog: astropy Table of DR3 sources
        catalog_coords: SkyCoord array for catalog
        radius_arcmin: search radius in arcminutes

    Returns:
        dict mapping target index -> list of matched source dicts (sorted by separation)
    """
    # Build target coordinate array
    target_coords = SkyCoord(
        ra=[t['ra'] for t in targets],
        dec=[t['dec'] for t in targets],
        unit='deg', frame='icrs'
    )

    print(f"  Cross-matching {len(targets)} targets against {len(catalog)} sources "
          f"(radius={radius_arcmin}')...")
    t0 = time.time()

    idx_targets, idx_catalog, sep2d, _ = search_around_sky(
        target_coords, catalog_coords, radius_arcmin * u.arcmin
    )

    elapsed = time.time() - t0
    print(f"  Found {len(idx_targets)} matches in {elapsed:.1f}s")

    # Determine flux conversion factor (Jy -> mJy)
    flux_unit = catalog['Total_flux'].unit
    if flux_unit is not None and 'mJy' in str(flux_unit):
        flux_to_mJy = 1.0
    else:
        # PyBDSF default is Jy
        flux_to_mJy = 1000.0

    # Determine axis conversion factor (PyBDSF Maj/Min are in degrees)
    maj_unit = catalog['Maj'].unit
    if maj_unit is not None and 'arcsec' in str(maj_unit):
        axis_to_arcsec = 1.0
    else:
        # PyBDSF default is degrees
        axis_to_arcsec = 3600.0

    # Group matches by target
    matches_by_target = {i: [] for i in range(len(targets))}

    for k in range(len(idx_targets)):
        ti = int(idx_targets[k])
        ci = int(idx_catalog[k])
        row = catalog[ci]

        # Map S_Code to resolved flag: M/C -> 'R', S -> 'U'
        s_code = str(row['S_Code']).strip()
        if s_code in ('M', 'C'):
            resolved = 'R'
        elif s_code == 'S':
            resolved = 'U'
        else:
            resolved = s_code

        source = {
            'ra': float(row['RA']),
            'dec': float(row['DEC']),
            'peak_flux': float(row['Peak_flux']) * flux_to_mJy,
            'total_flux': float(row['Total_flux']) * flux_to_mJy,
            'maj': float(row['Maj']) * axis_to_arcsec,
            'min': float(row['Min']) * axis_to_arcsec,
            'resolved': resolved,
            's_code': s_code,
            'separation_arcmin': sep2d[k].to(u.arcmin).value,
        }
        matches_by_target[ti].append(source)

    # Sort each target's matches by separation
    for ti in matches_by_target:
        matches_by_target[ti].sort(key=lambda x: x['separation_arcmin'])

    return matches_by_target


def main():
    csv_file = "LoVoCCS_target_list - lovoccs.csv"
    search_radius = 10.0  # arcminutes

    print("=" * 80)
    print("LoVoCCS - LoTSS DR3 Source Matching")
    print("=" * 80)
    print(f"Search radius: {search_radius} arcminutes")
    print("LoTSS DR3 covers ~88% of the northern sky (13.7M sources, 144 MHz, 6\" resolution)")
    print("Catalog: Shimwell et al. 2026, A&A")
    print()

    # Parse targets
    print("Parsing LoVoCCS target list...")
    targets = parse_lovoccs_csv(csv_file)
    print(f"Found {len(targets)} targets\n")

    # Download DR3 catalog if needed
    if not download_lotss_dr3_catalog():
        print("ERROR: Could not obtain LoTSS DR3 catalog. Exiting.")
        sys.exit(1)
    print()

    # Load catalog
    catalog, catalog_coords = load_lotss_dr3_catalog()
    if catalog is None:
        print("ERROR: Could not load LoTSS DR3 catalog. Exiting.")
        sys.exit(1)
    print()

    # Bulk cross-match
    matches_by_target = match_targets_lotss_dr3(
        targets, catalog, catalog_coords, radius_arcmin=search_radius
    )
    print()

    # Format results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    results = []
    detailed_matches = []

    for i, target in enumerate(targets):
        sources = matches_by_target[i]
        n_matches = len(sources)

        print(f"[{i+1:3d}/{len(targets)}] {target['name']:20s} "
              f"(RA={target['ra']:7.2f}, Dec={target['dec']:7.2f})...", end=" ")

        if n_matches > 0:
            closest = sources[0]
            closest_sep = closest['separation_arcmin']
            closest_flux = closest['total_flux']
            closest_resolved = closest['resolved']

            resolved_str = f", resolved={closest_resolved}" if closest_resolved is not None else ""
            print(f"✓ {n_matches} source(s) [closest: {closest_sep:.2f}' away{resolved_str}]")

            results.append({
                **target,
                'has_lotss_match': True,
                'n_lotss_sources': n_matches,
                'closest_sep_arcmin': closest_sep,
                'closest_total_flux_mJy': closest_flux,
                'closest_resolved': closest_resolved
            })

            for j, source in enumerate(sources):
                detailed_matches.append({
                    'cluster_id': target['id'],
                    'cluster_name': target['name'],
                    'cluster_ra': target['ra'],
                    'cluster_dec': target['dec'],
                    'source_rank': j + 1,
                    'source_ra': source['ra'],
                    'source_dec': source['dec'],
                    'separation_arcmin': source['separation_arcmin'],
                    'peak_flux_mJy_beam': source['peak_flux'],
                    'total_flux_mJy': source['total_flux'],
                    'major_axis_arcsec': source['maj'],
                    'minor_axis_arcsec': source['min'],
                    'resolved': source['resolved'],
                    's_code': source['s_code'],
                })
        else:
            print("✗ No matches")
            results.append({
                **target,
                'has_lotss_match': False,
                'n_lotss_sources': 0,
                'closest_sep_arcmin': None,
                'closest_total_flux_mJy': None,
                'closest_resolved': None
            })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    found = sum(1 for r in results if r['has_lotss_match'])
    not_found = sum(1 for r in results if r['has_lotss_match'] is False)
    total_sources = sum(r['n_lotss_sources'] for r in results if r['n_lotss_sources'] > 0)

    print(f"Total targets:              {len(results)}")
    print(f"With LoTSS matches:         {found} ({100*found/len(results):.1f}%)")
    print(f"Without LoTSS matches:      {not_found} ({100*not_found/len(results):.1f}%)")
    print(f"Total LoTSS sources:        {total_sources}")
    if found > 0:
        print(f"Average sources per match:  {total_sources/found:.1f}")

    # Count resolved sources
    if found > 0:
        resolved_count = sum(1 for r in results if r['has_lotss_match'] and r['closest_resolved'] == 'R')
        print(f"Clusters with resolved emission: {resolved_count} ({100*resolved_count/found:.1f}% of matches)")

    # Save summary results
    results_df = pd.DataFrame(results)
    output_file = "lovoccs_lotss_matches.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSummary results saved to: {output_file}")

    # Save detailed matches
    if detailed_matches:
        detailed_df = pd.DataFrame(detailed_matches)
        detailed_output = "lovoccs_lotss_matches_detailed.csv"
        detailed_df.to_csv(detailed_output, index=False)
        print(f"Detailed matches saved to: {detailed_output}")
        print(f"  (Contains {len(detailed_matches)} individual source matches)")

    # Show some statistics on matched targets
    if found > 0:
        print("\n" + "=" * 80)
        print("MATCHED TARGETS (sorted by number of sources)")
        print("=" * 80)
        matched_targets = [r for r in results if r['has_lotss_match']]
        matched_targets.sort(key=lambda x: x['n_lotss_sources'], reverse=True)

        print(f"{'Cluster':<20s} {'N_sources':>10s} {'Closest_sep':>12s} {'Flux_mJy':>12s} {'Resolved':>10s}")
        print("-" * 80)
        for r in matched_targets[:20]:  # Show top 20
            flux_str = f"{r['closest_total_flux_mJy']:.2f}" if r['closest_total_flux_mJy'] else "N/A"
            sep_str = f"{r['closest_sep_arcmin']:.2f}'" if r['closest_sep_arcmin'] else "N/A"
            resolved_str = str(r['closest_resolved']) if r['closest_resolved'] else "N/A"
            print(f"{r['name']:<20s} {r['n_lotss_sources']:>10d} {sep_str:>12s} {flux_str:>12s} {resolved_str:>10s}")

        if len(matched_targets) > 20:
            print(f"... and {len(matched_targets) - 20} more")


if __name__ == "__main__":
    main()

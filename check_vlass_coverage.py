#!/usr/bin/env python3
"""
Script to check which LoVoCCS targets are present in the VLASS survey.
"""

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
import sys

def parse_csv(filename):
    """Parse the LoVoCCS CSV file to extract target info."""
    # Read CSV, skipping the header rows
    df = pd.read_csv(filename, skiprows=1)

    # Extract relevant columns (name, ra, dec)
    # Based on the file structure: ID, name, ra(deg), dec(deg)
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

def query_vlass(ra, dec, radius_arcmin=5.0):
    """
    Query VLASS catalog using Vizier service.

    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    radius_arcmin : float
        Search radius in arcminutes

    Returns:
    --------
    int : Number of VLASS catalog sources found within the search radius
    """
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        # Create coordinate
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

        # Query Vizier for VLASS catalog
        # VLASS Epoch 1 Quick Look Catalog (Gordon+, 2021) is J/ApJS/255/30
        v = Vizier(columns=['*'], row_limit=-1)

        result = v.query_region(coord, radius=radius_arcmin*u.arcmin, catalog='J/ApJS/255/30')
        if result:
            return len(result[0])
        return 0

    except Exception as e:
        print(f"Error querying VLASS for RA={ra}, Dec={dec}: {str(e)}", file=sys.stderr)
        return -1

def main():
    csv_file = "LoVoCCS_target_list - lovoccs.csv"

    print("Parsing target list...")
    targets = parse_csv(csv_file)
    print(f"Found {len(targets)} targets\n")

    print("Querying VLASS catalog...")
    print("=" * 80)

    results = []
    for i, target in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] Checking {target['name']} (RA={target['ra']:.2f}, Dec={target['dec']:.2f})...", end=" ")
        sys.stdout.flush()

        count = query_vlass(target['ra'], target['dec'])

        if count > 0:
            print(f"✓ FOUND ({count} catalog sources)")
            results.append({**target, 'in_vlass': True, 'n_sources': count})
        elif count == 0:
            print("✗ NOT FOUND")
            results.append({**target, 'in_vlass': False, 'n_sources': 0})
        else:
            print("? ERROR")
            results.append({**target, 'in_vlass': None, 'n_sources': -1})

    print("\n" + "=" * 80)
    print("\nSUMMARY")
    print("=" * 80)

    found = sum(1 for r in results if r['in_vlass'])
    not_found = sum(1 for r in results if r['in_vlass'] is False)
    errors = sum(1 for r in results if r['in_vlass'] is None)

    print(f"Total targets: {len(results)}")
    print(f"Found in VLASS: {found} ({100*found/len(results):.1f}%)")
    print(f"Not in VLASS: {not_found} ({100*not_found/len(results):.1f}%)")
    print(f"Errors: {errors}")

    # Save results
    results_df = pd.DataFrame(results)
    output_file = "vlass_coverage_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Show targets found in VLASS
    if found > 0:
        print("\nTargets found in VLASS:")
        for r in results:
            if r['in_vlass']:
                print(f"  - {r['name']} ({r['n_sources']} catalog sources)")

if __name__ == "__main__":
    main()

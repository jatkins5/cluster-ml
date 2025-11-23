#!/usr/bin/env python3
"""
Script to match LoVoCCS galaxy cluster targets with eROSITA X-ray sources.

This script performs cross-matching between the LoVoCCS target list and the
eROSITA catalog to identify X-ray counterparts to the galaxy clusters.
"""

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u
from astroquery.vizier import Vizier
import sys

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

def query_erosita(ra, dec, radius_arcmin=5.0):
    """
    Query eROSITA catalog using Vizier service.

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
    dict : Dictionary containing match information:
        - n_matches: number of eROSITA sources within radius
        - sources: list of matched sources with their properties
    """
    try:
        # Create coordinate
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

        # Query Vizier for eROSITA catalog
        # eRASS1 (eROSITA All-Sky Survey) main catalog
        # Catalog: IX/68/eross1m (Merloni+, 2024)
        v = Vizier(columns=['*'], row_limit=-1)

        result = v.query_region(coord, radius=radius_arcmin*u.arcmin, catalog='J/A+A/685/A106')

        if result and len(result) > 0:
            table = result[0]
            n_matches = len(table)

            # Extract source information
            sources = []
            for row in table:
                source_info = {
                    'ra': float(row['RAJ2000']) if 'RAJ2000' in row.colnames else None,
                    'dec': float(row['DEJ2000']) if 'DEJ2000' in row.colnames else None,
                    'flux': float(row['F0_5_2']) if 'F0_5_2' in row.colnames else None,  # 0.5-2 keV flux
                    'det_ml': float(row['DET_LIKE']) if 'DET_LIKE' in row.colnames else None,  # Detection likelihood
                    'extent': float(row['EXT']) if 'EXT' in row.colnames else None,  # Source extent in arcsec
                }

                # Calculate separation from cluster center
                if source_info['ra'] is not None and source_info['dec'] is not None:
                    source_coord = SkyCoord(ra=source_info['ra']*u.degree,
                                          dec=source_info['dec']*u.degree,
                                          frame='icrs')
                    sep = coord.separation(source_coord).to(u.arcmin).value
                    source_info['separation_arcmin'] = sep
                else:
                    source_info['separation_arcmin'] = None

                sources.append(source_info)

            # Sort sources by separation from cluster center
            sources = sorted(sources, key=lambda x: x['separation_arcmin'] if x['separation_arcmin'] is not None else 999)

            return {'n_matches': n_matches, 'sources': sources}

        return {'n_matches': 0, 'sources': []}

    except Exception as e:
        print(f"Error querying eROSITA for RA={ra}, Dec={dec}: {str(e)}", file=sys.stderr)
        return {'n_matches': -1, 'sources': []}

def main():
    csv_file = "LoVoCCS_target_list - lovoccs.csv"
    search_radius = 5.0  # arcminutes

    print("=" * 80)
    print("LoVoCCS - eROSITA Source Matching")
    print("=" * 80)
    print(f"Search radius: {search_radius} arcminutes\n")

    print("Parsing LoVoCCS target list...")
    targets = parse_lovoccs_csv(csv_file)
    print(f"Found {len(targets)} targets\n")

    print("Querying eROSITA catalog...")
    print("=" * 80)

    results = []
    detailed_matches = []

    for i, target in enumerate(targets, 1):
        print(f"[{i:3d}/{len(targets)}] {target['name']:20s} (RA={target['ra']:7.2f}, Dec={target['dec']:7.2f})...", end=" ")
        sys.stdout.flush()

        match_result = query_erosita(target['ra'], target['dec'], radius_arcmin=search_radius)
        n_matches = match_result['n_matches']
        sources = match_result['sources']

        if n_matches > 0:
            # Get closest source
            closest = sources[0] if sources else None
            closest_sep = closest['separation_arcmin'] if closest else None
            closest_flux = closest['flux'] if closest else None

            print(f"✓ {n_matches} source(s) [closest: {closest_sep:.2f}' away]")

            results.append({
                **target,
                'has_erosita_match': True,
                'n_erosita_sources': n_matches,
                'closest_sep_arcmin': closest_sep,
                'closest_flux_0.5_2keV': closest_flux
            })

            # Store detailed information for all matched sources
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
                    'flux_0.5_2keV': source['flux'],
                    'det_likelihood': source['det_ml'],
                    'extent_arcsec': source['extent']
                })

        elif n_matches == 0:
            print("✗ No matches")
            results.append({
                **target,
                'has_erosita_match': False,
                'n_erosita_sources': 0,
                'closest_sep_arcmin': None,
                'closest_flux_0.5_2keV': None
            })

        else:
            print("? ERROR")
            results.append({
                **target,
                'has_erosita_match': None,
                'n_erosita_sources': -1,
                'closest_sep_arcmin': None,
                'closest_flux_0.5_2keV': None
            })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    found = sum(1 for r in results if r['has_erosita_match'])
    not_found = sum(1 for r in results if r['has_erosita_match'] is False)
    errors = sum(1 for r in results if r['has_erosita_match'] is None)
    total_sources = sum(r['n_erosita_sources'] for r in results if r['n_erosita_sources'] > 0)

    print(f"Total targets:              {len(results)}")
    print(f"With eROSITA matches:       {found} ({100*found/len(results):.1f}%)")
    print(f"Without eROSITA matches:    {not_found} ({100*not_found/len(results):.1f}%)")
    print(f"Errors:                     {errors}")
    print(f"Total eROSITA sources:      {total_sources}")
    print(f"Average sources per match:  {total_sources/found:.1f}" if found > 0 else "N/A")

    # Save summary results
    results_df = pd.DataFrame(results)
    output_file = "lovoccs_erosita_matches.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSummary results saved to: {output_file}")

    # Save detailed matches
    if detailed_matches:
        detailed_df = pd.DataFrame(detailed_matches)
        detailed_output = "lovoccs_erosita_matches_detailed.csv"
        detailed_df.to_csv(detailed_output, index=False)
        print(f"Detailed matches saved to: {detailed_output}")
        print(f"  (Contains {len(detailed_matches)} individual source matches)")

    # Show some statistics on matched targets
    if found > 0:
        print("\n" + "=" * 80)
        print("MATCHED TARGETS (sorted by number of sources)")
        print("=" * 80)
        matched_targets = [r for r in results if r['has_erosita_match']]
        matched_targets.sort(key=lambda x: x['n_erosita_sources'], reverse=True)

        print(f"{'Cluster':<20s} {'N_sources':>10s} {'Closest_sep':>12s} {'Flux_0.5-2keV':>15s}")
        print("-" * 80)
        for r in matched_targets[:20]:  # Show top 20
            flux_str = f"{r['closest_flux_0.5_2keV']:.2e}" if r['closest_flux_0.5_2keV'] else "N/A"
            sep_str = f"{r['closest_sep_arcmin']:.2f}'" if r['closest_sep_arcmin'] else "N/A"
            print(f"{r['name']:<20s} {r['n_erosita_sources']:>10d} {sep_str:>12s} {flux_str:>15s}")

        if len(matched_targets) > 20:
            print(f"... and {len(matched_targets) - 20} more")

if __name__ == "__main__":
    main()

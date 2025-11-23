#!/usr/bin/env python3
"""
Script to match LoVoCCS galaxy cluster targets with LoTSS radio sources.

This script performs cross-matching between the LoVoCCS target list and the
LOFAR Two-metre Sky Survey (LoTSS) DR2 catalog to identify radio counterparts.
"""

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
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

def query_lotss(ra, dec, radius_arcmin=5.0):
    """
    Query LoTSS catalog using Vizier service.

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
        - n_matches: number of LoTSS sources within radius
        - sources: list of matched sources with their properties
    """
    try:
        # Create coordinate
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

        # Query Vizier for LoTSS catalog
        # LoTSS DR2 catalog: J/A+A/659/A1 (Shimwell+, 2022)
        # This is the main LoTSS DR2 value-added catalog
        # There's also J/A+A/678/A151, the value-added catalog with optical IDs
        v = Vizier(columns=['*'], row_limit=-1)

        result = v.query_region(coord, radius=radius_arcmin*u.arcmin, catalog='J/A+A/678/A151/catalog')

        if result and len(result) > 0:
            table = result[0]
            n_matches = len(table)

            # Extract source information
            sources = []
            for row in table:
                source_info = {
                    'ra': float(row['RA_ICRS']) if 'RA_ICRS' in row.colnames else (float(row['RAJ2000']) if 'RAJ2000' in row.colnames else None),
                    'dec': float(row['DE_ICRS']) if 'DE_ICRS' in row.colnames else (float(row['DEJ2000']) if 'DEJ2000' in row.colnames else None),
                    'peak_flux': float(row['Speak']) if 'Speak' in row.colnames else None,  # Peak flux density in mJy/beam
                    'total_flux': float(row['Stotal']) if 'Stotal' in row.colnames else None,  # Total flux density in mJy
                    'maj': float(row['Maj']) if 'Maj' in row.colnames else None,  # Major axis in arcsec
                    'min': float(row['Min']) if 'Min' in row.colnames else None,  # Minor axis in arcsec
                    'resolved': row['Resolved'] if 'Resolved' in row.colnames else None,  # Resolved flag
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
        print(f"Error querying LoTSS for RA={ra}, Dec={dec}: {str(e)}", file=sys.stderr)
        return {'n_matches': -1, 'sources': []}

def main():
    csv_file = "LoVoCCS_target_list - lovoccs.csv"
    search_radius = 10.0  # arcminutes

    print("=" * 80)
    print("LoVoCCS - LoTSS Source Matching")
    print("=" * 80)
    print(f"Search radius: {search_radius} arcminutes")
    print("Note: LoTSS DR2 covers ~27% of the sky (5720 deg²) in the northern hemisphere")
    print("      Sky coverage: 0h < RA < 24h, +25° < Dec < +70°")
    print()

    print("Parsing LoVoCCS target list...")
    targets = parse_lovoccs_csv(csv_file)
    print(f"Found {len(targets)} targets\n")

    print("Querying LoTSS catalog...")
    print("=" * 80)

    results = []
    detailed_matches = []

    for i, target in enumerate(targets, 1):
        print(f"[{i:3d}/{len(targets)}] {target['name']:20s} (RA={target['ra']:7.2f}, Dec={target['dec']:7.2f})...", end=" ")
        sys.stdout.flush()

        match_result = query_lotss(target['ra'], target['dec'], radius_arcmin=search_radius)
        n_matches = match_result['n_matches']
        sources = match_result['sources']

        if n_matches > 0:
            # Get closest source
            closest = sources[0] if sources else None
            closest_sep = closest['separation_arcmin'] if closest else None
            closest_flux = closest['total_flux'] if closest else None
            closest_resolved = closest['resolved'] if closest else None

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
                    'peak_flux_mJy_beam': source['peak_flux'],
                    'total_flux_mJy': source['total_flux'],
                    'major_axis_arcsec': source['maj'],
                    'minor_axis_arcsec': source['min'],
                    'resolved': source['resolved']
                })

        elif n_matches == 0:
            print("✗ No matches")
            results.append({
                **target,
                'has_lotss_match': False,
                'n_lotss_sources': 0,
                'closest_sep_arcmin': None,
                'closest_total_flux_mJy': None,
                'closest_resolved': None
            })

        else:
            print("? ERROR")
            results.append({
                **target,
                'has_lotss_match': None,
                'n_lotss_sources': -1,
                'closest_sep_arcmin': None,
                'closest_total_flux_mJy': None,
                'closest_resolved': None
            })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    found = sum(1 for r in results if r['has_lotss_match'])
    not_found = sum(1 for r in results if r['has_lotss_match'] is False)
    errors = sum(1 for r in results if r['has_lotss_match'] is None)
    total_sources = sum(r['n_lotss_sources'] for r in results if r['n_lotss_sources'] > 0)

    print(f"Total targets:              {len(results)}")
    print(f"With LoTSS matches:         {found} ({100*found/len(results):.1f}%)")
    print(f"Without LoTSS matches:      {not_found} ({100*not_found/len(results):.1f}%)")
    print(f"Errors:                     {errors}")
    print(f"Total LoTSS sources:        {total_sources}")
    print(f"Average sources per match:  {total_sources/found:.1f}" if found > 0 else "N/A")

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

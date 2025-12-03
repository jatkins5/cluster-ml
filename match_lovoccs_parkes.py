#!/usr/bin/env python3
"""
Script to match LoVoCCS galaxy cluster targets with Parkes radio sources.

This script performs cross-matching between the LoVoCCS target list and the
Parkes radio survey catalogs (via VizieR) to identify radio counterparts.

Catalogs queried:
- PMN (VIII/38) - Parkes-MIT-NRAO 4.85 GHz surveys
- PKSCAT90 (VIII/15) - Parkes Radio Sources Catalogue
"""

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier
import sys
import time

# Define Parkes catalogs to query
PARKES_CATALOGS = {
    'PMN': {
        'vizier_id': 'VIII/38',
        'description': 'Parkes-MIT-NRAO 4.85 GHz survey',
        'freq_mhz': 4850,
        'ra_col': 'RAJ2000',
        'dec_col': 'DEJ2000',
        'flux_col': 'S4850',  # Flux density at 4850 MHz in mJy
        'flux_unit': 'mJy',
        'name_col': 'PMN',
    },
    'PKSCAT90': {
        'vizier_id': 'VIII/15',
        'description': 'Parkes Radio Sources Catalogue',
        'freq_mhz': 2700,  # Primary freq, has multi-frequency data
        'ra_col': 'RAJ2000',
        'dec_col': 'DEJ2000',
        'flux_col': 'S2700',  # Flux at 2700 MHz in Jy
        'flux_unit': 'Jy',
        'name_col': 'Bname',
        'extra_flux_cols': ['S80', 'S178', 'S408', 'S1410', 'S5000'],
    },
}

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


def get_source_coords(row, catalog_info, cluster_coord):
    """Extract source coordinates and calculate separation from cluster."""
    ra_col = catalog_info['ra_col']
    dec_col = catalog_info['dec_col']

    try:
        if ra_col in row.colnames and dec_col in row.colnames:
            ra_val = row[ra_col]
            dec_val = row[dec_col]

            # Handle different coordinate formats
            if isinstance(ra_val, str):
                # Sexagesimal format
                source_coord = SkyCoord(ra_val, dec_val,
                                       unit=(u.hourangle, u.degree), frame='icrs')
            else:
                # Decimal degrees
                source_coord = SkyCoord(ra=float(ra_val)*u.degree,
                                       dec=float(dec_val)*u.degree, frame='icrs')

            source_ra = source_coord.ra.degree
            source_dec = source_coord.dec.degree
            sep = cluster_coord.separation(source_coord).to(u.arcmin).value
            return source_ra, source_dec, sep
    except Exception:
        pass
    return None, None, None


def query_single_catalog(coord, catalog_name, catalog_info, radius_arcmin):
    """Query a single VizieR catalog for Parkes sources."""
    sources = []

    try:
        v = Vizier(columns=['*'], row_limit=-1)
        result = v.query_region(coord, radius=radius_arcmin*u.arcmin,
                               catalog=catalog_info['vizier_id'])

        if result and len(result) > 0:
            for table in result:
                for row in table:
                    source_ra, source_dec, sep = get_source_coords(row, catalog_info, coord)

                    # Get flux
                    flux_val = None
                    flux_col = catalog_info.get('flux_col')
                    if flux_col and flux_col in row.colnames:
                        try:
                            flux_val = float(row[flux_col])
                        except (ValueError, TypeError):
                            pass

                    # Get source name
                    name_val = None
                    name_col = catalog_info.get('name_col')
                    if name_col and name_col in row.colnames:
                        name_val = str(row[name_col])

                    # Get velocity if available (for HI surveys)
                    velocity = None
                    vel_col = catalog_info.get('velocity_col')
                    if vel_col and vel_col in row.colnames:
                        try:
                            velocity = float(row[vel_col])
                        except (ValueError, TypeError):
                            pass

                    # Get extra flux columns for PKSCAT90
                    extra_fluxes = {}
                    for ecol in catalog_info.get('extra_flux_cols', []):
                        if ecol in row.colnames:
                            try:
                                val = row[ecol]
                                if val is not None and not np.ma.is_masked(val):
                                    extra_fluxes[ecol] = float(val)
                                else:
                                    extra_fluxes[ecol] = None
                            except (ValueError, TypeError):
                                extra_fluxes[ecol] = None

                    source_info = {
                        'catalog': catalog_name,
                        'catalog_desc': catalog_info['description'],
                        'ra': source_ra,
                        'dec': source_dec,
                        'separation_arcmin': sep,
                        'source_name': name_val,
                        'flux': flux_val,
                        'flux_unit': catalog_info.get('flux_unit'),
                        'freq_mhz': catalog_info.get('freq_mhz'),
                        'velocity_kms': velocity,
                    }
                    source_info.update(extra_fluxes)
                    sources.append(source_info)

    except Exception as e:
        # Silently continue if a catalog query fails
        pass

    return sources


def query_parkes(ra, dec, radius_arcmin=5.0):
    """
    Query all Parkes catalogs in VizieR.

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
        - n_matches: number of Parkes sources within radius
        - sources: list of matched sources with their properties
        - by_catalog: breakdown of matches per catalog
    """
    try:
        coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

        all_sources = []
        by_catalog = {}

        for cat_name, cat_info in PARKES_CATALOGS.items():
            sources = query_single_catalog(coord, cat_name, cat_info, radius_arcmin)
            all_sources.extend(sources)
            by_catalog[cat_name] = len(sources)
            # Small delay to avoid overwhelming VizieR
            time.sleep(0.1)

        # Sort all sources by separation
        all_sources = sorted(all_sources,
                            key=lambda x: x['separation_arcmin'] if x['separation_arcmin'] is not None else 999)

        return {
            'n_matches': len(all_sources),
            'sources': all_sources,
            'by_catalog': by_catalog
        }

    except Exception as e:
        print(f"Error querying Parkes for RA={ra}, Dec={dec}: {str(e)}", file=sys.stderr)
        return {'n_matches': -1, 'sources': [], 'by_catalog': {}, 'error': str(e)}

def main():
    csv_file = "LoVoCCS_target_list - lovoccs.csv"
    search_radius = 5.0  # arcminutes

    print("=" * 80)
    print("LoVoCCS - Parkes Source Matching (via VizieR)")
    print("=" * 80)
    print(f"Search radius: {search_radius} arcminutes")
    print("\nQuerying the following Parkes catalogs:")
    for cat_name, cat_info in PARKES_CATALOGS.items():
        print(f"  - {cat_name}: {cat_info['description']} ({cat_info['vizier_id']})")
    print()

    print("Parsing LoVoCCS target list...")
    targets = parse_lovoccs_csv(csv_file)
    print(f"Found {len(targets)} targets\n")

    print("Querying Parkes catalogs via VizieR...")
    print("=" * 80)

    results = []
    detailed_matches = []
    catalog_totals = {cat: 0 for cat in PARKES_CATALOGS.keys()}

    for i, target in enumerate(targets, 1):
        print(f"[{i:3d}/{len(targets)}] {target['name']:20s} (RA={target['ra']:7.2f}, Dec={target['dec']:7.2f})...", end=" ")
        sys.stdout.flush()

        match_result = query_parkes(target['ra'], target['dec'], radius_arcmin=search_radius)
        n_matches = match_result['n_matches']
        sources = match_result['sources']
        by_catalog = match_result.get('by_catalog', {})

        # Update catalog totals
        for cat, count in by_catalog.items():
            catalog_totals[cat] += count

        if n_matches > 0:
            # Get closest source
            closest = sources[0] if sources else None
            closest_sep = closest['separation_arcmin'] if closest and closest['separation_arcmin'] is not None else None
            closest_flux = closest['flux'] if closest else None
            closest_catalog = closest['catalog'] if closest else None

            # Build catalog breakdown string
            cat_str = ", ".join(f"{c}:{by_catalog.get(c, 0)}" for c in PARKES_CATALOGS.keys() if by_catalog.get(c, 0) > 0)

            sep_str = f"{closest_sep:.2f}'" if closest_sep is not None else "?"
            print(f"found {n_matches} [{cat_str}] closest: {sep_str} ({closest_catalog})")

            results.append({
                **target,
                'has_parkes_match': True,
                'n_parkes_sources': n_matches,
                'n_pmn': by_catalog.get('PMN', 0),
                'n_pkscat90': by_catalog.get('PKSCAT90', 0),
                'closest_sep_arcmin': closest_sep,
                'closest_catalog': closest_catalog,
                'closest_flux': closest_flux,
                'closest_flux_unit': closest['flux_unit'] if closest else None,
            })

            # Store detailed information for all matched sources
            for j, source in enumerate(sources):
                detailed_matches.append({
                    'cluster_id': target['id'],
                    'cluster_name': target['name'],
                    'cluster_ra': target['ra'],
                    'cluster_dec': target['dec'],
                    'source_rank': j + 1,
                    'catalog': source['catalog'],
                    'catalog_desc': source['catalog_desc'],
                    'source_name': source['source_name'],
                    'source_ra': source['ra'],
                    'source_dec': source['dec'],
                    'separation_arcmin': source['separation_arcmin'],
                    'flux': source['flux'],
                    'flux_unit': source['flux_unit'],
                    'freq_mhz': source['freq_mhz'],
                    'velocity_kms': source.get('velocity_kms'),
                    # PKSCAT90 multi-frequency fluxes
                    'S80_Jy': source.get('S80'),
                    'S178_Jy': source.get('S178'),
                    'S408_Jy': source.get('S408'),
                    'S1410_Jy': source.get('S1410'),
                    'S2700_Jy': source.get('S2700'),
                    'S5000_Jy': source.get('S5000'),
                })

        elif n_matches == 0:
            print("no matches")
            results.append({
                **target,
                'has_parkes_match': False,
                'n_parkes_sources': 0,
                'n_pmn': 0,
                'n_pkscat90': 0,
                'closest_sep_arcmin': None,
                'closest_catalog': None,
                'closest_flux': None,
                'closest_flux_unit': None,
            })

        else:
            error_msg = match_result.get('error', 'Unknown error')
            print(f"ERROR: {error_msg}")
            results.append({
                **target,
                'has_parkes_match': None,
                'n_parkes_sources': -1,
                'n_pmn': -1,
                'n_pkscat90': -1,
                'closest_sep_arcmin': None,
                'closest_catalog': None,
                'closest_flux': None,
                'closest_flux_unit': None,
            })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    found = sum(1 for r in results if r['has_parkes_match'])
    not_found = sum(1 for r in results if r['has_parkes_match'] is False)
    errors = sum(1 for r in results if r['has_parkes_match'] is None)
    total_sources = sum(r['n_parkes_sources'] for r in results if r['n_parkes_sources'] > 0)

    print(f"Total targets:              {len(results)}")
    print(f"With Parkes matches:        {found} ({100*found/len(results):.1f}%)")
    print(f"Without Parkes matches:     {not_found} ({100*not_found/len(results):.1f}%)")
    print(f"Errors:                     {errors}")
    print(f"Total Parkes sources:       {total_sources}")
    if found > 0:
        print(f"Average sources per match:  {total_sources/found:.1f}")

    print("\nBreakdown by catalog:")
    for cat_name, cat_info in PARKES_CATALOGS.items():
        print(f"  {cat_name:12s}: {catalog_totals[cat_name]:4d} sources")

    # Save summary results
    results_df = pd.DataFrame(results)
    output_file = "lovoccs_parkes_matches.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSummary results saved to: {output_file}")

    # Save detailed matches
    if detailed_matches:
        detailed_df = pd.DataFrame(detailed_matches)
        detailed_output = "lovoccs_parkes_matches_detailed.csv"
        detailed_df.to_csv(detailed_output, index=False)
        print(f"Detailed matches saved to: {detailed_output}")
        print(f"  (Contains {len(detailed_matches)} individual source matches)")

    # Show some statistics on matched targets
    if found > 0:
        print("\n" + "=" * 80)
        print("MATCHED TARGETS (sorted by number of sources)")
        print("=" * 80)
        matched_targets = [r for r in results if r['has_parkes_match']]
        matched_targets.sort(key=lambda x: x['n_parkes_sources'], reverse=True)

        print(f"{'Cluster':<20s} {'Total':>6s} {'PMN':>5s} {'PKS90':>6s} {'Closest':>8s}")
        print("-" * 60)
        for r in matched_targets[:20]:  # Show top 20
            sep_str = f"{r['closest_sep_arcmin']:.2f}'" if r['closest_sep_arcmin'] else "N/A"
            print(f"{r['name']:<20s} {r['n_parkes_sources']:>6d} {r['n_pmn']:>5d} {r['n_pkscat90']:>6d} {sep_str:>8s}")

        if len(matched_targets) > 20:
            print(f"... and {len(matched_targets) - 20} more")

if __name__ == "__main__":
    main()

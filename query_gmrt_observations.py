#!/usr/bin/env python3
"""
Query GMRT (Giant Metrewave Radio Telescope) data for LoVoCCS galaxy clusters.

This script queries HEASARC GMRT catalogs (primarily TGSS ADR1 at 150 MHz) to find
radio sources near LoVoCCS cluster positions. TGSS ADR1 covers 90% of the sky
(declinations -53 to +90 degrees) with 0.62 million sources.

Available GMRT catalogs in HEASARC:
- gmrtas150m: TGSS ADR1 All-Sky 150-MHz (primary, largest coverage)
- gmrtha325m: Herschel-ATLAS/GAMA Fields 325-MHz
- gmrtelain1/2: ELAIS-N1/N2 fields 610-MHz
- gmrtlhcat/2/3: Lockman Hole 610-MHz
- And others for specific survey fields

Output:
- lovoccs_gmrt_matches.csv: Summary with one row per cluster
- lovoccs_gmrt_matches_detailed.csv: All GMRT sources near each cluster
"""

import pandas as pd
import numpy as np
import time
import sys
import argparse
from astropy.coordinates import SkyCoord
import astropy.units as u


# TGSS coverage limits
TGSS_DEC_MIN = -53.0
TGSS_DEC_MAX = 90.0


def normalize_cluster_name(name):
    """Normalize cluster name for consistency."""
    name = str(name).strip()

    # Normalize RXC names: "RXC J1217.6 + 0339" -> "RXC J1217.6+0339"
    # Remove spaces around + or - in coordinates
    if name.startswith('RXC') or name.startswith('RX '):
        # Handle "RXC J1217.6 + 0339" -> "RXC J1217.6+0339"
        name = name.replace(' + ', '+').replace(' - ', '-')
        # Handle "RX J0820.9+0751" format
        name = name.replace('RX ', 'RX J') if name.startswith('RX ') and 'RX J' not in name else name

    # Normalize APMCC names
    if name.startswith('APMCC'):
        # "APMCC 699" -> "APMCC_699" for consistency, or keep as is
        pass

    return name


def parse_lovoccs_csv(filename):
    """Parse the LoVoCCS CSV file to extract target info."""
    # Don't skip any rows - let pandas use the first row as header
    df = pd.read_csv(filename, skiprows=0)

    targets = []
    for idx, row in df.iterrows():
        if idx >= 107:
            break
        try:
            target_id = row.iloc[0]
            name = normalize_cluster_name(row.iloc[3])
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


def query_heasarc_gmrt(ra, dec, radius_arcmin=5.0, catalog='gmrtas150m'):
    """
    Query HEASARC for GMRT sources near a position.

    Parameters:
    -----------
    ra, dec : float
        Position in degrees (ICRS)
    radius_arcmin : float
        Search radius in arcminutes
    catalog : str
        HEASARC catalog name (default: gmrtas150m for TGSS)

    Returns:
    --------
    list : List of source dictionaries, or None on error
    """
    from astroquery.heasarc import Heasarc

    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    try:
        result = Heasarc.query_region(
            coord,
            catalog=catalog,
            radius=radius_arcmin * u.arcmin
        )

        if result is None or len(result) == 0:
            return []

        sources = []
        colnames = [c.lower() for c in result.colnames]

        for row in result:
            # Get RA/Dec (try both cases)
            ra_val = None
            dec_val = None
            if 'ra' in colnames:
                ra_val = float(row['ra'])
            elif 'RA' in result.colnames:
                ra_val = float(row['RA'])

            if 'dec' in colnames:
                dec_val = float(row['dec'])
            elif 'DEC' in result.colnames:
                dec_val = float(row['DEC'])

            source = {
                'ra': ra_val,
                'dec': dec_val,
            }

            # Get flux - try various column names (TGSS uses int_flux_150_mhz)
            source['flux_mJy'] = None
            for flux_col in ['int_flux_150_mhz', 'total_flux', 'flux_150_mhz', 'flux']:
                if flux_col in colnames:
                    try:
                        source['flux_mJy'] = float(row[flux_col])
                        break
                    except (ValueError, TypeError):
                        continue

            # Get flux error
            source['flux_err_mJy'] = None
            for err_col in ['int_flux_150_mhz_error', 'e_total_flux', 'flux_150_mhz_error']:
                if err_col in colnames:
                    try:
                        source['flux_err_mJy'] = float(row[err_col])
                        break
                    except (ValueError, TypeError):
                        continue

            # Get peak flux
            source['peak_flux_mJy'] = None
            for peak_col in ['flux_150_mhz', 'peak_flux', 'peak_flux_150mhz']:
                if peak_col in colnames:
                    try:
                        source['peak_flux_mJy'] = float(row[peak_col])
                        break
                    except (ValueError, TypeError):
                        continue

            # Source size (for extended source detection)
            source['major_axis_arcsec'] = None
            source['minor_axis_arcsec'] = None
            if 'maj' in colnames:
                try:
                    source['major_axis_arcsec'] = float(row['maj'])
                except (ValueError, TypeError):
                    pass
            if 'min' in colnames:
                try:
                    source['minor_axis_arcsec'] = float(row['min'])
                except (ValueError, TypeError):
                    pass

            # Source name
            source['source_name'] = None
            for name_col in ['name', 'source_name']:
                if name_col in colnames:
                    source['source_name'] = str(row[name_col])
                    break

            source['catalog'] = catalog

            sources.append(source)

        return sources

    except Exception as e:
        print(f"  Error querying {catalog}: {e}", file=sys.stderr)
        return None


def angular_separation_arcmin(ra1, dec1, ra2, dec2):
    """Calculate angular separation in arcminutes between two positions."""
    c1 = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree, frame='icrs')
    c2 = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree, frame='icrs')
    return c1.separation(c2).arcmin


def process_cluster_sources(cluster, sources, frequency_mhz=150):
    """
    Process GMRT sources for a cluster and return summary + detailed records.

    Parameters:
    -----------
    cluster : dict
        Cluster info with id, name, ra, dec
    sources : list
        List of source dicts from query_heasarc_gmrt
    frequency_mhz : float
        Observation frequency

    Returns:
    --------
    summary : dict
        Summary row for this cluster
    detailed : list
        List of detailed source records
    """
    summary = {
        'id': cluster['id'],
        'name': cluster['name'],
        'ra': cluster['ra'],
        'dec': cluster['dec'],
        'in_tgss_coverage': TGSS_DEC_MIN <= cluster['dec'] <= TGSS_DEC_MAX,
        'has_gmrt_match': False,
        'n_gmrt_sources': 0,
        'closest_sep_arcmin': None,
        'closest_flux_mJy': None,
        'gmrt_frequency_mhz': frequency_mhz
    }

    detailed = []

    if not sources:
        return summary, detailed

    # Calculate separations and sort by distance
    for source in sources:
        if source['ra'] is not None and source['dec'] is not None:
            source['separation_arcmin'] = angular_separation_arcmin(
                cluster['ra'], cluster['dec'],
                source['ra'], source['dec']
            )
        else:
            source['separation_arcmin'] = None

    # Filter sources with valid separations and sort
    valid_sources = [s for s in sources if s['separation_arcmin'] is not None]
    valid_sources.sort(key=lambda x: x['separation_arcmin'])

    if valid_sources:
        summary['has_gmrt_match'] = True
        summary['n_gmrt_sources'] = len(valid_sources)
        summary['closest_sep_arcmin'] = valid_sources[0]['separation_arcmin']
        summary['closest_flux_mJy'] = valid_sources[0].get('flux_mJy')

        # Build detailed records
        for rank, source in enumerate(valid_sources, 1):
            detailed.append({
                'cluster_id': cluster['id'],
                'cluster_name': cluster['name'],
                'cluster_ra': cluster['ra'],
                'cluster_dec': cluster['dec'],
                'source_rank': rank,
                'source_name': source.get('source_name'),
                'source_ra': source['ra'],
                'source_dec': source['dec'],
                'separation_arcmin': source['separation_arcmin'],
                'flux_mJy': source.get('flux_mJy'),
                'flux_err_mJy': source.get('flux_err_mJy'),
                'peak_flux_mJy': source.get('peak_flux_mJy'),
                'major_axis_arcsec': source.get('major_axis_arcsec'),
                'minor_axis_arcsec': source.get('minor_axis_arcsec'),
                'frequency_mhz': frequency_mhz,
                'catalog': source.get('catalog')
            })

    return summary, detailed


def main():
    parser = argparse.ArgumentParser(
        description='Query GMRT data for LoVoCCS clusters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Query all clusters using TGSS
  %(prog)s --clusters A780 A85       # Query specific clusters
  %(prog)s --radius 10               # Use 10 arcmin search radius
  %(prog)s --include-all-gmrt        # Query all GMRT catalogs
        """
    )
    parser.add_argument('--radius', type=float, default=5.0,
                       help='Search radius in arcminutes (default: 5.0)')
    parser.add_argument('--catalog', default='gmrtas150m',
                       help='Primary GMRT catalog to query (default: gmrtas150m)')
    parser.add_argument('--clusters', nargs='+', default=None,
                       help='Specific cluster names to query (default: all)')
    parser.add_argument('--output-prefix', default='lovoccs_gmrt',
                       help='Output file prefix (default: lovoccs_gmrt)')
    parser.add_argument('--include-all-gmrt', action='store_true',
                       help='Query all available GMRT catalogs (slower)')

    args = parser.parse_args()

    # Available GMRT catalogs in HEASARC
    all_gmrt_catalogs = [
        ('gmrtas150m', 150),    # TGSS ADR1 All-Sky
        ('gmrtha325m', 325),    # Herschel-ATLAS/GAMA
        ('gmrtelain1', 610),    # ELAIS-N1
        ('gmrtelain2', 610),    # ELAIS-N2
        ('gmrtlhcat', 610),     # Lockman Hole
        ('gmrtlhcat2', 610),    # Lockman Hole 2
        ('gmrtlhcat3', 610),    # Lockman Hole 3
        ('gmrtspxfls', 610),    # Spitzer FLS
        ('gmrtxl240m', 240),    # XMM-LSS 240 MHz
        ('gmrtxl610m', 610),    # XMM-LSS 610 MHz
    ]

    if args.include_all_gmrt:
        catalogs_to_query = all_gmrt_catalogs
    else:
        # Just the primary catalog
        freq = 150 if args.catalog == 'gmrtas150m' else 610
        catalogs_to_query = [(args.catalog, freq)]

    print("=" * 80)
    print("LoVoCCS - GMRT Source Query")
    print("=" * 80)
    print()
    print(f"Search radius: {args.radius} arcmin")
    print(f"Catalogs: {[c[0] for c in catalogs_to_query]}")
    print()

    # Load cluster list
    print("Loading LoVoCCS target list...")
    csv_file = "LoVoCCS_target_list - lovoccs.csv"
    targets = parse_lovoccs_csv(csv_file)
    print(f"Found {len(targets)} clusters")

    # Filter to specific clusters if requested
    if args.clusters:
        targets = [t for t in targets if t['name'] in args.clusters]
        print(f"Filtering to {len(targets)} specified clusters")

    print()

    # Count clusters in TGSS coverage
    in_coverage = sum(1 for t in targets if TGSS_DEC_MIN <= t['dec'] <= TGSS_DEC_MAX)
    print(f"Clusters in TGSS coverage (dec > {TGSS_DEC_MIN}): {in_coverage}/{len(targets)}")
    print()

    # Query each cluster
    all_summaries = []
    all_detailed = []

    print("=" * 80)
    print("Querying HEASARC GMRT catalogs...")
    print("=" * 80)

    for i, target in enumerate(targets, 1):
        in_cov = "Y" if TGSS_DEC_MIN <= target['dec'] <= TGSS_DEC_MAX else "N"
        print(f"\n[{i:3d}/{len(targets)}] {target['name']:25s} "
              f"(RA={target['ra']:.2f}, Dec={target['dec']:.2f}) [TGSS:{in_cov}]")

        cluster_sources = []

        for catalog, freq_mhz in catalogs_to_query:
            sources = query_heasarc_gmrt(
                target['ra'],
                target['dec'],
                radius_arcmin=args.radius,
                catalog=catalog
            )

            if sources is None:
                # Error occurred
                continue
            elif sources:
                print(f"  {catalog}: {len(sources)} sources")
                # Add frequency info
                for s in sources:
                    s['frequency_mhz'] = freq_mhz
                cluster_sources.extend(sources)

            # Small delay between catalog queries
            time.sleep(0.2)

        # Process all sources for this cluster
        # Use frequency of primary catalog for summary
        primary_freq = catalogs_to_query[0][1]
        summary, detailed = process_cluster_sources(target, cluster_sources, primary_freq)

        if summary['has_gmrt_match']:
            print(f"  -> {summary['n_gmrt_sources']} total sources, "
                  f"closest: {summary['closest_sep_arcmin']:.2f}', "
                  f"{summary['closest_flux_mJy']:.1f} mJy" if summary['closest_flux_mJy'] else "")
        else:
            print(f"  -> No GMRT sources found")

        all_summaries.append(summary)
        all_detailed.extend(detailed)

        # Delay between clusters
        time.sleep(0.2)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    n_with_match = sum(1 for s in all_summaries if s['has_gmrt_match'])
    n_in_coverage = sum(1 for s in all_summaries if s['in_tgss_coverage'])
    total_sources = sum(s['n_gmrt_sources'] for s in all_summaries)

    print(f"Total clusters queried:      {len(all_summaries)}")
    print(f"Clusters in TGSS coverage:   {n_in_coverage}")
    print(f"Clusters with GMRT sources:  {n_with_match}")
    print(f"Total GMRT sources found:    {total_sources}")

    # Save results
    summary_file = f"{args.output_prefix}_matches.csv"
    detailed_file = f"{args.output_prefix}_matches_detailed.csv"

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSummary saved to: {summary_file}")

    if all_detailed:
        df_detailed = pd.DataFrame(all_detailed)
        df_detailed.to_csv(detailed_file, index=False)
        print(f"Detailed results saved to: {detailed_file}")

    # Show clusters with most sources
    if n_with_match > 0:
        print("\nTop 10 clusters by GMRT source count:")
        print("-" * 60)
        df_sorted = df_summary[df_summary['has_gmrt_match']].sort_values(
            'n_gmrt_sources', ascending=False
        ).head(10)
        for _, row in df_sorted.iterrows():
            flux_str = f"{row['closest_flux_mJy']:.1f} mJy" if pd.notna(row['closest_flux_mJy']) else "N/A"
            print(f"  {row['name']:25s} {row['n_gmrt_sources']:3d} sources, "
                  f"closest: {row['closest_sep_arcmin']:.2f}', {flux_str}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

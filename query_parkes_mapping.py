#!/usr/bin/env python3
"""
Query ATOA for Parkes mapping observations with spatial diversity.

This script queries the Australia Telescope Online Archive (ATOA) to find
Parkes observations that could be combined into spatial maps. It uses a
larger search radius than typical point-source queries to find observations
at multiple different sky positions around each cluster.

Usage:
    python query_parkes_mapping.py --scan-all --output mappable_clusters.csv
    python query_parkes_mapping.py --cluster A780 --radius 2.0 --min-positions 5
"""

import pandas as pd
import numpy as np
import pyvo
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class CoverageReport:
    """Report on spatial coverage for a cluster."""
    cluster_name: str
    cluster_ra: float
    cluster_dec: float
    n_observations: int
    n_unique_positions: int
    ra_range: Tuple[float, float]  # (min, max) degrees
    dec_range: Tuple[float, float]
    spatial_extent_arcmin: float
    freq_bands: List[str]
    can_make_image: bool
    reason: str


def parse_lovoccs_csv(filename):
    """Parse the LoVoCCS CSV file to extract target info."""
    df = pd.read_csv(filename, skiprows=1)

    targets = []
    for idx, row in df.iterrows():
        if idx >= 107:
            break
        try:
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


def query_mapping_observations(service, ra, dec, radius_deg=2.0, max_results=500):
    """
    Query ATOA for Parkes observations that could form a map.

    Uses a larger search radius to find observations at multiple positions.

    Parameters:
    -----------
    service : pyvo.dal.TAPService
        The ATOA TAP service connection
    ra, dec : float
        Center position in degrees (ICRS)
    radius_deg : float
        Search radius in degrees (default 2.0 for mapping)
    max_results : int
        Maximum number of results to return

    Returns:
    --------
    list : List of observation dictionaries with position info
    """
    query = f"""
    SELECT TOP {max_results}
        obs_id,
        target_name,
        obs_collection,
        s_ra,
        s_dec,
        frequency,
        bandwidth,
        t_exptime,
        t_min,
        t_max,
        access_url,
        access_estsize
    FROM ivoa.obscore
    WHERE facility_name = 'ATOA'
      AND instrument_name = 'Parkes'
      AND 1=CONTAINS(POINT('ICRS', s_ra, s_dec), CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))
    ORDER BY s_ra, s_dec, frequency
    """

    try:
        result = service.search(query)

        observations = []
        for row in result:
            observations.append({
                'obs_id': str(row['obs_id']),
                'target_name': str(row['target_name']) if row['target_name'] else None,
                'project': str(row['obs_collection']) if row['obs_collection'] else None,
                'ra': float(row['s_ra']) if row['s_ra'] else None,
                'dec': float(row['s_dec']) if row['s_dec'] else None,
                'freq_mhz': float(row['frequency']) if row['frequency'] else None,
                'bandwidth_mhz': float(row['bandwidth']) if row['bandwidth'] else None,
                'exptime_s': float(row['t_exptime']) if row['t_exptime'] and not np.isnan(row['t_exptime']) else None,
                'mjd_start': float(row['t_min']) if row['t_min'] else None,
                'mjd_end': float(row['t_max']) if row['t_max'] else None,
                'access_url': str(row['access_url']) if row['access_url'] else None,
                'size_kb': int(row['access_estsize']) if row['access_estsize'] else None,
            })

        return observations

    except Exception as e:
        print(f"  Error querying: {e}", file=sys.stderr)
        return []


def analyze_coverage(observations: List[dict], cluster_name: str,
                     cluster_ra: float, cluster_dec: float,
                     position_tolerance_arcmin: float = 1.0,
                     min_positions: int = 5) -> CoverageReport:
    """
    Analyze spatial coverage of observations.

    Parameters:
    -----------
    observations : list
        List of observation dictionaries
    cluster_name : str
        Name of the cluster
    cluster_ra, cluster_dec : float
        Cluster center coordinates
    position_tolerance_arcmin : float
        Positions within this tolerance are considered the same
    min_positions : int
        Minimum unique positions required for mapping

    Returns:
    --------
    CoverageReport : Analysis results
    """
    if not observations:
        return CoverageReport(
            cluster_name=cluster_name,
            cluster_ra=cluster_ra,
            cluster_dec=cluster_dec,
            n_observations=0,
            n_unique_positions=0,
            ra_range=(0, 0),
            dec_range=(0, 0),
            spatial_extent_arcmin=0,
            freq_bands=[],
            can_make_image=False,
            reason="No observations found"
        )

    # Filter out observations with missing coordinates
    valid_obs = [o for o in observations if o['ra'] is not None and o['dec'] is not None]

    if not valid_obs:
        return CoverageReport(
            cluster_name=cluster_name,
            cluster_ra=cluster_ra,
            cluster_dec=cluster_dec,
            n_observations=len(observations),
            n_unique_positions=0,
            ra_range=(0, 0),
            dec_range=(0, 0),
            spatial_extent_arcmin=0,
            freq_bands=[],
            can_make_image=False,
            reason="No observations with valid coordinates"
        )

    # Extract positions
    ras = np.array([o['ra'] for o in valid_obs])
    decs = np.array([o['dec'] for o in valid_obs])

    # Find unique positions (within tolerance)
    tolerance_deg = position_tolerance_arcmin / 60.0
    unique_positions = []

    for ra, dec in zip(ras, decs):
        is_unique = True
        for ura, udec in unique_positions:
            # Simple distance check (good enough for small areas)
            dist = np.sqrt((ra - ura)**2 + (dec - udec)**2)
            if dist < tolerance_deg:
                is_unique = False
                break
        if is_unique:
            unique_positions.append((ra, dec))

    n_unique = len(unique_positions)

    # Calculate spatial extent
    ra_range = (ras.min(), ras.max())
    dec_range = (decs.min(), decs.max())

    # Extent in arcminutes
    ra_extent = (ra_range[1] - ra_range[0]) * 60 * np.cos(np.radians(cluster_dec))
    dec_extent = (dec_range[1] - dec_range[0]) * 60
    spatial_extent = np.sqrt(ra_extent**2 + dec_extent**2)

    # Get frequency bands
    freqs = [o['freq_mhz'] for o in valid_obs if o['freq_mhz'] is not None]
    freq_bands = []
    if freqs:
        # Group into bands (within 10% of each other)
        freqs_sorted = sorted(set(freqs))
        current_band = [freqs_sorted[0]]
        for f in freqs_sorted[1:]:
            if f / current_band[0] < 1.1:
                current_band.append(f)
            else:
                freq_bands.append(f"{np.mean(current_band):.0f} MHz")
                current_band = [f]
        freq_bands.append(f"{np.mean(current_band):.0f} MHz")

    # Determine if mapping is possible
    can_make_image = n_unique >= min_positions

    if not can_make_image:
        if n_unique == 1:
            reason = f"Only 1 unique position (need {min_positions}+ for mapping)"
        else:
            reason = f"Only {n_unique} unique positions (need {min_positions}+ for mapping)"
    else:
        reason = f"{n_unique} unique positions over {spatial_extent:.1f} arcmin"

    return CoverageReport(
        cluster_name=cluster_name,
        cluster_ra=cluster_ra,
        cluster_dec=cluster_dec,
        n_observations=len(valid_obs),
        n_unique_positions=n_unique,
        ra_range=ra_range,
        dec_range=dec_range,
        spatial_extent_arcmin=spatial_extent,
        freq_bands=freq_bands,
        can_make_image=can_make_image,
        reason=reason
    )


def find_mappable_clusters(service, targets: List[dict],
                           search_radius: float = 2.0,
                           min_positions: int = 5,
                           verbose: bool = True) -> List[CoverageReport]:
    """
    Scan all clusters to find those with mapping-suitable data.

    Parameters:
    -----------
    service : pyvo.dal.TAPService
        The ATOA TAP service
    targets : list
        List of target dictionaries with name, ra, dec
    search_radius : float
        Search radius in degrees
    min_positions : int
        Minimum unique positions for mapping
    verbose : bool
        Print progress

    Returns:
    --------
    list : List of CoverageReport objects
    """
    reports = []

    for i, target in enumerate(targets, 1):
        if verbose:
            print(f"[{i:3d}/{len(targets)}] {target['name']:<25} ", end="", flush=True)

        observations = query_mapping_observations(
            service,
            target['ra'],
            target['dec'],
            radius_deg=search_radius,
            max_results=500
        )

        report = analyze_coverage(
            observations,
            target['name'],
            target['ra'],
            target['dec'],
            min_positions=min_positions
        )

        reports.append(report)

        if verbose:
            status = "YES" if report.can_make_image else "no "
            print(f"{status} | {report.n_observations:4d} obs | {report.n_unique_positions:3d} pos | {report.reason}")

        # Be polite to the server
        time.sleep(0.3)

    return reports


def reports_to_dataframe(reports: List[CoverageReport]) -> pd.DataFrame:
    """Convert coverage reports to a DataFrame."""
    data = []
    for r in reports:
        data.append({
            'cluster_name': r.cluster_name,
            'ra': r.cluster_ra,
            'dec': r.cluster_dec,
            'n_observations': r.n_observations,
            'n_unique_positions': r.n_unique_positions,
            'ra_min': r.ra_range[0] if r.ra_range != (0, 0) else None,
            'ra_max': r.ra_range[1] if r.ra_range != (0, 0) else None,
            'dec_min': r.dec_range[0] if r.dec_range != (0, 0) else None,
            'dec_max': r.dec_range[1] if r.dec_range != (0, 0) else None,
            'spatial_extent_arcmin': r.spatial_extent_arcmin,
            'freq_bands': '; '.join(r.freq_bands),
            'can_make_image': r.can_make_image,
            'reason': r.reason
        })
    return pd.DataFrame(data)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Query ATOA for Parkes mapping observations'
    )
    parser.add_argument('--scan-all', action='store_true',
                       help='Scan all LoVoCCS clusters')
    parser.add_argument('--cluster', type=str, default=None,
                       help='Query specific cluster by name')
    parser.add_argument('--radius', type=float, default=2.0,
                       help='Search radius in degrees (default: 2.0)')
    parser.add_argument('--min-positions', type=int, default=5,
                       help='Minimum unique positions for mapping (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file for results')
    parser.add_argument('--targets-file', type=str,
                       default='LoVoCCS_target_list - lovoccs.csv',
                       help='Path to LoVoCCS targets CSV')

    args = parser.parse_args()

    if not args.scan_all and not args.cluster:
        parser.error("Must specify --scan-all or --cluster")

    print("=" * 80)
    print("Parkes Mapping Observation Query")
    print("=" * 80)
    print(f"Search radius: {args.radius} degrees")
    print(f"Minimum positions: {args.min_positions}")
    print()

    # Connect to ATOA
    print("Connecting to ATOA TAP service...")
    try:
        service = pyvo.dal.TAPService("https://atoavo.atnf.csiro.au/tap")
        print("Connected!\n")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    # Load targets
    print(f"Loading targets from {args.targets_file}...")
    targets = parse_lovoccs_csv(args.targets_file)
    print(f"Found {len(targets)} clusters\n")

    # Filter to specific cluster if requested
    if args.cluster:
        targets = [t for t in targets if t['name'] == args.cluster]
        if not targets:
            print(f"Cluster '{args.cluster}' not found in target list")
            sys.exit(1)
        print(f"Querying cluster: {args.cluster}\n")

    print("=" * 80)
    print("Querying ATOA for mapping observations...")
    print("=" * 80)
    print()

    reports = find_mappable_clusters(
        service,
        targets,
        search_radius=args.radius,
        min_positions=args.min_positions
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    mappable = [r for r in reports if r.can_make_image]
    print(f"Total clusters queried:      {len(reports)}")
    print(f"Clusters with mapping data:  {len(mappable)}")

    if mappable:
        print(f"\nClusters suitable for mapping:")
        for r in sorted(mappable, key=lambda x: -x.n_unique_positions):
            print(f"  {r.cluster_name:<25} {r.n_unique_positions:3d} positions, "
                  f"{r.spatial_extent_arcmin:.1f}' extent, {r.n_observations} obs")

    # Save results
    if args.output:
        df = reports_to_dataframe(reports)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")

    # Also save detailed observations for mappable clusters
    if mappable and args.output:
        detail_file = args.output.replace('.csv', '_details.csv')
        print(f"\nDownloading detailed observation lists for mappable clusters...")

        all_obs = []
        for r in mappable:
            target = next(t for t in targets if t['name'] == r.cluster_name)
            obs = query_mapping_observations(
                service, target['ra'], target['dec'],
                radius_deg=args.radius, max_results=500
            )
            for o in obs:
                o['cluster_name'] = r.cluster_name
            all_obs.extend(obs)
            time.sleep(0.3)

        if all_obs:
            obs_df = pd.DataFrame(all_obs)
            obs_df.to_csv(detail_file, index=False)
            print(f"Detailed observations saved to: {detail_file}")


if __name__ == "__main__":
    main()

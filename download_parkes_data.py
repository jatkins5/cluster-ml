#!/usr/bin/env python3
"""
Download Parkes radio data from ATOA for LoVoCCS galaxy clusters.

IMPORTANT: The ATOA archive contains raw Parkes observation data in RPFITS format.
These are NOT processed images - they are raw telescope data that requires
specialized software (Livedata, AIPS, MIRIAD) to reduce into images/spectra.

This script:
1. Queries the ATOA TAP service for Parkes observations near each cluster
2. Provides download URLs for the RPFITS files
3. Optionally downloads the files (requires OPAL authentication)

RPFITS files contain:
- Spectral line data (e.g., HI 21cm observations)
- Continuum observations
- Pulsar timing data

Authentication:
- ATOA requires OPAL credentials to download data
- Set environment variables OPAL_USERNAME and OPAL_PASSWORD, or
- Use --username and --password arguments, or
- Create a ~/.atoa_credentials file with username on line 1, password on line 2
"""

import pandas as pd
import numpy as np
import pyvo
import os
import sys
import time
import requests
from pathlib import Path
import getpass


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


def query_parkes_observations(service, ra, dec, radius_deg=0.5, max_results=100):
    """
    Query ATOA for Parkes observations near a position.

    Parameters:
    -----------
    service : pyvo.dal.TAPService
        The ATOA TAP service connection
    ra, dec : float
        Position in degrees (ICRS)
    radius_deg : float
        Search radius in degrees
    max_results : int
        Maximum number of results to return (0 for unlimited)

    Returns:
    --------
    list : List of observation dictionaries
    """
    top_clause = f"TOP {max_results}" if max_results > 0 else ""
    query = f"""
    SELECT {top_clause}
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
    ORDER BY frequency, t_min
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


def select_diverse_observations(observations, max_total, position_tolerance_arcmin=1.0):
    """
    Select observations prioritizing spatial diversity.

    Instead of just taking the first N observations (which may all be at
    the same position), this function ensures observations at different
    sky positions are included.

    Parameters:
    -----------
    observations : list
        List of observation dictionaries with 'ra' and 'dec' keys
    max_total : int
        Maximum total observations to return
    position_tolerance_arcmin : float
        Positions within this tolerance are considered the same

    Returns:
    --------
    list : Selected observations with spatial diversity
    """
    if not observations or max_total <= 0:
        return observations[:max_total] if max_total > 0 else observations

    tolerance_deg = position_tolerance_arcmin / 60.0

    # Group observations by unique position
    position_groups = []  # List of (ra, dec, [observations])

    for obs in observations:
        if obs['ra'] is None or obs['dec'] is None:
            continue

        # Find if this position matches an existing group
        found_group = False
        for group in position_groups:
            group_ra, group_dec, group_obs = group
            dist = np.sqrt((obs['ra'] - group_ra)**2 + (obs['dec'] - group_dec)**2)
            if dist < tolerance_deg:
                group_obs.append(obs)
                found_group = True
                break

        if not found_group:
            position_groups.append((obs['ra'], obs['dec'], [obs]))

    n_positions = len(position_groups)

    if n_positions == 0:
        return []

    # Calculate how many observations per position
    # Distribute evenly, with remainder going to positions with more data
    base_per_position = max_total // n_positions
    remainder = max_total % n_positions

    selected = []

    # Sort groups by number of observations (ascending) so smaller groups get picked first
    # This ensures we don't over-allocate to groups with few observations
    position_groups.sort(key=lambda g: len(g[2]))

    for i, (ra, dec, group_obs) in enumerate(position_groups):
        # How many to take from this group
        n_to_take = base_per_position
        if i >= n_positions - remainder:
            n_to_take += 1

        # Take up to n_to_take from this group
        selected.extend(group_obs[:n_to_take])

    # If we still have room (some groups had fewer than their allocation),
    # fill with remaining observations from larger groups
    if len(selected) < max_total:
        already_selected = set(o['obs_id'] for o in selected)
        for ra, dec, group_obs in sorted(position_groups, key=lambda g: -len(g[2])):
            for obs in group_obs:
                if obs['obs_id'] not in already_selected:
                    selected.append(obs)
                    already_selected.add(obs['obs_id'])
                    if len(selected) >= max_total:
                        break
            if len(selected) >= max_total:
                break

    return selected[:max_total]


def get_credentials(args):
    """Get OPAL credentials from various sources."""
    username = None
    password = None

    # 1. Command line arguments
    if args.username:
        username = args.username
    if args.password:
        password = args.password

    # 2. Environment variables
    if not username:
        username = os.environ.get('OPAL_USERNAME')
    if not password:
        password = os.environ.get('OPAL_PASSWORD')

    # 3. Credentials file
    cred_file = Path.home() / '.atoa_credentials'
    if cred_file.exists() and (not username or not password):
        try:
            lines = cred_file.read_text().strip().split('\n')
            if len(lines) >= 2:
                if not username:
                    username = lines[0].strip()
                if not password:
                    password = lines[1].strip()
        except Exception:
            pass

    # 4. Interactive prompt
    if not username:
        username = input("OPAL Username: ")
    if not password:
        password = getpass.getpass("OPAL Password: ")

    return username, password


def create_authenticated_session(username, password):
    """Create an authenticated session with ATOA."""
    session = requests.Session()

    # Get the login page first to get session cookie
    login_page = session.get("https://atoa.atnf.csiro.au/login.jsp")

    # Submit login form
    login_data = {
        '_action': 'login',
        'j_username': username,
        'j_password': password
    }

    login_response = session.post(
        "https://atoa.atnf.csiro.au/login",
        data=login_data,
        allow_redirects=True
    )

    # Check if login succeeded by looking for login page redirect or error
    if 'login.jsp' in login_response.url or 'error' in login_response.text.lower():
        # Try to find specific error message
        if 'Invalid' in login_response.text or 'incorrect' in login_response.text.lower():
            return None, "Invalid username or password"
        return None, "Login failed - check credentials"

    return session, None


def download_file(url, output_path, session=None, chunk_size=8192):
    """Download a file from URL to local path."""
    try:
        if session:
            response = session.get(url, stream=True, timeout=300)
        else:
            response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        # Check if we got HTML instead of data (auth redirect)
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            return False, "Authentication required or session expired"

        # Check file size is reasonable
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) < 10000:
            # Suspiciously small - might be error page
            content = response.content
            if b'<!DOCTYPE' in content or b'<html' in content:
                return False, "Received HTML instead of data"
            # Write small file anyway
            with open(output_path, 'wb') as f:
                f.write(content)
            return True, None

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)

        return True, None
    except Exception as e:
        return False, str(e)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Query/download Parkes data for LoVoCCS clusters')
    parser.add_argument('--download', action='store_true',
                       help='Actually download files (default: just list URLs)')
    parser.add_argument('--output-dir', default='parkes_data',
                       help='Directory to save downloaded files')
    parser.add_argument('--radius', type=float, default=0.5,
                       help='Search radius in degrees (default: 0.5)')
    parser.add_argument('--max-per-cluster', type=int, default=50,
                       help='Max observations per cluster (default: 50)')
    parser.add_argument('--freq-min', type=float, default=None,
                       help='Minimum frequency in MHz')
    parser.add_argument('--freq-max', type=float, default=None,
                       help='Maximum frequency in MHz')
    parser.add_argument('--clusters', nargs='+', default=None,
                       help='Specific cluster names to query (default: all)')
    parser.add_argument('--username', default=None,
                       help='OPAL username (or set OPAL_USERNAME env var)')
    parser.add_argument('--password', default=None,
                       help='OPAL password (or set OPAL_PASSWORD env var)')
    parser.add_argument('--prioritize-diversity', action='store_true',
                       help='Prioritize observations at different sky positions for mapping')

    args = parser.parse_args()

    print("=" * 80)
    print("LoVoCCS - Parkes ATOA Data Query/Download")
    print("=" * 80)
    print()
    print("NOTE: ATOA contains raw RPFITS files, NOT processed images.")
    print("      These require specialized software (Livedata/AIPS/MIRIAD) to reduce.")
    print()

    # Connect to ATOA
    print("Connecting to ATOA TAP service...")
    try:
        service = pyvo.dal.TAPService("https://atoavo.atnf.csiro.au/tap")
        print("Connected!\n")
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(1)

    # Parse cluster list
    print("Loading LoVoCCS target list...")
    csv_file = "LoVoCCS_target_list - lovoccs.csv"
    targets = parse_lovoccs_csv(csv_file)
    print(f"Found {len(targets)} clusters\n")

    # Filter to specific clusters if requested
    if args.clusters:
        targets = [t for t in targets if t['name'] in args.clusters]
        print(f"Filtering to {len(targets)} specified clusters\n")

    # Create output directory and authenticate if downloading
    session = None
    if args.download:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"Download directory: {output_dir}\n")

        # Authenticate with ATOA
        print("Authenticating with ATOA...")
        username, password = get_credentials(args)
        session, error = create_authenticated_session(username, password)
        if error:
            print(f"Authentication failed: {error}")
            print("Please check your OPAL credentials.")
            sys.exit(1)
        print("Authentication successful!\n")

    # Query each cluster
    all_observations = []

    print("=" * 80)
    diversity_str = ", prioritizing diversity" if args.prioritize_diversity else ""
    print(f"Querying ATOA (radius={args.radius} deg, max={args.max_per_cluster} per cluster{diversity_str})")
    print("=" * 80)

    for i, target in enumerate(targets, 1):
        print(f"\n[{i:3d}/{len(targets)}] {target['name']} (RA={target['ra']:.2f}, Dec={target['dec']:.2f})")

        # If prioritizing diversity, query all observations first, then select diverse subset
        if args.prioritize_diversity:
            observations = query_parkes_observations(
                service,
                target['ra'],
                target['dec'],
                radius_deg=args.radius,
                max_results=0  # No limit - get all
            )
        else:
            observations = query_parkes_observations(
                service,
                target['ra'],
                target['dec'],
                radius_deg=args.radius,
                max_results=args.max_per_cluster
            )

        # Filter by frequency if requested
        if args.freq_min:
            observations = [o for o in observations if o['freq_mhz'] and o['freq_mhz'] >= args.freq_min]
        if args.freq_max:
            observations = [o for o in observations if o['freq_mhz'] and o['freq_mhz'] <= args.freq_max]

        # Apply diversity selection if requested
        if args.prioritize_diversity and len(observations) > args.max_per_cluster:
            n_before = len(observations)
            observations = select_diverse_observations(observations, args.max_per_cluster)
            # Count unique positions
            positions = set((o['ra'], o['dec']) for o in observations if o['ra'] and o['dec'])
            print(f"  Diversity selection: {n_before} -> {len(observations)} obs at {len(positions)} positions")

        if observations:
            # Group by frequency
            freq_counts = {}
            for obs in observations:
                freq = obs['freq_mhz']
                if freq:
                    freq_key = f"{freq:.0f} MHz"
                    freq_counts[freq_key] = freq_counts.get(freq_key, 0) + 1

            freq_str = ", ".join(f"{k}:{v}" for k, v in sorted(freq_counts.items()))
            total_size_mb = sum(o['size_kb'] or 0 for o in observations) / 1024

            print(f"  Found {len(observations)} observations ({total_size_mb:.1f} MB)")
            print(f"  Frequencies: {freq_str}")

            # Add cluster info to observations
            for obs in observations:
                obs['cluster_name'] = target['name']
                obs['cluster_ra'] = target['ra']
                obs['cluster_dec'] = target['dec']

            all_observations.extend(observations)

            # Download if requested
            if args.download:
                cluster_dir = output_dir / target['name'].replace(' ', '_')
                cluster_dir.mkdir(exist_ok=True)

                for obs in observations:
                    if obs['access_url']:
                        filename = obs['obs_id']
                        filepath = cluster_dir / filename

                        if filepath.exists():
                            # Check if existing file is valid (not HTML error page)
                            if filepath.stat().st_size > 10000:
                                print(f"  Skipping {filename} (already exists)")
                                continue
                            else:
                                # Small file might be error page, re-download
                                filepath.unlink()

                        print(f"  Downloading {filename}...", end=" ")
                        sys.stdout.flush()
                        success, error = download_file(obs['access_url'], filepath, session=session)
                        if success:
                            size_mb = filepath.stat().st_size / 1024 / 1024
                            print(f"OK ({size_mb:.1f} MB)")
                        else:
                            print(f"FAILED: {error}")
                            if filepath.exists():
                                filepath.unlink()  # Remove partial/error file
        else:
            print("  No observations found")

        # Small delay to be polite to the server
        time.sleep(0.2)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    clusters_with_data = len(set(o['cluster_name'] for o in all_observations))
    total_size_gb = sum(o['size_kb'] or 0 for o in all_observations) / 1024 / 1024

    print(f"Total clusters queried:     {len(targets)}")
    print(f"Clusters with Parkes data:  {clusters_with_data}")
    print(f"Total observations found:   {len(all_observations)}")
    print(f"Total data size:            {total_size_gb:.2f} GB")

    # Save observation list
    if all_observations:
        df = pd.DataFrame(all_observations)
        output_csv = "lovoccs_parkes_atoa_observations.csv"
        df.to_csv(output_csv, index=False)
        print(f"\nObservation list saved to: {output_csv}")

        # Show top clusters by data volume
        print("\nTop 10 clusters by data volume:")
        print("-" * 60)
        cluster_sizes = df.groupby('cluster_name')['size_kb'].sum().sort_values(ascending=False)
        for name, size_kb in cluster_sizes.head(10).items():
            n_obs = len(df[df['cluster_name'] == name])
            print(f"  {name:25s} {n_obs:4d} obs, {size_kb/1024:.1f} MB")

    print("\n" + "=" * 80)
    print("NOTE: Downloaded files are raw RPFITS format.")
    print("To process them, you'll need:")
    print("  - Livedata (for spectral line data)")
    print("  - AIPS or MIRIAD (for general reduction)")
    print("  - Python packages: astropy (for basic FITS), or rpfits")
    print("=" * 80)


if __name__ == "__main__":
    main()

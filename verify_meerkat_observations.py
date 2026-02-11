#!/usr/bin/env python3
"""
Verify MeerKAT observations for LoVoCCS clusters by cross-referencing
with the MeerKAT Galaxy Cluster Legacy Survey (MGCLS) DR1 catalog
and querying the SARAO archive API directly.

MGCLS observed 115 galaxy clusters with MeerKAT L-band (2018-2019).
Reference: Knowles et al. 2022, A&A 657, A56

The SARAO archive API provides access to ALL MeerKAT observations,
including programs beyond MGCLS.
"""

import csv
import re
import json
import asyncio
import webbrowser
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import parse_qs, urlparse
import math

# Optional dependencies for SARAO API
try:
    import aiohttp
    from gql import gql, Client
    from gql.transport.aiohttp import AIOHTTPTransport
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


# SARAO API configuration (custom PKCE flow)
SARAO_BASE_URL = "https://archive.sarao.ac.za"
SARAO_AUTH_INIT_URL = f"{SARAO_BASE_URL}/_auth/pkce-cli-auth-url"
SARAO_AUTH_EXCHANGE_URL = f"{SARAO_BASE_URL}/_auth/pkce-cli-auth-complete"
SARAO_AUTH_REFRESH_URL = f"{SARAO_BASE_URL}/_auth/pkce-cli-refresh"
SARAO_GRAPHQL_URL = f"{SARAO_BASE_URL}/graphql"
TOKENS_FILE = Path(__file__).parent / "tokens.json"


@dataclass
class Cluster:
    name: str
    ra: float  # degrees
    dec: float  # degrees
    meerkat_flag: Optional[int] = None  # 0 or 1 from CSV
    in_mgcls: bool = False
    mgcls_name: Optional[str] = None
    match_type: Optional[str] = None  # 'name', 'coordinate', or 'both'
    # SARAO API results
    sarao_observation_count: int = 0
    sarao_proposal_ids: list = field(default_factory=list)
    sarao_schedule_blocks: list = field(default_factory=list)


def normalize_cluster_name(name: str) -> str:
    """Normalize cluster name for matching."""
    name = name.strip().upper()
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name)
    # Normalize Abell variants
    name = re.sub(r'^ABELL\s*', 'A', name)
    name = re.sub(r'^A\s+', 'A', name)
    # Handle special cases
    name = name.replace('APMCC ', 'APMCC')
    return name


def extract_abell_number(name: str) -> Optional[int]:
    """Extract Abell catalog number if present."""
    match = re.search(r'A(?:BELL)?\s*(\d+)', name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# =============================================================================
# SARAO Archive API Functions
# =============================================================================

def load_tokens() -> Optional[dict]:
    """Load saved tokens from file."""
    if TOKENS_FILE.exists():
        try:
            with open(TOKENS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    return None


def save_tokens(tokens: dict):
    """Save tokens to file."""
    with open(TOKENS_FILE, 'w') as f:
        json.dump(tokens, f, indent=2)


def refresh_access_token(refresh_token: str) -> Optional[dict]:
    """Refresh the access token using the refresh token."""
    import requests

    try:
        resp = requests.post(
            SARAO_AUTH_REFRESH_URL,
            json={'refresh_token': refresh_token},
            timeout=30
        )
        if resp.status_code == 200:
            tokens = resp.json()
            save_tokens(tokens)
            return tokens
        return None
    except Exception as e:
        print(f"Token refresh failed: {e}")
        return None


def authenticate_sarao() -> Optional[str]:
    """
    Perform PKCE authentication with SARAO archive.
    Returns access token or None if authentication fails.

    SARAO uses a custom PKCE flow:
    1. GET /_auth/pkce-cli-auth-url -> returns auth_url and state
    2. User opens auth_url in browser and logs in
    3. User pastes redirect URL back to CLI
    4. POST /_auth/pkce-cli-auth-complete with code and state -> returns tokens
    """
    import requests

    if not HAS_AIOHTTP:
        print("Error: aiohttp package required for SARAO API access.")
        print("Install with: pip install aiohttp")
        return None

    # Try to load existing tokens
    tokens = load_tokens()
    if tokens:
        access_token = tokens.get('access_token')
        refresh_token = tokens.get('refresh_token')

        # Try refresh if we have a refresh token
        if refresh_token:
            print("Attempting to refresh SARAO access token...")
            new_tokens = refresh_access_token(refresh_token)
            if new_tokens:
                print("Token refreshed successfully.")
                return new_tokens.get('access_token')
            print("Token refresh failed, need to re-authenticate.")

    # Need to perform full authentication
    print("\n" + "="*60)
    print("SARAO Archive Authentication Required")
    print("="*60)

    # Step 1: Get authentication URL from SARAO
    print("\nRequesting authentication URL from SARAO...")
    try:
        resp = requests.get(SARAO_AUTH_INIT_URL, timeout=30)
        if resp.status_code != 200:
            print(f"Failed to get auth URL: {resp.status_code} - {resp.text}")
            return None
        auth_data = resp.json()
        auth_url = auth_data.get('auth_url')
        state = auth_data.get('state')
        if not auth_url or not state:
            print(f"Invalid auth response: {auth_data}")
            return None
    except Exception as e:
        print(f"Failed to contact SARAO: {e}")
        return None

    # Step 2: Direct user to authenticate
    print("\nOpening browser for SARAO login...")
    print(f"If browser doesn't open, visit:\n{auth_url}\n")
    webbrowser.open(auth_url)

    print("After logging in, you will be redirected to a page.")
    print("Copy the ENTIRE URL from your browser's address bar and paste it below.\n")
    redirect_url = input("Paste redirect URL here: ").strip()

    if not redirect_url:
        print("No URL provided. Authentication cancelled.")
        return None

    # Step 3: Extract code and state from redirect URL
    try:
        parsed = urlparse(redirect_url)
        query_params = parse_qs(parsed.query)
        code = query_params.get('code', [None])[0]
        state_returned = query_params.get('state', [None])[0]

        if not code:
            print(f"No authorization code found in URL: {redirect_url}")
            return None
        if state_returned != state:
            print(f"State mismatch: expected {state}, got {state_returned}")
            return None
    except Exception as e:
        print(f"Failed to parse redirect URL: {e}")
        return None

    # Step 4: Exchange code for tokens (GET request with query params)
    print("\nExchanging authorization code for tokens...")
    try:
        # Add manual=true to the query params and make GET request
        query_params['manual'] = ['true']
        from urllib.parse import urlencode
        exchange_params = {k: v[0] if len(v) == 1 else v for k, v in query_params.items()}
        resp = requests.get(
            SARAO_AUTH_EXCHANGE_URL,
            params=exchange_params,
            timeout=30
        )
        if resp.status_code != 200:
            print(f"Token exchange failed: {resp.status_code} - {resp.text}")
            return None
        tokens = resp.json()
    except Exception as e:
        print(f"Token exchange failed: {e}")
        return None

    if 'access_token' not in tokens:
        print(f"Invalid token response: {tokens}")
        return None

    save_tokens(tokens)
    print("Authentication successful! Tokens saved to tokens.json")
    return tokens.get('access_token')


async def query_sarao_cone_search(
    access_token: str,
    ra: float,
    dec: float,
    radius: float = 0.167  # ~10 arcmin in degrees
) -> dict:
    """
    Query SARAO archive for observations at given coordinates.

    Args:
        access_token: SARAO OAuth2 access token
        ra: Right ascension in degrees
        dec: Declination in degrees
        radius: Search radius in degrees (default 10 arcmin)

    Returns:
        Dict with observation count, proposal IDs, and schedule block codes
    """
    transport = AIOHTTPTransport(
        url=SARAO_GRAPHQL_URL,
        headers={'Authorization': f'Bearer {access_token}'}
    )

    # Use fetch_schema_from_transport=True to properly serialize JSON scalar types
    async with Client(transport=transport, fetch_schema_from_transport=True) as session:
        query = gql("""
            query ($filters: [SolrFilterInput!]) {
                captureBlocks(filters: $filters, limit: 100) {
                    pageInfo {
                        totalCount
                    }
                    records {
                        ScheduleBlockIdCode
                        ProposalId
                        Description
                        StartTime
                    }
                }
            }
        """)

        # Filter value requires STRING values for ra, dec, radius
        variables = {
            'filters': [{
                'field': 'radec',
                'value': {'ra': str(ra), 'dec': str(dec), 'radius': str(radius)}
            }]
        }

        try:
            result = await session.execute(query, variable_values=variables)

            records = result.get('captureBlocks', {}).get('records', [])
            total_count = result.get('captureBlocks', {}).get('pageInfo', {}).get('totalCount', 0)

            proposals = list(set(r.get('ProposalId', '') for r in records if r.get('ProposalId')))
            schedule_blocks = [r.get('ScheduleBlockIdCode', '') for r in records if r.get('ScheduleBlockIdCode')]

            return {
                'count': total_count,
                'proposals': proposals,
                'schedule_blocks': schedule_blocks,
                'records': records
            }
        except Exception as e:
            print(f"  Error querying SARAO API: {e}")
            return {'count': 0, 'proposals': [], 'schedule_blocks': [], 'error': str(e)}


async def query_all_clusters_sarao(
    access_token: str,
    clusters: list,
    radius: float = 0.167
) -> list:
    """
    Query SARAO archive for all clusters.

    Args:
        access_token: SARAO OAuth2 access token
        clusters: List of Cluster objects with ra/dec coordinates
        radius: Search radius in degrees

    Returns:
        Updated list of clusters with SARAO observation info
    """
    print(f"\nQuerying SARAO archive for {len(clusters)} clusters...")
    print(f"Search radius: {radius:.3f} deg ({radius*60:.1f} arcmin)")

    # Query clusters in batches to avoid overwhelming the API
    batch_size = 10
    total = len(clusters)

    for i in range(0, total, batch_size):
        batch = clusters[i:i+batch_size]
        print(f"\n  Processing clusters {i+1}-{min(i+batch_size, total)} of {total}...")

        tasks = []
        for cluster in batch:
            if cluster.ra != 0 and cluster.dec != 0:
                tasks.append((cluster, query_sarao_cone_search(access_token, cluster.ra, cluster.dec, radius)))

        # Execute batch concurrently
        for cluster, task in tasks:
            result = await task
            cluster.sarao_observation_count = result.get('count', 0)
            cluster.sarao_proposal_ids = result.get('proposals', [])
            cluster.sarao_schedule_blocks = result.get('schedule_blocks', [])

            if cluster.sarao_observation_count > 0:
                print(f"    {cluster.name}: {cluster.sarao_observation_count} observations "
                      f"({', '.join(cluster.sarao_proposal_ids[:3])}{'...' if len(cluster.sarao_proposal_ids) > 3 else ''})")

        # Small delay between batches to be nice to the API
        if i + batch_size < total:
            await asyncio.sleep(0.5)

    return clusters


def run_sarao_verification(clusters: list, radius: float = 0.167) -> list:
    """
    Run SARAO archive verification for all clusters.

    This is the main entry point for SARAO API queries.
    Handles authentication and runs the async queries.
    """
    if not HAS_AIOHTTP:
        print("\nSARAO API verification skipped - missing dependencies.")
        print("Install with: pip install aiohttp")
        return clusters

    # Authenticate
    access_token = authenticate_sarao()
    if not access_token:
        print("\nSARAO authentication failed. Skipping API verification.")
        return clusters

    # Run queries
    return asyncio.run(query_all_clusters_sarao(access_token, clusters, radius))


# =============================================================================
# Utility Functions
# =============================================================================

def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Calculate angular separation in arcminutes using haversine formula."""
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)

    # Haversine formula
    dlon = ra2_rad - ra1_rad
    dlat = dec2_rad - dec1_rad
    a = math.sin(dlat/2)**2 + math.cos(dec1_rad) * math.cos(dec2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Convert to arcminutes (c is in radians, 1 radian = 3437.75 arcmin)
    return math.degrees(c) * 60


def parse_mgcls_table(filepath: str) -> list[dict]:
    """Parse the MGCLS table1.dat file.

    The format is approximately fixed-width:
    - Col 1: Selection type (R or X)
    - Col 3-22: Cluster name (may include numbers like "Abell 85")
    - Col 23-30: RA (degrees)
    - Col 31-39: Dec (degrees)
    - Col 40-44: Redshift
    - etc.
    """
    clusters = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            # The data appears to be roughly fixed-width with names ending around column 20-22
            # and RA starting around column 23. Let's parse more carefully.
            sel_type = line[0]

            # Find the pattern: name followed by a decimal coordinate
            # RA values are like "3.3842" or "351.3321" (with a decimal point)
            # Look for a float with decimal that's a valid RA (0-360)
            match = re.search(r'\s+(\d{1,3}\.\d+)\s+(-?\d{1,2}\.\d+)\s+(\d\.\d+)', line)
            if not match:
                continue

            ra = float(match.group(1))
            dec = float(match.group(2))
            redshift = float(match.group(3))

            # Name is everything between selection type and RA
            name_end = match.start(1)
            name = line[1:name_end].strip()

            # Handle merger flag (+) in name
            name = name.replace('+', '').strip()

            # Check for alternate name at end of line (after the fixed columns)
            # The pattern is: RA Dec z Xflag Optflag Radioflag size nsub merge [altname]
            alt_name = None
            remaining = line[match.end(3):].strip()
            parts = remaining.split()
            # Skip: X_flag, Opt_flag, Radio_flag, size, n_sub, merge_flag
            # These are 6 fields, so anything after is alternate name
            if len(parts) > 6:
                alt_name = ' '.join(parts[6:])

            clusters.append({
                'name': name,
                'ra': ra,
                'dec': dec,
                'redshift': redshift,
                'alt_name': alt_name,
                'selection': sel_type
            })

    return clusters


def parse_lovoccs_csv(filepath: str) -> list[tuple]:
    """Parse the LoVoCCS cluster comparison CSV.

    Returns list of (Cluster, citation) tuples.
    """
    clusters = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['LoVoCCS '].strip() if 'LoVoCCS ' in row else row.get('LoVoCCS', '').strip()
            meerkat_str = row.get('MeerKAT', '').strip()
            citation = row.get('Citation', '').strip()

            # Handle empty or non-numeric MeerKAT values
            try:
                meerkat_flag = int(meerkat_str)
            except ValueError:
                meerkat_flag = None

            clusters.append((Cluster(
                name=name,
                ra=0.0,  # Will be filled from coordinates file
                dec=0.0,
                meerkat_flag=meerkat_flag
            ), citation))

    return clusters


def normalize_coord_name(name: str) -> str:
    """Normalize cluster name for coordinate matching.

    Handles variations like 'RXC J1217.6 + 0339' vs 'RXC J1217.6+0339'.
    """
    # Remove extra spaces around + and -
    name = re.sub(r'\s*\+\s*', '+', name)
    name = re.sub(r'\s*-\s*', '-', name)
    # Normalize whitespace
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def parse_lovoccs_coordinates(filepath: str) -> dict[str, tuple[float, float]]:
    """Parse LoVoCCS coordinates from meerkat_observable.csv.

    Returns dict mapping both original and normalized names to coordinates.
    """
    coords = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name'].strip()
            ra = float(row['ra'])
            dec = float(row['dec'])
            # Store under both original and normalized names
            coords[name] = (ra, dec)
            normalized = normalize_coord_name(name)
            if normalized != name:
                coords[normalized] = (ra, dec)
    return coords


def match_clusters(lovoccs: list[Cluster], mgcls: list[dict], coord_threshold: float = 5.0) -> list[Cluster]:
    """
    Match LoVoCCS clusters to MGCLS clusters by name and/or coordinates.

    Args:
        lovoccs: List of LoVoCCS clusters
        mgcls: List of MGCLS cluster dicts
        coord_threshold: Maximum angular separation in arcminutes for coordinate matching

    Returns:
        Updated list of LoVoCCS clusters with match information
    """
    # Build lookup structures for MGCLS
    mgcls_by_abell = {}
    mgcls_by_norm_name = {}

    for m in mgcls:
        norm = normalize_cluster_name(m['name'])
        mgcls_by_norm_name[norm] = m

        abell_num = extract_abell_number(m['name'])
        if abell_num:
            mgcls_by_abell[abell_num] = m

        # Also check alternate names
        if m['alt_name']:
            alt_norm = normalize_cluster_name(m['alt_name'])
            mgcls_by_norm_name[alt_norm] = m
            alt_abell = extract_abell_number(m['alt_name'])
            if alt_abell:
                mgcls_by_abell[alt_abell] = m

    for cluster in lovoccs:
        norm_name = normalize_cluster_name(cluster.name)
        abell_num = extract_abell_number(cluster.name)

        matched_mgcls = None
        match_type = None

        # Try name matching first
        if norm_name in mgcls_by_norm_name:
            matched_mgcls = mgcls_by_norm_name[norm_name]
            match_type = 'name'
        elif abell_num and abell_num in mgcls_by_abell:
            matched_mgcls = mgcls_by_abell[abell_num]
            match_type = 'name'

        # If no name match and we have coordinates, try coordinate matching
        if matched_mgcls is None and cluster.ra != 0 and cluster.dec != 0:
            best_sep = float('inf')
            for m in mgcls:
                sep = angular_separation(cluster.ra, cluster.dec, m['ra'], m['dec'])
                if sep < best_sep:
                    best_sep = sep
                    if sep <= coord_threshold:
                        matched_mgcls = m
                        match_type = 'coordinate'

        # Verify coordinate match if we had a name match
        if matched_mgcls and cluster.ra != 0 and cluster.dec != 0:
            sep = angular_separation(cluster.ra, cluster.dec,
                                    matched_mgcls['ra'], matched_mgcls['dec'])
            if sep <= coord_threshold:
                if match_type == 'name':
                    match_type = 'both'

        if matched_mgcls:
            cluster.in_mgcls = True
            cluster.mgcls_name = matched_mgcls['name']
            cluster.match_type = match_type

    return lovoccs


def main(skip_sarao_api: bool = False, sarao_search_radius: float = 0.167):
    """
    Main verification function.

    Args:
        skip_sarao_api: If True, skip SARAO API queries (useful for testing)
        sarao_search_radius: Cone search radius in degrees (default ~10 arcmin)
    """
    base_path = Path(__file__).parent

    # Parse data files
    print("Loading MGCLS catalog...")
    mgcls_path = base_path / 'mgcls_table1.dat'
    if not mgcls_path.exists():
        print(f"Error: MGCLS data file not found at {mgcls_path}")
        print("Download from: https://cdsarc.cds.unistra.fr/ftp/J/A+A/657/A56/table1.dat")
        return

    mgcls_clusters = parse_mgcls_table(mgcls_path)
    print(f"  Loaded {len(mgcls_clusters)} MGCLS clusters")

    print("\nLoading LoVoCCS cluster list...")
    lovoccs_csv_path = base_path / 'Cluster Comparison.xlsx - Sheet2.csv'
    lovoccs_data = parse_lovoccs_csv(lovoccs_csv_path)
    lovoccs_clusters = [item[0] for item in lovoccs_data]
    citations = {item[0].name: item[1] for item in lovoccs_data}
    print(f"  Loaded {len(lovoccs_clusters)} LoVoCCS clusters")

    print("\nLoading LoVoCCS coordinates...")
    coords_path = base_path / 'lovoccs_meerkat_observable.csv'
    coords = parse_lovoccs_coordinates(coords_path)
    print(f"  Loaded coordinates for {len(coords)} clusters")

    # Add coordinates to clusters (try both original and normalized names)
    for cluster in lovoccs_clusters:
        if cluster.name in coords:
            cluster.ra, cluster.dec = coords[cluster.name]
        else:
            # Try normalized name
            normalized = normalize_coord_name(cluster.name)
            if normalized in coords:
                cluster.ra, cluster.dec = coords[normalized]

    # Match clusters against MGCLS
    print("\nMatching clusters against MGCLS...")
    lovoccs_clusters = match_clusters(lovoccs_clusters, mgcls_clusters)

    # Query SARAO API for ALL MeerKAT observations
    if not skip_sarao_api:
        print("\n" + "="*80)
        print("SARAO ARCHIVE API VERIFICATION")
        print("="*80)
        lovoccs_clusters = run_sarao_verification(lovoccs_clusters, radius=sarao_search_radius)
    else:
        print("\nSkipping SARAO API verification (--skip-sarao-api flag)")

    # Count SARAO results
    n_with_sarao_obs = sum(1 for c in lovoccs_clusters if c.sarao_observation_count > 0)

    # Find discrepancies - now considering SARAO API results
    discrepancies = []
    for cluster in lovoccs_clusters:
        if cluster.meerkat_flag is None:
            continue

        # A cluster has MeerKAT data if it's in MGCLS OR found in SARAO archive
        has_meerkat = cluster.in_mgcls or cluster.sarao_observation_count > 0
        expected = 1 if has_meerkat else 0
        actual = cluster.meerkat_flag

        if expected != actual:
            if has_meerkat:
                if cluster.in_mgcls:
                    issue = 'SHOULD BE 1 (in MGCLS)'
                else:
                    issue = f'SHOULD BE 1 ({cluster.sarao_observation_count} obs found in SARAO archive)'
            else:
                issue = 'CSV says 1 but NOT found in MGCLS or SARAO archive'

            discrepancies.append({
                'name': cluster.name,
                'ra': cluster.ra,
                'dec': cluster.dec,
                'csv_meerkat': actual,
                'in_mgcls': cluster.in_mgcls,
                'mgcls_name': cluster.mgcls_name,
                'match_type': cluster.match_type,
                'sarao_count': cluster.sarao_observation_count,
                'sarao_proposals': cluster.sarao_proposal_ids,
                'issue': issue
            })

    # Print results
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)

    # Summary stats
    n_in_mgcls = sum(1 for c in lovoccs_clusters if c.in_mgcls)
    n_marked_meerkat = sum(1 for c in lovoccs_clusters if c.meerkat_flag == 1)
    n_sarao_only = sum(1 for c in lovoccs_clusters
                       if c.sarao_observation_count > 0 and not c.in_mgcls)
    n_total_with_meerkat = sum(1 for c in lovoccs_clusters
                               if c.in_mgcls or c.sarao_observation_count > 0)

    print(f"\nLoVoCCS clusters in MGCLS: {n_in_mgcls}")
    print(f"LoVoCCS clusters with SARAO archive observations: {n_with_sarao_obs}")
    print(f"  - In MGCLS: {n_with_sarao_obs - n_sarao_only}")
    print(f"  - Non-MGCLS programs only: {n_sarao_only}")
    print(f"Total clusters with MeerKAT data (MGCLS + SARAO): {n_total_with_meerkat}")
    print(f"LoVoCCS clusters marked as having MeerKAT in CSV: {n_marked_meerkat}")
    print(f"Discrepancies found: {len(discrepancies)}")

    # Print matches
    print("\n" + "-"*80)
    print("CLUSTERS MATCHED TO MGCLS:")
    print("-"*80)
    for cluster in lovoccs_clusters:
        if cluster.in_mgcls:
            marker = " **MISMATCH**" if cluster.meerkat_flag != 1 else ""
            print(f"  {cluster.name:25} -> {cluster.mgcls_name:20} ({cluster.match_type}){marker}")

    # Print discrepancies
    if discrepancies:
        print("\n" + "-"*80)
        print("DISCREPANCIES (CSV MeerKAT flag vs MGCLS/SARAO):")
        print("-"*80)

        # Split into groups based on data sources
        should_be_1_mgcls = [d for d in discrepancies if d['in_mgcls'] and d['csv_meerkat'] == 0]
        should_be_1_sarao = [d for d in discrepancies if not d['in_mgcls'] and d['sarao_count'] > 0 and d['csv_meerkat'] == 0]
        shouldnt_be_1 = [d for d in discrepancies if not d['in_mgcls'] and d['sarao_count'] == 0 and d['csv_meerkat'] == 1]

        if should_be_1_mgcls:
            print("\nClusters marked 0 but ARE in MGCLS (should be 1):")
            for d in should_be_1_mgcls:
                print(f"  {d['name']:25} -> MGCLS: {d['mgcls_name']} ({d['match_type']})")

        if should_be_1_sarao:
            print("\nClusters marked 0 but FOUND in SARAO archive (should be 1):")
            for d in should_be_1_sarao:
                proposals = ', '.join(d['sarao_proposals'][:3])
                if len(d['sarao_proposals']) > 3:
                    proposals += '...'
                print(f"  {d['name']:25} -> {d['sarao_count']} observations ({proposals})")

        if shouldnt_be_1:
            print("\nClusters marked 1 but NOT found in MGCLS or SARAO archive:")
            for d in shouldnt_be_1:
                print(f"  {d['name']:25} (RA={d['ra']:.2f}, Dec={d['dec']:.2f})")

    # Save results to CSV
    output_path = base_path / 'meerkat_verification_results.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cluster_name', 'ra', 'dec', 'csv_meerkat_flag', 'in_mgcls',
                        'mgcls_name', 'match_type', 'sarao_obs_count', 'sarao_proposals',
                        'has_meerkat_data', 'discrepancy', 'notes'])

        for cluster in lovoccs_clusters:
            if cluster.meerkat_flag is None:
                continue

            has_meerkat = cluster.in_mgcls or cluster.sarao_observation_count > 0
            expected = 1 if has_meerkat else 0
            discrepancy = 'YES' if expected != cluster.meerkat_flag else ''
            notes = ''
            if discrepancy:
                if cluster.in_mgcls:
                    notes = 'Should be marked 1 - cluster is in MGCLS'
                elif cluster.sarao_observation_count > 0:
                    notes = f'Should be marked 1 - {cluster.sarao_observation_count} observations in SARAO archive'
                else:
                    notes = 'Marked 1 but not found in MGCLS or SARAO archive'

            writer.writerow([
                cluster.name,
                f"{cluster.ra:.4f}" if cluster.ra else '',
                f"{cluster.dec:.4f}" if cluster.dec else '',
                cluster.meerkat_flag,
                1 if cluster.in_mgcls else 0,
                cluster.mgcls_name or '',
                cluster.match_type or '',
                cluster.sarao_observation_count,
                ';'.join(cluster.sarao_proposal_ids) if cluster.sarao_proposal_ids else '',
                1 if has_meerkat else 0,
                discrepancy,
                notes
            ])

    print(f"\nResults saved to: {output_path}")

    # Also save MGCLS cluster list as CSV for reference
    mgcls_csv_path = base_path / 'mgcls_clusters.csv'
    with open(mgcls_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'ra', 'dec', 'alt_name', 'selection_type'])
        for m in mgcls_clusters:
            writer.writerow([m['name'], m['ra'], m['dec'], m['alt_name'] or '', m['selection']])

    print(f"MGCLS cluster list saved to: {mgcls_csv_path}")

    # Print summary of SARAO API findings
    if n_with_sarao_obs > 0:
        print("\n" + "-"*80)
        print("CLUSTERS WITH MEERKAT OBSERVATIONS (from SARAO API):")
        print("-"*80)
        print(f"\nFound {n_with_sarao_obs} LoVoCCS clusters with MeerKAT observations in SARAO archive.\n")

        # Show clusters found via SARAO that are NOT in MGCLS (i.e., other programs)
        non_mgcls_with_obs = [c for c in lovoccs_clusters
                              if c.sarao_observation_count > 0 and not c.in_mgcls]
        if non_mgcls_with_obs:
            print("Non-MGCLS clusters with MeerKAT observations:")
            for c in sorted(non_mgcls_with_obs, key=lambda x: -x.sarao_observation_count):
                proposals = ', '.join(c.sarao_proposal_ids[:3])
                if len(c.sarao_proposal_ids) > 3:
                    proposals += '...'
                csv_flag = "CSV=1" if c.meerkat_flag == 1 else "CSV=0 **FIX**"
                print(f"  {c.name:25} {c.sarao_observation_count:3d} obs  ({proposals})  [{csv_flag}]")

    # Print clusters marked as having MeerKAT but not found anywhere
    if shouldnt_be_1:
        print("\n" + "-"*80)
        print("CLUSTERS MARKED AS HAVING MEERKAT BUT NOT FOUND:")
        print("-"*80)
        print("\nThese clusters are marked MeerKAT=1 in CSV but were NOT found in")
        print("either MGCLS or SARAO archive. May need manual verification.\n")

        for d in sorted(shouldnt_be_1, key=lambda x: x['name']):
            print(f"  {d['name']:25} RA={d['ra']:8.4f}  Dec={d['dec']:8.4f}")

    # Create a CSV for any remaining clusters to check manually
    clusters_to_check = []

    if shouldnt_be_1:
        for d in shouldnt_be_1:
            clusters_to_check.append({
                'name': d['name'],
                'ra': d['ra'],
                'dec': d['dec'],
                'reason': 'CSV=1 but not found in MGCLS or SARAO archive'
            })

    # Also check clusters marked 0 that have citations but no SARAO observations
    clusters_with_citations_no_obs = [
        c for c in lovoccs_clusters
        if c.meerkat_flag == 0 and not c.in_mgcls and c.sarao_observation_count == 0
        and citations.get(c.name, '')
    ]

    if clusters_with_citations_no_obs:
        print("\n" + "-"*80)
        print("CLUSTERS WITH CITATIONS BUT NO MEERKAT OBSERVATIONS FOUND:")
        print("-"*80)
        print("\nThese clusters have citations but no MeerKAT data in MGCLS or SARAO.")
        print("Citations may reference other telescopes or planned observations.\n")

        for cluster in clusters_with_citations_no_obs:
            citation = citations.get(cluster.name, '')
            print(f"  {cluster.name:25} Citation: {citation[:60]}...")
            if cluster.ra and cluster.dec:
                clusters_to_check.append({
                    'name': cluster.name,
                    'ra': cluster.ra,
                    'dec': cluster.dec,
                    'reason': f'Has citation but no SARAO obs: {citation[:40]}'
                })

    # Save combined search CSV for manual follow-up
    if clusters_to_check:
        search_csv_path = base_path / 'meerkat_archive_search.csv'
        with open(search_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'ra', 'dec', 'reason'])
            for c in clusters_to_check:
                writer.writerow([c['name'], c['ra'], c['dec'], c['reason']])
        print(f"\nClusters for manual follow-up saved to: {search_csv_path}")
        print(f"Total clusters needing manual verification: {len(clusters_to_check)}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify MeerKAT observations for LoVoCCS clusters'
    )
    parser.add_argument(
        '--skip-sarao-api',
        action='store_true',
        help='Skip SARAO archive API queries (only check MGCLS)'
    )
    parser.add_argument(
        '--search-radius',
        type=float,
        default=0.167,
        help='Cone search radius in degrees (default: 0.167 = ~10 arcmin)'
    )

    args = parser.parse_args()
    main(skip_sarao_api=args.skip_sarao_api, sarao_search_radius=args.search_radius)

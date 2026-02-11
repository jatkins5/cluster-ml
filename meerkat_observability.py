#!/usr/bin/env python3
"""
Script to generate a list of LoVoCCS clusters observable by MeerKAT,
with columns indicating coverage by existing surveys (VLASS, LoTSS, FIRST, eROSITA, Parkes).
"""

import pandas as pd
import numpy as np


# MeerKAT observatory parameters
MEERKAT_LAT = -30.71  # degrees
MEERKAT_MIN_ELEVATION = 15  # degrees
MEERKAT_MAX_DEC = 90 - MEERKAT_MIN_ELEVATION + MEERKAT_LAT  # ~+44.29 degrees


def parse_lovoccs_csv(filename):
    """Parse the LoVoCCS CSV file to extract target info."""
    # Don't skip rows - let pandas use row 0 as header
    # The file has: ou, ID, &, name, &, ra(deg), &, dec(deg), ...
    df = pd.read_csv(filename)

    targets = []
    for idx, row in df.iterrows():
        if idx >= 107:  # Stop at summary rows
            break
        try:
            target_id = row.iloc[0]
            name = str(row.iloc[3]).strip()
            ra = float(row.iloc[5])
            dec = float(row.iloc[7])

            if not np.isnan(ra) and not np.isnan(dec) and name and name != 'nan':
                targets.append({
                    'id': int(target_id),
                    'name': name,
                    'ra': ra,
                    'dec': dec
                })
        except (ValueError, IndexError):
            continue

    return pd.DataFrame(targets)


def calculate_meerkat_observability(dec):
    """
    Calculate MeerKAT observability metrics for a given declination.

    Parameters:
    -----------
    dec : float
        Declination in degrees

    Returns:
    --------
    dict with:
        - meerkat_observable: bool (True if observable)
        - max_elevation_deg: float (maximum elevation achievable)
    """
    # Maximum elevation: 90 - |dec - lat|
    max_elevation = 90 - abs(dec - MEERKAT_LAT)

    # Observable if max elevation >= minimum elevation
    observable = max_elevation >= MEERKAT_MIN_ELEVATION

    return {
        'meerkat_observable': observable,
        'max_elevation_deg': round(max_elevation, 2)
    }


def load_survey_results():
    """Load all survey match CSV files."""
    surveys = {}

    # VLASS coverage
    try:
        vlass = pd.read_csv('vlass_coverage_results.csv')
        surveys['vlass'] = vlass[['id', 'in_vlass', 'n_sources']].rename(
            columns={'n_sources': 'n_vlass_sources'}
        )
    except FileNotFoundError:
        print("Warning: vlass_coverage_results.csv not found")
        surveys['vlass'] = None

    # LoTSS matches
    try:
        lotss = pd.read_csv('lovoccs_lotss_matches.csv')
        surveys['lotss'] = lotss[['id', 'has_lotss_match', 'n_lotss_sources']]
    except FileNotFoundError:
        print("Warning: lovoccs_lotss_matches.csv not found")
        surveys['lotss'] = None

    # FIRST matches
    try:
        first = pd.read_csv('lovoccs_first_matches.csv')
        surveys['first'] = first[['id', 'has_first_match', 'n_first_sources']]
    except FileNotFoundError:
        print("Warning: lovoccs_first_matches.csv not found")
        surveys['first'] = None

    # eROSITA matches
    try:
        erosita = pd.read_csv('lovoccs_erosita_matches.csv')
        surveys['erosita'] = erosita[['id', 'has_erosita_match', 'n_erosita_sources']]
    except FileNotFoundError:
        print("Warning: lovoccs_erosita_matches.csv not found")
        surveys['erosita'] = None

    # Parkes matches
    try:
        parkes = pd.read_csv('lovoccs_parkes_matches.csv')
        surveys['parkes'] = parkes[['id', 'has_parkes_match', 'n_parkes_sources']]
    except FileNotFoundError:
        print("Warning: lovoccs_parkes_matches.csv not found")
        surveys['parkes'] = None

    # Mappable clusters (Parkes imaging capability)
    try:
        mappable = pd.read_csv('mappable-clusters.csv')
        surveys['mappable'] = mappable[['cluster_name', 'can_make_image']].rename(
            columns={'cluster_name': 'name', 'can_make_image': 'parkes_can_map'}
        )
    except FileNotFoundError:
        print("Warning: mappable-clusters.csv not found")
        surveys['mappable'] = None

    return surveys


def merge_survey_data(base_df, surveys):
    """Merge all survey data onto the base dataframe."""
    result = base_df.copy()

    # Merge surveys that use 'id' as key
    for survey_name in ['vlass', 'lotss', 'first', 'erosita', 'parkes']:
        if surveys.get(survey_name) is not None:
            result = result.merge(surveys[survey_name], on='id', how='left')

    # Merge mappable clusters (uses 'name' as key)
    if surveys.get('mappable') is not None:
        result = result.merge(surveys['mappable'], on='name', how='left')

    return result


def count_surveys_with_data(row):
    """Count number of surveys with detections for a cluster."""
    count = 0

    if row.get('in_vlass', False):
        count += 1
    if row.get('has_lotss_match', False):
        count += 1
    if row.get('has_first_match', False):
        count += 1
    if row.get('has_erosita_match', False):
        count += 1
    if row.get('has_parkes_match', False):
        count += 1

    return count


def main():
    csv_file = "LoVoCCS_target_list - lovoccs.csv"
    output_file = "lovoccs_meerkat_observable.csv"

    print("Parsing LoVoCCS target list...")
    base_df = parse_lovoccs_csv(csv_file)
    print(f"Found {len(base_df)} targets")

    print("\nCalculating MeerKAT observability...")
    observability = base_df['dec'].apply(calculate_meerkat_observability)
    base_df['meerkat_observable'] = observability.apply(lambda x: x['meerkat_observable'])
    base_df['max_elevation_deg'] = observability.apply(lambda x: x['max_elevation_deg'])

    print("\nLoading survey results...")
    surveys = load_survey_results()

    print("Merging survey data...")
    result = merge_survey_data(base_df, surveys)

    # Fill NaN values for boolean columns
    bool_cols = ['in_vlass', 'has_lotss_match', 'has_first_match',
                 'has_erosita_match', 'has_parkes_match', 'parkes_can_map']
    for col in bool_cols:
        if col in result.columns:
            result[col] = result[col].fillna(False)

    # Fill NaN values for count columns
    count_cols = ['n_vlass_sources', 'n_lotss_sources', 'n_first_sources',
                  'n_erosita_sources', 'n_parkes_sources']
    for col in count_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    # Count surveys with data
    result['n_surveys_with_data'] = result.apply(count_surveys_with_data, axis=1)

    # Reorder columns for output
    output_cols = ['id', 'name', 'ra', 'dec', 'meerkat_observable', 'max_elevation_deg']

    # Add survey columns if present
    survey_cols = ['in_vlass', 'has_lotss_match', 'has_first_match',
                   'has_erosita_match', 'has_parkes_match', 'parkes_can_map',
                   'n_surveys_with_data']
    for col in survey_cols:
        if col in result.columns:
            output_cols.append(col)

    result = result[output_cols]

    # Save results
    result.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(result)
    observable = result['meerkat_observable'].sum()
    print(f"Total clusters: {total}")
    print(f"MeerKAT observable (dec < +{MEERKAT_MAX_DEC:.1f}): {observable} ({100*observable/total:.1f}%)")

    print(f"\nDeclination range: {result['dec'].min():.2f} to {result['dec'].max():.2f} deg")
    print(f"Max elevation range: {result['max_elevation_deg'].min():.1f} to {result['max_elevation_deg'].max():.1f} deg")

    # Survey coverage summary
    print("\nSurvey coverage (among MeerKAT-observable clusters):")
    obs_df = result[result['meerkat_observable']]

    if 'in_vlass' in result.columns:
        vlass_count = obs_df['in_vlass'].sum()
        print(f"  VLASS: {vlass_count} ({100*vlass_count/len(obs_df):.1f}%)")

    if 'has_lotss_match' in result.columns:
        lotss_count = obs_df['has_lotss_match'].sum()
        print(f"  LoTSS: {lotss_count} ({100*lotss_count/len(obs_df):.1f}%)")

    if 'has_first_match' in result.columns:
        first_count = obs_df['has_first_match'].sum()
        print(f"  FIRST: {first_count} ({100*first_count/len(obs_df):.1f}%)")

    if 'has_erosita_match' in result.columns:
        erosita_count = obs_df['has_erosita_match'].sum()
        print(f"  eROSITA: {erosita_count} ({100*erosita_count/len(obs_df):.1f}%)")

    if 'has_parkes_match' in result.columns:
        parkes_count = obs_df['has_parkes_match'].sum()
        print(f"  Parkes: {parkes_count} ({100*parkes_count/len(obs_df):.1f}%)")

    if 'parkes_can_map' in result.columns:
        parkes_map_count = obs_df['parkes_can_map'].sum()
        print(f"  Parkes mappable: {parkes_map_count} ({100*parkes_map_count/len(obs_df):.1f}%)")

    # Show clusters not observable by MeerKAT
    not_observable = result[~result['meerkat_observable']]
    if len(not_observable) > 0:
        print(f"\nClusters NOT observable by MeerKAT ({len(not_observable)}):")
        for _, row in not_observable.iterrows():
            print(f"  - {row['name']} (dec={row['dec']:.2f}, max_el={row['max_elevation_deg']:.1f})")


if __name__ == "__main__":
    main()

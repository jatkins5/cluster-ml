#!/usr/bin/env python3
"""
Download and display eROSITA image cutouts for LoVoCCS targets.

This script downloads X-ray images from the eROSITA All-Sky Survey
for galaxy clusters that have eROSITA matches.
"""

import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, LogStretch, AsinhStretch
from astroquery.skyview import SkyView
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def download_erosita_cutout(ra, dec, size=0.5, name="target"):
    """
    Download an eROSITA image cutout using SkyView.

    Parameters:
    -----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    size : float
        Image size in degrees (default 0.5 deg = 30 arcmin)
    name : str
        Target name for the filename

    Returns:
    --------
    HDU object with the image data
    """
    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')

    print(f"Downloading eROSITA cutout for {name}")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print(f"  Size: {size:.2f} degrees ({size*60:.1f} arcmin)")
    print()

    try:
        # Try to get eROSITA data from SkyView
        # Available surveys: eRASS1 (multiple bands)
        surveys_to_try = [
            'eRASS1 (0.2-0.6 keV)',  # Soft band
            'eRASS1 (0.6-2.3 keV)',  # Medium band
            'eRASS1 (2.3-5.0 keV)',  # Hard band
        ]

        print("Querying SkyView for eROSITA data...")
        for survey in surveys_to_try:
            try:
                print(f"  Trying survey: {survey}")
                hdu_list = SkyView.get_images(
                    position=coord,
                    survey=survey,
                    radius=size*u.degree,
                    pixels=512  # Good resolution
                )

                if hdu_list and len(hdu_list) > 0:
                    print(f"  ✓ Successfully retrieved {survey}")
                    hdu = hdu_list[0]

                    # Save the FITS file
                    band_name = survey.split('(')[1].split(')')[0].replace(' ', '_').replace('-', '_')
                    filename = f"erosita_{name.replace(' ', '_')}_{band_name}.fits"
                    hdu.writeto(filename, overwrite=True)

                    # Get some statistics
                    data = hdu[0].data
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        max_val = np.max(valid_data)
                        mean_val = np.mean(valid_data)
                        print(f"  Max counts: {max_val:.4e}, Mean counts: {mean_val:.4e}")

                    print(f"  Saved FITS file: {filename}")
                    return hdu, survey

            except Exception as e:
                print(f"  ✗ Failed for {survey}: {str(e)}")
                continue

        print("No eROSITA data found from SkyView")
        return None, None

    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def display_erosita_image(hdu, name="target", survey="eROSITA"):
    """
    Display an eROSITA X-ray image with proper WCS coordinates.

    Parameters:
    -----------
    hdu : HDU object
        FITS HDU containing the image data
    name : str
        Target name for the title
    survey : str
        Survey name for the title
    """
    # Extract data and WCS
    data = hdu[0].data
    header = hdu[0].header

    # Remove extra dimensions if needed
    data = np.squeeze(data)

    # Create WCS
    wcs = WCS(header)
    if wcs.naxis > 2:
        wcs = wcs.celestial

    # Create figure with WCS projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=wcs)

    # Replace zeros and negative values with NaN for better visualization
    data_plot = data.copy()
    data_plot[data_plot <= 0] = np.nan

    # Normalize the image with log stretch for X-ray data
    # X-ray images typically have high dynamic range
    try:
        norm = ImageNormalize(data_plot, stretch=AsinhStretch(a=0.1))
    except:
        # Fallback if normalization fails
        norm = None

    # Display the image with a X-ray appropriate colormap
    im = ax.imshow(data_plot, origin='lower', cmap='inferno', norm=norm, interpolation='gaussian')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Counts', fontsize=12)

    # Labels
    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)

    # Extract energy band from survey name
    if '(' in survey and ')' in survey:
        band = survey.split('(')[1].split(')')[0]
        title = f'eROSITA {band}: {name}'
    else:
        title = f'eROSITA: {name}'

    ax.set_title(title, fontsize=14, fontweight='bold')

    # Grid
    ax.grid(color='white', ls='--', alpha=0.3)

    # Add statistics text box
    valid_data = data_plot[~np.isnan(data_plot)]
    if len(valid_data) > 0:
        stats_text = f'Max: {np.max(valid_data):.2e}\nMean: {np.mean(valid_data):.2e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Save figure
    band_name = band.replace(' ', '_').replace('-', '_') if '(' in survey else 'xray'
    output_file = f"erosita_{name.replace(' ', '_')}_{band_name}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved image: {output_file}")

    plt.close()

def main():
    # Read the results file to find targets with eROSITA matches
    try:
        results = pd.read_csv("lovoccs_erosita_matches.csv")
    except FileNotFoundError:
        print("Error: lovoccs_erosita_matches.csv not found!")
        print("Please run match_lovoccs_erosita.py first.")
        return

    # Filter for targets with eROSITA matches
    matched = results[results['has_erosita_match'] == True].copy()

    if len(matched) == 0:
        print("No targets with eROSITA matches found!")
        return

    # Sort by number of sources
    matched = matched.sort_values('n_erosita_sources', ascending=False)

    print("LoVoCCS targets with eROSITA matches (sorted by number of sources):")
    print("=" * 80)
    for idx, row in matched.head(10).iterrows():
        sep_str = f"{row['closest_sep_arcmin']:.2f}'" if pd.notna(row['closest_sep_arcmin']) else "N/A"
        flux_str = f"{row['closest_flux_0.5_2keV']:.2e}" if pd.notna(row['closest_flux_0.5_2keV']) else "N/A"
        print(f"  {row['name']:30s} - {row['n_erosita_sources']:2d} sources, "
              f"closest: {sep_str}, flux: {flux_str}")
    print()

    # Select three targets: high, medium, and low source counts
    target_high = matched.iloc[0]

    # Medium: middle of the list
    target_med = matched.iloc[len(matched)//2] if len(matched) > 1 else target_high

    # Low: near the end
    target_low = matched.iloc[-1] if len(matched) > 2 else target_med

    targets = [
        (target_high, "high"),
        (target_med, "medium"),
        (target_low, "low")
    ]

    # Process each target
    for target, category in targets:
        print(f"\nProcessing {category} coverage target: {target['name']} "
              f"({target['n_erosita_sources']} eROSITA sources)")
        print("=" * 80)
        print()

        # Download and display
        result = download_erosita_cutout(
            ra=target['ra'],
            dec=target['dec'],
            size=0.5,  # 0.5 degrees = 30 arcmin
            name=target['name']
        )

        if result[0] is not None:
            hdu, survey = result
            display_erosita_image(hdu, name=target['name'], survey=survey)
        else:
            print(f"Could not retrieve eROSITA image for {target['name']}")

        print()

    print("\n" + "=" * 80)
    print("Image download complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download and display FIRST image cutouts for LoVoCCS targets.

This script downloads 1.4 GHz radio images from the FIRST survey
for galaxy clusters that have FIRST matches.
"""

import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch
from astroquery.image_cutouts.first import First
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def download_first_cutout(ra, dec, size=0.5, name="target"):
    """
    Download a FIRST image cutout using astroquery.

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

    # Convert size from degrees to arcmin
    size_arcmin = size * 60.0

    print(f"Downloading FIRST cutout for {name}")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print(f"  Size: {size:.2f} degrees ({size_arcmin:.1f} arcmin)")
    print()

    try:
        print("Querying FIRST archive...")
        hdu_list = First.get_images(coord, image_size=size_arcmin*u.arcmin)

        if hdu_list and len(hdu_list) > 0:
            print(f"  ✓ Successfully retrieved FIRST image")
            hdu = hdu_list[0]

            # Save the FITS file
            filename = f"first_{name.replace(' ', '_')}.fits"
            hdu.writeto(filename, overwrite=True)

            # Get some statistics
            data = hdu.data
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                max_val = np.max(valid_data)
                mean_val = np.mean(valid_data)
                print(f"  Max flux: {max_val:.4e} Jy/beam, Mean: {mean_val:.4e} Jy/beam")

            print(f"  Saved FITS file: {filename}")
            return hdu

        print("No FIRST data found")
        return None

    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def display_first_image(hdu, name="target"):
    """
    Display a FIRST radio image with proper WCS coordinates.

    Parameters:
    -----------
    hdu : HDU object
        FITS HDU containing the image data
    name : str
        Target name for the title
    """
    # Extract data and WCS
    data = hdu.data
    header = hdu.header

    # Remove extra dimensions if needed
    data = np.squeeze(data)

    # Create WCS
    wcs = WCS(header)
    if wcs.naxis > 2:
        wcs = wcs.celestial

    # Create figure with WCS projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=wcs)

    # Normalize the image with asinh stretch for radio data
    # Radio images typically have high dynamic range
    try:
        norm = ImageNormalize(data, stretch=AsinhStretch())
    except:
        # Fallback if normalization fails
        norm = None

    # Display the image with a radio-appropriate colormap
    im = ax.imshow(data, origin='lower', cmap='viridis', norm=norm, interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux (Jy/beam)', fontsize=12)

    # Labels
    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)
    ax.set_title(f'FIRST 1.4 GHz: {name}', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(color='white', ls='--', alpha=0.3)

    # Add statistics text box
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        stats_text = f'Max: {np.max(valid_data):.2e} Jy/beam\nMean: {np.mean(valid_data):.2e} Jy/beam'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Save figure
    output_file = f"first_{name.replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved image: {output_file}")

    plt.close()

def main():
    # Read the results file to find targets with FIRST matches
    try:
        results = pd.read_csv("lovoccs_first_matches.csv")
    except FileNotFoundError:
        print("Error: lovoccs_first_matches.csv not found!")
        print("Please run match_lovoccs_first.py first.")
        return

    # Filter for targets with FIRST matches
    matched = results[results['has_first_match'] == True].copy()

    if len(matched) == 0:
        print("No targets with FIRST matches found!")
        return

    # Sort by number of sources
    matched = matched.sort_values('n_first_sources', ascending=False)

    print("LoVoCCS targets with FIRST matches (sorted by number of sources):")
    print("=" * 80)
    for idx, row in matched.head(10).iterrows():
        sep_str = f"{row['closest_sep_arcmin']:.2f}'" if pd.notna(row['closest_sep_arcmin']) else "N/A"
        flux_str = f"{row['closest_int_flux_mJy']:.2f} mJy" if pd.notna(row['closest_int_flux_mJy']) else "N/A"
        extended_str = "extended" if row['closest_is_extended'] else "point-like"
        print(f"  {row['name']:30s} - {row['n_first_sources']:2d} sources, "
              f"closest: {sep_str} ({extended_str}), flux: {flux_str}")
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
              f"({target['n_first_sources']} FIRST sources)")
        print("=" * 80)
        print()

        # Download and display
        hdu = download_first_cutout(
            ra=target['ra'],
            dec=target['dec'],
            size=0.3,  # 0.3 degrees = 18 arcmin
            name=target['name']
        )

        if hdu is not None:
            display_first_image(hdu, name=target['name'])
        else:
            print(f"Could not retrieve FIRST image for {target['name']}")

        print()

    print("\n" + "=" * 80)
    print("Image download complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

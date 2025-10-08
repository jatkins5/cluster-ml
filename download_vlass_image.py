#!/usr/bin/env python3
"""
Download and display a VLASS image cutout for a LoVoCCS target.
"""

import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch
from astroquery.skyview import SkyView
import numpy as np

def download_vlass_cutout(ra, dec, size=0.5, name="target"):
    """
    Download a VLASS image cutout using the CIRADA cutout service.

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

    print(f"Downloading VLASS cutout for {name}")
    print(f"  Position: RA={ra:.4f}, Dec={dec:.4f}")
    print(f"  Size: {size:.2f} degrees ({size*60:.1f} arcmin)")
    print()

    # Use the CIRADA VLASS cutout service
    try:
        import requests
        from io import BytesIO

        # Convert size to arcminutes for the cutout service
        size_arcmin = size * 60

        # CIRADA cutout service URL
        url = f"https://cutouts.cirada.ca/vlass_cutout"
        params = {
            'ra': ra,
            'dec': dec,
            'size': size_arcmin,  # size in arcminutes
            'units': 'arcmin'
        }

        print("Requesting cutout from CIRADA service...")
        response = requests.get(url, params=params, timeout=60)

        if response.status_code == 200:
            # Load the FITS data
            hdu_list = fits.open(BytesIO(response.content))

            # Save the FITS file
            filename = f"vlass_{name.replace(' ', '_')}.fits"
            hdu_list.writeto(filename, overwrite=True)
            print(f"Saved FITS file: {filename}")
            return hdu_list
        else:
            print(f"Error: HTTP {response.status_code}")
            print("Trying alternative NVSS survey instead...")

            # Fallback to NVSS if VLASS not available
            hdu_list = SkyView.get_images(
                position=coord,
                survey='NVSS',
                pixels=1000,
                radius=size*u.degree
            )

            if hdu_list:
                filename = f"nvss_{name.replace(' ', '_')}.fits"
                hdu_list[0].writeto(filename, overwrite=True)
                print(f"Saved NVSS FITS file instead: {filename}")
                return hdu_list[0]
            return None

    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        print("Trying NVSS as fallback...")

        try:
            # Fallback to NVSS
            hdu_list = SkyView.get_images(
                position=coord,
                survey='NVSS',
                pixels=1000,
                radius=size*u.degree
            )

            if hdu_list:
                filename = f"nvss_{name.replace(' ', '_')}.fits"
                hdu_list[0].writeto(filename, overwrite=True)
                print(f"Saved NVSS FITS file: {filename}")
                return hdu_list[0]
        except Exception as e2:
            print(f"NVSS fallback also failed: {str(e2)}")

        return None

def display_vlass_image(hdu, name="target"):
    """
    Display a VLASS image with proper WCS coordinates.

    Parameters:
    -----------
    hdu : HDU object
        FITS HDU containing the image data
    name : str
        Target name for the title
    """
    # Extract data and WCS
    data = hdu[0].data
    wcs = WCS(hdu[0].header)

    # Remove NaN values for better visualization
    data = np.squeeze(data)  # Remove extra dimensions

    # Create figure with WCS projection
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=wcs)

    # Normalize the image with asinh stretch for better dynamic range
    norm = ImageNormalize(data, stretch=AsinhStretch())

    # Display the image
    im = ax.imshow(data, origin='lower', cmap='viridis', norm=norm)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux (Jy/beam)', fontsize=12)

    # Labels
    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)
    ax.set_title(f'VLASS Image: {name}', fontsize=14, fontweight='bold')

    # Grid
    ax.grid(color='white', ls='--', alpha=0.3)

    # Save figure
    output_file = f"vlass_{name.replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved image: {output_file}")

    plt.show()

def main():
    # Read the results file to find targets with VLASS coverage
    results = pd.read_csv("vlass_coverage_results.csv")

    # Filter for targets in VLASS
    in_vlass = results[results['in_vlass'] == True].sort_values('n_obs', ascending=False)

    print("LoVoCCS targets in VLASS (sorted by number of observations):")
    print("=" * 70)
    for idx, row in in_vlass.head(10).iterrows():
        print(f"  {row['name']:30s} - {row['n_obs']:2d} observations")
    print()

    # Select three targets: high, medium, and low observation counts
    # High: first one
    target_high = in_vlass.iloc[0]

    # Medium: middle of the list
    target_med = in_vlass.iloc[len(in_vlass)//2]

    # Low: near the end (but not the absolute last to avoid single observation)
    low_obs = in_vlass[in_vlass['n_obs'] <= 3]
    if len(low_obs) > 0:
        target_low = low_obs.iloc[len(low_obs)//2]
    else:
        target_low = in_vlass.iloc[-3]

    targets = [
        (target_high, "high"),
        (target_med, "medium"),
        (target_low, "low")
    ]

    for target, category in targets:
        print(f"\nProcessing {category} coverage target: {target['name']} ({target['n_obs']} observations)")
        print("=" * 70)
        print()

        # Download and display
        hdu = download_vlass_cutout(
            ra=target['ra'],
            dec=target['dec'],
            size=0.3,  # 0.3 degrees = 18 arcmin
            name=target['name']
        )

        if hdu:
            display_vlass_image(hdu, name=target['name'])

        print("\n")

if __name__ == "__main__":
    main()

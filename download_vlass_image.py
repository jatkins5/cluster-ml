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
from astroquery.cadc import Cadc
import numpy as np

def download_vlass_cutout(ra, dec, size=0.5, name="target"):
    """
    Download a VLASS image cutout using CADC.

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

    try:
        cadc = Cadc()

        # Query for VLASS images at this position
        print("Querying CADC for VLASS data...")
        result = cadc.query_region(
            coord,
            collection='VLASS',
            radius=size*u.degree
        )

        if result is None or len(result) == 0:
            print("No VLASS data found at this position")
            return None

        print(f"Found {len(result)} VLASS observations")
        print(f"Available columns: {result.colnames[:10]}...")

        # Get the first observation
        obs = result[0]
        print(f"\nUsing observation: {obs.get('observationID', 'N/A')}")

        # Get images
        print("Downloading images...")
        cutout_hdu_list = cadc.get_images(
            coord,
            radius=size*u.degree,
            collection='VLASS'
        )

        if cutout_hdu_list and len(cutout_hdu_list) > 0:
            print(f"Downloaded {len(cutout_hdu_list)} image(s)")

            # Find the image that actually contains our target
            best_hdu = None
            best_flux = -1

            for i, hdu in enumerate(cutout_hdu_list):
                data = np.squeeze(hdu[0].data)
                wcs_full = WCS(hdu[0].header)
                wcs_2d = wcs_full.celestial if wcs_full.naxis > 2 else wcs_full

                # Check if target is in this image
                target_pix = wcs_2d.world_to_pixel_values(ra, dec)
                ny, nx = data.shape
                in_image = (0 <= target_pix[0] < nx) and (0 <= target_pix[1] < ny)

                if in_image:
                    max_flux = np.nanmax(data)
                    obs_id = hdu[0].header.get('OBJECT', f'image_{i}')
                    print(f"  Image {i+1} ({obs_id}): contains target, max flux = {max_flux:.6f} Jy/beam")

                    # Keep the image with highest max flux
                    if max_flux > best_flux:
                        best_flux = max_flux
                        best_hdu = hdu

            if best_hdu is not None:
                # Save the FITS file
                filename = f"vlass_{name.replace(' ', '_')}.fits"
                best_hdu.writeto(filename, overwrite=True)
                print(f"\nSaved FITS file: {filename} (max flux: {best_flux:.6f} Jy/beam)")
                return best_hdu
            else:
                print("Warning: No image contains the target position!")
                # Fall back to first image
                filename = f"vlass_{name.replace(' ', '_')}.fits"
                cutout_hdu_list[0].writeto(filename, overwrite=True)
                print(f"Saved FITS file: {filename} (using first image)")
                return cutout_hdu_list[0]
        else:
            print("No image data returned")
            return None

    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        import traceback
        traceback.print_exc()
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
    header = hdu[0].header

    # Remove extra dimensions (e.g., frequency and Stokes)
    data = np.squeeze(data)  # Remove extra dimensions

    # Create a 2D WCS from the full WCS
    # VLASS images often have 4 dimensions (RA, Dec, Frequency, Stokes)
    # We need to slice to just the spatial dimensions
    full_wcs = WCS(header)
    if full_wcs.naxis > 2:
        # Drop the extra axes (typically frequency and Stokes)
        wcs = full_wcs.celestial
    else:
        wcs = full_wcs

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

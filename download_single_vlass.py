#!/usr/bin/env python3
"""
Download a single VLASS image for a specific target.
Usage: python download_single_vlass.py <target_name> <ra> <dec> <n_sources>
Example: python download_single_vlass.py A2443 336.51 17.38 10
"""

import sys
from astroquery.cadc import Cadc
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch

def download_vlass_image(name, ra, dec, n_sources, size=0.3):
    """Download and display a VLASS image for a single target."""

    coord = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    cadc = Cadc()

    print(f'Downloading VLASS data for {name} ({n_sources} catalog sources)...')
    print(f'Position: RA={ra:.4f}, Dec={dec:.4f}')
    print()

    # Download images
    images = cadc.get_images(coord, radius=size*u.degree, collection='VLASS')
    print(f'Downloaded {len(images)} image(s)')

    # Find best image containing target
    best_hdu = None
    best_flux = -1

    for i, hdu in enumerate(images):
        data = np.squeeze(hdu[0].data)
        wcs_full = WCS(hdu[0].header)
        wcs_2d = wcs_full.celestial if wcs_full.naxis > 2 else wcs_full

        target_pix = wcs_2d.world_to_pixel_values(ra, dec)
        ny, nx = data.shape
        in_image = (0 <= target_pix[0] < nx) and (0 <= target_pix[1] < ny)

        if in_image:
            max_flux = np.nanmax(data)
            obs_id = hdu[0].header.get('OBJECT', f'image_{i}')
            print(f'  Image {i+1} ({obs_id}): contains target, max flux = {max_flux:.6f} Jy/beam')

            if max_flux > best_flux:
                best_flux = max_flux
                best_hdu = hdu

    if best_hdu is None:
        print('ERROR: No image contains the target position!')
        return False

    # Save FITS
    safe_name = name.replace(' ', '_').replace('+', 'p')
    filename = f'vlass_{safe_name}.fits'
    best_hdu.writeto(filename, overwrite=True)
    print(f'\nSaved: {filename} (max flux: {best_flux:.6f} Jy/beam)')

    # Create image
    data = np.squeeze(best_hdu[0].data)
    wcs_full = WCS(best_hdu[0].header)
    wcs = wcs_full.celestial if wcs_full.naxis > 2 else wcs_full

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection=wcs)

    norm = ImageNormalize(data, stretch=AsinhStretch())
    im = ax.imshow(data, origin='lower', cmap='viridis', norm=norm)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flux (Jy/beam)', fontsize=12)

    ax.set_xlabel('RA (J2000)', fontsize=12)
    ax.set_ylabel('Dec (J2000)', fontsize=12)
    ax.set_title(f'VLASS Image: {name}', fontsize=14, fontweight='bold')
    ax.grid(color='white', ls='--', alpha=0.3)

    output_file = f'vlass_{safe_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_file}')

    return True

def main():
    if len(sys.argv) < 5:
        print("Usage: python download_single_vlass.py <target_name> <ra> <dec> <n_sources>")
        print("Example: python download_single_vlass.py A2443 336.51 17.38 10")
        print()
        print("Suggested targets with ~10 catalog sources:")
        print("  A2443          336.51  17.38  10")
        print("  A3532          194.32 -30.38  10")
        print("  A4068          359.98 -39.48  10")
        print("  RXC_J1217+0339 184.42   3.66   9")
        print("  A3528          193.67 -29.22   9")
        sys.exit(1)

    name = sys.argv[1]
    ra = float(sys.argv[2])
    dec = float(sys.argv[3])
    n_sources = int(sys.argv[4])

    success = download_vlass_image(name, ra, dec, n_sources)

    if success:
        print("\n✓ Download complete!")
    else:
        print("\n✗ Download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()

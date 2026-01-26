#!/usr/bin/env python3
"""
Assemble Parkes single-dish spectra into spatial images.

This script takes RPFITS files from multiple sky positions and grids them
onto a regular RA/Dec pixel grid to create spatial maps. Each RPFITS file
contains spectrum data at a single pointing position.

Usage:
    python assemble_parkes_images.py parkes_data/A780/ --scan-only
    python assemble_parkes_images.py parkes_data/A780/ --output hydra_a.fits
    python assemble_parkes_images.py parkes_data/A780/ --output hydra_a_cube.fits --mode cube
"""

import numpy as np
import pandas as pd
import os
import sys
import shutil
import tempfile
import subprocess
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datetime import datetime

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import AsinhStretch, ImageNormalize
import matplotlib.pyplot as plt


@dataclass
class ParkesObservation:
    """Data from a single Parkes observation."""
    file_path: str
    source_name: str
    ra_deg: float
    dec_deg: float
    freq_mhz: float
    bandwidth_mhz: float
    n_channels: int
    n_time_samples: int
    spectrum: Optional[np.ndarray] = None  # [nchan] averaged spectrum
    frequencies: Optional[np.ndarray] = None  # [nchan] in Hz


@dataclass
class CoverageAnalysis:
    """Analysis of spatial coverage."""
    n_observations: int
    n_unique_positions: int
    ra_range: Tuple[float, float]
    dec_range: Tuple[float, float]
    spatial_extent_arcmin: float
    center_ra: float
    center_dec: float
    freq_mhz: float
    beam_size_arcmin: float
    can_make_image: bool
    reason: str
    observations: List[ParkesObservation]


def extract_rpfits_metadata_subprocess(rpf_file: str, output_pickle: str) -> bool:
    """
    Extract metadata from RPFITS file using a subprocess.

    This runs casatools in a separate process to avoid segfaults from
    affecting the main process.
    """
    script = f'''
import sys
import os
import shutil
import tempfile
import pickle
import numpy as np

# Suppress casatools logging
os.environ["CASATOOLS_LOG_LEVEL"] = "SEVERE"

import casatools

rpf_file = {repr(rpf_file)}
output_file = {repr(output_pickle)}

try:
    # Create temporary MS
    with tempfile.TemporaryDirectory() as tmpdir:
        ms_file = os.path.join(tmpdir, "temp.ms")

        # Convert RPFITS to MS
        af = casatools.atcafiller()
        result = af.open(msname=ms_file, filenames=[rpf_file])
        if not result:
            raise RuntimeError("Failed to open")
        result = af.fill()
        if not result:
            raise RuntimeError("Failed to fill")

        # Extract metadata
        tb = casatools.table()
        metadata = {{}}

        # Get position from FIELD table
        tb.open(ms_file + '/FIELD')
        phase_dir = tb.getcol('PHASE_DIR')
        metadata['source_name'] = tb.getcol('NAME')[0]
        metadata['ra_deg'] = float(np.degrees(phase_dir[0, 0, 0]))
        metadata['dec_deg'] = float(np.degrees(phase_dir[1, 0, 0]))
        tb.close()

        # Get frequency from SPECTRAL_WINDOW
        tb.open(ms_file + '/SPECTRAL_WINDOW')
        metadata['freq_hz'] = float(tb.getcol('REF_FREQUENCY')[0])
        metadata['bandwidth_hz'] = float(tb.getcol('TOTAL_BANDWIDTH')[0])
        metadata['n_channels'] = int(tb.getcol('NUM_CHAN')[0])
        chan_freq = tb.getcol('CHAN_FREQ')[:, 0]
        metadata['frequencies'] = chan_freq.tolist()
        tb.close()

        # Get data shape from main table
        tb.open(ms_file)
        metadata['n_rows'] = int(tb.nrows())
        data = tb.getcol('DATA')  # [npol, nchan, nrow]
        metadata['data_shape'] = list(data.shape)

        # Compute averaged spectrum
        spectrum = np.abs(data).mean(axis=(0, 2))
        metadata['spectrum'] = spectrum.tolist()
        tb.close()

        # Save to pickle
        with open(output_file, 'wb') as f:
            pickle.dump(metadata, f)

    sys.exit(0)

except Exception as e:
    # Write error to pickle
    with open(output_file, 'wb') as f:
        pickle.dump({{'error': str(e)}}, f)
    sys.exit(1)
'''

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        timeout=60
    )

    return result.returncode == 0


def extract_observation_from_file(rpf_file: str) -> Optional[dict]:
    """
    Extract observation metadata from an RPFITS file.

    Uses subprocess isolation to handle casatools instability.
    """
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        output_pickle = f.name

    try:
        success = extract_rpfits_metadata_subprocess(rpf_file, output_pickle)

        if os.path.exists(output_pickle):
            with open(output_pickle, 'rb') as f:
                metadata = pickle.load(f)

            if 'error' in metadata:
                return None

            # Convert lists back to numpy arrays
            metadata['spectrum'] = np.array(metadata['spectrum'])
            metadata['frequencies'] = np.array(metadata['frequencies'])
            return metadata

        return None

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
    finally:
        if os.path.exists(output_pickle):
            os.unlink(output_pickle)


def scan_rpfits_directory(data_dir: str, verbose: bool = True) -> List[ParkesObservation]:
    """
    Scan a directory for RPFITS files and extract metadata.

    Uses subprocess isolation to handle casatools instability.

    Parameters:
    -----------
    data_dir : str
        Directory containing RPFITS files
    verbose : bool
        Print progress

    Returns:
    --------
    list : List of ParkesObservation objects
    """
    data_path = Path(data_dir)
    rpf_files = sorted(data_path.glob('*.rpf'))

    if not rpf_files:
        print(f"No .rpf files found in {data_dir}")
        return []

    if verbose:
        print(f"Found {len(rpf_files)} RPFITS files in {data_dir}")
        print()

    observations = []

    for i, rpf_file in enumerate(rpf_files, 1):
        if verbose:
            print(f"[{i:3d}/{len(rpf_files)}] {rpf_file.name}... ", end="", flush=True)

        try:
            metadata = extract_observation_from_file(str(rpf_file))

            if metadata is None:
                if verbose:
                    print("FAILED (subprocess error)")
                continue

            obs = ParkesObservation(
                file_path=str(rpf_file),
                source_name=metadata['source_name'],
                ra_deg=metadata['ra_deg'],
                dec_deg=metadata['dec_deg'],
                freq_mhz=metadata['freq_hz'] / 1e6,
                bandwidth_mhz=metadata['bandwidth_hz'] / 1e6,
                n_channels=metadata['n_channels'],
                n_time_samples=metadata['n_rows'],
                spectrum=metadata['spectrum'],
                frequencies=metadata['frequencies']
            )
            observations.append(obs)

            if verbose:
                print(f"RA={obs.ra_deg:.4f}, Dec={obs.dec_deg:.4f}, "
                      f"Freq={obs.freq_mhz:.1f} MHz")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")

    return observations


def analyze_spatial_coverage(observations: List[ParkesObservation],
                             position_tolerance_arcmin: float = 1.0,
                             min_positions: int = 5) -> CoverageAnalysis:
    """
    Analyze spatial coverage of observations.

    Parameters:
    -----------
    observations : list
        List of ParkesObservation objects
    position_tolerance_arcmin : float
        Positions within this tolerance are considered the same
    min_positions : int
        Minimum unique positions for mapping

    Returns:
    --------
    CoverageAnalysis : Analysis results
    """
    if not observations:
        return CoverageAnalysis(
            n_observations=0,
            n_unique_positions=0,
            ra_range=(0, 0),
            dec_range=(0, 0),
            spatial_extent_arcmin=0,
            center_ra=0,
            center_dec=0,
            freq_mhz=0,
            beam_size_arcmin=0,
            can_make_image=False,
            reason="No observations",
            observations=[]
        )

    # Extract positions
    ras = np.array([o.ra_deg for o in observations])
    decs = np.array([o.dec_deg for o in observations])
    freqs = np.array([o.freq_mhz for o in observations])

    # Find unique positions
    tolerance_deg = position_tolerance_arcmin / 60.0
    unique_positions = []

    for ra, dec in zip(ras, decs):
        is_unique = True
        for ura, udec in unique_positions:
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

    center_ra = np.mean(ras)
    center_dec = np.mean(decs)

    # Extent in arcminutes
    ra_extent = (ra_range[1] - ra_range[0]) * 60 * np.cos(np.radians(center_dec))
    dec_extent = (dec_range[1] - dec_range[0]) * 60
    spatial_extent = np.sqrt(ra_extent**2 + dec_extent**2)

    # Beam size (Parkes: ~14.4 arcmin at 1 GHz)
    mean_freq_ghz = np.mean(freqs) / 1000
    beam_size_arcmin = 14.4 / mean_freq_ghz if mean_freq_ghz > 0 else 14.4

    # Determine if mapping is possible
    can_make_image = n_unique >= min_positions

    if not can_make_image:
        if n_unique == 1:
            reason = f"Only 1 unique position (need {min_positions}+ for mapping)"
        else:
            reason = f"Only {n_unique} unique positions (need {min_positions}+ for mapping)"
    else:
        reason = f"{n_unique} unique positions over {spatial_extent:.1f} arcmin"

    return CoverageAnalysis(
        n_observations=len(observations),
        n_unique_positions=n_unique,
        ra_range=ra_range,
        dec_range=dec_range,
        spatial_extent_arcmin=spatial_extent,
        center_ra=center_ra,
        center_dec=center_dec,
        freq_mhz=np.mean(freqs),
        beam_size_arcmin=beam_size_arcmin,
        can_make_image=can_make_image,
        reason=reason,
        observations=observations
    )


def create_wcs_header(center_ra: float, center_dec: float,
                      pixel_size_arcmin: float,
                      nx: int, ny: int,
                      freq_hz: float = None) -> WCS:
    """Create a WCS header for the output image."""
    w = WCS(naxis=2)

    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.cdelt = [-pixel_size_arcmin / 60, pixel_size_arcmin / 60]
    w.wcs.crval = [center_ra, center_dec]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.cunit = ['deg', 'deg']

    return w


def grid_spectra_to_image(coverage: CoverageAnalysis,
                          pixel_size_arcmin: float = None,
                          output_mode: str = 'continuum') -> Tuple[np.ndarray, WCS, dict]:
    """
    Grid spectra onto a regular pixel grid.

    Parameters:
    -----------
    coverage : CoverageAnalysis
        Coverage analysis with observations
    pixel_size_arcmin : float
        Pixel size in arcminutes. If None, uses beam_size / 3
    output_mode : str
        'continuum' for 2D image, 'cube' for 3D datacube

    Returns:
    --------
    tuple : (data array, WCS, metadata dict)
    """
    if not coverage.can_make_image:
        raise ValueError(f"Cannot create image: {coverage.reason}")

    observations = coverage.observations

    # Default pixel size: beam / 3 (Nyquist-ish sampling)
    if pixel_size_arcmin is None:
        pixel_size_arcmin = coverage.beam_size_arcmin / 3

    # Calculate image dimensions
    ra_extent_arcmin = (coverage.ra_range[1] - coverage.ra_range[0]) * 60 * np.cos(np.radians(coverage.center_dec))
    dec_extent_arcmin = (coverage.dec_range[1] - coverage.dec_range[0]) * 60

    # Add padding (1 beam on each side)
    padding = coverage.beam_size_arcmin
    nx = int(np.ceil((ra_extent_arcmin + 2 * padding) / pixel_size_arcmin))
    ny = int(np.ceil((dec_extent_arcmin + 2 * padding) / pixel_size_arcmin))

    # Minimum size
    nx = max(nx, 10)
    ny = max(ny, 10)

    # Create WCS
    wcs = create_wcs_header(
        coverage.center_ra,
        coverage.center_dec,
        pixel_size_arcmin,
        nx, ny
    )

    # Initialize data and weight arrays
    if output_mode == 'continuum':
        data = np.zeros((ny, nx))
        weights = np.zeros((ny, nx))
    else:  # cube
        # Get common frequency grid from first observation
        n_chan = observations[0].n_channels
        data = np.zeros((n_chan, ny, nx))
        weights = np.zeros((n_chan, ny, nx))

    # Gaussian beam kernel (FWHM to sigma)
    beam_sigma_arcmin = coverage.beam_size_arcmin / 2.355

    # Grid each observation
    for obs in observations:
        # Convert observation position to pixel
        px, py = wcs.world_to_pixel_values(obs.ra_deg, obs.dec_deg)
        px, py = int(np.round(float(px))), int(np.round(float(py)))

        # Skip if outside image bounds
        if px < 0 or px >= nx or py < 0 or py >= ny:
            continue

        # Get spectrum value
        if output_mode == 'continuum':
            # Collapse frequency axis
            value = np.mean(obs.spectrum) if obs.spectrum is not None else 1.0

            # Apply Gaussian beam weighting to nearby pixels
            beam_radius_pix = int(np.ceil(2 * coverage.beam_size_arcmin / pixel_size_arcmin))

            for dy in range(-beam_radius_pix, beam_radius_pix + 1):
                for dx in range(-beam_radius_pix, beam_radius_pix + 1):
                    ix, iy = px + dx, py + dy
                    if 0 <= ix < nx and 0 <= iy < ny:
                        # Distance in arcminutes
                        dist_arcmin = np.sqrt(dx**2 + dy**2) * pixel_size_arcmin
                        # Gaussian weight
                        weight = np.exp(-0.5 * (dist_arcmin / beam_sigma_arcmin)**2)
                        data[iy, ix] += value * weight
                        weights[iy, ix] += weight
        else:
            # Cube mode - preserve frequency axis
            if obs.spectrum is not None:
                beam_radius_pix = int(np.ceil(2 * coverage.beam_size_arcmin / pixel_size_arcmin))

                for dy in range(-beam_radius_pix, beam_radius_pix + 1):
                    for dx in range(-beam_radius_pix, beam_radius_pix + 1):
                        ix, iy = px + dx, py + dy
                        if 0 <= ix < nx and 0 <= iy < ny:
                            dist_arcmin = np.sqrt(dx**2 + dy**2) * pixel_size_arcmin
                            weight = np.exp(-0.5 * (dist_arcmin / beam_sigma_arcmin)**2)
                            # Ensure spectrum length matches
                            spec = obs.spectrum[:n_chan] if len(obs.spectrum) >= n_chan else np.pad(obs.spectrum, (0, n_chan - len(obs.spectrum)))
                            data[:, iy, ix] += spec * weight
                            weights[:, iy, ix] += weight

    # Normalize by weights
    with np.errstate(divide='ignore', invalid='ignore'):
        data = np.where(weights > 0, data / weights, 0)

    # Metadata
    metadata = {
        'n_observations': len(observations),
        'n_unique_positions': coverage.n_unique_positions,
        'beam_size_arcmin': coverage.beam_size_arcmin,
        'pixel_size_arcmin': pixel_size_arcmin,
        'freq_mhz': coverage.freq_mhz,
        'spatial_extent_arcmin': coverage.spatial_extent_arcmin,
        'output_mode': output_mode
    }

    return data, wcs, metadata


def write_fits_image(data: np.ndarray, wcs: WCS, output_path: str,
                     metadata: dict):
    """Write image data to a FITS file with proper headers."""
    header = wcs.to_header()

    # Add metadata
    header['BUNIT'] = 'arbitrary'
    header['TELESCOP'] = 'Parkes'
    header['INSTRUME'] = 'Single-dish'
    header['NOBS'] = (metadata['n_observations'], 'Number of observations')
    header['NPOS'] = (metadata['n_unique_positions'], 'Number of unique positions')
    header['BMAJ'] = (metadata['beam_size_arcmin'] / 60, 'Beam FWHM (deg)')
    header['BMIN'] = (metadata['beam_size_arcmin'] / 60, 'Beam FWHM (deg)')
    header['PIXSCALE'] = (metadata['pixel_size_arcmin'], 'Pixel scale (arcmin)')
    header['FREQ'] = (metadata['freq_mhz'] * 1e6, 'Reference frequency (Hz)')
    header['DATE'] = (datetime.utcnow().isoformat(), 'Creation date')
    header['HISTORY'] = 'Created by assemble_parkes_images.py'

    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(output_path, overwrite=True)


def create_visualization(data: np.ndarray, wcs: WCS, output_path: str,
                         metadata: dict, title: str = None):
    """Create a PNG visualization of the image."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection=wcs)

    # Handle cube vs 2D
    if data.ndim == 3:
        # Collapse to 2D for visualization
        plot_data = np.mean(data, axis=0)
    else:
        plot_data = data

    # Normalize with asinh stretch
    valid_data = plot_data[plot_data > 0]
    if len(valid_data) > 0:
        norm = ImageNormalize(plot_data, stretch=AsinhStretch(),
                              vmin=np.percentile(valid_data, 1),
                              vmax=np.percentile(valid_data, 99))
    else:
        norm = None

    im = ax.imshow(plot_data, origin='lower', cmap='viridis', norm=norm)

    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Parkes Map ({metadata["n_observations"]} obs, '
                     f'{metadata["freq_mhz"]:.0f} MHz)')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Intensity (arbitrary)')

    # Add beam indicator
    beam_size_deg = metadata['beam_size_arcmin'] / 60
    beam_circle = plt.Circle((0.1, 0.1), beam_size_deg / 2,
                             transform=ax.get_transform('world'),
                             fill=False, color='white', linewidth=2)
    # Note: This won't show correctly without more work, just indicating position

    # Info text
    info_text = (f"Beam: {metadata['beam_size_arcmin']:.1f}'\n"
                 f"Pixel: {metadata['pixel_size_arcmin']:.1f}'\n"
                 f"Positions: {metadata['n_unique_positions']}")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            color='white')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Assemble Parkes spectra into spatial images'
    )
    parser.add_argument('data_dir', type=str,
                       help='Directory containing RPFITS files')
    parser.add_argument('--scan-only', action='store_true',
                       help='Only scan and report coverage, do not create image')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output FITS file path')
    parser.add_argument('--mode', type=str, default='continuum',
                       choices=['continuum', 'cube'],
                       help='Output mode: continuum (2D) or cube (3D)')
    parser.add_argument('--pixel-size', type=float, default=None,
                       help='Pixel size in arcminutes (default: beam/3)')
    parser.add_argument('--min-positions', type=int, default=5,
                       help='Minimum unique positions for mapping')
    parser.add_argument('--no-png', action='store_true',
                       help='Do not create PNG visualization')

    args = parser.parse_args()

    print("=" * 80)
    print("Parkes Image Assembly")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Mode: {args.mode}")
    if args.pixel_size:
        print(f"Pixel size: {args.pixel_size} arcmin")
    print()

    # Scan directory
    print("Scanning RPFITS files...")
    print("-" * 80)
    observations = scan_rpfits_directory(args.data_dir)

    if not observations:
        print("\nNo valid observations found.")
        sys.exit(1)

    # Analyze coverage
    print("\n" + "-" * 80)
    print("Analyzing spatial coverage...")
    coverage = analyze_spatial_coverage(
        observations,
        min_positions=args.min_positions
    )

    print(f"\nCoverage Analysis:")
    print(f"  Observations:      {coverage.n_observations}")
    print(f"  Unique positions:  {coverage.n_unique_positions}")
    print(f"  RA range:          {coverage.ra_range[0]:.4f} to {coverage.ra_range[1]:.4f} deg")
    print(f"  Dec range:         {coverage.dec_range[0]:.4f} to {coverage.dec_range[1]:.4f} deg")
    print(f"  Spatial extent:    {coverage.spatial_extent_arcmin:.1f} arcmin")
    print(f"  Mean frequency:    {coverage.freq_mhz:.1f} MHz")
    print(f"  Beam size:         {coverage.beam_size_arcmin:.1f} arcmin")
    print(f"  Can make image:    {'YES' if coverage.can_make_image else 'NO'}")
    print(f"  Status:            {coverage.reason}")

    if args.scan_only:
        print("\n(Scan only mode - not creating image)")
        sys.exit(0)

    if not coverage.can_make_image:
        print(f"\nCannot create image: {coverage.reason}")
        print("Suggestion: Use convert_rpfits_to_png.py for individual spectra")
        sys.exit(1)

    # Create image
    print("\n" + "-" * 80)
    print("Creating image...")

    data, wcs, metadata = grid_spectra_to_image(
        coverage,
        pixel_size_arcmin=args.pixel_size,
        output_mode=args.mode
    )

    print(f"  Image shape: {data.shape}")
    print(f"  Pixel size:  {metadata['pixel_size_arcmin']:.2f} arcmin")

    # Output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.data_dir) / f"parkes_map_{args.mode}.fits"

    print(f"\nWriting FITS: {output_path}")
    write_fits_image(data, wcs, str(output_path), metadata)

    # PNG visualization
    if not args.no_png:
        png_path = output_path.with_suffix('.png')
        print(f"Writing PNG:  {png_path}")
        create_visualization(data, wcs, str(png_path), metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()

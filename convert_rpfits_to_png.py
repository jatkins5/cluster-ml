#!/usr/bin/env python3
"""
Convert Parkes RPFITS files to PNG images.

This script uses CASA tools to read RPFITS files and generate
spectrum plots as PNG images.

Usage:
    python convert_rpfits_to_png.py <rpfits_file>
    python convert_rpfits_to_png.py parkes_data/A780/*.rpf
"""

import casatools
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil
from pathlib import Path


def rpfits_to_ms(rpf_file, ms_file):
    """Convert RPFITS file to CASA MeasurementSet."""
    af = casatools.atcafiller()

    # Remove existing ms if present
    if os.path.exists(ms_file):
        shutil.rmtree(ms_file)

    result = af.open(msname=ms_file, filenames=[rpf_file])
    if not result:
        raise RuntimeError(f"Failed to open {rpf_file}")

    result = af.fill()
    if not result:
        raise RuntimeError(f"Failed to fill MS from {rpf_file}")

    return True


def extract_spectrum(ms_file):
    """Extract spectrum data from MeasurementSet."""
    ms = casatools.ms()
    ms.open(ms_file)

    # Get data and frequency info
    metadata = ms.getdata(['data', 'axis_info'])

    data = metadata['data']  # [npol, nchan, nrow]
    freq_info = metadata['axis_info']['freq_axis']
    freqs = freq_info['chan_freq'][:, 0]  # Hz

    # Get source info
    try:
        field_info = ms.getdata(['field_id'])
        # Try to get source name from ms
        tb = casatools.table()
        tb.open(ms_file + '/FIELD')
        source_name = tb.getcol('NAME')[0]
        tb.close()
    except:
        source_name = "Unknown"

    ms.close()

    # Average over polarizations and time, take amplitude
    spectrum = np.abs(data).mean(axis=(0, 2))

    return freqs, spectrum, source_name, data.shape


def plot_spectrum(freqs, spectrum, source_name, output_file, title=None):
    """Create a spectrum plot and save as PNG."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Convert frequency to MHz
    freqs_mhz = freqs / 1e6

    ax.plot(freqs_mhz, spectrum, 'b-', linewidth=0.5)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Amplitude')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Parkes Spectrum: {source_name}')

    ax.grid(True, alpha=0.3)

    # Add some stats
    freq_center = freqs_mhz.mean()
    bandwidth = freqs_mhz.max() - freqs_mhz.min()
    ax.text(0.02, 0.98, f'Center: {freq_center:.1f} MHz\nBW: {bandwidth:.1f} MHz\nChannels: {len(freqs)}',
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    return output_file


def plot_waterfall(ms_file, output_file):
    """Create a time-frequency waterfall plot if multiple time samples exist."""
    ms = casatools.ms()
    ms.open(ms_file)

    metadata = ms.getdata(['data', 'axis_info', 'time'])
    data = metadata['data']  # [npol, nchan, nrow]
    freqs = metadata['axis_info']['freq_axis']['chan_freq'][:, 0]
    times = metadata['time']

    ms.close()

    if data.shape[2] < 2:
        print("  Only 1 time sample, skipping waterfall plot")
        return None

    # Average over polarizations, take amplitude
    waterfall = np.abs(data).mean(axis=0)  # [nchan, ntime]

    fig, ax = plt.subplots(figsize=(12, 8))

    freqs_mhz = freqs / 1e6
    times_rel = (times - times.min()) / 60  # minutes from start

    im = ax.imshow(waterfall, aspect='auto', origin='lower',
                   extent=[times_rel.min(), times_rel.max(), freqs_mhz.min(), freqs_mhz.max()],
                   cmap='viridis')

    ax.set_xlabel('Time (minutes from start)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Time-Frequency Waterfall')

    plt.colorbar(im, ax=ax, label='Amplitude')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

    return output_file


def convert_rpfits(rpf_file, output_dir=None):
    """Convert a single RPFITS file to PNG spectrum plot."""
    rpf_path = Path(rpf_file).resolve()

    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = rpf_path.parent

    out_dir.mkdir(parents=True, exist_ok=True)

    basename = rpf_path.stem
    ms_file = str(out_dir / f"{basename}.ms")
    png_file = str(out_dir / f"{basename}_spectrum.png")
    waterfall_file = str(out_dir / f"{basename}_waterfall.png")

    print(f"\nProcessing: {rpf_path.name}")

    # Convert to MS
    print("  Converting RPFITS -> MeasurementSet...")
    try:
        rpfits_to_ms(str(rpf_path), ms_file)
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

    # Extract and plot spectrum
    print("  Extracting spectrum...")
    try:
        freqs, spectrum, source_name, data_shape = extract_spectrum(ms_file)
        print(f"  Source: {source_name}, Data shape: {data_shape}")
    except Exception as e:
        print(f"  ERROR extracting spectrum: {e}")
        return None

    # Plot spectrum
    print(f"  Creating spectrum plot: {png_file}")
    title = f"{rpf_path.name}\nSource: {source_name}"
    plot_spectrum(freqs, spectrum, source_name, png_file, title=title)

    # Try waterfall plot
    try:
        wf = plot_waterfall(ms_file, waterfall_file)
        if wf:
            print(f"  Created waterfall plot: {waterfall_file}")
    except Exception as e:
        print(f"  Waterfall plot failed: {e}")

    # Clean up MS (optional - comment out to keep)
    # shutil.rmtree(ms_file)

    return png_file


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Handle glob patterns
    import glob
    files = []
    for arg in sys.argv[1:]:
        if '*' in arg or '?' in arg:
            files.extend(glob.glob(arg))
        else:
            files.append(arg)

    if not files:
        print("No files found")
        sys.exit(1)

    print(f"Processing {len(files)} file(s)...")

    results = []
    for f in files:
        if os.path.isfile(f):
            result = convert_rpfits(f)
            if result:
                results.append(result)

    print(f"\n{'='*60}")
    print(f"Processed {len(results)}/{len(files)} files successfully")
    if results:
        print("Output files:")
        for r in results:
            print(f"  {r}")


if __name__ == "__main__":
    main()

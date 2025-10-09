# VLASS Coverage Check for LoVoCCS Targets

This repository contains scripts to check which LoVoCCS (Local Volume Complete Cluster Survey) galaxy cluster targets are present in the VLASS (Very Large Array Sky Survey) dataset.

## Contents

- `check_vlass_coverage.py` - Main script to query VLASS Epoch 1 catalog for target coverage
- `LoVoCCS_target_list - lovoccs.csv` - Input CSV file containing 106 galaxy cluster targets with coordinates
- `vlass_coverage_results.csv` - Output results showing which targets were found in VLASS

## Results Summary

**76 out of 106 targets (71.7%) have radio sources in VLASS Epoch 1**

The script queries the VLASS Epoch 1 Quick Look Catalog (Gordon+, 2021; catalog ID: J/ApJS/255/30) available through Vizier, using a 5 arcminute search radius around each target position. For each target, we count the number of distinct catalog radio sources detected within this radius (not the number of observations/epochs).

### Targets Not Found (30 targets)

The 30 targets not found are primarily:
- Targets with dec < -40° (outside VLASS coverage area)
- A few targets within the nominal coverage that may be in gaps between tiles or below detection threshold

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install astroquery pandas astropy numpy
```

## Usage

```bash
source venv/bin/activate
python check_vlass_coverage.py
```

The script will:
1. Parse the target list CSV
2. Query the VLASS Epoch 1 catalog via Vizier for each target
3. Print progress and summary to console
4. Save detailed results to `vlass_coverage_results.csv`

## Output Format

The results CSV contains:
- `id` - Target ID from input file
- `name` - Target name (cluster designation)
- `ra` - Right Ascension in degrees
- `dec` - Declination in degrees
- `in_vlass` - Boolean indicating if target was found in VLASS
- `n_sources` - **Number of distinct radio sources** in the VLASS catalog within 5 arcmin of the target

## Notes

- **Epoch Coverage**: Currently only queries Epoch 1. Epochs 2 and 3 have been released but are not yet easily accessible via Vizier or standard TAP services.
- **Positional Accuracy**: Epoch 1 has positional accuracy of ~0.5-1 arcsec. The 5 arcmin search radius should be sufficient to account for this.
- **VLASS Sky Coverage**: VLASS covers declinations > -40°. Targets with dec < -40° will not be found.

## Example VLASS Images

We downloaded VLASS radio images for three targets with varying coverage using `download_vlass_image.py`:

**A1644 - High source count (26 catalog sources)**
- Relatively clean field with faint diffuse emission
- Max flux: 0.18 Jy/beam
- Shows VLASS's high resolution (~2.5" vs NVSS's 45")
- Many catalog sources are faint and not prominently visible in this cutout

**A1307 - Medium source count (4 catalog sources)**
- Very bright central source (0.007 Jy/beam) - likely the cluster's BCG or central AGN
- **Strong diagonal stripe artifacts** across the image - typical imaging artifacts from interferometric deconvolution
- These patterns are common in radio interferometry, especially around bright sources

**A1736 - Low source count (2 catalog sources)**
- Multiple faint sources in the field (0.02 Jy/beam)
- **Prominent diagonal striping artifacts** throughout the image
- Extended structures visible in the lower right (may be real diffuse emission or artifacts)
- Artifacts are more prominent in some VLASS tiles due to imaging quality variations

### Image Visualization

The images use an asinh (inverse hyperbolic sine) stretch which is approximately:
- **Linear** for faint/low values (preserves subtle features)
- **Logarithmic** for bright/high values (compresses bright sources)

This allows visualization of both bright radio galaxies and fainter extended emission in the same image, which is essential for radio astronomy data with large dynamic range.

### Technical Notes

The script:
1. Queries CADC for VLASS observations overlapping the target coordinates
2. Downloads multiple image tiles (VLASS sky is tiled)
3. **Automatically selects the tile that contains the target position** with the highest flux
4. Extracts 2D images from 4D FITS cubes for display

This ensures the displayed image actually contains the cluster center, not just an overlapping tile.

### Usage

```bash
source venv/bin/activate
python download_vlass_image.py
```

Note: VLASS images are downloaded from CADC using astroquery. FITS files are gitignored.

## Future Work

- Add support for VLASS Epochs 2 and 3 when they become available via Vizier
- Implement direct catalog download from CIRADA/CADC if needed

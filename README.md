# VLASS Coverage Check for LoVoCCS Targets

This repository contains scripts to check which LoVoCCS (Local Volume Complete Cluster Survey) galaxy cluster targets are present in the VLASS (Very Large Array Sky Survey) dataset.

## Contents

- `check_vlass_coverage.py` - Main script to query VLASS Epoch 1 catalog for target coverage
- `LoVoCCS_target_list - lovoccs.csv` - Input CSV file containing 106 galaxy cluster targets with coordinates
- `vlass_coverage_results.csv` - Output results showing which targets were found in VLASS

## Results Summary

**76 out of 106 targets (71.7%) were found in VLASS Epoch 1**

The script queries the VLASS Epoch 1 Quick Look Catalog (Gordon+, 2021; catalog ID: J/ApJS/255/30) available through Vizier, using a 5 arcminute search radius around each target position.

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
- `in_vlass` - Boolean indicating if target was found
- `n_obs` - Number of VLASS sources detected within search radius

## Notes

- **Epoch Coverage**: Currently only queries Epoch 1. Epochs 2 and 3 have been released but are not yet easily accessible via Vizier or standard TAP services.
- **Positional Accuracy**: Epoch 1 has positional accuracy of ~0.5-1 arcsec. The 5 arcmin search radius should be sufficient to account for this.
- **VLASS Sky Coverage**: VLASS covers declinations > -40°. Targets with dec < -40° will not be found.

## Future Work

- Add support for VLASS Epochs 2 and 3 when they become available via Vizier
- Implement direct catalog download from CIRADA/CADC if needed
- Add image cutout retrieval for detected sources

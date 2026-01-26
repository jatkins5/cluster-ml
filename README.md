# Multi-wavelength Survey Coverage for LoVoCCS Targets

This repository contains scripts to cross-match LoVoCCS (Local Volume Complete Cluster Survey) galaxy cluster targets with multi-wavelength survey data, including radio (VLASS, LoTSS, FIRST) and X-ray (eROSITA) observations.

## Contents

### Scripts
- `check_vlass_coverage.py` - Query VLASS Epoch 1 catalog for radio source coverage
- `download_vlass_image.py` - Download and visualize VLASS radio images
- `match_lovoccs_erosita.py` - Cross-match LoVoCCS targets with eROSITA X-ray sources
- `match_lovoccs_lotss.py` - Cross-match LoVoCCS targets with LoTSS radio sources
- `match_lovoccs_first.py` - Cross-match LoVoCCS targets with FIRST radio sources
- `download_first_image.py` - Download and visualize FIRST radio images
- `match_lovoccs_parkes.py` - Cross-match LoVoCCS targets with Parkes radio catalogs (PMN, PKSCAT90)
- `download_parkes_data.py` - Download raw Parkes RPFITS data from ATOA (requires OPAL authentication)
- `convert_rpfits_to_png.py` - Convert Parkes RPFITS spectral data to PNG spectrum plots
- `query_parkes_mapping.py` - Query ATOA for Parkes mapping observations with spatial diversity
- `assemble_parkes_images.py` - Assemble Parkes single-dish spectra into spatial images

### Data Files
- `LoVoCCS_target_list - lovoccs.csv` - Input CSV file containing 106 galaxy cluster targets with coordinates
- `vlass_coverage_results.csv` - Output results showing which targets were found in VLASS
- `lovoccs_erosita_matches.csv` - Summary of eROSITA X-ray matches for each cluster
- `lovoccs_erosita_matches_detailed.csv` - Detailed information for all matched X-ray sources
- `lovoccs_lotss_matches.csv` - Summary of LoTSS radio matches for each cluster
- `lovoccs_lotss_matches_detailed.csv` - Detailed information for all matched LoTSS sources
- `lovoccs_first_matches.csv` - Summary of FIRST radio matches for each cluster
- `lovoccs_first_matches_detailed.csv` - Detailed information for all matched FIRST sources
- `lovoccs_parkes_matches.csv` - Summary of Parkes catalog matches for each cluster
- `lovoccs_parkes_matches_detailed.csv` - Detailed information for all matched Parkes sources

## Results Summary

### VLASS Radio Coverage

**76 out of 106 targets (71.7%) have radio sources in VLASS Epoch 1**

The script queries the VLASS Epoch 1 Quick Look Catalog (Gordon+, 2021; catalog ID: J/ApJS/255/30) available through Vizier, using a 5 arcminute search radius around each target position. For each target, we count the number of distinct catalog radio sources detected within this radius (not the number of observations/epochs).

**Targets Not Found (30 targets):**
- Targets with dec < -40° (outside VLASS coverage area)
- A few targets within the nominal coverage that may be in gaps between tiles or below detection threshold

### eROSITA X-ray Matches

**LoVoCCS targets matched with eROSITA All-Sky Survey (eRASS1)**

The matching script queries the eROSITA eRASS1 main catalog (Merloni+, 2024; catalog ID: J/A+A/685/A106) via Vizier, using a 5 arcminute search radius around each cluster center. For each match, we record:
- Number of X-ray sources within the search radius
- Separation of the closest source from the cluster center
- X-ray flux in the 0.5-2 keV band
- Detection likelihood and source extent

The script generates two output files:
- **Summary file** (`lovoccs_erosita_matches.csv`): One row per cluster with basic match statistics
- **Detailed file** (`lovoccs_erosita_matches_detailed.csv`): Individual properties for each matched X-ray source

### LoTSS Radio Matches

**2 out of 106 LoVoCCS targets matched with LOFAR Two-metre Sky Survey (LoTSS) DR2**

The matching script queries the LoTSS DR2 value-added catalog (Shimwell+, 2022; catalog ID: J/A+A/678/A151) via Vizier, using a 10 arcminute search radius around each cluster center. LoTSS provides low-frequency (144 MHz) radio observations with excellent sensitivity to diffuse emission. For each match, we record:
- Number of radio sources within the search radius
- Separation of the closest source from the cluster center
- Peak and total flux density (mJy)
- Source morphology (major/minor axes)
- **Resolved flag** - indicates extended emission (important for cluster radio halos/relics)

**Sky coverage**: LoTSS DR2 covers ~27% of the sky (5720 deg²) in the northern hemisphere (0h < RA < 24h, +25° < Dec < +70°). The low match rate (2/106) reflects that most LoVoCCS targets fall outside the LoTSS coverage area or at declinations below +25°.

The script generates two output files:
- **Summary file** (`lovoccs_lotss_matches.csv`): One row per cluster with basic match statistics
- **Detailed file** (`lovoccs_lotss_matches_detailed.csv`): Individual properties for each matched radio source

### FIRST Radio Matches

**LoVoCCS targets matched with FIRST (Faint Images of the Radio Sky at Twenty Centimeters)**

The matching script queries the FIRST catalog (White+, 2020; catalog ID: VIII/92/first14) via Vizier, using a 5 arcminute search radius around each cluster center. FIRST provides 1.4 GHz radio observations with excellent resolution and sensitivity to point sources and compact emission. For each match, we record:
- Number of radio sources within the search radius
- Separation of the closest source from the cluster center
- Peak and integrated flux density (mJy)
- Source morphology (major/minor axes)
- **Extended source flag** - automatically calculated for sources with major axis > 5.4" (beam size)

**Sky coverage**: FIRST covers 10,575 deg², mostly northern sky (Dec > -40°)

**Survey properties**:
- Frequency: 1.4 GHz (20 cm)
- Resolution: ~5" (between VLASS's ~2.5" and LoTSS's ~6")
- Sensitivity: ~1 mJy/beam RMS
- Best for: Point sources, AGN, and moderately extended emission

The script generates two output files:
- **Summary file** (`lovoccs_first_matches.csv`): One row per cluster with basic match statistics
- **Detailed file** (`lovoccs_first_matches_detailed.csv`): Individual properties for each matched radio source

### Parkes Radio Catalog Matches

**28 out of 106 LoVoCCS targets (26.4%) matched with Parkes radio catalogs**

The matching script queries Parkes survey catalogs via VizieR, using a 5 arcminute search radius around each cluster center:

- **PMN** (VIII/38) - Parkes-MIT-NRAO 4.85 GHz survey
- **PKSCAT90** (VIII/15) - Parkes Radio Sources Catalogue, multi-frequency (80 MHz to 5 GHz)

**Results breakdown by catalog:**
- PMN: 28 sources (4.85 GHz continuum)
- PKSCAT90: 12 sources (historical multi-frequency catalog)

**Notable matches:**
- A780 (Hydra A): 23.5 Jy at 2700 MHz - famous radio galaxy
- A2052: 2.3 Jy at 2700 MHz with multi-frequency detections

The script generates two output files:
- **Summary file** (`lovoccs_parkes_matches.csv`): One row per cluster with match counts per catalog
- **Detailed file** (`lovoccs_parkes_matches_detailed.csv`): Individual properties for each matched source including multi-frequency flux measurements

### Parkes Raw Data (ATOA)

In addition to catalog matching, raw Parkes observation data can be downloaded from the Australia Telescope Online Archive (ATOA). These are **RPFITS format spectral data files**, not images.

**Important notes:**
- ATOA requires OPAL authentication to download data
- Parkes is a single-dish telescope - it produces spectra, not interferometric images
- The RPFITS files contain raw telescope data requiring specialized software to reduce
- Use `convert_rpfits_to_png.py` to visualize spectra as PNG plots

**18 out of 106 clusters** have Parkes observations within 0.25 degrees in ATOA, totaling ~1.8 GB of data.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install astroquery pandas astropy numpy matplotlib pyvo casatools
```

## Usage

### VLASS Radio Coverage Check

```bash
source venv/bin/activate
python check_vlass_coverage.py
```

The script will:
1. Parse the target list CSV
2. Query the VLASS Epoch 1 catalog via Vizier for each target
3. Print progress and summary to console
4. Save detailed results to `vlass_coverage_results.csv`

**Output format:**
- `id` - Target ID from input file
- `name` - Target name (cluster designation)
- `ra` - Right Ascension in degrees
- `dec` - Declination in degrees
- `in_vlass` - Boolean indicating if target was found in VLASS
- `n_sources` - Number of distinct radio sources in the VLASS catalog within 5 arcmin

### eROSITA X-ray Source Matching

```bash
source venv/bin/activate
python match_lovoccs_erosita.py
```

The script will:
1. Parse the LoVoCCS target list
2. Query the eROSITA eRASS1 catalog via Vizier for each cluster
3. Calculate separations and extract source properties
4. Save summary results to `lovoccs_erosita_matches.csv`
5. Save detailed source information to `lovoccs_erosita_matches_detailed.csv`

**Summary output format:**
- `id`, `name`, `ra`, `dec` - Cluster identification
- `has_erosita_match` - Boolean indicating if X-ray sources were found
- `n_erosita_sources` - Number of eROSITA sources within 5 arcmin
- `closest_sep_arcmin` - Angular separation to closest X-ray source
- `closest_flux_0.5_2keV` - Flux of closest source in 0.5-2 keV band

**Detailed output format:**
- Cluster information (id, name, coordinates)
- Source rank (sorted by separation)
- Source coordinates and separation
- X-ray flux, detection likelihood, and extent

### LoTSS Radio Source Matching

```bash
source venv/bin/activate
python match_lovoccs_lotss.py
```

The script will:
1. Parse the LoVoCCS target list
2. Query the LoTSS DR2 catalog via Vizier for each cluster
3. Calculate separations and extract source properties
4. Save summary results to `lovoccs_lotss_matches.csv`
5. Save detailed source information to `lovoccs_lotss_matches_detailed.csv`

**Summary output format:**
- `id`, `name`, `ra`, `dec` - Cluster identification
- `has_lotss_match` - Boolean indicating if radio sources were found
- `n_lotss_sources` - Number of LoTSS sources within 10 arcmin
- `closest_sep_arcmin` - Angular separation to closest radio source
- `closest_total_flux_mJy` - Total flux of closest source in mJy
- `closest_resolved` - Flag indicating if closest source is resolved/extended

**Detailed output format:**
- Cluster information (id, name, coordinates)
- Source rank (sorted by separation)
- Source coordinates and separation
- Peak and total flux density, source morphology, resolved flag

### FIRST Radio Source Matching

```bash
source venv/bin/activate
python match_lovoccs_first.py
```

The script will:
1. Parse the LoVoCCS target list
2. Query the FIRST catalog via Vizier for each cluster
3. Calculate separations and extract source properties
4. Save summary results to `lovoccs_first_matches.csv`
5. Save detailed source information to `lovoccs_first_matches_detailed.csv`

**Summary output format:**
- `id`, `name`, `ra`, `dec` - Cluster identification
- `has_first_match` - Boolean indicating if radio sources were found
- `n_first_sources` - Number of FIRST sources within 5 arcmin
- `closest_sep_arcmin` - Angular separation to closest radio source
- `closest_int_flux_mJy` - Integrated flux of closest source in mJy
- `closest_is_extended` - Flag indicating if closest source is extended (maj > 5.4")

**Detailed output format:**
- Cluster information (id, name, coordinates)
- Source rank (sorted by separation)
- Source coordinates and separation
- Peak and integrated flux density, source morphology, RMS noise, sidelobe probability

### Parkes Radio Catalog Matching

```bash
source venv/bin/activate
python match_lovoccs_parkes.py
```

The script will:
1. Parse the LoVoCCS target list
2. Query multiple Parkes catalogs (HIPASS, PMN, PKSCAT90) via VizieR for each cluster
3. Calculate separations and extract source properties
4. Save summary results to `lovoccs_parkes_matches.csv`
5. Save detailed source information to `lovoccs_parkes_matches_detailed.csv`

### Parkes Raw Data Download

```bash
source venv/bin/activate
# Query what's available (no download)
python download_parkes_data.py --clusters A780

# Download with authentication (will prompt for OPAL credentials)
python download_parkes_data.py --download --clusters A780

# Or set credentials via environment variables
export OPAL_USERNAME="your_username"
export OPAL_PASSWORD="your_password"
python download_parkes_data.py --download --clusters A780

# Prioritize spatial diversity for mapping (recommended for image assembly)
python download_parkes_data.py --download --clusters A780 --prioritize-diversity --radius 2.0
```

**Spatial Diversity Selection (`--prioritize-diversity`):**

When downloading observations for image assembly, you typically want observations at many different sky positions rather than many observations at the same position. The `--prioritize-diversity` flag ensures the selected observations are spread across different pointings:

- Without flag: Takes first N observations (may all be at same position)
- With flag: Queries all available observations, then selects a diverse subset ensuring different sky positions are represented

This is particularly useful when combined with `--max-per-cluster` to get a manageable number of observations while maximizing spatial coverage for mapping.

### Convert Parkes Spectra to PNG

```bash
source venv/bin/activate
# Convert a single file
python convert_rpfits_to_png.py parkes_data/A780/2003-11-27_1941-P440.rpf

# Convert all files for a cluster
python convert_rpfits_to_png.py parkes_data/A780/*.rpf
```

This generates spectrum plots (`*_spectrum.png`) and time-frequency waterfall plots (`*_waterfall.png`) for each RPFITS file.

### Parkes Image Assembly

Parkes is a single-dish telescope, so each observation produces one spectrum at one sky position. To create spatial maps, you need observations at multiple different positions. The workflow is:

**Step 1: Find clusters with mapping data**
```bash
source venv/bin/activate

# Scan all clusters to find those with mapping-suitable observations
python query_parkes_mapping.py --scan-all --output mappable_clusters.csv

# Or query a specific cluster
python query_parkes_mapping.py --cluster A780 --radius 2.0 --min-positions 5
```

This queries ATOA with a larger search radius (2 degrees) to find observations at multiple sky positions. Results show which clusters have enough spatial diversity for mapping:
- Number of unique pointing positions
- Spatial extent covered
- Frequency bands available

**Step 2: Download mapping observations**

Once you've identified clusters with mapping data, download the observations using `download_parkes_data.py` with a larger search radius:

```bash
python download_parkes_data.py --download --clusters A780 --radius 2.0
```

**Step 3: Assemble into images**
```bash
source venv/bin/activate

# First scan the downloaded data to check coverage
python assemble_parkes_images.py parkes_data/A780/ --scan-only

# Create a continuum (2D) image
python assemble_parkes_images.py parkes_data/A780/ --output hydra_a.fits

# Create a spectral cube (3D)
python assemble_parkes_images.py parkes_data/A780/ --output hydra_a_cube.fits --mode cube

# Specify custom pixel size (default is beam/3)
python assemble_parkes_images.py parkes_data/A780/ --output hydra_a.fits --pixel-size 5
```

The script will:
1. Scan all RPFITS files and extract coordinates/spectra (using subprocess isolation for stability)
2. Analyze spatial coverage and determine if mapping is possible
3. Grid spectra onto a regular RA/Dec pixel grid using Gaussian beam weighting
4. Output FITS file with proper WCS headers and PNG visualization

**Requirements:**
- Minimum 5 unique sky positions for meaningful mapping
- Observations should be at compatible frequencies
- Parkes beam size is ~14.4 arcmin at 1 GHz (scales as 14.4/freq_GHz)

**Example output for A780:**
```
Coverage Analysis:
  Observations:      472
  Unique positions:  12
  Spatial extent:    216.4 arcmin
  Mean frequency:    685.0 MHz
  Beam size:         21.0 arcmin
  Can make image:    YES
```

### Image Visualization

**VLASS radio images:**
```bash
python download_vlass_image.py
```

This script will:
1. Read the VLASS coverage results
2. Select representative targets (high, medium, low source counts)
3. Download image cutouts from CADC
4. Generate publication-quality visualizations with WCS coordinates
5. Save both FITS and PNG files

**FIRST radio images:**
```bash
python download_first_image.py
```

This script will:
1. Read the FIRST match results
2. Select representative targets (high, medium, low source counts)
3. Download image cutouts from the FIRST archive using astroquery
4. Generate publication-quality visualizations with WCS coordinates
5. Save both FITS and PNG files

**eROSITA X-ray images:**

eROSITA images are not available through automated download tools yet. To view X-ray images for matched sources:

1. Upload `lovoccs_erosita_matches.csv` to [Aladin](https://aladin.u-strasbg.fr/)
2. In Aladin, load the eROSITA survey data layer
3. The matched coordinates will be overlaid on the X-ray images

Alternatively, you can use the detailed matches file (`lovoccs_erosita_matches_detailed.csv`) to view individual X-ray source positions.

## Notes

### VLASS
- **Epoch Coverage**: Currently only queries Epoch 1. Epochs 2 and 3 have been released but are not yet easily accessible via Vizier or standard TAP services.
- **Positional Accuracy**: Epoch 1 has positional accuracy of ~0.5-1 arcsec. The 5 arcmin search radius should be sufficient to account for this.
- **Sky Coverage**: VLASS covers declinations > -40°. Targets with dec < -40° will not be found.

### eROSITA
- **Catalog**: Uses eRASS1 (eROSITA All-Sky Survey, first all-sky scan)
- **Extended Sources**: eROSITA sources include an extent measurement useful for identifying extended cluster emission
- **Sky Coverage**: eRASS1 covers the entire sky

### LoTSS
- **Catalog**: Uses LoTSS DR2 value-added catalog (Shimwell+, 2022)
- **Frequency**: 144 MHz - excellent for detecting diffuse, steep-spectrum emission
- **Resolution**: ~6 arcsec - can resolve cluster-scale structures
- **Sky Coverage**: Northern hemisphere, 0h < RA < 24h, +25° < Dec < +70° (~5720 deg²)
- **Resolved Sources**: Catalog includes resolved flag to identify extended emission (radio halos/relics)

### FIRST
- **Catalog**: Uses FIRST 2014 catalog (White+, 2020)
- **Frequency**: 1.4 GHz (20 cm) - intermediate frequency between VLASS and LoTSS
- **Resolution**: ~5 arcsec - good for both point sources and moderately extended emission
- **Sky Coverage**: 10,575 deg², mostly northern sky (Dec > -40°)
- **Sensitivity**: ~1 mJy/beam RMS
- **Extended Sources**: Script automatically flags sources with major axis > 5.4" (beam size)

### Parkes
- **Catalogs**: PMN, PKSCAT90 via VizieR
- **Frequencies**: 4850 MHz (PMN), multi-frequency (PKSCAT90)
- **Telescope**: Single-dish (64m) - produces spectra, not interferometric images
- **Sky Coverage**: Southern sky, various coverage per catalog
- **Raw Data**: Available from ATOA in RPFITS format (requires OPAL account)
- **Data Products**: Raw spectral/continuum data requiring reduction with Livedata/AIPS/MIRIAD
- **Visualization**: Use `convert_rpfits_to_png.py` with casatools to view spectra

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

**A2415 - Very high source count (24 catalog sources)**
- Relatively clean field appearance with max flux 0.185 Jy/beam
- Despite 24 catalog sources within 5 arcmin, most are faint or at the edges of the cutout
- Demonstrates that high catalog source counts don't always translate to crowded-looking images
- Many sources may be near the detection threshold

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

### VLASS
- Add support for VLASS Epochs 2 and 3 when they become available via Vizier
- Implement direct catalog download from CIRADA/CADC if needed

### eROSITA
- Download cutouts manually from eROSITA website https://erosita.mpe.mpg.de/dr1/erodat/

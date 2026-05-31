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
- `verify_meerkat_observations.py` - Verify MeerKAT observations via MGCLS/SARAO archive
- `download_meerkat_image.py` - Download and visualize MGCLS radio images

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
- `meerkat_verification_results.csv` - MeerKAT observation verification (MGCLS + SARAO archive)
- `mgcls_clusters.csv` - MGCLS DR1 cluster catalog (115 clusters, coordinates, selection type)
- `mgcls_table1.dat` - MGCLS Table 1 from Knowles et al. 2022

## Results Summary

### VLASS Radio Coverage

**76 out of 106 targets (71.7%) have radio sources in VLASS Epoch 1**

The script queries the VLASS Epoch 1 Quick Look Catalog (Gordon+, 2021; catalog ID: J/ApJS/255/30) available through Vizier, using a 5 arcminute search radius around each target position. For each target, we count the number of distinct catalog radio sources detected within this radius (not the number of observations/epochs).

**Targets Not Found (30 targets):**
- Targets with dec < -40° (outside VLASS coverage area)
- A few targets within the nominal coverage that may be in gaps between tiles or below detection threshold

### eROSITA X-ray Matches

**59 out of 106 LoVoCCS targets matched with eROSITA All-Sky Survey (eRASS1)**

The matching script queries the eROSITA eRASS1 main catalog (Merloni+, 2024; catalog ID: J/A+A/685/A106) via Vizier, using a 5 arcminute search radius around each cluster center. For each match, we record:
- Number of X-ray sources within the search radius
- Separation of the closest source from the cluster center
- X-ray flux in the 0.5-2 keV band
- Detection likelihood and source extent

The script generates two output files:
- **Summary file** (`lovoccs_erosita_matches.csv`): One row per cluster with basic match statistics
- **Detailed file** (`lovoccs_erosita_matches_detailed.csv`): Individual properties for each matched X-ray source

### LoTSS Radio Matches

**26 out of 106 LoVoCCS targets matched with LOFAR Two-metre Sky Survey (LoTSS) DR3**

The matching script bulk cross-matches the LoVoCCS target list against the LoTSS DR3 PyBDSF source catalog (Shimwell et al. 2026; `LoTSS_DR3_v1.0.srl.fits`, ~13.7M sources), which is downloaded locally and matched using `astropy.coordinates.search_around_sky` with a 10 arcminute search radius. LoTSS provides low-frequency (144 MHz) radio observations with excellent sensitivity to diffuse emission. For each match, we record:
- Number of radio sources within the search radius
- Separation of the closest source from the cluster center
- Peak and total flux density (mJy)
- Source morphology (major/minor axes)
- **Resolved flag** - indicates extended emission (important for cluster radio halos/relics)

**Sky coverage**: LoTSS DR3 covers ~88% of the northern sky at 6" angular resolution (9" below declination +10°), as described in Shimwell et al. 2026. The match rate of 26/106 reflects LoVoCCS targets that fall outside the LOFAR footprint, predominantly at southern declinations.

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

### MeerKAT (MGCLS)

**18 out of 106 LoVoCCS targets (17.0%) are in the MGCLS DR1 survey**

The MeerKAT Galaxy Cluster Legacy Survey (MGCLS; Knowles et al. 2022, A&A 657, A56) observed 115 galaxy clusters with MeerKAT L-band (~1.28 GHz, ~8" resolution). Cross-matching with LoVoCCS identifies 18 overlapping targets.

**Data products available:**
- **Basic products**: 16-plane FITS cubes per cluster
  - Plane 0: Stokes I continuum at ~1283 MHz reference frequency
  - Plane 1: Spectral index map
  - Planes 2-15: 14 frequency channel images
- **Enhanced products**: Higher-quality mosaics with improved calibration

**Additional SARAO archive observations:** Beyond MGCLS, several LoVoCCS targets have MeerKAT observations from other programs (e.g., A780/Hydra A with 63 observations). These require SARAO authentication and visibility-level processing — see "Processing MeerKAT Visibility Data" below.

**DOI:** [10.48479/7epd-w356](https://doi.org/10.48479/7epd-w356)

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
2. Download the LoTSS DR3 PyBDSF source catalog (`LoTSS_DR3_v1.0.srl.fits`) if not already present
3. Cross-match all clusters against the catalog in bulk via `search_around_sky` with a 10 arcmin radius
4. Calculate separations and extract source properties
5. Save summary results to `lovoccs_lotss_matches.csv`
6. Save detailed source information to `lovoccs_lotss_matches_detailed.csv`

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

**MeerKAT MGCLS radio images:**
```bash
# Download sample of 3 clusters (A85, A3667, A133)
python download_meerkat_image.py

# Download all 18 MGCLS targets
python download_meerkat_image.py --all

# Download specific clusters
python download_meerkat_image.py --clusters A85 A2597

# Skip already-downloaded targets
python download_meerkat_image.py --all --skip-existing

# Regenerate PNGs from existing FITS files
python download_meerkat_image.py --all --png-only

# Validate that FITS files cover target positions
python download_meerkat_image.py --all --validate

# Download enhanced products instead of basic
python download_meerkat_image.py --clusters A85 --product-type enhanced
```

The script will:
1. Read the MGCLS verification results to identify the 18 overlapping clusters
2. Fetch a JWT token from the MGCLS DR1 archive page
3. Download FITS cubes and extract the Stokes I continuum plane (plane 0)
4. Generate publication-quality PNG visualizations with WCS coordinates
5. Save both FITS and PNG files to `meerkat_images/`

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

### MeerKAT
- **Survey**: MGCLS DR1 (Knowles et al. 2022, A&A 657, A56)
- **Frequency**: L-band ~1.28 GHz (23 cm)
- **Resolution**: ~8 arcsec (basic products), ~5 arcsec (enhanced)
- **Sensitivity**: ~3-5 µJy/beam RMS (typical)
- **Sky Coverage**: 115 targeted galaxy clusters (southern sky)
- **Archive**: [DOI 10.48479/7epd-w356](https://doi.org/10.48479/7epd-w356)
- **Data Products**: FITS cubes with continuum, spectral index, and channel images

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

## Processing MeerKAT Visibility Data

For LoVoCCS targets with MeerKAT observations in the SARAO archive (beyond MGCLS pre-made images), visibility data can be processed into images using these pipelines:

### processMeerKAT (IDIA)
- **URL**: https://github.com/idia-astro/pipelines
- Full calibration + imaging pipeline for MeerKAT data
- SLURM-based, designed for IDIA/ilifu cluster but adaptable
- Requires CASA 6.5+
- Handles RFI flagging, cross-calibration, self-calibration, and imaging

### oxkat (Ian Heywood)
- **URL**: https://github.com/IanHeywood/oxkat
- Semi-automated pipeline using WSClean + CubiCal
- Well-suited for continuum imaging of individual targets
- Modular design allows customization of calibration steps

### CARACal
- **URL**: https://github.com/caracal-pipeline/caracal
- End-to-end containerized pipeline (formerly MeerKATHI)
- Supports both continuum and spectral line processing
- Uses Stimela containerization framework for reproducibility

### katdal
- **URL**: https://pypi.org/project/katdal/
- Python library for programmatic access to SARAO archive data
- `pip install katdal` — read MeerKAT visibility data directly
- Useful for data inspection, flagging, and custom processing workflows

**Note:** Processing MeerKAT visibilities requires significant compute resources (~100 GB RAM, GPU for imaging) and SARAO archive authentication. The MGCLS pre-made images (`download_meerkat_image.py`) are recommended for initial analysis.

## ML Training Dataset (TNG-Cluster Radio Simulation)

### Overview

`Radio_Data/` contains simulated radio emission data for 352 TNG-Cluster galaxy clusters at z=0 (snapshot 99), used to train an ML model to predict **time since collision (TSC)** from radio observations.

### Data Files

- `Radio_Data/radio_FOF{haloID}_sub{subID}.npz` — 352 files, one per cluster:
  - `pos`: (N, 3) float64 — 3D shock particle positions in physical kpc
  - `w`: (N,) float64 — per-particle radio emission weight (pre-computed at reference DSA conditions; see below)
- `Radio_Data/TNG-Cluster_Catalog.hdf5` — cluster properties (R500c, halo mass, zform, etc.) for all 352 clusters
- `Radio_Data/normalized_psi_table.npz` — DSA radio emission lookup table Ψ(s, e_min); the `w` weights were computed using this table during `Radio_generation.py`
- `feats_labels_dict_tngcluster.pkl` — pre-computed merger labels for all 352 clusters across snapshots 72–99

### How `w` relates to radio emission

Each particle's weight encodes:
```
w_i = 5.2e23 × E_i × B_i^(1 + s_i/2) / (B_i² + B_CMB²) × Ψ_norm(s_i, e_min_i)
```
where E is energy dissipation, B is magnetic field strength, s is DSA spectral index (from Mach number), and Ψ_norm is the normalised radio emission efficiency from the lookup table. The total radio power of a cluster is `sum(w)`.

### Label: Time Since Collision (TSC)

Two approaches are available for the training label:

#### 1. `label_score` from `feats_labels_dict_tngcluster.pkl` (available now)

The pkl file provides a continuous **merger activity score** per cluster per snapshot. The score sweeps over a time window parameter τ from 0.1 to 4.0 Gyr:

```
label_score_all_tau{τ}  — mergers within ±τ Gyr of the snapshot (past + future)
label_score_pre_tau{τ}  — mergers within τ Gyr before the snapshot only
```

At snapshot 99 (z=0), `all == pre` since there are no future snapshots. The score accumulates merger mass ratio contributions weighted by proximity in time, so:
- Score ≈ 0 at small τ → no recent major merger
- Score crossing ~1 at τ* → last significant merger occurred ~τ* Gyr ago

This is a **soft, continuous TSC proxy** tunable via τ. Use `label_score_all_tau{T}` at snap 99 as the regression or classification target.

#### 2. Merger-catalog TSC (`TSC_Cutimages/`)

Ground-truth TSC derived from the **Lee et al. (2024) TNG-Cluster merger catalog** ([arXiv:2311.06340](https://arxiv.org/abs/2311.06340)). The catalog (`cluster_mergers.hdf5`) tracks 2,083 individual merger events across 346 unique halos, recording collision snapshots, mass ratios, velocities, and orbital histories.

**How TSC is computed** (see `TSC_Cutimages/TSC.ipynb`):

1. Load all collision snapshots (`Snap_coll`) for each halo from the merger catalog
2. Keep only collisions at or before snapshot 99 (z=0)
3. Take the **most recent** collision snapshot per halo
4. Convert snapshot to cosmic time using Planck15 cosmology
5. TSC = t(snap 99) − t(last collision snapshot), in Gyr

**Statistics** (saved in `TSC_Cutimages/TSC_eachhalo_snap99.hdf5`):
- 352 halos total, 6 have NaN (no recorded collision in catalog)
- Range: 0.0 – 7.7 Gyr; mean 1.57, median 1.01 Gyr
- Right-skewed distribution: most clusters merged recently, long tail of old mergers

The folder also contains **centered radio particle data** (`radio_centered_FOF*_sub*.hdf5`), generated by `Cut_images.py`. Each file stores particle positions re-centered on the halo center (from the group catalog) and radio power, ready for image generation:
- `centered_pos_radio`: (N, 3) positions in kpc, centered on halo
- `power_radio`: (N,) radio power in W/Hz
- Attributes: `halo_pos`, `halo_r` (R_crit200 in kpc), FOF/sub IDs

#### 3. Pseudo-TSC (interpolated from label_score)

A proxy TSC derived from the pkl label_score curve: the τ value at which `label_score_all` first crosses 0.5, interpolated linearly. Capped at 4.0 Gyr for 34 quiescent clusters whose score never reaches 0.5. See `build_dataset.py` for implementation.

| Property | `label_score` (pkl) | Merger-catalog TSC | Pseudo-TSC |
|---|---|---|---|
| Definition | Soft score: Σ(mass_ratio × time kernel) | Δt since last collision snapshot | τ where label_score crosses 0.5 |
| Tunable timescale | Yes, via τ parameter | No | No |
| Includes multiple mergers | Yes (weighted sum) | Most recent only | Implicitly (via score curve) |
| Range | 0–3.5 (dimensionless) | 0–7.7 Gyr | 0.1–4.0 Gyr (capped) |
| Missing values | None | 6/352 (no collision) | 34/352 capped at 4.0 |
| Source | `feats_labels_dict_tngcluster.pkl` | `TSC_Cutimages/TSC_eachhalo_snap99.hdf5` | `dataset.h5 → labels/pseudo_tsc` |

### Building the Training Dataset

#### Radio dataset

```bash
source venv/bin/activate
python build_dataset.py                          # defaults: 128×128 px, 4×R500c extent
python build_dataset.py --img-size 256           # higher resolution
python build_dataset.py --extent-r500 3.0        # tighter crop
python build_dataset.py --output my_dataset.h5   # custom output path
```

Output HDF5 (`dataset.h5`) structure:
```
images/                  (352, 3, 128, 128) float32  — arcsinh-normalised projections
  [dim 1: xy, yz, xz projections]
meta/
  halo_id                (352,) int64
  mass_ratio             (352,) float32
  r500c_kpc              (352,) float32
labels/
  tau_gyr                (40,)  float32  — τ values from 0.1 to 4.0 Gyr
  label_score_all        (352, 40) float32
  label_score_pre        (352, 40) float32
```

Loading example:
```python
import h5py, numpy as np

with h5py.File("dataset.h5", "r") as f:
    images = f["images"][:]               # (352, 3, 128, 128)
    labels = f["labels/label_score_all"][:, 9]  # tau=1.0 Gyr (index 9)
    halo_ids = f["meta/halo_id"][:]
```

#### X-ray dataset

Simulated Chandra ACIS-I observations from `TNGCluster_Xray_Snap99/`, generated via `pyxsim` + `soxs` (see `X-ray_chandra_Snap0_x.py` in each projection directory). Each cluster has mock 2 Ms Chandra observations in 3 viewing axes, with exposure-corrected count-rate images in the 0.1–2.0 keV band.

**Raw data:** 4880×4880 FITS images at ~0.492 arcsec/pixel (Chandra ACIS-I native scale), FoV = 2×R500c per cluster, one `_img.fits` per halo per projection.

**Projection alignment with radio:**

| Radio projection | Axes | X-ray directory |
|---|---|---|
| 0 (xy) | view along z | `snap99_z/` |
| 1 (yz) | view along x | `snap99_x/` |
| 2 (xz) | view along y | `snap99_y/` |

```bash
python build_xray_dataset.py                              # 128×128 (38× block-average)
python build_xray_dataset.py --img-size 256               # 256×256 (19× block-average)
python build_xray_dataset.py --output dataset_xray.h5     # custom output path
```

The script center-crops the 4880×4880 raw images to 4864×4864 (drops 8 px per side, <0.2% of FoV) so the target size divides evenly, then block-averages (exact area-mean) to the target resolution. Arcsinh normalization is applied after downsampling.

Output HDF5 (`dataset_xray_128.h5`) structure:
```
images/                  (352, 3, 128, 128) float32  — arcsinh(block-averaged counts/s/px)
  [dim 1: xy, yz, xz — same order as radio dataset.h5]
meta/
  halo_id                (352,) int64  — same ordering as dataset.h5
```

## Model Results

### XGBoost Baseline (`train_xgboost.py`)

**Input:** 17 tabular morphology features per (cluster × projection) from `feats_labels_dict_tngcluster.pkl`
(BIC scores, position/velocity dispersions, elongation ratio, R500c, mass ratio).
1056 samples total (352 clusters × 3 projections). **Target:** `label_score_all_tau1.0` at snap 99.

**5-fold GroupKFold CV (grouped by cluster):**

| Fold | R² | MAE | RMSE |
|------|-----|-----|------|
| 1 | 0.374 | 0.501 | 0.688 |
| 2 | −0.071 | 0.471 | 0.639 |
| 3 | 0.366 | 0.439 | 0.568 |
| 4 | 0.225 | 0.540 | 0.727 |
| 5 | 0.378 | 0.391 | 0.539 |
| **mean ± std** | **0.254 ± 0.173** | **0.469 ± 0.051** | **0.632 ± 0.071** |
| **OOF** | **0.290** | | |

**Notes:**
- Tabular features explain ~29% of label variance with no image information — meaningful signal
- MAE of ~0.47 on a 0–3.5 label scale corresponds to roughly ±0.5 Gyr TSC error
- Fold 2 negative R² is a small-dataset artefact: that fold has 9 zero-label clusters (vs 0 in folds 1 and 5), causing the model trained without them to mispredict the tails
- High fold variance (σ=0.173) reflects the 70-cluster held-out sets being too small for stable R² estimates, not model instability — MAE/RMSE are much more consistent (σ≈0.05–0.07)
- Spatial structure not captured by tabular features; CNN expected to improve on this

### Shallow CNN (`train_cnn.py`)

**Input:** 128×128 arcsinh-normalised radio emission images from `dataset.h5`.
Each projection treated as an independent sample (1056 total); augmented 8× on-the-fly (4 rotations × 2 flips).
Architecture: 4 conv blocks (32→64→128→256 channels) + global avg pool + 2-layer MLP head.
Trained with AdamW + cosine LR schedule + Huber loss. Best checkpoint per fold saved.
**Target:** `label_score_all_tau1.0` at snap 99.

**5-fold GroupKFold CV (grouped by cluster):**

| Fold | R² | MAE | RMSE |
|------|-----|-----|------|
| 1 | 0.452 | 0.503 | 0.644 |
| 2 | 0.305 | 0.425 | 0.515 |
| 3 | 0.525 | 0.359 | 0.491 |
| 4 | 0.454 | 0.432 | 0.610 |
| 5 | 0.546 | 0.351 | 0.460 |
| **mean ± std** | **0.456 ± 0.084** | **0.414 ± 0.055** | **0.544 ± 0.071** |
| **OOF** | **0.472** | | |

### Comparison

All results use 5-fold CV grouped by cluster. Label definitions:
- **tau=1.0**: `label_score_all_tau1.0` — soft merger activity score in last 1 Gyr
- **pseudo-TSC**: interpolated tau where score first crosses 0.5 — proxy for Gyr since last major merger (34/352 capped at 4.0 Gyr)
- **merger-TSC**: ground-truth time since last collision from Lee et al. merger catalog (0–7.7 Gyr, 6/352 dropped)

| Model | Label | OOF R² | MAE | RMSE | R² std |
|---|---|---|---|---|---|
| XGBoost (tabular) | tau=1.0 | 0.290 | 0.469 | 0.632 | 0.173 |
| XGBoost (tabular) | pseudo-TSC | 0.333 | 0.711 | 0.962 | 0.169 |
| Shallow CNN (images) | tau=1.0 | 0.472 | 0.414 | 0.544 | 0.084 |
| Shallow CNN (images) | pseudo-TSC | 0.529 | 0.587 | 0.806 | 0.095 |
| Shallow CNN (images) | merger-TSC (δ=2.0, 120ep) | 0.511 | 0.798 | 1.073 | 0.102 |
| Shallow CNN (images) | merger-TSC (TSC≤2 Gyr, 120ep) | 0.149 | 0.439 | 0.541 | 0.025 |
| Shallow CNN (256px, 4×R500c) | merger-TSC (δ=2.0, 120ep) | 0.507 | 0.771 | 1.075 | 0.097 |
| Shallow CNN (256px, 2×R500c) | merger-TSC (δ=2.0, 120ep) | 0.480 | 0.783 | 1.103 | 0.109 |
| Pooled CNN | pseudo-TSC | **0.564** | — | — | — |
| Pooled CNN | merger-TSC (δ=0.5) | 0.494 | 0.779 | 1.103 | 0.150 |
| Pooled CNN | merger-TSC (δ=2.0) | 0.528 | 0.782 | 1.067 | 0.188 |
| Pooled CNN | merger-TSC (δ=2.0, 120ep) | **0.544** | 0.778 | 1.047 | 0.196 |
| X-ray-only CNN | merger-TSC (δ=2.0, 120ep) | 0.321 | 0.938 | 1.265 | 0.104 |
| Dual-encoder (radio+X-ray) | merger-TSC (δ=2.0, 120ep) | 0.511 | 0.777 | 1.073 | 0.078 |

Key observations:
- CNN consistently outperforms XGBoost, confirming spatial structure in radio images carries signal beyond tabular morphology features
- Pseudo-TSC label improves CNN OOF R² by +0.057 and substantially reduces the train/val gap, suggesting it is a cleaner regression target
- Merger-catalog TSC is a harder target than pseudo-TSC (wider range 0–7.7 Gyr, right-skewed), but the pooled CNN still achieves R²=0.544 with tuned Huber delta and more epochs
- Increasing Huber delta from 0.5 to 2.0 improved merger-TSC results (+0.034 OOF R²), as the default delta was too aggressive for the wider label range
- Log-transforming merger-TSC labels did not help (OOF R² 0.443–0.480) — compressing the tail also compressed the discriminative part of the distribution
- **Shallow CNN with merger-TSC (R²=0.511) is the most practical model** — it processes a single projection, so it can be applied directly to real observations where only one line-of-sight view is available. It retains most of the pooled CNN's performance (0.544) with lower fold variance (std 0.102 vs 0.196)
- High fold variance persists across pooled CNN merger-TSC runs (R² std ~0.15–0.20), driven by Fold 1 consistently underperforming (R²~0.12), likely due to cluster composition in that split
- **Filtering to recent mergers (TSC ≤ 2 Gyr, 258 clusters) collapsed R² to 0.149** — the CNN's performance on the full set is largely driven by separating "recently merged" vs "long ago merged" (a coarse distinction), not precisely timing recent mergers. Within the recent-merger subset, radio morphologies are too similar for the model to distinguish 0.3 vs 1.5 Gyr
- **Higher resolution images did not help**: 256×256 at 4×R500c (R²=0.507) matched the 128px baseline, and 256×256 at 2×R500c (R²=0.480) was slightly worse — the tighter crop likely cuts off outer relic structure. The bottleneck is not pixel resolution but the difficulty of learning geometric features (like relic separation) implicitly from raw images with only ~350 training samples
- **X-ray alone is substantially weaker than radio** (OOF R² 0.321 vs 0.511). X-ray surface brightness traces thermal gas (density² × temperature½), which correlates with mass and relaxation state but is a weaker TSC indicator than radio relics/halos. Training was also less stable — some folds showed wildly negative R² mid-training before recovering at the best-epoch checkpoint
- **Adding X-ray to radio via dual-encoder fusion did not improve over radio-only** (R² 0.511 vs 0.511). The dual encoder uses separate CNN backbones for each modality with concatenated features before the MLP head, but the X-ray channel contributes no complementary information — the model overfits harder (train R² ~0.9 vs val R² ~0.3–0.5) with twice the parameters. The bottleneck remains what the CNN can extract from images at this sample size, not missing modalities

## Generative Modeling (Diffusion)

Exploratory work on generating synthetic radio cluster maps with a diffusion
model, motivated by eventually conditioning on a mass map to produce realistic
synthetic observations. The first question — whether a diffusion model can be
trained on only ~350 independent clusters without simply memorizing them — has
been answered for radio at 64px.

**Pipeline:**
- `build_diffusion_data.py` — projects raw radio NPZ to 64px maps with a
  *tuned, invertible* `arcsinh(x / a)` stretch (robust scale `a` = median
  positive pixel, high-percentile clip). Unlike `dataset.h5`'s parameter-free
  `np.arcsinh`, the stretch + scaling parameters are stored so generated
  samples can be mapped back to physical units for evaluation.
- `train_diffusion.py` — 8.5M-param UNet DDPM (cosine schedule, EMA),
  cluster-level train/val split (3 projections kept in the same fold, per the
  no-leakage rule), 8× rotation/flip augmentation.
- `submit_diffusion.sh` (baseline, p99.9 clip) and `submit_diffusion_v2.sh`
  (sharpness tune, p99.99 clip) — SLURM `gpu`, ~400 epochs (~minutes on one
  GPU).

Two runs of interest are checked in: a **baseline** (`diffusion_out/`,
p99.9 bright-pixel clip) that established feasibility, and a **v2 sharpness
tune** (`diffusion_out_v2/`, p99.99 clip) that preserves compact bright knots
the original clip was wiping out. Single-knob change between the two; same
UNet, schedule, seed, and arcsinh knee `a`.

**Fidelity (low-*k* power ratio, gen / val):**

| Space | baseline | v2 | Note |
|---|---|---|---|
| physical (`a·sinh(...)`) | 9.08× | **1.49×** | physical-space PSD is `sinh()`-inflated — the inverse stretch is violently exponential (`sinh(20.8) ≈ 5e8`), so tiny high-end errors blow up here. Less clipping → less high-end error → 6× drop in the inflated metric. |
| **normalized [-1, 1]** (training space) | 1.20× | **1.00×** | the meaningful readout; both runs are essentially correct on large scales, v2 is perfect |
| DC (total flux, normalized) | 0.91× | **0.99×** | total-flux scatter matched |
| log10 physical | 0.10× | — | baseline-only diagnostic: gen had *less* large-scale variance, i.e. mildly over-smoothed bright knots → motivated v2 |

An earlier write-up called the 9× physical-space figure a "low-*k* excess
defect"; the diagnostic in `diffusion_out/psd_diag.png` showed it was almost
entirely the `a·sinh()` inverse exaggerating a small bright-end error. The
v2 clip relaxation then directly addressed that residual — bright knots
survive into training, the model learns them, and both physical and
normalized PSDs come into line.

**Memorization** is read with a brightness-corrected metric (see "Metric
upgrade" below) — both runs are clean:

| Metric | baseline | v2 |
|---|---|---|
| `gen→train` NN, **mean-subtracted** / train-train baseline | 1.02 | 0.90 |
| brightness-vs-NN correlation | +0.90 | +0.93 |
| visual nearest-neighbour panel | n/a | `diffusion_out_v2/nn_check.png` — no copied morphologies |

Baseline ratio 1.02 means generated samples sit at almost exactly the same
structural distance from the training set as training images sit from each
other (no copying). v2 ratio 0.90 is mildly inside that envelope but still
above the 0.85 memorization-alert threshold, and the visual NN panel
confirms zero copied cluster morphologies — the "close" samples are
near-empty fields matching the dataset's faintest training image, which is
generic faint-cluster behaviour, not memorization.

**Metric upgrade.** Re-running the upgraded `evaluate()` on the existing
samples surfaced that **both runs have brightness-vs-NN correlation ≈ 0.9**.
On heavy-zero-content data like cluster maps, raw L2 NN is dominated by
"how empty is this image" rather than by morphology, so a pair of near-empty
fields trivially gets small L2. The baseline's earlier "no memorization,
NN ratio 1.008" reading was lucky — we'd have been blind to a real signal
in either direction. `evaluate()` now prints:

1. **Mean-subtracted NN distance** (primary, structural, brightness-independent),
   reported as a ratio against the same-statistic train-train baseline.
2. **Brightness-vs-NN correlation** as an automatic guard; if `|corr| ≥ 0.5`
   the printed summary and figure title flag raw L2 as unreliable.
3. **Normalized-space PSD panel** alongside the physical one (flagged as
   sinh-inflated for continuity), with the scalar low-*k* and DC ratios.

Memorization-alert rule: MS ratio < 0.85 *and* brightness-corr < 0.5.

**Figures (all in `diffusion_out/` and `diffusion_out_v2/`):**
- `eval_final.png` — per-run fidelity + memorization readout (v2 figure
  predates the metric upgrade; future runs use the new layout automatically)
- `sim_vs_gen.png` — side-by-side simulated vs. generated radio maps
  (baseline)
- `psd_diag.png` — three-space PSD diagnostic (baseline) showing the
  physical-space metric inflation
- `nn_check.png` — v2 generated samples and their nearest training images,
  confirming no morphology copying

**Implication.** The core risk for the synthetic-observation idea is
retired for radio@64px: v2 reproduces both large-scale power and total flux
to ~1.00× of held-out clusters in the meaningful space, with no copied
morphologies under a brightness-corrected memorization check. Conditioning
is worth pursuing. Because the unconditional generator already works, the
high-value conditioner is a **total / DM-dominated mass map** (what weak
lensing reconstructs, physically independent of the radio/X-ray being
generated) rather than a gas mass map (free from the existing gas-only
cutouts but largely redundant with the X-ray channel). The DM-map path
requires downloading dark-matter particles for the 352 halos from the
public TNG-Cluster release. Next steps: adopt v2 as the new baseline →
repeat the feasibility test for X-ray → scope the DM-particle download.

## Relic-Distance Labels (negative result)

Phase 0 of a planned effort to teach the CNN about radio relics' merger
geometry — specifically the Lee correlation `d_relic ∝ TSC`. The
shallow-CNN result that motivated this is that restricting to recent
mergers (TSC ≤ 2 Gyr) collapses R² to 0.149 (see Comparison table) — the
CNN is doing coarse "recent merger vs not" classification rather than
*timing* recent mergers, and observed double/single radio relics carry
exactly that timing information through their distance from the BCG.

Since we don't have observational relic catalogs (Lee's labelled sample
isn't available), the plan was to derive relic positions programmatically
from the simulation data, validate against Lee's published relation, then
either feed (TSC, d_relic) into a multi-task CNN or condition the
diffusion model on it for relic-geometry-aware synthetic data. The Lee
validation is the prerequisite — without it the labels mean nothing.

**Detectors built:**
- `detect_relics.py` — Mach-derived: reads
  `/oscar/data/idellant/Chuiyang/TNGCluster_Cutout/snap99/cutout_sub*_FOF*.hdf5`,
  filters `PartType0` gas cells by Mach ≥ threshold, projects M² × ρ as
  a shock surface-brightness proxy, finds local maxima per projection,
  filters peaks to the annulus `r_min_frac < r < r_max_frac × R500c`.
  Outputs `relic_catalog.h5` (or `_v2`) joined with `pseudo_tsc` /
  `merger_tsc` (from `TSC_Cutimages/TSC_eachhalo_snap99.hdf5`) and
  `mass_ratio`.
- `detect_relics_radio.py` — independent radio-map peak detection on
  `dataset_512.h5`. Threshold set from the annulus max (the central halo
  outshines any peripheral relic, so a global-max threshold would silently
  kill every outer peak). Same output schema for direct comparison.
- `plot_relic_validation_v3.py` — per-projection treatment (each
  `halo × proj` is one observational analog), restriction to the
  bow-shock-active TSC regime, four-panel stratified scatter.

**Four independent validations, all negative:**

| Run | Detector | Label | Slicing | corr(TSC, d) on cleanest subset |
|---|---|---|---|---|
| v1 | Mach ≥ 2, peak ≥ 5%, sep ≥ 300 kpc | pseudo-TSC, 3-proj mean | clean (n ≤ 2), n=32 | +0.07 |
| v2 | Mach ≥ 3, peak ≥ 25%, sep ≥ 600 kpc, r ≤ 2 R500c | merger-TSC, 3-proj mean | major + clean (mr ≥ 0.25), n=11 | −0.10 |
| v3 | (v2 catalog re-sliced) | merger-TSC, per-projection | active + significant + single relic (n=1), n=46 | **−0.149** |
| radio | independent peak detection on `dataset_512.h5` | merger-TSC, per-projection | active + significant (mr ≥ 0.1), n=132 | **−0.146** |

Detector cleanliness was *not* the issue: v2 produced a median of 2
relics per projection, matching the observed double-relic mode, with a
small zero-shock tail for quiescent clusters. The clean-subset / `major +
clean` / `single-relic` strata are progressively closer to Lee's
observational sample, but the correlation gets *more* negative there,
not less. The dominant pattern across all three runs is "high-TSC
clusters have *smaller* d_max" — physically consistent with bow shocks
dissipating or escaping the cluster, leaving only inner residual shocks
in late-stage mergers.

**Why this probably failed** (none disprovable without more work):

1. **Bow-shock dissipation timescale.** Lee's positive relation assumes
   the bow shock keeps propagating outward over Gyr. In TNG-Cluster the
   high-Mach cells at old merger-TSC are likely *not* the original bow
   shock (escaped or dissipated) but residual internal / sloshing /
   accretion shocks, often at smaller radii. That alone gives an
   aggregate negative trend.
2. **Selection-effect mismatch.** Lee's ~28-cluster observational sample
   is selected for *having visible relics*. The TNG-Cluster sample is
   unbiased — most z=0 halos are not in the active relic-bearing state.
   Even "major + clean" includes many quiescent systems where what we're
   picking up isn't a Lee-type relic at all.
3. **Possible label issue.** `TSC_eachhalo_snap99.hdf5` was pre-joined by
   a previous pipeline; we haven't traced whether it's "since the most
   recent merger of any mass" vs. "since the most recent significant
   merger." A late minor merger could override the bow-shock-relevant
   major merger.

**Decision:** drop relic-distance labels as the route to teaching the
CNN about merger geometry. Pivot to **TSC-scalar-conditional diffusion**
— synthesise (image, TSC) augmentation samples using the labels we
already have, mix into CNN training, evaluate whether synthetic data
moves the needle on the recent-merger subset. This is the original
"Idea B" (synthetic-data augmentation as general methodology) without
the relic-geometry dependence.

The radio-map detector independently confirms the result: with the
threshold set from the *annulus* max (since the central halo outshines
every peripheral relic) and the same radial filter / minimum separation
as the Mach v2 detector, the active-regime correlation is −0.013 (i.e.
zero) and gets *more* negative as we tighten to major mergers (−0.146).
Two completely independent derivations of "where is the radio relic" —
one from gas-cell shock physics, one from the synchrotron map itself —
agree that the Lee signal isn't present in TNG-Cluster at this resolution.

**Figures:**
- `relic_v2_validation.png`, `relic_v2_count_hist.png` — Mach v2,
  merger-TSC, mass-ratio stratification
- `relic_v3_validation.png` — Mach v2 catalog re-sliced per-projection
  in the active TSC regime; the four-panel "where does the signal live"
  diagnostic
- `relic_radio_validation.png` — independent radio-map peak detection;
  same four-panel layout, same null result

## TSC-Conditional Diffusion (in progress)

After the relic-detection negative result, the planned synthetic-data
augmentation pipeline pivoted to its simplest form: condition the
diffusion model directly on the TSC scalar we already have (no relic
geometry required), generate `(image, TSC)` augmentation pairs, mix them
into CNN training, and see whether the recent-merger subset where the
CNN currently collapses to R² 0.149 improves. The gate question before
that augmentation experiment: does conditioning actually *take* —
i.e., do generated samples vary morphologically with the TSC condition
in a way that matches real clusters at that TSC?

**Implementation** (`train_diffusion.py` upgraded in place; unconditional
mode preserved as the default):
- `--condition` flag enables a Fourier-embed of the scalar label, MLP-
  projected to the time-embedding dimension and **summed into the
  timestep embedding** before any ResBlocks see it. A learned **null
  token** replaces the condition embedding when the label is NaN; this
  enables classifier-free guidance.
- Training uses CFG dropout (`--cond-drop-prob 0.1`, standard) so the
  model sees the null occasionally and learns both `p(x|c)` and `p(x)`.
- Sampling uses CFG (`--cfg-scale`), with `--sample-tsc` to choose the
  grid of condition values. End-of-run produces:
  - `cond_grid_final.png` — sample grid, rows by TSC value
  - `cond_trend_final.png` — gen brightness vs TSC overlaid with real
    val brightness vs TSC (the aggregate-stat trend)
  - `eval_cond_final.png` — the standard fidelity + memorization readout
- `plot_cond_nn_check.py` (separate diagnostic): for each gen sample,
  find its nearest training neighbour by mean-subtracted L2, and
  produce two figures stratified by gen TSC bin:
  - `nn_check_cond.png` — visual `(gen, NN train)` pairs
  - `cond_leakage.png` — histogram of "what TSC does the nearest
    training neighbour live at?" overlaid with the overall training TSC
    distribution

**v1 (cfg=1.5) result — gate failure, attribution clear.**

Training and aggregate-stat trend look encouraging:

| Signal | Value |
|---|---|
| `corr(condition, gen brightness)` | **−0.458** (well past the 0.3 "condition is taking" threshold) |
| Gen brightness curve vs real-val curve | Same direction; gen has stronger monotone shift, real val noisier at the tail |
| Visual `cond_grid_final.png` | Clear morphological progression down the rows: low TSC → bright, extended, complex; high TSC → sparse, dim, point-like |

But the NN check is decisive in the wrong direction. The
`cond_leakage.png` panels show that the orange "NN-train TSC" histogram
**peaks at ~3 Gyr regardless of the gen TSC condition** — the dashed
red gen-condition line sits well away from the orange peak in every
panel. The per-bin table:

| gen TSC | n | gen→train NN | ⟨NN train TSC⟩ | train-train NN @ same TSC |
|---|---|---|---|---|
| 0.30 | 16 | 22.5 | 2.68 ± 1.35 | 23.2 |
| 1.00 | 16 | 17.6 | 2.76 ± 1.07 | 22.7 |
| 2.00 | 16 | 20.6 | 2.83 ± 1.15 | 18.7 |
| 4.00 | 16 | 16.9 | 3.26 ± 0.40 | 11.8 |
| 6.00 | 16 |  7.3 | **3.15 ± 0.00** |  8.2 |

The std=0.00 at TSC=6 is a smoking gun: **all 16 generated samples at
the highest condition find their nearest neighbour at the same single
training image** (one at TSC≈3, nowhere near 6). Mean-subtracted gen NN
distance there (7.3) is also *below* the train-train baseline (8.2) —
classic memorization at the tail.

**Interpretation.** The model has learned the condition as a **global
brightness knob** but not as a morphological selector. The trend in
aggregate stats (`corr=−0.46`) and the visible row progression in
`cond_grid` are real, but the per-sample morphology comes from a small
set of "average cluster template" memories. Synthetic samples at, say,
TSC=4 don't actually look like real TSC=4 clusters — they look like
average mid-TSC (~3 Gyr) clusters dimmed down. **Mixing these into CNN
training would not teach the geometric signal we want.**

Likely cause: adding the condition embedding only to the timestep
embedding is the weakest form of conditioning. The timestep signal
dominates and the small constant condition offset gets washed out,
especially given ~12 effective training clusters per TSC bin.

**CFG sanity-check (sample-only, no retrain):** the model's
`--sample-only` mode was added to `train_diffusion.py` so we could
re-load the trained EMA and resample at higher CFG without paying the
training cost. Three CFG values compared on the same checkpoint:

| | CFG=1.5 | CFG=3.0 | CFG=5.0 |
|---|---|---|---|
| `corr(cond, gen brightness)` | −0.46 | −0.68 | **−0.70** |
| ⟨NN train TSC⟩ @ gen=0.3 | 2.68 ± 1.35 | 2.35 ± 1.44 | **2.03 ± 1.29** |
| ⟨NN train TSC⟩ @ gen=1.0 | 2.76 ± 1.07 | **1.91 ± 1.63** | 2.43 ± 2.11 |
| ⟨NN train TSC⟩ @ gen=2.0 | 2.83 ± 1.15 | 2.85 ± 0.93 | 3.16 ± 1.48 |
| ⟨NN train TSC⟩ @ gen=4.0 | 3.26 ± 0.40 | 3.15 ± **0.00** | 3.15 ± **0.00** |
| ⟨NN train TSC⟩ @ gen=6.0 | 3.15 ± **0.00** | 3.15 ± **0.00** | 3.15 ± **0.00** |
| `gen→train` NN @ gen=6.0 | 7.3 | 5.7 | **5.6** (memorised harder) |

**Two outcomes that change the plan:**

1. **In the data-dense regime (TSC < 2 Gyr), CFG=5 fixes the
   conditioning.** The cfg=5 `cond_leakage_cfg5.png` panels show the
   gen=0.3 orange "NN-train-TSC" histogram concentrating near low TSC
   (overlapping the dashed condition line), where at cfg=1.5 it was
   stuck at ~3 Gyr. Per-bin mean NN-train-TSC at gen=0.3 moved
   2.68 → 2.35 → **2.03** across the CFG sweep, and brightness
   correlation strengthened from −0.46 to −0.70. The conditioning
   *is* in the weights; cfg=1.5 was just under-amplifying it.
2. **At the high-TSC tail (gen ≥ 4 Gyr), CFG amplifies memorisation
   rather than fixing it.** All 16 generated samples at TSC=4 and TSC=6
   map to the *same single training image* at TSC≈3 (std = 0.00 at all
   CFG values), and the NN distance there shrinks from 7.3 → 5.6 as CFG
   grows. The high-TSC tail of the training set has too few clusters
   (probably <10 across all 3 projections × ~3 clusters), and CFG
   sharpens whatever pattern the model has — which there is a copy of
   the few examples that exist. **No architecture change (including
   AdaGN) can manufacture data that isn't there**, so the tail is a
   fundamental data-density limit, not a conditioning-architecture
   limit.

**AdaGN dropped from the plan.** It would have helped if the
conditioning signal had been too weak in the weights, but cfg=5 shows
the signal *is* there and just needs amplification at sample time. The
remaining problem is data, not architecture.

**Convenient alignment:** the regime where the conditioner works
cleanly (TSC ∈ [0.3, 2.0] Gyr) is exactly the regime where the CNN
currently collapses (the TSC ≤ 2 Gyr restriction reduced merger-TSC R²
from 0.511 to 0.149 — see Comparison table). So the augmentation
experiment can be restricted to the data-dense range without giving up
any of the practical value.

**Phase 2 — augmentation experiment at 64px (inconclusive):**

Pipeline (`submit_cnn_aug.sh`): sample 540 synthetic `(image, TSC)`
pairs at CFG=5 in TSC ∈ [0.3, 2.0] Gyr → train two CNNs at 64px on
`diffusion_radio_64_v2.h5` with everything else identical except whether
those synthetic samples are mixed in. Three seeds per condition for noise
floor; cluster-level split fixed at seed=0 across all runs.

Results (3 seeds each):

| Metric | baseline | aug | Δ | Δ / joint σ |
|---|---|---|---|---|
| R² all (TSC 0–7.7) | +0.097 ± 0.056 | +0.107 ± 0.042 | +0.010 | 0.14 (noise) |
| **R² recent (TSC ≤ 2)** | **−0.894 ± 0.209** | **−0.573 ± 0.098** | **+0.321** | **1.39** |
| R² very recent (TSC ≤ 1) | −7.455 ± 0.205 | −5.617 ± 0.617 | +1.838 | 2.83 |
| R² late (TSC > 2) | −4.777 ± 0.691 | −5.029 ± 0.442 | −0.252 | 0.31 (noise) |

**Honest read.** Augmentation moves the recent-merger R² in the right
direction by Δ ≈ +0.32 (well past joint seed noise, ~1.4σ on recent and
2.8σ on very-recent), with no measurable effect on the late-merger
subset or overall R². But **both conditions are catastrophic in absolute
terms on the recent subset** — both negative R², meaning the CNN at 64px
just can't learn the recent-merger signal regardless of augmentation. The
`comparison.png` scatter shows both flatlining at the dataset-mean ~1.0–
1.3 Gyr for true TSC ∈ [0, 2]. The existing pooled CNN at 128px got R²
0.149 on this subset; at 64px we're below the resolution floor where that
signal lives. So the result is *consistent with* "synthetic augmentation
helps" but does not validate it usefully — the methodology improved a
catastrophic baseline to a less-catastrophic one.

**Phase 2 v2 — 128px scale-up (in progress):**

The decisive test is at 128px where the unaugmented baseline is +0.149
on the recent-merger subset. Same methodology, scaled up:

1. Build `diffusion_radio_128_v2.h5` (tuned arcsinh, just `--img-size 128`).
2. Train conditional diffusion at 128px (same UNet config, more pixels).
3. CNN baseline-vs-aug experiment at 128px (architecture scaled with
   resolution; 3 seeds each).
4. Compare. If Δ on recent-merger R² holds at this scale, we land near
   ~0.47 — a meaningful improvement on the published 0.149. If Δ
   collapses, the 64px improvement was an artifact of the catastrophic
   baseline and the methodology isn't actually adding signal.

**Figures (sample suite per CFG):**
- `diffusion_out_cond/cond_grid_final{,_cfg3,_cfg5}.png` — sample grid by TSC row
- `diffusion_out_cond/cond_trend_final{,_cfg3,_cfg5}.png` — gen brightness vs TSC
- `diffusion_out_cond/eval_cond_final{,_cfg3,_cfg5}.png` — fidelity + memorisation
- `diffusion_out_cond/nn_check_cond{,_cfg3,_cfg5}.png` — visual `(gen, NN train)` pairs
- `diffusion_out_cond/cond_leakage{,_cfg3,_cfg5}.png` — the decisive
  per-bin NN-TSC histogram diagnostic
- `cnn_aug_out/comparison.png` — 64px CNN baseline vs aug predictions
  (three panels: all TSC, ≤ 2 Gyr, ≤ 1 Gyr)

## Future Work

### Pretrained Vision Backbone (Linear Probing)

The idea: freeze a pretrained vision backbone, extract embeddings, train a linear regression head on top. This is called **linear probing**.

**Key finding from the literature:** [Bridging the Gap (arXiv:2409.11175)](https://arxiv.org/abs/2409.11175) benchmarked MAE, DINOv1, DINOv2, MSN, SigLIP, and AM-RADIO on radio astronomy tasks. Best performers on radio classification were SigLIP and AM-RADIO, but all models were tested on *discrete* radio sources (FR I/II morphologies), not diffuse cluster emission (halos/relics). No established pretrained model exists for diffuse emission.

**Available options, roughly in order of relevance:**
- **DINOv2** — strongest general-purpose feature extractor for linear probing; avoids ImageNet texture bias better than ResNet; available via `torch.hub`; best bet if trying this approach
- **Radio Galaxy Zoo SSL model** ([RASTI 2024](https://academic.oup.com/rasti/article/3/1/19/7491070)) — only model trained on actual radio survey images (FIRST), but on discrete jet/lobe morphologies, not halos
- **ResNet/SigLIP/AM-RADIO** — ImageNet-pretrained; large domain gap from natural images to arcsinh radio maps

**Bottom line:** none of these were trained on diffuse cluster emission. The shallow CNN trained directly on this data may well outperform all of them. Worth trying DINOv2 as the best available option, but expectations should be low. Fine-tuning (unfreezing the backbone) is likely to overfit badly at N=352.

### Explicit Geometric Features

Higher resolution images (256×256), narrower crops (2×R500c), and multi-modal fusion (radio + X-ray) all failed to improve merger-TSC prediction, confirming the CNN cannot implicitly learn the relic separation geometry that Lee et al. (2024) showed correlates with TSC (Pearson r=0.83). With only ~350 clusters, the model lacks the data to discover this relationship from raw pixels. Possible approaches:

- **Extract relic separation as a feature** — identify relics in the radio images (e.g. via thresholding + connected components) and measure their separation normalized by R_500c. Feed this alongside the CNN embedding or use it directly.
- **Hybrid model** — CNN image features concatenated with hand-crafted geometric features (relic separation, relic luminosity, morphology asymmetry) before the regression head.
- **X-ray morphological features** — established X-ray merger diagnostics (centroid shift, concentration parameter, power ratios, asymmetry) could be extracted from the simulated Chandra images and used as explicit features. The CNN at 128px resolution with 346 training samples cannot learn these implicitly, but they are straightforward to compute from the images directly.
- **More training data** — TNG300-1 adds ~120 relic systems (Lee et al. 2024); combining with TNG-Cluster's 352 may help the CNN learn finer structure.

### VLASS
- Add support for VLASS Epochs 2 and 3 when they become available via Vizier
- Implement direct catalog download from CIRADA/CADC if needed

### eROSITA
- Download cutouts manually from eROSITA website https://erosita.mpe.mpg.de/dr1/erodat/

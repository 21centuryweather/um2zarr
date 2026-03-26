# um2zarr

Convert Unified Model (UM) fieldsfiles to cloud-optimised Zarr format.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
[![Status](https://img.shields.io/badge/status-beta-green.svg)](https://github.com/ACCESS-Community-Hub/um2zarr)

---

## Installation

```bash
git clone https://github.com/ACCESS-Community-Hub/um2zarr.git
cd um2zarr
conda install -c conda-forge scitools-iris mo-pack cftime
pip install -e .

# Optional extras
pip install -e ".[notebooks]"   # Jupyter gallery deps
pip install -e ".[cloud]"       # s3fs, gcsfs, adlfs
pip install -e ".[dev]"         # pytest, black, mypy
```

---

## Quick start

```bash
# Single file
um2zarr convert input.pp output.zarr

# With chunking and compression
um2zarr convert input.pp output.zarr \
  --chunk-lat 256 --chunk-lon 256 \
  --compression zstd --compression-level 3 \
  --workers 4

# Batch convert a directory, resume on re-run
um2zarr convert data/ output/ \
  --resume --on-error skip \
  --write-catalogue output/catalogue.yaml

# CMIP6-compliant output
um2zarr convert input.pp output.zarr \
  --cmor --cmor-experiment-id historical \
  --cmor-source-id ACCESS-GA9 --cmor-variant-label r1i1p1f1

# Rechunk an existing store
um2zarr-rechunk input.zarr output_ts.zarr --target timeseries
```

Or via a YAML config file:

```bash
um2zarr convert input.pp output.zarr --config config.example.yaml
```

---

## Python API

```python
from um2zarr import ConversionOrchestrator, ProcessingConfig

config = ProcessingConfig(
    chunk_time=1, chunk_lat=256, chunk_lon=256,
    compression='zstd', compression_level=3,
    workers=4, resume=True, on_error='skip',
)
orchestrator = ConversionOrchestrator(config)
orchestrator.convert_file('input.pp', 'output.zarr')
```

---

## Features

**Conversion**
- UM fieldsfiles and PP files via iris/mule
- Lazy Dask-parallel processing, configurable chunking and compression
- Append mode for extending existing Zarr stores along the time axis
- Ensemble dimension support

**Reliability**
- `CheckpointManager` — JSON sidecar tracks completed files; re-runs skip them automatically
- `--on-error skip | retry | abort` modes
- Dask graph size monitoring with automatic time-axis splitting

**Metadata**
- 4 700+ entry STASH-to-CF mapping table with supplemental entries and custom override support
- CMIP6 / CMOR output — variable renaming, units conversion, global attribute injection, CF validation

**Ecosystem**
- Intake / intake-esm catalogue generation (`--write-catalogue`)
- Post-conversion rechunking (`um2zarr-rechunk`) with `timeseries`, `map`, and `profile` presets
- Notebook gallery in `notebooks/` — six worked examples using real UM test data

---

## Notebook gallery

Six Jupyter notebooks in `notebooks/` using a 12 km GAL9 NWP test dataset:

| Notebook | Topics |
|---|---|
| `00_gallery_index.ipynb` | Environment check, dataset overview |
| `01_quickstart.ipynb` | Single-file conversion, xarray, quick plot |
| `02_exploring_um_data.ipynb` | Binary header inspection, STASH inventory |
| `03_stash_metadata.ipynb` | CF mapping, CMIP6 cross-reference |
| `04_batch_conversion.ipynb` | Batch convert, checkpointing, resume |
| `05_cmor_output.ipynb` | CMIP6 renaming, units, global attributes |
| `06_rechunking.ipynb` | Timeseries vs map presets, benchmarks |

---

## Architecture

```
Orchestration  →  ConversionOrchestrator · CheckpointManager · CLI
Processing     →  StashMetadataManager · CMORProcessor · ChunkManager
I/O            →  UMFileReader · ZarrWriter · IntakeCatalogueWriter
```

---

## Dependencies

| Package | Role |
|---|---|
| `scitools-iris` | UM file reading |
| `xarray` | Dataset representation |
| `zarr` | Output format (v2 and v3) |
| `dask[distributed]` | Parallel processing |
| `cftime` | Calendar-aware time encoding |
| `click` + `pyyaml` | CLI and config |

---

## Development

```bash
pytest                          # run tests
black um2zarr/                  # format
mypy um2zarr/                   # type-check
```

See [ROADMAP.md](ROADMAP.md) for planned features.

---

## License

MIT — see [LICENSE](LICENSE).

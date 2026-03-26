"""
Post-write rechunking for different downstream access patterns.

After the initial conversion (which produces time-chunked Zarr stores optimised
for *writing*), this module lets users reshape the chunk layout without
re-reading any UM source files.

Three preset layouts are exposed via the ``RechunkTarget`` enum:

=========== =====================================================
Preset      Optimal chunk shape
=========== =====================================================
timeseries  ``{time: N, lat: 1, lon: 1}`` — point extraction
map         ``{time: 1, lat: M, lon: M}`` — spatial slicing
profile     ``{time: 1, lev: L, lat: 1, lon: 1}`` — vertical
=========== =====================================================

Usage::

    from um2zarr.processing.rechunker import rechunk_store, RechunkTarget

    rechunk_store(
        store_path="output.zarr",
        target=RechunkTarget.MAP,
        max_mem="4GB",
    )

The preferred backend is the ``rechunker`` library (zero-copy, uses a staging
store).  If rechunker is not installed, a fallback path based on xarray and
Zarr is used (reads and rewrites the whole store).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional rechunker import
# ---------------------------------------------------------------------------

try:
    import rechunker as _rechunker_lib

    HAS_RECHUNKER = True
except ImportError:
    HAS_RECHUNKER = False

try:
    import zarr

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class RechunkTarget(str, Enum):
    """Preset chunk shapes for common access patterns."""

    TIMESERIES = "timeseries"  # Extract single-point time series
    MAP = "map"  # Read full horizontal slices
    PROFILE = "profile"  # Extract vertical profiles


def rechunk_store(
    store_path: str | Path,
    target: RechunkTarget | str = RechunkTarget.MAP,
    output_path: str | Path | None = None,
    max_mem: str = "2GB",
    time_chunk: int = 24,
    spatial_chunk: int = 128,
    level_chunk: int = 20,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Rechunk an existing um2zarr Zarr store for a specific access pattern.

    Parameters
    ----------
    store_path:
        Path to the source Zarr store (produced by ``um2zarr``).
    target:
        A ``RechunkTarget`` preset or the string ``'timeseries'``, ``'map'``,
        or ``'profile'``.
    output_path:
        Destination Zarr store.  Defaults to ``<store_path>.rechunked.zarr``
        alongside the source.
    max_mem:
        Maximum memory budget for the rechunker staging step, e.g. ``'4GB'``.
        Only used when the rechunker library backend is active.
    time_chunk:
        Number of time steps per chunk (timeseries preset).
    spatial_chunk:
        Number of grid points per horizontal chunk edge (map preset).
    level_chunk:
        Number of vertical levels per chunk (profile preset).
    overwrite:
        If *True*, delete any existing output store before writing.

    Returns
    -------
    dict
        Statistics including ``output_path``, ``backend``, ``success``, and
        any error details.
    """
    store_path = Path(store_path)
    target = RechunkTarget(str(target))

    if output_path is None:
        output_path = store_path.with_suffix("").with_suffix("").parent / (
            store_path.stem + ".rechunked.zarr"
        )
    output_path = Path(output_path)

    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
            logger.info(f"Removed existing output store: {output_path}")
        else:
            raise FileExistsError(
                f"Output store already exists: {output_path}.  "
                "Pass overwrite=True to replace it."
            )

    if not HAS_XARRAY:
        raise ImportError(
            "xarray is required for rechunking.  Install with: pip install xarray"
        )

    logger.info(
        f"Rechunking {store_path} → {output_path} "
        f"(target={target.value}, max_mem={max_mem})"
    )

    # Load source dataset lazily
    try:
        ds = xr.open_zarr(str(store_path), consolidated=None)
    except Exception as exc:
        return {
            "success": False,
            "error": f"Cannot open source store: {exc}",
            "output_path": str(output_path),
        }

    # Build per-variable target chunk dicts
    target_chunks = _build_target_chunks(
        ds, target, time_chunk, spatial_chunk, level_chunk
    )
    logger.debug(f"Target chunks: {target_chunks}")

    # Try preferred rechunker backend
    if HAS_RECHUNKER and HAS_ZARR:
        return _rechunk_with_rechunker(
            ds, store_path, output_path, target_chunks, max_mem
        )
    else:
        logger.warning(
            "rechunker library not found — falling back to xarray re-write. "
            "Install with: pip install rechunker"
        )
        return _rechunk_with_xarray(ds, output_path, target_chunks)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dim_names(ds: xr.Dataset) -> dict[str, str]:
    """Return a canonical mapping {role: actual_dim_name} by probing coordinates."""
    dims = set(ds.dims)

    # time
    time_dim = next(
        (d for d in dims if d in {"time", "t", "forecast_reference_time"}), None
    )
    # horizontal
    lat_dim = next((d for d in dims if d in {"lat", "latitude", "grid_latitude"}), None)
    lon_dim = next(
        (d for d in dims if d in {"lon", "longitude", "grid_longitude"}), None
    )
    # vertical
    lev_dim = next(
        (
            d
            for d in dims
            if d
            in {
                "lev",
                "level",
                "pressure",
                "model_level_number",
                "atmosphere_hybrid_sigma_pressure_coordinate",
            }
        ),
        None,
    )

    return {
        "time": time_dim,
        "lat": lat_dim,
        "lon": lon_dim,
        "lev": lev_dim,
    }


def _build_target_chunks(
    ds: xr.Dataset,
    target: RechunkTarget,
    time_chunk: int,
    spatial_chunk: int,
    level_chunk: int,
) -> dict[str, dict[str, int]]:
    """Return ``{var_name: {dim: chunk_size, ...}}`` for every data variable."""
    canon = _dim_names(ds)
    per_var: dict[str, dict[str, int]] = {}

    for var in ds.data_vars:
        da = ds[var]
        spec: dict[str, int] = {}

        for dim, size in da.sizes.items():
            role = next((k for k, v in canon.items() if v == dim), None)

            if target == RechunkTarget.TIMESERIES:
                # Maximise time axis, minimise horizontal
                if role == "time":
                    spec[dim] = min(time_chunk, size)
                elif role in ("lat", "lon"):
                    spec[dim] = 1
                elif role == "lev":
                    spec[dim] = min(level_chunk, size)
                else:
                    spec[dim] = size  # keep other dims intact

            elif target == RechunkTarget.MAP:
                # Minimise time axis, maximise spatial
                if role == "time":
                    spec[dim] = 1
                elif role in ("lat", "lon"):
                    spec[dim] = min(spatial_chunk, size)
                elif role == "lev":
                    spec[dim] = 1
                else:
                    spec[dim] = size

            else:  # PROFILE
                # Single time step, single horizontal point, full vertical
                if role == "time":
                    spec[dim] = 1
                elif role in ("lat", "lon"):
                    spec[dim] = 1
                elif role == "lev":
                    spec[dim] = min(level_chunk, size)
                else:
                    spec[dim] = size

        per_var[var] = spec

    return per_var


def _rechunk_with_rechunker(
    ds: xr.Dataset,
    source_path: Path,
    output_path: Path,
    target_chunks: dict[str, dict[str, int]],
    max_mem: str,
) -> dict[str, Any]:
    """Use the rechunker library for zero-copy staging rechunk."""
    import rechunker as rlib
    import zarr

    logger.info("Using rechunker library backend")

    # Build the rechunker target_store and temp_store
    with tempfile.TemporaryDirectory(prefix="um2zarr_rechunk_") as tmp_dir:
        temp_store = str(Path(tmp_dir) / "staging.zarr")

        # rechunker.rechunk works on zarr groups
        source_zarr = zarr.open_group(str(source_path), mode="r")
        target_zarr = zarr.open_group(str(output_path), mode="w")
        staging_zarr = zarr.open_group(temp_store, mode="w")

        # Build per-array chunk map for rechunker (only data_vars, not coords)
        rechunk_spec: dict[str, Any] = {}
        for var in ds.data_vars:
            if var in target_chunks:
                rechunk_spec[var] = target_chunks[var]
            else:
                rechunk_spec[var] = None  # keep existing chunks

        try:
            plan = rlib.rechunk(
                source_zarr,
                target_chunks=rechunk_spec,
                max_mem=max_mem,
                target_store=target_zarr,
                temp_store=staging_zarr,
            )
            plan.execute()

            # Copy coordinates and attributes unchanged
            ds_coords = ds.drop_vars(list(ds.data_vars))
            ds_coords.to_zarr(str(output_path), mode="a")

            logger.info(f"rechunker backend completed: {output_path}")
            return {
                "success": True,
                "output_path": str(output_path),
                "backend": "rechunker",
                "n_variables": len(rechunk_spec),
            }
        except Exception as exc:
            logger.error(f"rechunker backend failed: {exc}")
            # Clean up partial output
            if output_path.exists():
                shutil.rmtree(output_path)
            return {
                "success": False,
                "error": str(exc),
                "output_path": str(output_path),
                "backend": "rechunker",
            }


def _rechunk_with_xarray(
    ds: xr.Dataset,
    output_path: Path,
    target_chunks: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """
    Fallback: rechunk by reading the whole store into a re-chunked Dask graph
    and writing to a new store.  Uses more memory than the rechunker backend but
    requires no extra dependency.
    """
    logger.info("Using xarray fallback rechunk backend (higher memory usage)")

    try:
        # Apply per-variable rechunking — use the union of chunk specs as
        # dataset-level chunks (xarray takes the first matching spec)
        # Build a single {dim: size} dict from the per-var specs
        unified: dict[str, int] = {}
        for spec in target_chunks.values():
            for dim, sz in spec.items():
                unified.setdefault(dim, sz)

        ds_rechunked = ds.chunk(unified)
        ds_rechunked.to_zarr(str(output_path), mode="w")

        logger.info(f"xarray fallback rechunk completed: {output_path}")
        return {
            "success": True,
            "output_path": str(output_path),
            "backend": "xarray",
            "n_variables": len(ds.data_vars),
        }
    except Exception as exc:
        logger.error(f"xarray fallback rechunk failed: {exc}")
        if output_path.exists():
            shutil.rmtree(output_path)
        return {
            "success": False,
            "error": str(exc),
            "output_path": str(output_path),
            "backend": "xarray",
        }

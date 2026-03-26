"""
Intake catalogue generation for um2zarr Zarr stores.

After conversion, this module generates machine-readable catalogues that let
downstream users discover and open the produced Zarr stores with
`intake <https://intake.readthedocs.io>`_ or compatible tools.

Three catalogue formats are supported:

``intake``
    A YAML file using the ``intake-xarray`` ``zarr`` driver.  Each Zarr store
    is a separate source entry.  This is the simplest format and requires only
    ``intake`` + ``intake-xarray`` on the reader side.

``esm``
    A pair of files — a JSON catalogue descriptor and a CSV asset table —
    conforming to the
    `intake-esm <https://intake-esm.readthedocs.io>`_ specification.  Suitable
    for large collections where filtering by variable, experiment, or member is
    required.

``json``
    A plain provenance/metadata JSON file (no intake dependency required on the
    reader side).  Good for archiving conversion runs.

Usage::

    from um2zarr.io.catalogue_writer import IntakeCatalogueWriter

    writer = IntakeCatalogueWriter(catalogue_dir="/data/zarr/catalogue")
    writer.add_store(
        store_path="/data/zarr/run1.zarr",
        source_files=["/data/um/run1_001.pp", "/data/um/run1_002.pp"],
        dataset=ds,                  # xarray.Dataset, optional but recommended
        write_stats=stats,           # dict from ZarrWriter.write_dataset()
    )
    writer.write(format="intake")    # → /data/zarr/catalogue/catalogue.yaml
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import yaml as _yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from .._version import __version__ as _UM2ZARR_VERSION  # type: ignore[no-redef]
except Exception:
    try:
        from .. import __version__ as _UM2ZARR_VERSION  # type: ignore[assignment]
    except Exception:
        _UM2ZARR_VERSION = "unknown"


# ---------------------------------------------------------------------------
# Store-level metadata helpers
# ---------------------------------------------------------------------------


def _extract_dataset_metadata(
    dataset: Any,
    write_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Extract a rich metadata dict from an xarray Dataset and optional write
    statistics returned by ``ZarrWriter.write_dataset()``.

    Returns a plain dict that is JSON / YAML serialisable.
    """
    meta: dict[str, Any] = {}

    if write_stats:
        meta["compression"] = write_stats.get("compression", "unknown")
        meta["compression_level"] = write_stats.get("compression_level", None)
        meta["compressed_size_mb"] = write_stats.get("compressed_size_mb", None)
        meta["compression_ratio"] = write_stats.get("compression_ratio", None)
        meta["write_time_seconds"] = write_stats.get("write_time_seconds", None)
        meta["n_variables"] = write_stats.get("n_variables", None)

    if not HAS_XARRAY or dataset is None:
        return meta

    # Variable inventory
    variables = []
    for var in dataset.data_vars:
        da = dataset[var]
        entry: dict[str, Any] = {
            "name": var,
            "dtype": str(da.dtype),
            "dims": list(da.dims),
            "shape": list(da.shape),
        }
        for attr_key in ("standard_name", "long_name", "units", "cell_methods"):
            if attr_key in da.attrs:
                entry[attr_key] = da.attrs[attr_key]
        # Chunk shape
        if hasattr(da.data, "chunks"):
            entry["chunks"] = {
                dim: da.data.chunks[i][0]
                for i, dim in enumerate(da.dims)
                if da.data.chunks[i]
            }
        variables.append(entry)
    meta["variables"] = variables

    # Coordinate ranges
    coordinate_ranges: dict[str, Any] = {}
    for coord in dataset.coords:
        da = dataset.coords[coord]
        if da.size == 0:
            continue
        try:
            # Safely compute min/max only for 1-D numeric or datetime-like coords
            if da.ndim == 1:
                vals = da.values
                if hasattr(vals, "astype"):
                    coord_min = str(vals.min())
                    coord_max = str(vals.max())
                    coordinate_ranges[coord] = {
                        "min": coord_min,
                        "max": coord_max,
                        "size": int(da.size),
                        "dtype": str(da.dtype),
                    }
        except Exception:
            pass
    meta["coordinate_ranges"] = coordinate_ranges

    # Global attributes of interest
    global_attrs = {
        k: v for k, v in dataset.attrs.items() if isinstance(v, (str, int, float, bool))
    }
    if global_attrs:
        meta["global_attributes"] = global_attrs

    meta["dimensions"] = dict(dataset.sizes)

    return meta


# ---------------------------------------------------------------------------
# Main writer class
# ---------------------------------------------------------------------------


class IntakeCatalogueWriter:
    """
    Accumulate metadata for one or more Zarr stores and write an intake
    catalogue (YAML, ESM JSON+CSV, or plain JSON) when done.

    Parameters
    ----------
    catalogue_dir:
        Directory where catalogue files will be written.
    catalogue_name:
        Base name for the catalogue files (without extension).
        Defaults to ``'um2zarr_catalogue'``.
    description:
        Human-readable description embedded in the catalogue.
    extra_attributes:
        Dict of additional key-value pairs added to every catalogue entry
        (e.g. ``{'experiment_id': 'historical', 'source_id': 'ACCESS-CM2'}``).
    """

    def __init__(
        self,
        catalogue_dir: str | Path,
        catalogue_name: str = "um2zarr_catalogue",
        description: str = "UM climate model output converted to Zarr by um2zarr",
        extra_attributes: dict[str, Any] | None = None,
    ) -> None:
        self.catalogue_dir = Path(catalogue_dir)
        self.catalogue_name = catalogue_name
        self.description = description
        self.extra_attributes = extra_attributes or {}
        self._entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_store(
        self,
        store_path: str | Path,
        source_files: Sequence[str | Path] | None = None,
        dataset: Any = None,
        write_stats: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a Zarr store with the catalogue.

        Parameters
        ----------
        store_path:
            Absolute path to the Zarr store.
        source_files:
            List of UM source files that produced this store (provenance).
        dataset:
            The xarray Dataset written to the store.  If provided, richer
            metadata (variable list, coordinate ranges, chunk shapes) is
            extracted.
        write_stats:
            Stats dict returned by ``ZarrWriter.write_dataset()``.
        extra:
            Per-store extra key/value pairs (override ``extra_attributes``).
        """
        store_path = Path(store_path)
        entry: dict[str, Any] = {
            "store_path": str(store_path),
            "store_name": store_path.stem,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "um2zarr_version": _UM2ZARR_VERSION,
        }

        if source_files:
            entry["source_files"] = [str(p) for p in source_files]

        # Merge extra attributes (class-level then per-call)
        merged_extra = {**self.extra_attributes, **(extra or {})}
        entry.update(merged_extra)

        # Enrich with dataset metadata
        entry["metadata"] = _extract_dataset_metadata(dataset, write_stats)

        self._entries.append(entry)
        logger.info(f"Registered store for catalogue: {store_path.name}")

    def write(
        self,
        format: str = "intake",
        consolidated: bool = True,
    ) -> list[Path]:
        """
        Write the catalogue to disk.

        Parameters
        ----------
        format:
            One of ``'intake'`` (YAML), ``'esm'`` (JSON + CSV), or
            ``'json'`` (provenance JSON).  Defaults to ``'intake'``.
        consolidated:
            When True (default), the intake YAML uses
            ``consolidated: true`` so readers can skip metadata requests.

        Returns
        -------
        List of Paths of the files written.
        """
        if not self._entries:
            logger.warning("No stores registered — catalogue will be empty")

        self.catalogue_dir.mkdir(parents=True, exist_ok=True)

        fmt = format.lower()
        if fmt == "intake":
            return self._write_intake_yaml(consolidated=consolidated)
        elif fmt == "esm":
            return self._write_esm_catalogue()
        elif fmt == "json":
            return self._write_provenance_json()
        else:
            raise ValueError(
                f"Unknown catalogue format {format!r}.  "
                "Choose 'intake', 'esm', or 'json'."
            )

    # ------------------------------------------------------------------
    # Format writers
    # ------------------------------------------------------------------

    def _write_intake_yaml(self, consolidated: bool = True) -> list[Path]:
        """Write a YAML catalogue compatible with intake-xarray's zarr driver."""
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML catalogue output.  "
                "Install with: pip install pyyaml"
            )

        out_path = self.catalogue_dir / f"{self.catalogue_name}.yaml"

        sources: dict[str, Any] = {}
        for entry in self._entries:
            source_name = _safe_identifier(entry["store_name"])
            meta_block = entry.get("metadata", {})

            # Build compact description from variable list
            var_names = [v["name"] for v in meta_block.get("variables", [])]
            auto_desc = (
                f"Variables: {', '.join(var_names[:8])}"
                + (" …" if len(var_names) > 8 else "")
                if var_names
                else entry.get("description", self.description)
            )

            source_entry: dict[str, Any] = {
                "driver": "zarr",
                "description": auto_desc,
                "args": {
                    "urlpath": entry["store_path"],
                    "consolidated": consolidated,
                    "storage_options": {},
                },
                "metadata": {
                    "created_at": entry["created_at"],
                    "um2zarr_version": entry["um2zarr_version"],
                },
            }

            # Attach coordinate ranges and variable info under metadata
            if meta_block.get("coordinate_ranges"):
                source_entry["metadata"]["coordinate_ranges"] = meta_block[
                    "coordinate_ranges"
                ]
            if var_names:
                source_entry["metadata"]["variables"] = var_names
            if meta_block.get("compressed_size_mb") is not None:
                source_entry["metadata"]["compressed_size_mb"] = meta_block[
                    "compressed_size_mb"
                ]
            if meta_block.get("compression"):
                source_entry["metadata"]["compression"] = meta_block["compression"]
            if entry.get("source_files"):
                source_entry["metadata"]["source_files"] = entry["source_files"]

            # Propagate any extra user-supplied attributes
            for k, v in entry.items():
                if k not in {
                    "store_path",
                    "store_name",
                    "created_at",
                    "um2zarr_version",
                    "source_files",
                    "metadata",
                }:
                    source_entry["metadata"][k] = v

            sources[source_name] = source_entry

        catalogue_doc = {
            "description": self.description,
            "sources": sources,
        }

        with open(out_path, "w") as fh:
            _yaml.dump(
                catalogue_doc,
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

        logger.info(
            f"Intake YAML catalogue written: {out_path} ({len(sources)} source(s))"
        )
        return [out_path]

    def _write_esm_catalogue(self) -> list[Path]:
        """
        Write an intake-esm catalogue.

        Produces two files:
        - ``<name>.json``  — the catalogue descriptor
        - ``<name>.csv``   — the asset table (one row per Zarr store)

        The CSV columns include any extra_attributes registered at class-level
        plus ``path``, ``variables``, and core provenance fields.
        """
        json_path = self.catalogue_dir / f"{self.catalogue_name}.json"
        csv_path = self.catalogue_dir / f"{self.catalogue_name}.csv"

        # Collect all attribute columns across all entries
        # Core fixed columns first
        core_columns = [
            "store_name",
            "path",
            "variables",
            "created_at",
            "um2zarr_version",
        ]
        extra_keys: list[str] = []
        for entry in self._entries:
            for k in entry:
                if (
                    k not in core_columns + ["store_path", "source_files", "metadata"]
                    and k not in extra_keys
                ):
                    extra_keys.append(k)

        all_columns = core_columns + extra_keys

        # Write CSV asset table
        rows: list[dict[str, Any]] = []
        for entry in self._entries:
            meta = entry.get("metadata", {})
            var_names = [v["name"] for v in meta.get("variables", [])]
            row: dict[str, Any] = {
                "store_name": entry["store_name"],
                "path": entry["store_path"],
                "variables": " ".join(var_names),
                "created_at": entry["created_at"],
                "um2zarr_version": entry["um2zarr_version"],
            }
            for k in extra_keys:
                row[k] = entry.get(k, "")
            rows.append(row)

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_columns)
            writer.writeheader()
            writer.writerows(rows)

        # intake-esm attribute names come from extra_keys plus 'variables'
        attributes = [{"column_name": k, "vocabulary": ""} for k in extra_keys]
        attributes.insert(0, {"column_name": "variables", "vocabulary": ""})

        descriptor = {
            "esmcat_version": "0.0.1",
            "id": self.catalogue_name,
            "description": self.description,
            "catalog_file": str(csv_path),
            "attributes": attributes,
            "assets": {
                "column_name": "path",
                "format": "zarr",
            },
            "aggregation_control": {
                "variable_column_name": "variables",
                "groupby_attrs": [
                    k for k in extra_keys if k not in {"created_at", "um2zarr_version"}
                ],
                "aggregations": [],
            },
        }

        with open(json_path, "w") as fh:
            json.dump(descriptor, fh, indent=2, default=str)

        logger.info(
            f"intake-esm catalogue written: {json_path} + {csv_path.name} "
            f"({len(rows)} asset(s))"
        )
        return [json_path, csv_path]

    def _write_provenance_json(self) -> list[Path]:
        """Write a plain provenance JSON (no intake required for reading)."""
        out_path = self.catalogue_dir / f"{self.catalogue_name}.json"

        doc = {
            "um2zarr_version": _UM2ZARR_VERSION,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "description": self.description,
            "stores": self._entries,
        }

        with open(out_path, "w") as fh:
            json.dump(doc, fh, indent=2, default=str)

        logger.info(
            f"Provenance JSON written: {out_path} ({len(self._entries)} store(s))"
        )
        return [out_path]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"IntakeCatalogueWriter("
            f"dir={self.catalogue_dir!r}, "
            f"entries={len(self._entries)})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def write_catalogue(
    store_path: str | Path,
    dataset: Any = None,
    write_stats: dict[str, Any] | None = None,
    source_files: Sequence[str | Path] | None = None,
    catalogue_path: str | Path | None = None,
    format: str = "intake",
    extra_attributes: dict[str, Any] | None = None,
) -> list[Path]:
    """
    One-shot convenience wrapper: write a catalogue for a single Zarr store.

    Parameters
    ----------
    store_path:
        Path to the Zarr store.
    dataset:
        xarray Dataset written to the store (recommended but optional).
    write_stats:
        Stats dict returned by ``ZarrWriter.write_dataset()``.
    source_files:
        UM source files that produced the store.
    catalogue_path:
        Output path for the catalogue file.  If a directory is given, the
        default name ``um2zarr_catalogue.<ext>`` is used.  If a full file path
        is given, the stem becomes the catalogue name.  Defaults to the
        same directory as the store.
    format:
        ``'intake'``, ``'esm'``, or ``'json'``.
    extra_attributes:
        Dict of extra key-value pairs added to the catalogue entry.

    Returns
    -------
    List of Paths of the catalogue files written.
    """
    store_path = Path(store_path)

    if catalogue_path is None:
        cat_dir = store_path.parent
        cat_name = "um2zarr_catalogue"
    else:
        catalogue_path = Path(catalogue_path)
        if catalogue_path.suffix in {".yaml", ".yml", ".json", ".csv"}:
            cat_dir = catalogue_path.parent
            cat_name = catalogue_path.stem
        else:
            cat_dir = catalogue_path
            cat_name = "um2zarr_catalogue"

    writer = IntakeCatalogueWriter(
        catalogue_dir=cat_dir,
        catalogue_name=cat_name,
        extra_attributes=extra_attributes,
    )
    writer.add_store(
        store_path=store_path,
        source_files=source_files,
        dataset=dataset,
        write_stats=write_stats,
    )
    return writer.write(format=format)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _safe_identifier(name: str) -> str:
    """Return a Python-identifier-safe version of *name* for use as YAML key."""
    import re

    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if cleaned and cleaned[0].isdigit():
        cleaned = "s_" + cleaned
    return cleaned or "store"

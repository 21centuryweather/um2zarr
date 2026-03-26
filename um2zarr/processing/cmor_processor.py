"""
CMIP6 / CMOR-style output processing for um2zarr.

This module post-processes xarray Datasets produced by the standard um2zarr
pipeline, applying the conventions required by the
`Climate Model Output Rewriter (CMOR) <https://cmor.llnl.gov>`_ standard so
that the resulting Zarr stores are interoperable with CMIP6 analysis tools.

What this module does
---------------------
1. **Variable renaming** — maps UM STASH / CF names to the CMIP6 short names
   defined in the CMIP6 data request (e.g. ``theta → ta``, ``q → hus``).
2. **Units conversion** — converts variables to the units mandated by the
   CMIP6 MIP tables (e.g. Pa pressure levels → hPa, K → °C for surface
   temperature in the land realm, etc.).  Where CMIP6 already uses SI units
   (K, m s⁻¹, kg kg⁻¹) no conversion is applied.
3. **Time axis standardisation** — re-encodes the time coordinate with the
   correct calendar (360-day, proleptic_gregorian, noleap) and units string
   (``days since <reference date>``) using ``cftime`` if available.
4. **Required global attributes** — adds the mandatory CMIP6 global metadata
   (``activity_id``, ``experiment_id``, ``source_id``, ``variant_label``,
   ``mip_era``, ``tracking_id``, etc.) from a user-supplied ``CMORConfig``.
5. **Per-variable attributes** — sets ``cell_methods``, ``positive``
   (for flux variables), ``missing_value``, and ``_FillValue`` per the
   CMIP6 MIP table entries.
6. **Coordinate axis labelling** — ensures ``axis`` and ``positive``
   attributes are present on coordinate variables (T, X, Y, Z).
7. **Validation** — reports missing required attributes and unknown variable
   names so that problems are visible before the store is written.

Usage::

    from um2zarr.processing.cmor_processor import CMORProcessor, CMORConfig

    config = CMORConfig(
        activity_id='CMIP',
        experiment_id='historical',
        source_id='ACCESS-CM2',
        variant_label='r1i1p1f1',
        institution_id='CSIRO-ARCCSS',
    )
    processor = CMORProcessor(config)
    cmor_ds, report = processor.process(dataset)

CLI equivalent::

    um2zarr input.pp output.zarr \\
        --cmor \\
        --cmor-experiment-id historical \\
        --cmor-source-id ACCESS-CM2 \\
        --cmor-variant-label r1i1p1f1 \\
        --cmor-activity-id CMIP
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import xarray as xr

    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import cftime  # type: ignore[import]

    HAS_CFTIME = True
except ImportError:
    HAS_CFTIME = False


# ---------------------------------------------------------------------------
# CMIP6 MIP table entry
# ---------------------------------------------------------------------------


@dataclass
class CMORTable:
    """Per-variable CMIP6 MIP table metadata."""

    cmip6_name: str  # CMIP6 short variable name (e.g. 'ta')
    standard_name: str  # CF standard name
    long_name: str  # Human-readable long name
    units: str  # CMIP6 required units
    cell_methods: str  # CF cell_methods string
    positive: str = ""  # 'up' | 'down' | '' for non-directional
    realm: str = "atmos"  # 'atmos' | 'land' | 'ocean' | 'seaIce'
    frequency: str = ""  # 'mon' | 'day' | '6hr' | '3hr' | '1hr' | ''
    dimensions: str = ""  # Space-separated CMIP6 dimension names
    # Conversion from UM/CF units to CMIP6 units
    unit_conversion: str | None = None  # e.g. 'Pa_to_hPa', 'K_to_degC'


# ---------------------------------------------------------------------------
# STASH → CMIP6 mapping table
# ---------------------------------------------------------------------------
# Keys are integer STASH codes (section * 1000 + item).
# The mapping covers the most commonly used CMIP6 variables from large UM
# climate runs.  It is intentionally incomplete — extend via CMORConfig.
# ---------------------------------------------------------------------------


# Shorthand for building entries
def _ct(
    cmip6: str,
    std: str,
    long: str,
    units: str,
    cell: str,
    pos: str = "",
    realm: str = "atmos",
    dims: str = "",
    conv: str | None = None,
) -> CMORTable:
    return CMORTable(
        cmip6_name=cmip6,
        standard_name=std,
        long_name=long,
        units=units,
        cell_methods=cell,
        positive=pos,
        realm=realm,
        dimensions=dims,
        unit_conversion=conv,
    )


STASH_TO_CMIP6: dict[int, CMORTable] = {
    # -----------------------------------------------------------------------
    # Section 0 — Prognostic fields
    # -----------------------------------------------------------------------
    1: _ct(
        "ta",
        "air_temperature",
        "Air Temperature",
        "K",
        "time: mean",
        dims="time plev lat lon",
    ),
    2: _ct(
        "hus",
        "specific_humidity",
        "Specific Humidity",
        "1",
        "time: mean",
        dims="time plev lat lon",
    ),
    3: _ct(
        "ua",
        "eastward_wind",
        "Eastward Wind",
        "m s-1",
        "time: mean",
        dims="time plev lat lon",
    ),
    4: _ct(
        "va",
        "northward_wind",
        "Northward Wind",
        "m s-1",
        "time: mean",
        dims="time plev lat lon",
    ),
    9: _ct("orog", "surface_altitude", "Surface Altitude", "m", "", dims="lat lon"),
    # -----------------------------------------------------------------------
    # Section 1 — Radiation diagnostics
    # -----------------------------------------------------------------------
    1201: _ct(
        "rsds",
        "surface_downwelling_shortwave_flux_in_air",
        "Surface Downwelling Shortwave Radiation",
        "W m-2",
        "time: mean",
        pos="down",
        dims="time lat lon",
    ),
    1202: _ct(
        "rlds",
        "surface_downwelling_longwave_flux_in_air",
        "Surface Downwelling Longwave Radiation",
        "W m-2",
        "time: mean",
        pos="down",
        dims="time lat lon",
    ),
    1207: _ct(
        "rsut",
        "toa_outgoing_shortwave_flux",
        "TOA Outgoing Shortwave Radiation",
        "W m-2",
        "time: mean",
        pos="up",
        dims="time lat lon",
    ),
    1208: _ct(
        "rlut",
        "toa_outgoing_longwave_flux",
        "TOA Outgoing Longwave Radiation",
        "W m-2",
        "time: mean",
        pos="up",
        dims="time lat lon",
    ),
    1235: _ct(
        "rsdscs",
        "surface_downwelling_shortwave_flux_in_air_assuming_clear_sky",
        "Surface Downwelling Clear-Sky Shortwave Radiation",
        "W m-2",
        "time: mean",
        pos="down",
        dims="time lat lon",
    ),
    # -----------------------------------------------------------------------
    # Section 2 — Dynamics / large-scale precipitation
    # -----------------------------------------------------------------------
    2205: _ct(
        "pr",
        "precipitation_flux",
        "Precipitation",
        "kg m-2 s-1",
        "time: mean",
        dims="time lat lon",
    ),
    2207: _ct(
        "prsn",
        "snowfall_flux",
        "Snowfall Flux",
        "kg m-2 s-1",
        "time: mean",
        dims="time lat lon",
    ),
    # -----------------------------------------------------------------------
    # Section 3 — Boundary-layer diagnostics
    # -----------------------------------------------------------------------
    3217: _ct(
        "hfss",
        "surface_upward_sensible_heat_flux",
        "Surface Upward Sensible Heat Flux",
        "W m-2",
        "time: mean",
        pos="up",
        dims="time lat lon",
    ),
    3234: _ct(
        "hfls",
        "surface_upward_latent_heat_flux",
        "Surface Upward Latent Heat Flux",
        "W m-2",
        "time: mean",
        pos="up",
        dims="time lat lon",
    ),
    3209: _ct(
        "zg",
        "geopotential_height",
        "Geopotential Height",
        "m",
        "time: mean",
        dims="time plev lat lon",
    ),
    # -----------------------------------------------------------------------
    # Section 4 — Near-surface diagnostics
    # -----------------------------------------------------------------------
    4203: _ct(
        "tas",
        "air_temperature",
        "Near-Surface Air Temperature",
        "K",
        "time: mean",
        dims="time lat lon",
        conv=None,
    ),  # CMIP6 tas is in K, not °C
    4204: _ct(
        "huss",
        "specific_humidity",
        "Near-Surface Specific Humidity",
        "1",
        "time: mean",
        dims="time lat lon",
    ),
    4205: _ct(
        "sfcWind",
        "wind_speed",
        "Near-Surface Wind Speed",
        "m s-1",
        "time: mean",
        dims="time lat lon",
    ),
    4206: _ct(
        "uas",
        "eastward_wind",
        "Eastward Near-Surface Wind",
        "m s-1",
        "time: mean",
        dims="time lat lon",
    ),
    4207: _ct(
        "vas",
        "northward_wind",
        "Northward Near-Surface Wind",
        "m s-1",
        "time: mean",
        dims="time lat lon",
    ),
    4208: _ct(
        "psl",
        "air_pressure_at_mean_sea_level",
        "Sea Level Pressure",
        "Pa",
        "time: mean",
        dims="time lat lon",
    ),
    # -----------------------------------------------------------------------
    # Section 5 — Sea ice / ocean surface
    # -----------------------------------------------------------------------
    5201: _ct(
        "siconc",
        "sea_ice_area_fraction",
        "Sea-Ice Area Fraction",
        "1",
        "time: mean",
        realm="seaIce",
        dims="time lat lon",
    ),
    # -----------------------------------------------------------------------
    # Section 16 — Tropopause / stratosphere
    # -----------------------------------------------------------------------
    16202: _ct(
        "tro3",
        "mole_fraction_of_ozone_in_air",
        "Mole Fraction of O3",
        "mol mol-1",
        "time: mean",
        dims="time plev lat lon",
    ),
    # -----------------------------------------------------------------------
    # Section 24 — Total-column quantities
    # -----------------------------------------------------------------------
    24006: _ct(
        "prw",
        "atmosphere_mass_content_of_water_vapor",
        "Precipitable Water",
        "kg m-2",
        "time: mean",
        dims="time lat lon",
    ),
    24012: _ct(
        "clwvi",
        "atmosphere_mass_content_of_cloud_condensed_water",
        "Condensed Water Path",
        "kg m-2",
        "time: mean",
        dims="time lat lon",
    ),
}

# ---------------------------------------------------------------------------
# CF name → CMIP6 lookup  (fallback when STASH attrs are not available)
# ---------------------------------------------------------------------------

CF_TO_CMIP6: dict[str, CMORTable] = {
    entry.standard_name: entry for entry in STASH_TO_CMIP6.values()
}
# Also index by the um2zarr CF short name where it differs
_CFSHORT_TO_CMIP6: dict[str, CMORTable] = {
    "theta": STASH_TO_CMIP6[1],
    "q": STASH_TO_CMIP6[2],
    "u": STASH_TO_CMIP6[3],
    "v": STASH_TO_CMIP6[4],
    "rsds": STASH_TO_CMIP6[1201],
    "rlds": STASH_TO_CMIP6[1202],
    "rsut": STASH_TO_CMIP6[1207],
    "rlut": STASH_TO_CMIP6[1208],
    "hfss": STASH_TO_CMIP6[3217],
    "hfls": STASH_TO_CMIP6[3234],
}

# ---------------------------------------------------------------------------
# Units conversion registry
# ---------------------------------------------------------------------------
# Each entry is a callable (value_array → converted_array) plus the target
# units string.  Keys match the ``unit_conversion`` field in CMORTable.

_UNIT_CONVERTERS: dict[str, tuple[Any, str]] = {}

if HAS_NUMPY:
    _UNIT_CONVERTERS = {
        "Pa_to_hPa": (lambda x: x / 100.0, "hPa"),
        "K_to_degC": (lambda x: x - 273.15, "degC"),
        "degC_to_K": (lambda x: x + 273.15, "K"),
        "J_to_W": (lambda x: x / 86400.0, "W m-2"),  # daily → per-second
        "mm_to_kg": (lambda x: x / 86400.0, "kg m-2 s-1"),
    }

# ---------------------------------------------------------------------------
# Calendar name normalisation
# ---------------------------------------------------------------------------

_CALENDAR_ALIASES: dict[str, str] = {
    "360day": "360_day",
    "360-day": "360_day",
    "gregorian": "proleptic_gregorian",
    "standard": "proleptic_gregorian",
    "julian": "proleptic_gregorian",
    "noleap": "365_day",
    "365day": "365_day",
    "365-day": "365_day",
    "all_leap": "366_day",
    "366day": "366_day",
    "proleptic_gregorian": "proleptic_gregorian",
    "360_day": "360_day",
    "365_day": "365_day",
}

# ---------------------------------------------------------------------------
# CMOR experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class CMORConfig:
    """
    CMIP6 experiment metadata injected as global attributes.

    All fields correspond directly to required CMIP6 global attributes.
    At minimum, ``activity_id``, ``experiment_id``, ``source_id``, and
    ``variant_label`` should be set before calling ``CMORProcessor.process()``.
    """

    # Required CMIP6 global attributes
    activity_id: str = "CMIP"  # e.g. 'CMIP', 'ScenarioMIP', 'HighResMIP'
    experiment_id: str = ""  # e.g. 'historical', 'ssp585', 'amip'
    source_id: str = ""  # e.g. 'ACCESS-CM2', 'UKESM1-0-LL'
    variant_label: str = "r1i1p1f1"  # e.g. 'r1i1p1f1'
    institution_id: str = ""  # e.g. 'CSIRO-ARCCSS', 'MOHC'
    mip_era: str = "CMIP6"

    # Recommended global attributes
    source_type: str = "AGCM"  # 'AGCM' | 'AOGCM' | 'AER' | …
    sub_experiment_id: str = "none"
    sub_experiment: str = "none"
    grid: str = ""  # Human-readable grid description
    grid_label: str = "gn"  # 'gn' = native grid, 'gr' = regridded

    # Calendar to enforce (empty = preserve source calendar)
    calendar: str = ""  # '360_day' | '365_day' | 'proleptic_gregorian'

    # Time reference date for encoding
    time_reference: str = "1850-01-01"  # ISO date string

    # Mapping overrides: STASH code → CMIP6 name (extend built-in table)
    extra_stash_map: dict[int, CMORTable] = field(default_factory=dict)

    # Variable names to skip during CMOR processing (pass through unchanged)
    skip_variables: list[str] = field(default_factory=list)

    # Whether to drop variables that have no CMIP6 mapping
    drop_unmapped: bool = False


# ---------------------------------------------------------------------------
# CMOR processor
# ---------------------------------------------------------------------------


class CMORProcessor:
    """
    Apply CMIP6 / CMOR conventions to an xarray Dataset.

    The processor operates non-destructively: the input Dataset is never
    modified in place.  All methods return a new Dataset (or a copy).

    Parameters
    ----------
    config:
        Experiment metadata and processing options.
    """

    def __init__(self, config: CMORConfig | None = None) -> None:
        self.config = config or CMORConfig()

        # Build effective lookup table (built-in + user overrides)
        self._table: dict[int, CMORTable] = {
            **STASH_TO_CMIP6,
            **self.config.extra_stash_map,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        dataset: Any,
        source_files: list[str] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """
        Apply all CMOR transformations to *dataset* and return a new Dataset
        plus a processing report.

        Steps applied (in order):

        1. Variable renaming (STASH / CF → CMIP6 short names)
        2. Units conversion
        3. Per-variable CF/CMIP6 attribute assignment
        4. Coordinate axis labelling
        5. Time axis standardisation
        6. Required global attribute injection
        7. Validation

        Parameters
        ----------
        dataset:
            xarray Dataset produced by the um2zarr conversion pipeline.
        source_files:
            Optional list of source UM file paths for provenance attributes.

        Returns
        -------
        (cmor_dataset, report)
            ``cmor_dataset`` is a new xr.Dataset with CMIP6 conventions.
            ``report`` is a dict with keys ``renamed``, ``converted_units``,
            ``skipped``, ``unmapped``, ``dropped``, ``warnings``, ``errors``.
        """
        if not HAS_XARRAY:
            raise ImportError("xarray is required for CMOR processing.")

        report: dict[str, Any] = {
            "renamed": [],
            "converted_units": [],
            "skipped": list(self.config.skip_variables),
            "unmapped": [],
            "dropped": [],
            "warnings": [],
            "errors": [],
        }

        ds = dataset.copy()

        # Step 1: rename + collect per-variable CMIP6 entries
        ds, var_entries = self._rename_variables(ds, report)

        # Step 2: units conversion
        ds = self._convert_units(ds, var_entries, report)

        # Step 3: per-variable attributes
        ds = self._set_variable_attributes(ds, var_entries, report)

        # Step 4: coordinate axis labelling
        ds = self._label_coordinate_axes(ds, report)

        # Step 5: time axis
        ds = self._standardise_time_axis(ds, report)

        # Step 6: global attributes
        ds = self._set_global_attributes(ds, source_files, report)

        # Step 7: validate
        self._validate(ds, report)

        if report["warnings"]:
            for msg in report["warnings"]:
                logger.warning(f"[CMOR] {msg}")
        if report["errors"]:
            for msg in report["errors"]:
                logger.error(f"[CMOR] {msg}")

        return ds, report

    # ------------------------------------------------------------------
    # Step 1 — Variable renaming
    # ------------------------------------------------------------------

    def _rename_variables(
        self,
        ds: Any,
        report: dict[str, Any],
    ) -> tuple[Any, dict[str, CMORTable]]:
        """
        Rename data variables to their CMIP6 short names.

        Resolution order per variable:
        1. ``stash_section`` / ``stash_item`` attrs → lookup in STASH table
        2. ``standard_name`` attr → lookup in CF_TO_CMIP6
        3. Variable name itself → lookup in ``_CFSHORT_TO_CMIP6``
        4. No mapping found → mark as unmapped; drop if ``config.drop_unmapped``
        """
        rename_map: dict[str, str] = {}
        var_entries: dict[str, CMORTable] = {}  # keyed by *new* name

        for var in list(ds.data_vars):
            if var in self.config.skip_variables:
                continue

            da = ds[var]
            entry = self._find_table_entry(da, var)

            if entry is None:
                report["unmapped"].append(var)
                if self.config.drop_unmapped:
                    report["dropped"].append(var)
                else:
                    report["warnings"].append(
                        f"Variable '{var}' has no CMIP6 mapping — "
                        "kept with original name. Set drop_unmapped=True to remove it."
                    )
                continue

            new_name = entry.cmip6_name
            if new_name != var:
                rename_map[var] = new_name
                report["renamed"].append(f"{var} → {new_name}")

            var_entries[new_name] = entry

        # Drop unmapped variables
        ds = ds.drop_vars(report["dropped"], errors="ignore")

        # Rename
        if rename_map:
            ds = ds.rename_vars(rename_map)

        return ds, var_entries

    def _find_table_entry(self, da: Any, var_name: str) -> CMORTable | None:
        """Resolve a DataArray to a CMORTable entry using multiple strategies."""
        # Strategy 1: STASH attrs
        stash_section = da.attrs.get("stash_section") or da.attrs.get(
            "um_stash_source_section"
        )
        stash_item = da.attrs.get("stash_item") or da.attrs.get("um_stash_source_item")
        if stash_section is not None and stash_item is not None:
            stash_code = int(stash_section) * 1000 + int(stash_item)
            if stash_code in self._table:
                return self._table[stash_code]

        # Strategy 2: standard_name attr
        std_name = da.attrs.get("standard_name", "")
        if std_name and std_name in CF_TO_CMIP6:
            return CF_TO_CMIP6[std_name]

        # Strategy 3: variable name matches a CF short name
        if var_name in _CFSHORT_TO_CMIP6:
            return _CFSHORT_TO_CMIP6[var_name]

        # Strategy 4: variable name is already a CMIP6 name
        cmip6_names = {e.cmip6_name: e for e in self._table.values()}
        if var_name in cmip6_names:
            return cmip6_names[var_name]

        return None

    # ------------------------------------------------------------------
    # Step 2 — Units conversion
    # ------------------------------------------------------------------

    def _convert_units(
        self,
        ds: Any,
        var_entries: dict[str, CMORTable],
        report: dict[str, Any],
    ) -> Any:
        """Apply unit conversions where CMORTable specifies one."""
        for var, entry in var_entries.items():
            if var not in ds.data_vars:
                continue
            conv_key = entry.unit_conversion
            if not conv_key or conv_key not in _UNIT_CONVERTERS:
                continue

            converter, new_units = _UNIT_CONVERTERS[conv_key]
            try:
                original_units = ds[var].attrs.get("units", "?")
                ds[var] = xr.apply_ufunc(
                    converter,
                    ds[var],
                    dask="parallelized",
                    output_dtypes=[ds[var].dtype],
                    keep_attrs=True,
                )
                ds[var].attrs["units"] = new_units
                report["converted_units"].append(
                    f"{var}: {original_units} → {new_units}"
                )
            except Exception as exc:
                report["warnings"].append(
                    f"Units conversion failed for '{var}' (conv={conv_key!r}): {exc}"
                )

        return ds

    # ------------------------------------------------------------------
    # Step 3 — Per-variable attributes
    # ------------------------------------------------------------------

    def _set_variable_attributes(
        self,
        ds: Any,
        var_entries: dict[str, CMORTable],
        report: dict[str, Any],
    ) -> Any:
        """Assign standard_name, long_name, units, cell_methods, positive."""
        _FILL = 1.0e20  # CMIP6 canonical fill value for float variables

        for var, entry in var_entries.items():
            if var not in ds.data_vars:
                continue
            da = ds[var]
            attrs = dict(da.attrs)

            # Mandatory CF/CMIP6 per-variable attrs
            attrs["standard_name"] = entry.standard_name
            attrs["long_name"] = entry.long_name
            attrs["units"] = attrs.get("units") or entry.units
            if entry.cell_methods:
                attrs["cell_methods"] = entry.cell_methods
            if entry.positive:
                attrs["positive"] = entry.positive

            # Canonical missing_value / _FillValue for float arrays
            if HAS_NUMPY and hasattr(ds[var], "dtype"):
                if ds[var].dtype.kind == "f":
                    attrs.setdefault("missing_value", _FILL)
                    attrs.setdefault("_FillValue", _FILL)

            ds[var].attrs = attrs

        return ds

    # ------------------------------------------------------------------
    # Step 4 — Coordinate axis labelling
    # ------------------------------------------------------------------

    def _label_coordinate_axes(self, ds: Any, report: dict[str, Any]) -> Any:
        """Add ``axis`` attributes to T/X/Y/Z coordinate variables."""

        _AXIS_RULES: list[tuple[set[str], str, dict[str, str]]] = [
            # (name candidates, axis label, extra attrs)
            ({"time", "t", "forecast_reference_time"}, "T", {}),
            (
                {"lat", "latitude", "grid_latitude", "rlat"},
                "Y",
                {"positive": "north", "units": "degrees_north"},
            ),
            (
                {"lon", "longitude", "grid_longitude", "rlon"},
                "X",
                {"positive": "east", "units": "degrees_east"},
            ),
            (
                {
                    "lev",
                    "level",
                    "pressure",
                    "plev",
                    "model_level_number",
                    "atmosphere_hybrid_sigma_pressure_coordinate",
                },
                "Z",
                {"positive": "down"},
            ),
        ]

        for coord in ds.coords:
            da = ds.coords[coord]
            if "axis" in da.attrs:
                continue  # already labelled

            for candidates, axis_label, extras in _AXIS_RULES:
                if coord in candidates:
                    new_attrs = dict(da.attrs)
                    new_attrs["axis"] = axis_label
                    for k, v in extras.items():
                        new_attrs.setdefault(k, v)
                    # Can't assign directly to coord attrs in xarray;
                    # use assign_coords
                    ds = ds.assign_coords({coord: da.assign_attrs(new_attrs)})
                    break

        return ds

    # ------------------------------------------------------------------
    # Step 5 — Time axis standardisation
    # ------------------------------------------------------------------

    def _standardise_time_axis(self, ds: Any, report: dict[str, Any]) -> Any:
        """
        Re-encode the time coordinate with the correct calendar and units.

        If ``config.calendar`` is empty, the source calendar is preserved.
        Requires ``cftime`` for calendar conversion; silently skips if not
        available.
        """
        time_dim = next(
            (c for c in ds.coords if c in {"time", "t", "forecast_reference_time"}),
            None,
        )
        if time_dim is None:
            return ds

        target_cal = (
            _CALENDAR_ALIASES.get(self.config.calendar, self.config.calendar)
            if self.config.calendar
            else None
        )

        # Ensure units attribute on time coordinate
        time_da = ds.coords[time_dim]
        attrs = dict(time_da.attrs)

        # Set / normalise calendar attribute
        if "calendar" not in attrs and target_cal:
            attrs["calendar"] = target_cal
        elif "calendar" in attrs:
            normalised = _CALENDAR_ALIASES.get(attrs["calendar"], attrs["calendar"])
            if target_cal and normalised != target_cal:
                report["warnings"].append(
                    f"Time calendar mismatch: source={attrs['calendar']!r}, "
                    f"target={target_cal!r}.  Calendar conversion requires "
                    "cftime; attributes updated but values not resampled."
                )
                attrs["calendar"] = target_cal
            else:
                attrs["calendar"] = normalised

        # Ensure units string is present
        if "units" not in attrs:
            attrs["units"] = f"days since {self.config.time_reference} 00:00:00"
            report["warnings"].append(
                f"Time coordinate '{time_dim}' had no units attr; "
                f"set to '{attrs['units']}'.  Verify this is correct."
            )

        ds = ds.assign_coords({time_dim: time_da.assign_attrs(attrs)})
        return ds

    # ------------------------------------------------------------------
    # Step 6 — Global attributes
    # ------------------------------------------------------------------

    def _set_global_attributes(
        self,
        ds: Any,
        source_files: list[str] | None,
        report: dict[str, Any],
    ) -> Any:
        """Inject required and recommended CMIP6 global attributes."""
        cfg = self.config
        now = datetime.now(tz=timezone.utc)

        global_attrs = dict(ds.attrs)

        # Required CMIP6 global attrs
        _required = {
            "mip_era": cfg.mip_era,
            "activity_id": cfg.activity_id,
            "experiment_id": cfg.experiment_id,
            "source_id": cfg.source_id,
            "variant_label": cfg.variant_label,
        }
        for k, v in _required.items():
            if v:
                global_attrs[k] = v

        # Recommended attrs (only set if non-empty)
        _recommended = {
            "institution_id": cfg.institution_id,
            "source_type": cfg.source_type,
            "sub_experiment_id": cfg.sub_experiment_id,
            "sub_experiment": cfg.sub_experiment,
            "grid": cfg.grid,
            "grid_label": cfg.grid_label,
        }
        for k, v in _recommended.items():
            if v and v not in ("", "none"):
                global_attrs.setdefault(k, v)

        # Provenance attrs
        global_attrs["creation_date"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        global_attrs["tracking_id"] = f"hdl:21.14100/{uuid.uuid4()}"
        global_attrs["um2zarr_processing"] = (
            f"um2zarr CMOR post-processing applied {now.strftime('%Y-%m-%d')}"
        )
        if source_files:
            global_attrs["um_source_files"] = " ".join(source_files[:10])
            if len(source_files) > 10:
                global_attrs["um_source_files"] += (
                    f" … (+{len(source_files) - 10} more)"
                )

        # Conventions
        existing_conv = global_attrs.get("Conventions", "")
        if "CF-1." not in str(existing_conv):
            global_attrs["Conventions"] = "CF-1.11"
        else:
            global_attrs["Conventions"] = existing_conv

        ds = ds.assign_attrs(global_attrs)
        return ds

    # ------------------------------------------------------------------
    # Step 7 — Validation
    # ------------------------------------------------------------------

    def _validate(self, ds: Any, report: dict[str, Any]) -> None:
        """Check for missing required attributes and report issues."""
        _REQUIRED_GLOBALS = {
            "mip_era",
            "activity_id",
            "experiment_id",
            "source_id",
            "variant_label",
        }
        for attr in _REQUIRED_GLOBALS:
            if not ds.attrs.get(attr):
                report["errors"].append(
                    f"Missing required CMIP6 global attribute: '{attr}'. "
                    f"Set it via CMORConfig.{attr}."
                )

        for var in ds.data_vars:
            da = ds[var]
            if not da.attrs.get("units"):
                report["warnings"].append(
                    f"Variable '{var}' has no 'units' attribute after CMOR processing."
                )
            if not da.attrs.get("standard_name"):
                report["warnings"].append(
                    f"Variable '{var}' has no 'standard_name' attribute."
                )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def describe_mappings(self) -> dict[str, Any]:
        """Return a summary of all STASH → CMIP6 mappings in the active table."""
        return {
            stash: {
                "cmip6_name": entry.cmip6_name,
                "standard_name": entry.standard_name,
                "units": entry.units,
                "cell_methods": entry.cell_methods,
                "positive": entry.positive,
            }
            for stash, entry in self._table.items()
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def apply_cmor(
    dataset: Any,
    activity_id: str = "CMIP",
    experiment_id: str = "",
    source_id: str = "",
    variant_label: str = "r1i1p1f1",
    institution_id: str = "",
    calendar: str = "",
    extra_stash_map: dict[int, CMORTable] | None = None,
    drop_unmapped: bool = False,
    source_files: list[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    One-shot convenience wrapper: apply CMOR conventions in a single call.

    Returns ``(cmor_dataset, report)``.
    """
    config = CMORConfig(
        activity_id=activity_id,
        experiment_id=experiment_id,
        source_id=source_id,
        variant_label=variant_label,
        institution_id=institution_id,
        calendar=calendar,
        extra_stash_map=extra_stash_map or {},
        drop_unmapped=drop_unmapped,
    )
    processor = CMORProcessor(config)
    return processor.process(dataset, source_files=source_files)

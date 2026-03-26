"""
Core data structures for UM file processing.

Defines the fundamental data types used throughout the um2zarr pipeline.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class GridMetadata:
    """Container for UM grid metadata extracted from file headers."""

    grid_type: Literal["EG", "ND"]  # Endgame or New Dynamics
    dlat: float  # Latitude spacing in degrees
    dlon: float  # Longitude spacing in degrees
    z_rho: np.ndarray  # Heights at rho levels
    z_theta: np.ndarray  # Heights at theta levels

    def __post_init__(self):
        """Validate grid metadata after initialization."""
        if self.grid_type not in ["EG", "ND"]:
            raise ValueError(f"Invalid grid_type: {self.grid_type}")
        if not isinstance(self.z_rho, np.ndarray):
            self.z_rho = np.array(self.z_rho)
        if not isinstance(self.z_theta, np.ndarray):
            self.z_theta = np.array(self.z_theta)


@dataclass
class StashMetadata:
    """Container for STASH code metadata mapping."""

    long_name: str
    cf_name: str
    units: str
    standard_name: str
    unique_name: str

    @classmethod
    def from_legacy_tuple(cls, legacy_data: list[str]) -> "StashMetadata":
        """Create from legacy stashvar_cmip6.py tuple format."""
        # Handle legacy format: [long_name, cf_name, units, standard_name, unique_name]
        return cls(
            long_name=legacy_data[0] if legacy_data[0] else "",
            cf_name=legacy_data[1] if legacy_data[1] else "",
            units=legacy_data[2] if legacy_data[2] else "",
            standard_name=legacy_data[3] if legacy_data[3] else "",
            unique_name=legacy_data[4] if legacy_data[4] else "",
        )


@dataclass
class ProcessingConfig:
    """Configuration for UM to Zarr processing pipeline."""

    compression: str = "zstd"
    compression_level: int = 3
    use_dask: bool = True
    n_workers: int = 4
    memory_limit: str = "4GB"
    threads_per_worker: int = 1
    chunk_strategy: str = "balanced"
    validate_cf: bool = True
    simple_names: bool = False

    # Heaviside masking options (for pressure-level UM diagnostics)
    nomask: bool = False  # If True, skip all masking and pass data through unmodified
    hcrit: float = 0.5  # Heaviside critical threshold: mask where heaviside <= hcrit

    # dask_setup specific parameters
    workload_type: str = "io"
    adaptive_scaling: bool = True

    # Adaptive scaling floor (minimum live workers)
    min_workers: int = 1

    # External / HPC cluster connection
    scheduler_file: str | None = None  # Path to dask scheduler JSON file
    scheduler_address: str | None = None  # tcp://host:port of running scheduler

    # dask_setup resource hints
    reserve_memory_gb: float = 2.0  # GB of RAM to leave free per node
    suggest_chunks: bool = False  # Let dask_setup recommend chunk sizes

    # -------------------------------------------------------------------------
    # Phase 2: Append-mode time series (2.1)
    # -------------------------------------------------------------------------
    append: bool = False  # If True, open existing store in append mode
    append_dim: str = "time"  # Dimension along which to append
    resume: bool = False  # Skip files already in the store's time axis

    # -------------------------------------------------------------------------
    # Phase 2: Error recovery (2.5)
    # -------------------------------------------------------------------------
    on_error: str = "skip"  # 'skip' | 'abort' | 'retry' — per-file error policy

    # -------------------------------------------------------------------------
    # Phase 2: Ensemble dimension support (2.4)
    # -------------------------------------------------------------------------
    ensemble_dim: str | None = None  # e.g. 'realization' — create ensemble axis
    ensemble_member_pattern: str | None = (
        None  # Regex to extract member IDs from filenames
    )

    # -------------------------------------------------------------------------
    # Phase 2: Graph size monitoring (2.6)
    # -------------------------------------------------------------------------
    max_graph_size_mb: float = 50.0  # Auto-split Dask graphs that exceed this size

    # -------------------------------------------------------------------------
    # Phase 3: Intake catalogue generation (3.1)
    # -------------------------------------------------------------------------
    catalogue_path: str | None = None  # Write intake catalogue to this path / dir
    catalogue_format: str = "intake"  # 'intake' | 'esm' | 'json'

    # -------------------------------------------------------------------------
    # Phase 3: CMIP6 / CMOR-style output (3.3)
    # -------------------------------------------------------------------------
    cmor: bool = False  # If True, apply CMOR post-processing
    cmor_activity_id: str = "CMIP"
    cmor_experiment_id: str = ""
    cmor_source_id: str = ""
    cmor_variant_label: str = "r1i1p1f1"
    cmor_institution_id: str = ""
    cmor_calendar: str = ""  # '' = preserve source calendar
    cmor_drop_unmapped: bool = False  # Drop variables with no CMIP6 mapping

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_compressions = ["zstd", "blosc", "gzip", "bz2", "lz4"]
        if self.compression not in valid_compressions:
            raise ValueError(
                f"Invalid compression: {self.compression}. Must be one of {valid_compressions}"
            )

        if not 1 <= self.compression_level <= 9:
            raise ValueError("Compression level must be between 1 and 9")

        if self.n_workers < 1:
            raise ValueError("Number of workers must be at least 1")

        valid_strategies = ["memory", "balanced", "storage"]
        if self.chunk_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid chunk_strategy: {self.chunk_strategy}. Must be one of {valid_strategies}"
            )

        valid_workload_types = ["cpu", "io", "mixed"]
        if self.workload_type not in valid_workload_types:
            raise ValueError(
                f"Invalid workload_type: {self.workload_type}. Must be one of {valid_workload_types}"
            )

        if self.min_workers < 1:
            raise ValueError("min_workers must be at least 1")

        if self.reserve_memory_gb < 0:
            raise ValueError("reserve_memory_gb must be non-negative")

        if not 0.0 < self.hcrit < 1.0:
            raise ValueError("hcrit must be between 0 and 1 (exclusive)")

        valid_on_error = ("skip", "abort", "retry")
        if self.on_error not in valid_on_error:
            raise ValueError(
                f"on_error must be one of {valid_on_error}, got {self.on_error!r}"
            )

        if self.max_graph_size_mb <= 0:
            raise ValueError("max_graph_size_mb must be positive")

        valid_catalogue_formats = ("intake", "esm", "json")
        if self.catalogue_format not in valid_catalogue_formats:
            raise ValueError(
                f"catalogue_format must be one of {valid_catalogue_formats}, "
                f"got {self.catalogue_format!r}"
            )

        valid_calendars = (
            "",
            "360_day",
            "365_day",
            "proleptic_gregorian",
            "366_day",
            "noleap",
            "all_leap",
            "standard",
            "gregorian",
        )
        if self.cmor_calendar not in valid_calendars:
            raise ValueError(
                f"cmor_calendar must be one of {valid_calendars}, "
                f"got {self.cmor_calendar!r}"
            )


@dataclass
class ConversionResult:
    """Result of Iris cube to Xarray dataset conversion."""

    dataset: Any  # xr.Dataset - avoiding import here
    conversion_warnings: list[str]
    variables_converted: int
    variables_skipped: list[str]
    processing_time: float

    def __post_init__(self):
        """Initialize empty lists if None provided."""
        if self.conversion_warnings is None:
            self.conversion_warnings = []
        if self.variables_skipped is None:
            self.variables_skipped = []


@dataclass
class ProcessingResult:
    """Result of dataset processing pipeline."""

    dataset: Any  # xr.Dataset
    processing_time: float
    memory_usage: dict[str, float]
    warnings: list[str]

    def __post_init__(self):
        """Initialize empty collections if None provided."""
        if self.warnings is None:
            self.warnings = []
        if self.memory_usage is None:
            self.memory_usage = {}


class UM2ZarrError(Exception):
    """Base exception for all um2zarr errors."""

    pass


class UMFileError(UM2ZarrError):
    """Error reading or parsing UM file."""

    pass


class GridMetadataError(UM2ZarrError):
    """Error extracting grid metadata from UM file."""

    pass


class ConversionError(UM2ZarrError):
    """Error converting Iris cube to Xarray."""

    pass


class ProcessingError(UM2ZarrError):
    """Error in dataset processing."""

    pass


class ZarrWriteError(UM2ZarrError):
    """Error writing to Zarr store."""

    pass


class ValidationError(UM2ZarrError):
    """Error in CF compliance validation."""

    pass

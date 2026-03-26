"""
um2zarr: Modern UM to Zarr conversion pipeline

Converts Unified Model (UM) fieldsfiles to cloud-optimized Zarr format
using Xarray and Dask for efficient processing and storage.

This package provides a modern, scalable alternative to traditional
NetCDF conversion workflows while preserving all UM-specific metadata
and scientific accuracy.
"""

__version__ = "0.3.0"
__author__ = "Samuel Green"
__email__ = "sam.green@unsw.edu.au"

# Core data structures and configuration
from .core.data_structures import (
    ConversionError,
    ConversionResult,
    GridMetadata,
    GridMetadataError,
    ProcessingConfig,
    ProcessingError,
    ProcessingResult,
    StashMetadata,
    # Exception hierarchy
    UM2ZarrError,
    UMFileError,
    ValidationError,
    ZarrWriteError,
)
from .io.catalogue_writer import IntakeCatalogueWriter, write_catalogue

# I/O layer
from .io.um_reader import UMFileReader
from .io.zarr_writer import EncodingStrategy, ZarrWriter
from .orchestration.checkpoint import CheckpointManager

# Orchestration — high-level conversion API
from .orchestration.cli import ConversionOrchestrator, DataProcessingPipeline
from .processing.chunk_manager import ChunkingStrategy, ChunkManager
from .processing.cmor_processor import (
    CF_TO_CMIP6,
    STASH_TO_CMIP6,
    CMORConfig,
    CMORProcessor,
    CMORTable,
    apply_cmor,
)
from .processing.iris_converter import IrisToXarrayConverter
from .processing.rechunker import RechunkTarget, rechunk_store

# Processing layer
from .processing.stash_metadata import StashMetadataManager

__all__ = [
    # Version
    "__version__",
    # Data structures
    "GridMetadata",
    "StashMetadata",
    "ProcessingConfig",
    "ConversionResult",
    "ProcessingResult",
    # Exceptions
    "UM2ZarrError",
    "UMFileError",
    "GridMetadataError",
    "ConversionError",
    "ProcessingError",
    "ZarrWriteError",
    "ValidationError",
    # I/O
    "UMFileReader",
    "ZarrWriter",
    "EncodingStrategy",
    "IntakeCatalogueWriter",
    "write_catalogue",
    # Processing
    "StashMetadataManager",
    "IrisToXarrayConverter",
    "ChunkManager",
    "ChunkingStrategy",
    "rechunk_store",
    "RechunkTarget",
    "CMORProcessor",
    "CMORConfig",
    "CMORTable",
    "apply_cmor",
    "STASH_TO_CMIP6",
    "CF_TO_CMIP6",
    # Orchestration
    "ConversionOrchestrator",
    "DataProcessingPipeline",
    "CheckpointManager",
]

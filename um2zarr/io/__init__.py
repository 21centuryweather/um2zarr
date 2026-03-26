"""
I/O layer for UM file reading and Zarr writing.
"""

# Import GridMetadata from data structures
from ..core.data_structures import GridMetadata

# Try to import full UM reader (requires Mule), fall back to simple reader
# TEMPORARY: Skip Mule reader due to numpy 2.x compatibility issue
try:
    # Force import error to use simple reader for now
    raise ImportError("Temporarily disabled due to numpy compatibility")
    from .um_reader import UMFileReader

    HAS_MULE = True
except ImportError:
    from .um_reader_simple import UMFileReader

    HAS_MULE = False

# Try to import Zarr writer (requires Zarr)
try:
    from .zarr_writer import (
        EncodingStrategy,
        ZarrWriter,
        append_to_zarr_time_series,
        write_um_dataset_to_zarr,
    )

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

__all__ = [
    "UMFileReader",
    "GridMetadata",
]

if HAS_ZARR:
    __all__.extend(
        [
            "ZarrWriter",
            "EncodingStrategy",
            "write_um_dataset_to_zarr",
            "append_to_zarr_time_series",
        ]
    )

# Intake catalogue writer (no hard dependencies beyond PyYAML)
try:
    from .catalogue_writer import IntakeCatalogueWriter, write_catalogue

    HAS_CATALOGUE = True
    __all__.extend(["IntakeCatalogueWriter", "write_catalogue"])
except ImportError:
    HAS_CATALOGUE = False

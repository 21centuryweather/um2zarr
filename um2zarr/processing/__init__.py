"""
Processing core modules for UM data transformation.

This module contains the core processing components that handle the conversion
from Iris cubes to Xarray datasets, including metadata management, coordinate
processing, masking, and optimization for Zarr storage.
"""

from .chunk_manager import ChunkManager
from .coordinate_processor import CoordinateProcessor
from .dtype_optimizer import DtypeOptimizer
from .iris_converter import IrisToXarrayConverter
from .masking_engine import MaskingEngine
from .stash_metadata import StashMetadata, StashMetadataManager

__all__ = [
    "IrisToXarrayConverter",
    "StashMetadataManager",
    "StashMetadata",
    "CoordinateProcessor",
    "MaskingEngine",
    "DtypeOptimizer",
    "ChunkManager",
]

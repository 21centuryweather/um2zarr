"""
Coordinate processing for CF compliance - stub implementation.
"""


import xarray as xr

from ..core.data_structures import GridMetadata


class CoordinateProcessor:
    """Process coordinates for CF compliance - stub for testing."""

    def fix_coordinates(
        self, dataset: xr.Dataset, grid_metadata: GridMetadata
    ) -> xr.Dataset:
        """Apply coordinate fixes (placeholder)."""
        return dataset

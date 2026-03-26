"""
Utility functions for um2zarr.

This module provides various utility functions including compatibility
shims and helper functions.
"""

from .numpy_compat import check_numpy_mule_compatibility, ensure_numpy_compatibility

__all__ = ["ensure_numpy_compatibility", "check_numpy_mule_compatibility"]

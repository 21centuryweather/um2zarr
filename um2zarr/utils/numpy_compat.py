"""
NumPy compatibility utilities.

Provides compatibility shims for NumPy functions that have been deprecated
or removed in newer versions, specifically to support legacy libraries
like mule that depend on old NumPy APIs.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def ensure_numpy_compatibility():
    """
    Ensure compatibility with legacy libraries that use deprecated NumPy functions.

    This function adds back deprecated NumPy functions as aliases to their
    modern equivalents, allowing older libraries like mule to work with
    newer NumPy versions.
    """
    # Check if numpy.product exists (was removed in NumPy 2.0)
    if not hasattr(np, "product"):
        logger.debug("Adding numpy.product compatibility alias for legacy libraries")
        # Add the missing function as an alias to numpy.prod
        np.product = np.prod

    # Check for other potential compatibility issues
    numpy_version = tuple(map(int, np.__version__.split(".")[:2]))

    if numpy_version >= (2, 0):
        logger.info(f"Running with NumPy {np.__version__} - compatibility layer active")

        # Add any other deprecated functions that might be needed
        # This can be extended as needed for other legacy dependencies

        # Example: if other functions were removed, we could add them here:
        # if not hasattr(np, 'some_old_function'):
        #     np.some_old_function = np.some_new_function

    else:
        logger.debug(
            f"Running with NumPy {np.__version__} - no compatibility fixes needed"
        )


def check_numpy_mule_compatibility():
    """
    Check if NumPy and mule versions are compatible.

    Returns:
        bool: True if compatible, False if compatibility issues detected
    """
    try:
        import mule

        numpy_version = tuple(map(int, np.__version__.split(".")[:2]))

        # Check for known incompatible combinations
        if numpy_version >= (2, 0):
            # Try to detect if mule will work with this NumPy version
            logger.info(
                f"Detected NumPy {np.__version__} with mule - applying compatibility fixes"
            )
            ensure_numpy_compatibility()

            # Test if the fix worked by trying to access the function mule needs
            if hasattr(np, "product"):
                logger.debug("✅ NumPy compatibility layer successfully applied")
                return True
            else:
                logger.error("❌ Failed to apply NumPy compatibility layer")
                return False
        else:
            logger.debug(f"NumPy {np.__version__} should be compatible with mule")
            return True

    except ImportError:
        logger.info("mule not available - NumPy compatibility check skipped")
        return True

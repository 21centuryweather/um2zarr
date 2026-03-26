"""
UM file reader using Iris and Mule for metadata extraction.

Provides the interface between UM fieldsfiles and the rest of the pipeline,
handling all UM-specific file format details using Iris and Mule.
"""

import logging
import time
from pathlib import Path

try:
    import iris
    import iris.cube
    import iris.exceptions

    # Configure Iris to use microsecond precision for dates (recommended)
    iris.FUTURE.date_microseconds = True
except ImportError:
    raise ImportError(
        "Iris is required for UM file reading. Install with: conda install -c conda-forge iris"
    )

# Apply NumPy compatibility shim for mule compatibility with NumPy 2.x
try:
    # Import compatibility layer before mule to ensure numpy.product is available
    from ..utils.numpy_compat import ensure_numpy_compatibility

    ensure_numpy_compatibility()
except ImportError:
    pass  # If utils module not available, continue without compatibility layer

try:
    import mule
except ImportError:
    raise ImportError(
        "Mule is required for UM metadata extraction. Please install mule package."
    )

from ..core.data_structures import GridMetadata, GridMetadataError, UMFileError

logger = logging.getLogger(__name__)


class UMFileReader:
    """Read UM files and extract metadata using Iris and Mule."""

    def __init__(self):
        """Initialize UM file reader."""
        self._setup_iris_warnings()

    def _setup_iris_warnings(self) -> None:
        """Configure Iris to reduce verbose warnings."""
        # Suppress common Iris warnings that aren't actionable
        import warnings

        warnings.filterwarnings("ignore", category=UserWarning, module="iris")

    def load_file(self, filepath: Path) -> tuple[list[iris.cube.Cube], GridMetadata]:
        """
        Load UM file and extract all metadata.

        Args:
            filepath: Path to UM fieldsfile

        Returns:
            Tuple of (iris cubes list, grid metadata)

        Raises:
            UMFileError: If file cannot be read or parsed
            GridMetadataError: If essential grid info missing
        """
        start_time = time.time()
        filepath = Path(filepath)

        if not filepath.exists():
            raise UMFileError(f"UM file not found: {filepath}")

        logger.info(f"Loading UM file: {filepath}")

        try:
            # Extract grid metadata using Mule first
            grid_metadata = self.extract_grid_metadata(filepath)
            logger.debug(f"Extracted grid metadata: {grid_metadata}")

            # Load all fields as Iris cubes
            cubes = iris.load(str(filepath))
            logger.info(f"Loaded {len(cubes)} cubes in {time.time() - start_time:.2f}s")

            if not cubes:
                raise UMFileError(f"No data cubes found in file: {filepath}")

            # Sort cubes by STASH code for consistent processing
            cubes = self._sort_cubes_by_stash(cubes)

            return cubes, grid_metadata

        except iris.exceptions.IrisError as e:
            raise UMFileError(f"Iris failed to load file {filepath}: {e}") from e
        except Exception as e:
            raise UMFileError(f"Unexpected error loading {filepath}: {e}") from e

    def extract_grid_metadata(self, filepath: Path) -> GridMetadata:
        """
        Extract grid metadata using Mule.

        Args:
            filepath: Path to UM fieldsfile

        Returns:
            GridMetadata object with grid information

        Raises:
            GridMetadataError: If essential metadata cannot be extracted
        """
        try:
            # Load UM file with Mule to access headers
            ff = mule.load_umfile(str(filepath))

            # Extract grid staggering type
            grid_staggering = ff.fixed_length_header.grid_staggering
            if grid_staggering == 6:
                grid_type = "EG"  # ENDGame grid
            elif grid_staggering == 3:
                grid_type = "ND"  # New Dynamics grid
            else:
                raise GridMetadataError(
                    f"Unknown grid staggering type: {grid_staggering}. "
                    f"Expected 6 (ENDGame) or 3 (New Dynamics)"
                )

            # Extract grid spacing
            dlat = ff.real_constants.row_spacing
            dlon = ff.real_constants.col_spacing

            if dlat <= 0 or dlon <= 0:
                raise GridMetadataError(
                    f"Invalid grid spacing: dlat={dlat}, dlon={dlon}"
                )

            # Extract vertical level heights
            z_rho = ff.level_dependent_constants.zsea_at_rho
            z_theta = ff.level_dependent_constants.zsea_at_theta

            logger.debug(
                f"Grid type: {grid_type}, spacing: {dlat}°×{dlon}°, "
                f"levels: {len(z_rho)} rho, {len(z_theta)} theta"
            )

            return GridMetadata(
                grid_type=grid_type, dlat=dlat, dlon=dlon, z_rho=z_rho, z_theta=z_theta
            )

        except Exception as e:
            # More detailed error reporting for debugging
            import traceback

            logger.debug(
                f"Full traceback for grid metadata error:\n{traceback.format_exc()}"
            )
            raise GridMetadataError(
                f"Failed to extract grid metadata from {filepath}: {e}"
            ) from e

    def filter_cubes_by_stash(
        self,
        cubes: list[iris.cube.Cube],
        include_list: list[int] | None = None,
        exclude_list: list[int] | None = None,
    ) -> list[iris.cube.Cube]:
        """
        Filter cubes based on STASH codes.

        Args:
            cubes: List of Iris cubes
            include_list: STASH codes to include (None = include all)
            exclude_list: STASH codes to exclude (None = exclude none)

        Returns:
            Filtered list of cubes

        Raises:
            ValueError: If both include_list and exclude_list provided
        """
        if include_list and exclude_list:
            raise ValueError("include_list and exclude_list are mutually exclusive")

        if not include_list and not exclude_list:
            return cubes

        filtered_cubes = []

        for cube in cubes:
            try:
                # Extract STASH code from cube attributes
                stash = cube.attributes.get("STASH")
                if stash is None:
                    logger.warning(f"Cube missing STASH attribute: {cube.name()}")
                    continue

                # Convert to itemcode format (section * 1000 + item)
                itemcode = 1000 * stash.section + stash.item

                # Apply filtering logic
                if include_list:
                    if itemcode in include_list:
                        filtered_cubes.append(cube)
                    else:
                        logger.debug(
                            f"Excluding STASH {itemcode} (not in include list)"
                        )

                elif exclude_list:
                    if itemcode not in exclude_list:
                        filtered_cubes.append(cube)
                    else:
                        logger.debug(f"Excluding STASH {itemcode} (in exclude list)")

            except AttributeError as e:
                logger.warning(f"Error processing STASH for cube {cube.name()}: {e}")
                continue

        logger.info(f"Filtered {len(cubes)} cubes to {len(filtered_cubes)} cubes")
        return filtered_cubes

    def _sort_cubes_by_stash(self, cubes: list[iris.cube.Cube]) -> list[iris.cube.Cube]:
        """Sort cubes by STASH code for consistent processing order."""

        def stash_key(cube):
            try:
                stash = cube.attributes.get("STASH")
                if stash:
                    return (stash.section, stash.item)
                else:
                    # Put cubes without STASH at the end
                    return (999, 999)
            except (AttributeError, KeyError):
                return (999, 999)

        return sorted(cubes, key=stash_key)

    def get_cube_summary(self, cubes: list[iris.cube.Cube]) -> list[dict]:
        """
        Get summary information about cubes for debugging/logging.

        Args:
            cubes: List of Iris cubes

        Returns:
            List of dictionaries with cube summary info
        """
        summaries = []

        for i, cube in enumerate(cubes):
            try:
                stash = cube.attributes.get("STASH")
                stash_code = (
                    f"{stash.section:02d}i{stash.item:03d}" if stash else "unknown"
                )

                summary = {
                    "index": i,
                    "name": cube.name(),
                    "stash_code": stash_code,
                    "shape": cube.shape,
                    "dtype": str(cube.dtype),
                    "units": str(cube.units),
                    "ndim": cube.ndim,
                    "coords": [coord.name() for coord in cube.coords()],
                }
                summaries.append(summary)

            except Exception as e:
                logger.warning(f"Error getting summary for cube {i}: {e}")
                summaries.append(
                    {
                        "index": i,
                        "name": getattr(cube, "name", lambda: "unknown")(),
                        "error": str(e),
                    }
                )

        return summaries

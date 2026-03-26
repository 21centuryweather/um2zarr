"""
Simplified UM file reader for testing - works without mule.

This version extracts grid metadata from Iris cubes directly,
avoiding the numpy compatibility issues with mule.
"""

import logging
import time
from pathlib import Path

import numpy as np

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

from ..core.data_structures import GridMetadata, GridMetadataError, UMFileError

logger = logging.getLogger(__name__)


class UMFileReader:
    """Simplified UM file reader that extracts metadata from Iris cubes."""

    def __init__(self):
        """Initialize UM file reader."""
        self._setup_iris_warnings()

    def _setup_iris_warnings(self) -> None:
        """Configure Iris to reduce verbose warnings."""
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
            # Load all fields as Iris cubes
            cubes = iris.load(str(filepath))
            logger.info(f"Loaded {len(cubes)} cubes in {time.time() - start_time:.2f}s")

            if not cubes:
                raise UMFileError(f"No data cubes found in file: {filepath}")

            # Extract grid metadata from cubes
            grid_metadata = self.extract_grid_metadata_from_cubes(cubes)
            logger.debug(f"Extracted grid metadata: {grid_metadata}")

            # Sort cubes by STASH code for consistent processing
            cubes = self._sort_cubes_by_stash(cubes)

            return cubes, grid_metadata

        except iris.exceptions.IrisError as e:
            raise UMFileError(f"Iris failed to load file {filepath}: {e}") from e
        except Exception as e:
            raise UMFileError(f"Unexpected error loading {filepath}: {e}") from e

    def extract_grid_metadata_from_cubes(
        self, cubes: list[iris.cube.Cube]
    ) -> GridMetadata:
        """
        Extract grid metadata from Iris cubes (fallback method).

        Args:
            cubes: List of loaded Iris cubes

        Returns:
            GridMetadata object with available information

        Raises:
            GridMetadataError: If essential metadata cannot be extracted
        """
        try:
            if not cubes:
                raise GridMetadataError("No cubes available for metadata extraction")

            # Use first cube to get basic grid info
            sample_cube = cubes[0]

            # Try to determine grid spacing from coordinates
            dlat = dlon = 1.0  # Default fallback
            grid_type = "ND"  # Default to New Dynamics

            # Extract latitude spacing
            try:
                lat_coord = sample_cube.coord("latitude")
                if len(lat_coord.points) > 1:
                    dlat = float(abs(lat_coord.points[1] - lat_coord.points[0]))
            except iris.exceptions.CoordinateNotFoundError:
                try:
                    lat_coord = sample_cube.coord("grid_latitude")
                    if len(lat_coord.points) > 1:
                        dlat = float(abs(lat_coord.points[1] - lat_coord.points[0]))
                except iris.exceptions.CoordinateNotFoundError:
                    logger.warning(
                        "Could not find latitude coordinate for grid spacing"
                    )

            # Extract longitude spacing
            try:
                lon_coord = sample_cube.coord("longitude")
                if len(lon_coord.points) > 1:
                    dlon = float(abs(lon_coord.points[1] - lon_coord.points[0]))
            except iris.exceptions.CoordinateNotFoundError:
                try:
                    lon_coord = sample_cube.coord("grid_longitude")
                    if len(lon_coord.points) > 1:
                        dlon = float(abs(lon_coord.points[1] - lon_coord.points[0]))
                except iris.exceptions.CoordinateNotFoundError:
                    logger.warning(
                        "Could not find longitude coordinate for grid spacing"
                    )

            # Try to get vertical level information
            z_rho = z_theta = np.array([0.0])  # Fallback

            try:
                # Look for model level coordinate
                for cube in cubes:
                    try:
                        level_coord = cube.coord("model_level_number")
                        if (
                            hasattr(level_coord, "attributes")
                            and "positive" in level_coord.attributes
                        ):
                            # This is likely a vertical coordinate with level info
                            z_rho = np.arange(len(level_coord.points), dtype=float)
                            z_theta = z_rho.copy()
                            break
                    except iris.exceptions.CoordinateNotFoundError:
                        continue
            except Exception:
                logger.warning("Could not extract vertical level information")

            logger.debug(
                f"Grid type: {grid_type}, spacing: {dlat}°×{dlon}°, "
                f"levels: {len(z_rho)} rho, {len(z_theta)} theta"
            )

            return GridMetadata(
                grid_type=grid_type, dlat=dlat, dlon=dlon, z_rho=z_rho, z_theta=z_theta
            )

        except Exception as e:
            raise GridMetadataError(f"Failed to extract grid metadata from cubes: {e}")

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

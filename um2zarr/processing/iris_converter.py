"""
Convert Iris cubes to Xarray datasets preserving all UM metadata.

This module handles the critical transition from Iris (UM-specific) to Xarray
(general scientific computing), ensuring no metadata is lost in the process.
"""

import logging
import time
from typing import Any

import numpy as np
import xarray as xr

try:
    import iris
    import iris.cube

    # Configure Iris to use microsecond precision for dates (recommended)
    iris.FUTURE.date_microseconds = True
except ImportError:
    raise ImportError(
        "Iris is required for Iris cube conversion. Install with: conda install -c conda-forge iris"
    )

from ..core.data_structures import ConversionError, ConversionResult, GridMetadata

logger = logging.getLogger(__name__)


class IrisToXarrayConverter:
    """Convert Iris cubes to Xarray datasets preserving all metadata."""

    def __init__(self, stash_manager: Any | None = None):
        """
        Initialize converter.

        Args:
            stash_manager: Optional STASH metadata manager for variable naming
        """
        self.stash_manager = stash_manager

    def convert_cubes_to_dataset(
        self,
        cubes: list[iris.cube.Cube],
        grid_metadata: GridMetadata,
        simple_names: bool = False,
    ) -> ConversionResult:
        """
        Convert list of Iris cubes to single Xarray dataset.

        Args:
            cubes: Iris cubes to convert
            grid_metadata: UM grid metadata from file header
            simple_names: Use simple variable names (fld_s01i002)

        Returns:
            ConversionResult with dataset and conversion metadata

        Raises:
            ConversionError: If critical cube cannot be converted
        """
        start_time = time.time()
        data_vars = {}
        warnings = []
        variables_skipped = []

        logger.info(f"Converting {len(cubes)} Iris cubes to Xarray dataset")

        for i, cube in enumerate(cubes):
            try:
                # Convert cube to DataArray
                data_array = self.convert_cube_to_dataarray(cube)

                # Generate variable name
                var_name = self._generate_variable_name(cube, simple_names)

                if var_name:
                    # Check for duplicate variable names
                    if var_name in data_vars:
                        # Make unique by appending index
                        original_name = var_name
                        var_name = f"{var_name}_{i}"
                        warnings.append(
                            f"Duplicate variable name '{original_name}', renamed to '{var_name}'"
                        )

                    data_vars[var_name] = data_array
                    logger.debug(f"Converted cube {i}: {var_name} {data_array.shape}")
                else:
                    variables_skipped.append(
                        f"Cube {i}: {cube.name()} (no variable name)"
                    )
                    logger.warning(
                        f"Skipped cube {i}: could not generate variable name"
                    )

            except Exception as e:
                variables_skipped.append(f"Cube {i}: {cube.name()} ({str(e)})")
                logger.error(f"Failed to convert cube {i} '{cube.name()}': {e}")
                continue

        if not data_vars:
            raise ConversionError("No cubes could be converted to variables")

        # Create dataset with global attributes
        # Use merge with compat='override' to handle coordinate conflicts
        try:
            dataset = xr.Dataset(
                data_vars=data_vars, attrs=self._create_global_attributes(grid_metadata)
            )
        except ValueError as e:
            if "conflicting values" in str(e):
                logger.warning(
                    f"Coordinate conflicts detected, using override mode: {e}"
                )
                # Create individual datasets and merge with override
                datasets = []
                for var_name, data_array in data_vars.items():
                    ds = xr.Dataset({var_name: data_array})
                    datasets.append(ds)

                if datasets:
                    dataset = datasets[0]
                    for ds in datasets[1:]:
                        dataset = xr.merge([dataset, ds], compat="override")

                    # Add global attributes after merging
                    dataset.attrs.update(self._create_global_attributes(grid_metadata))
                else:
                    raise ConversionError("No datasets to merge") from e
            else:
                raise

        # Optimize the dataset graph if it contains Dask arrays
        dataset = self._optimize_dataset_after_creation(dataset)

        processing_time = time.time() - start_time
        logger.info(f"Converted {len(data_vars)} variables in {processing_time:.2f}s")

        return ConversionResult(
            dataset=dataset,
            conversion_warnings=warnings,
            variables_converted=len(data_vars),
            variables_skipped=variables_skipped,
            processing_time=processing_time,
        )

    def convert_cube_to_dataarray(self, cube: iris.cube.Cube) -> xr.DataArray:
        """
        Convert single Iris cube to Xarray DataArray.

        Args:
            cube: Iris cube to convert

        Returns:
            Xarray DataArray with preserved metadata

        Raises:
            ConversionError: If cube conversion fails
        """
        try:
            # Extract data preserving lazy evaluation if available
            if cube.has_lazy_data():
                data = cube.lazy_data()
            else:
                data = cube.data

            # Build dimensions and coordinates
            dims = []
            coords = {}

            # Process dimensional coordinates
            for i, dim_coord in enumerate(cube.dim_coords):
                dim_name = self._standardize_dim_name(dim_coord.name())
                dims.append(dim_name)

                # Create coordinate with attributes
                coord_attrs = {
                    "units": str(dim_coord.units),
                    "long_name": dim_coord.long_name or dim_coord.name(),
                }

                if dim_coord.standard_name:
                    coord_attrs["standard_name"] = dim_coord.standard_name

                # Add bounds if present
                coord_data = dim_coord.points
                if dim_coord.has_bounds():
                    # Store bounds as separate variable
                    bounds_name = f"{dim_name}_bounds"
                    coords[bounds_name] = (["bounds_dim", "bnds"], dim_coord.bounds)
                    coord_attrs["bounds"] = bounds_name

                coords[dim_name] = (dim_name, coord_data, coord_attrs)

            # Process auxiliary coordinates (skip if they would conflict)
            for aux_coord in cube.aux_coords:
                coord_dims = cube.coord_dims(aux_coord)
                if coord_dims:
                    coord_name = self._standardize_coord_name(aux_coord.name())
                    coord_dim_names = [dims[i] for i in coord_dims]

                    # Skip coordinates that might conflict across cubes
                    if coord_name in [
                        "level_height",
                        "sigma",
                        "forecast_period",
                        "surface_altitude",
                        "forecast_reference_time",
                    ]:
                        logger.debug(
                            f"Skipping potentially conflicting coordinate: {coord_name}"
                        )
                        continue

                    coord_attrs = {
                        "units": str(aux_coord.units),
                        "long_name": aux_coord.long_name or aux_coord.name(),
                    }

                    if aux_coord.standard_name:
                        coord_attrs["standard_name"] = aux_coord.standard_name

                    coords[coord_name] = (
                        coord_dim_names,
                        aux_coord.points,
                        coord_attrs,
                    )

            # Create variable attributes
            attrs = self._extract_cube_attributes(cube)

            return xr.DataArray(
                data=data, dims=dims, coords=coords, attrs=attrs, name=cube.name()
            )

        except Exception as e:
            raise ConversionError(f"Failed to convert cube '{cube.name()}': {e}") from e

    def _standardize_dim_name(self, iris_name: str) -> str:
        """Convert Iris dimension names to CF-standard names."""
        name_mapping = {
            "grid_latitude": "y",
            "grid_longitude": "x",
            "latitude": "latitude",
            "longitude": "longitude",
            "time": "time",
            "model_level_number": "model_level",
            "pressure": "pressure",
            "height": "height",
            "depth": "depth",
            "atmosphere_hybrid_height_coordinate": "model_level",
            "atmosphere_hybrid_sigma_pressure_coordinate": "model_level",
        }
        return name_mapping.get(iris_name, iris_name.replace(" ", "_").lower())

    def _standardize_coord_name(self, iris_name: str) -> str:
        """Convert Iris coordinate names to CF-standard names."""
        name_mapping = {
            "forecast_reference_time": "forecast_reference_time",
            "forecast_period": "forecast_period",
            "level_height": "level_height",
            "sigma": "sigma",
            "surface_altitude": "surface_altitude",
        }
        return name_mapping.get(iris_name, iris_name.replace(" ", "_").lower())

    def _extract_cube_attributes(self, cube: iris.cube.Cube) -> dict[str, Any]:
        """Extract all relevant attributes from Iris cube."""
        attrs = {}

        # Basic metadata
        if cube.units:
            attrs["units"] = str(cube.units)
        if cube.standard_name:
            attrs["standard_name"] = cube.standard_name
        if cube.long_name:
            attrs["long_name"] = cube.long_name
        if hasattr(cube, "var_name") and cube.var_name:
            attrs["var_name"] = cube.var_name

        # STASH information (critical for UM data)
        if "STASH" in cube.attributes:
            stash = cube.attributes["STASH"]
            attrs["stash_section"] = stash.section
            attrs["stash_item"] = stash.item
            attrs["stash_model"] = stash.model

        # Cell methods (for temporal/spatial averaging info)
        if cube.cell_methods:
            cell_methods_str = self._format_cell_methods(cube.cell_methods)
            if cell_methods_str:
                attrs["cell_methods"] = cell_methods_str

        # Copy other cube attributes
        for key, value in cube.attributes.items():
            if key != "STASH":  # Already handled above
                # Convert to JSON-serializable types
                if isinstance(value, np.ndarray):
                    attrs[key] = value.tolist()
                elif hasattr(value, "__dict__"):
                    # Skip complex objects that can't be serialized
                    attrs[f"{key}_str"] = str(value)
                else:
                    attrs[key] = value

        return attrs

    def _format_cell_methods(self, cell_methods) -> str:
        """Format cell methods for CF compliance."""
        if not cell_methods:
            return ""

        method_strings = []
        for method in cell_methods:
            coord_names = " ".join(method.coord_names) if method.coord_names else ""
            intervals = " ".join(method.intervals) if method.intervals else ""
            comments = f" ({method.comments})" if method.comments else ""

            method_str = f"{coord_names}: {method.method}"
            if intervals:
                method_str += f" (interval: {intervals})"
            method_str += comments

            method_strings.append(method_str)

        return " ".join(method_strings)

    def _optimize_dataset_after_creation(self, dataset: xr.Dataset) -> xr.Dataset:
        """Optimize Dask graph for dataset after creation from Iris cubes."""
        try:
            from ..orchestration.dask_integration import (
                check_graph_size_and_warn,
                optimize_dask_graph,
            )

            # Check if dataset contains Dask arrays
            has_dask_arrays = any(
                hasattr(data_array.data, "chunks")
                for data_array in dataset.data_vars.values()
            )

            if not has_dask_arrays:
                logger.debug("No Dask arrays in dataset, skipping graph optimization")
                return dataset

            logger.debug("Optimizing dataset graph after Iris-to-Xarray conversion")

            # Check graph size and warn if large
            check_graph_size_and_warn(
                dataset,
                operation_name="Iris-to-Xarray conversion",
                warn_threshold_mb=8.0,
                error_threshold_mb=30.0,
            )

            # Apply graph optimizations
            optimized_dataset = optimize_dask_graph(
                dataset, optimize_graph=True, fuse_operations=True
            )

            return optimized_dataset

        except ImportError:
            logger.debug("Dask integration not available, skipping graph optimization")
            return dataset
        except Exception as e:
            logger.warning(
                f"Dataset graph optimization failed, continuing with original: {e}"
            )
            return dataset

    def _generate_variable_name(
        self, cube: iris.cube.Cube, simple: bool = False
    ) -> str | None:
        """Generate appropriate variable name for cube."""

        # Extract STASH information
        stash = cube.attributes.get("STASH")
        if not stash:
            # Fallback to cube name
            return self._clean_name(cube.name()) if cube.name() else None

        if simple:
            return f"fld_s{stash.section:02d}i{stash.item:03d}"

        # Use STASH manager if available
        if self.stash_manager:
            itemcode = 1000 * stash.section + stash.item
            var_name = self.stash_manager.get_variable_name(
                itemcode,
                has_max=any(m.method == "maximum" for m in cube.cell_methods),
                has_min=any(m.method == "minimum" for m in cube.cell_methods),
            )
            if var_name:
                return var_name

        # Fallback to cube name
        return self._clean_name(cube.name()) if cube.name() else None

    def _clean_name(self, name: str) -> str:
        """Clean variable name to be CF-compliant."""
        if not name:
            return ""

        # Replace spaces and special characters
        cleaned = name.replace(" ", "_").replace("-", "_")
        # Remove non-alphanumeric characters except underscores
        cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")
        # Ensure it starts with a letter
        if cleaned and not cleaned[0].isalpha():
            cleaned = "var_" + cleaned

        return cleaned.lower()

    def _create_global_attributes(self, grid_metadata: GridMetadata) -> dict[str, Any]:
        """Create global attributes for the dataset."""
        attrs = {
            "Conventions": "CF-1.8",
            "title": "UM data converted to Zarr format",
            "institution": "ACCESS-NRI",
            "source": "Unified Model (UM) fieldsfiles",
            "grid_type": grid_metadata.grid_type,
            "grid_spacing_lat": grid_metadata.dlat,
            "grid_spacing_lon": grid_metadata.dlon,
            "vertical_levels_rho": len(grid_metadata.z_rho),
            "vertical_levels_theta": len(grid_metadata.z_theta),
        }

        return attrs

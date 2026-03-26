"""
Data processing utilities for UM-to-Zarr conversion.

Provides masking, type coercion, and coordinate fixing functionality
to reproduce legacy um2netcdf4.py behavior using modern Xarray operations.
"""

import logging
from typing import Any

import numpy as np
import xarray as xr
from netCDF4 import default_fillvals

try:
    import dask.array as da

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

logger = logging.getLogger(__name__)

# STASH section/item numbers for UM heaviside functions
# These variables are masks for pressure-level diagnostics:
#   30301 = heaviside on theta (T) levels  → masks temperature-grid pressure fields
#   30302 = heaviside on UV (rho) levels   → masks wind-grid pressure fields
_HEAVISIDE_T_STASH = (30, 301)
_HEAVISIDE_UV_STASH = (30, 302)

# UM grid type codes that indicate a UV (rho / wind) staggered grid
_UV_GRID_CODES = {11, 18}


def _find_heaviside_vars(dataset: xr.Dataset) -> dict[str, str | None]:
    """
    Scan *dataset* for heaviside variables using STASH section/item attributes.

    Returns a dict with keys ``'t'`` and ``'uv'``, each either a variable name
    (str) or ``None`` if the corresponding heaviside array is absent.

    The function checks variable attributes for ``stash_section`` / ``stash_item``
    (lower-case, as set by IrisToXarrayConverter) or the upper-case alternatives
    written by some older converters.
    """
    result: dict[str, str | None] = {"t": None, "uv": None}
    for var_name, da in dataset.data_vars.items():
        attrs = da.attrs
        # Accept both lower and upper case keys
        section = attrs.get("stash_section", attrs.get("STASH_SECTION", None))
        item = attrs.get("stash_item", attrs.get("STASH_ITEM", None))
        try:
            section = int(section)
            item = int(item)
        except (TypeError, ValueError):
            continue
        if (section, item) == _HEAVISIDE_T_STASH:
            result["t"] = var_name
        elif (section, item) == _HEAVISIDE_UV_STASH:
            result["uv"] = var_name
    return result


def _is_pressure_level_var(da: xr.DataArray) -> bool:
    """Return True if *da* has a pressure-like dimension or coordinate."""
    for name in list(da.dims) + list(da.coords):
        nl = name.lower()
        if "pressure" in nl or nl == "plev":
            return True
    return False


def _is_uv_grid_var(da: xr.DataArray) -> bool:
    """
    Return True if *da* lives on the UV (rho / wind) staggered grid.

    The Iris converter stores the original UM grid type code in the
    ``grid_code`` attribute (integer).  UV grid codes are 11 and 18.
    Falls back to inspecting coordinate names (lon_u / lat_v patterns).
    """
    grid_code = da.attrs.get("grid_code", da.attrs.get("STASH_GRID", None))
    if grid_code is not None:
        try:
            return int(grid_code) in _UV_GRID_CODES
        except (TypeError, ValueError):
            pass
    # Coordinate name heuristic
    for coord in da.coords:
        if coord in ("lon_u", "lat_v"):
            return True
    return False


class MaskingOptions:
    """Configuration for masking operations."""

    def __init__(self, nomask: bool = False, hcrit: float = 0.5, simple: bool = False):
        """
        Initialize masking options.

        Args:
            nomask: If True, remove all masks from data
            hcrit: Critical value for heaviside function masking (default 0.5)
            simple: If True, use simple processing without complex masking
        """
        self.nomask = nomask
        self.hcrit = hcrit
        self.simple = simple


class DataTypeOptions:
    """Configuration for data type handling."""

    def __init__(
        self,
        use64bit: bool = False,
        preserve_int: bool = True,
        float_precision: str = "float32",
    ):
        """
        Initialize data type options.

        Args:
            use64bit: If True, preserve 64-bit precision
            preserve_int: If True, try to preserve integer types where appropriate
            float_precision: Default float precision ('float32' or 'float64')
        """
        self.use64bit = use64bit
        self.preserve_int = preserve_int
        self.float_precision = float_precision


def apply_heaviside_mask(
    data_array: xr.DataArray, heaviside_array: xr.DataArray, hcrit: float = 0.5
) -> xr.DataArray:
    """
    Apply heaviside function masking to pressure level data.

    This reproduces the legacy apply_mask function behavior using Xarray operations.

    Args:
        data_array: Data to mask
        heaviside_array: Heaviside function data for masking
        hcrit: Critical value for masking (default 0.5)

    Returns:
        Masked data array

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>>
        >>> # Create test data
        >>> data = xr.DataArray(
        ...     np.random.random((10, 5, 4)),
        ...     dims=['time', 'pressure', 'lat'],
        ...     coords={'pressure': [1000, 850, 700, 500, 200]}
        ... )
        >>> heaviside = xr.DataArray(
        ...     np.array([1.0, 0.8, 0.6, 0.4, 0.2])[None, :, None],
        ...     dims=['time', 'pressure', 'lat'],
        ...     coords={'pressure': [1000, 850, 700, 500, 200]}
        ... )
        >>>
        >>> # Apply masking
        >>> masked = apply_heaviside_mask(data, heaviside, hcrit=0.5)
        >>> # Values where heaviside <= 0.5 should be masked
        >>> print(masked.shape)
        (10, 5, 4)
    """
    logger.debug(f"Applying heaviside mask with hcrit={hcrit}")

    # Check if shapes match directly
    if data_array.shape == heaviside_array.shape:
        # Direct application - shapes match
        logger.debug("Shapes match - applying direct masking")

        # Create mask where heaviside <= hcrit
        mask = heaviside_array <= hcrit

        # Apply mask using xarray where (equivalent to numpy masked_array)
        # Use heaviside as divisor (reproducing original c.data/heaviside.data)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (data_array / heaviside_array).where(~mask)

    else:
        # Need to match coordinates - extract matching levels
        logger.debug("Shapes don't match - attempting coordinate matching")

        # Find pressure coordinate
        pressure_coord = None
        for coord in data_array.coords:
            if "pressure" in coord.lower() or coord == "plev":
                pressure_coord = coord
                break

        if pressure_coord is None:
            raise ValueError("Could not find pressure coordinate for masking")

        # Get pressure values from data array
        data_pressures = data_array.coords[pressure_coord].values

        # Select matching levels from heaviside function
        heaviside_subset = heaviside_array.sel({pressure_coord: data_pressures})

        # Apply masking
        mask = heaviside_subset <= hcrit
        with np.errstate(divide="ignore", invalid="ignore"):
            result = (data_array / heaviside_subset).where(~mask)

    # Convert to float32 (matching legacy behavior)
    result = result.astype(np.float32)

    logger.debug(f"Masking complete - result shape: {result.shape}")
    return result


def remove_all_masks(data_array: xr.DataArray) -> xr.DataArray:
    """
    Remove all masks from a data array (nomask option).

    Args:
        data_array: Input data array

    Returns:
        Data array with masks removed

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>>
        >>> # Create masked data
        >>> data = np.ma.masked_array([1, 2, 3, 4, 5], mask=[0, 1, 0, 1, 0])
        >>> arr = xr.DataArray(data, dims=['x'])
        >>>
        >>> # Remove masks
        >>> unmasked = remove_all_masks(arr)
        >>> print(unmasked.values)  # Should have no masked values
        [1. 2. 3. 4. 5.]
    """
    logger.debug("Removing all masks from data")

    # If data is masked, fill masked values and remove mask
    if hasattr(data_array.values, "mask"):
        # Fill masked values with the array's fill value or NaN
        result = data_array.copy()
        if hasattr(data_array.values, "fill_value"):
            result.values = data_array.values.filled(data_array.values.fill_value)
        else:
            result.values = data_array.values.filled(np.nan)
    else:
        result = data_array.copy()

    logger.debug("Mask removal complete")
    return result


def coerce_data_types(dataset: xr.Dataset, options: DataTypeOptions) -> xr.Dataset:
    """
    Apply data type coercion following legacy um2netcdf4 logic.

    Args:
        dataset: Input dataset
        options: Data type options

    Returns:
        Dataset with coerced data types

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>>
        >>> # Create dataset with mixed types
        >>> ds = xr.Dataset({
        ...     'temp': (['x'], np.array([1.0, 2.0, 3.0], dtype=np.float64)),
        ...     'count': (['x'], np.array([1, 2, 3], dtype=np.int64))
        ... })
        >>>
        >>> # Apply type coercion
        >>> options = DataTypeOptions(use64bit=False)
        >>> ds_coerced = coerce_data_types(ds, options)
        >>> print(ds_coerced['temp'].dtype)  # Should be float32
        float32
        >>> print(ds_coerced['count'].dtype)  # Should be int32
        int32
    """
    logger.debug("Applying data type coercion")

    result = dataset.copy()

    for var_name, data_array in result.data_vars.items():
        original_dtype = data_array.dtype

        # Handle float types
        if data_array.dtype.kind == "f":  # float types
            if not options.use64bit and data_array.dtype == np.float64:
                logger.debug(f"Converting {var_name} from float64 to float32")
                result[var_name] = data_array.astype(np.float32)

        # Handle integer types
        elif data_array.dtype.kind == "i":  # integer types
            if not options.use64bit and data_array.dtype == np.int64:
                logger.debug(f"Converting {var_name} from int64 to int32")
                result[var_name] = data_array.astype(np.int32)

    # Also handle coordinates
    for coord_name, coord_array in result.coords.items():
        if coord_array.dtype == np.int64:
            logger.debug(f"Converting coordinate {coord_name} from int64 to int32")
            result = result.assign_coords({coord_name: coord_array.astype(np.int32)})

    logger.debug("Data type coercion complete")
    return result


def set_fill_values(dataset: xr.Dataset) -> xr.Dataset:
    """
    Set appropriate fill values for data variables.

    Reproduces the legacy cubewrite fill value logic.

    Args:
        dataset: Input dataset

    Returns:
        Dataset with fill values set in encoding

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>>
        >>> ds = xr.Dataset({
        ...     'temp': (['x'], np.array([1.0, 2.0, 3.0], dtype=np.float32)),
        ...     'count': (['x'], np.array([1, 2, 3], dtype=np.int16))
        ... })
        >>>
        >>> ds_with_fills = set_fill_values(ds)
        >>> print(ds_with_fills['temp'].encoding['_FillValue'])  # Should be 1e20
        1e+20
    """
    logger.debug("Setting fill values")

    result = dataset.copy()

    for var_name, data_array in result.data_vars.items():
        dtype = data_array.dtype

        if dtype.kind == "f":  # float types
            fill_value = 1e20
        else:  # integer types
            # Use netCDF default fill values
            dtype_key = f"{dtype.kind}{dtype.itemsize}"
            fill_value = default_fillvals.get(dtype_key, -999)

        # Set in encoding for zarr/netcdf output
        if var_name not in result.encoding:
            result.encoding[var_name] = {}
        result[var_name].encoding["_FillValue"] = fill_value

        # Also set as attribute (for legacy compatibility)
        result[var_name].attrs["missing_value"] = fill_value

        logger.debug(f"Set fill value for {var_name} ({dtype}): {fill_value}")

    logger.debug("Fill value setting complete")
    return result


def fix_coordinate_names(
    dataset: xr.Dataset, grid_type: str, dlat: float, dlon: float
) -> xr.Dataset:
    """
    Fix latitude/longitude coordinate names based on grid staggering.

    Reproduces the fix_latlon_coord logic from legacy code.

    Args:
        dataset: Input dataset
        grid_type: Grid type ('EG' for EndGame, 'ND' for New Dynamics)
        dlat: Latitude spacing
        dlon: Longitude spacing

    Returns:
        Dataset with corrected coordinate names

    Examples:
        >>> import xarray as xr
        >>> import numpy as np
        >>>
        >>> # Create dataset with lat/lon coordinates
        >>> ds = xr.Dataset(
        ...     data_vars={'temp': (['lat', 'lon'], np.random.random((5, 6)))},
        ...     coords={
        ...         'lat': np.linspace(-90, 90, 5),
        ...         'lon': np.linspace(0, 360, 6, endpoint=False)
        ...     }
        ... )
        >>>
        >>> ds_fixed = fix_coordinate_names(ds, 'EG', 1.0, 1.0)
        >>> # Coordinate names should be updated based on grid properties
    """
    logger.debug(f"Fixing coordinate names for {grid_type} grid")

    result = dataset.copy()

    # Find latitude coordinate
    lat_coord = None
    for coord in result.coords:
        if "lat" in coord.lower():
            lat_coord = coord
            break

    # Find longitude coordinate
    lon_coord = None
    for coord in result.coords:
        if "lon" in coord.lower():
            lon_coord = coord
            break

    if lat_coord is None or lon_coord is None:
        logger.warning("Could not find lat/lon coordinates to fix")
        return result

    # Get coordinate values
    lat_values = result[lat_coord].values
    lon_values = result[lon_coord].values

    # Determine new coordinate names based on legacy logic
    new_lat_name = lat_coord
    new_lon_name = lon_coord

    # Latitude naming logic
    if len(lat_values) == 180:
        new_lat_name = "lat_river"
    elif (lat_values[0] == -90 and grid_type == "EG") or (
        np.allclose(-90.0 + 0.5 * dlat, lat_values[0]) and grid_type == "ND"
    ):
        new_lat_name = "lat_v"
    else:
        new_lat_name = "lat"

    # Longitude naming logic
    if len(lon_values) == 360:
        new_lon_name = "lon_river"
    elif (lon_values[0] == 0 and grid_type == "EG") or (
        np.allclose(0.5 * dlon, lon_values[0]) and grid_type == "ND"
    ):
        new_lon_name = "lon_u"
    else:
        new_lon_name = "lon"

    # Rename coordinates if they changed
    rename_dict = {}
    if new_lat_name != lat_coord:
        rename_dict[lat_coord] = new_lat_name
        logger.debug(f"Renaming {lat_coord} to {new_lat_name}")

    if new_lon_name != lon_coord:
        rename_dict[lon_coord] = new_lon_name
        logger.debug(f"Renaming {lon_coord} to {new_lon_name}")

    if rename_dict:
        result = result.rename(rename_dict)

    logger.debug("Coordinate name fixing complete")
    return result


def fix_level_coordinates(
    dataset: xr.Dataset,
    z_rho: np.ndarray | None = None,
    z_theta: np.ndarray | None = None,
) -> xr.Dataset:
    """
    Fix level coordinate names to distinguish rho and theta levels.

    Args:
        dataset: Input dataset
        z_rho: Reference rho level heights
        z_theta: Reference theta level heights

    Returns:
        Dataset with corrected level coordinate names
    """
    logger.debug("Fixing level coordinate names")

    result = dataset.copy()

    # Find model level coordinates
    level_coord = None
    height_coord = None
    sigma_coord = None

    for coord in result.coords:
        coord_lower = coord.lower()
        if "model_level" in coord_lower and "number" in coord_lower:
            level_coord = coord
        elif "level_height" in coord_lower:
            height_coord = coord
        elif "sigma" in coord_lower:
            sigma_coord = coord

    if not all([level_coord, height_coord, sigma_coord]):
        logger.debug(
            "Not all level coordinates found - skipping level coordinate fixing"
        )
        return result

    if z_rho is None or z_theta is None:
        logger.debug("Reference levels not provided - skipping level coordinate fixing")
        return result

    # Get height values
    height_values = result[height_coord].values

    # Check if these are rho or theta levels
    if len(height_values) > 0:
        d_rho = np.abs(height_values[0] - z_rho)
        d_theta = np.abs(height_values[0] - z_theta)

        rename_dict = {}

        if d_rho.min() < 1e-6:
            # These are rho levels
            rename_dict[level_coord] = "model_rho_level_number"
            rename_dict[height_coord] = "rho_level_height"
            rename_dict[sigma_coord] = "sigma_rho"
            logger.debug("Identified as rho levels")

        elif d_theta.min() < 1e-6:
            # These are theta levels
            rename_dict[level_coord] = "model_theta_level_number"
            rename_dict[height_coord] = "theta_level_height"
            rename_dict[sigma_coord] = "sigma_theta"
            logger.debug("Identified as theta levels")

        if rename_dict:
            result = result.rename(rename_dict)

    logger.debug("Level coordinate fixing complete")
    return result


def fix_pressure_coordinates(dataset: xr.Dataset) -> xr.Dataset:
    """
    Fix pressure coordinates to follow CF conventions.

    Args:
        dataset: Input dataset

    Returns:
        Dataset with fixed pressure coordinates
    """
    logger.debug("Fixing pressure coordinates")

    result = dataset.copy()

    # Find pressure coordinate
    pressure_coord = None
    for coord in result.coords:
        if "pressure" in coord.lower() or coord == "plev":
            pressure_coord = coord
            break

    if pressure_coord is None:
        logger.debug("No pressure coordinate found")
        return result

    # Set positive attribute and convert units
    pressure_array = result[pressure_coord]

    # Set CF attributes
    pressure_array.attrs["positive"] = "down"

    # Convert to Pa if not already
    if "units" in pressure_array.attrs:
        current_units = pressure_array.attrs["units"]
        if current_units.lower() in ["hpa", "mbar", "mb"]:
            # Convert from hPa/mbar to Pa
            new_pressure = pressure_array * 100
            new_pressure.attrs = pressure_array.attrs.copy()
            new_pressure.attrs["units"] = "Pa"
            result = result.assign_coords({pressure_coord: new_pressure})
            logger.debug("Converted pressure units from hPa to Pa")

    # Round to avoid floating point precision issues
    pressure_values = result[pressure_coord].values
    rounded_values = np.round(pressure_values, 5)
    # Create new coordinate with rounded values, preserving attributes
    pressure_attrs = result[pressure_coord].attrs.copy()
    result = result.assign_coords(
        {pressure_coord: (pressure_coord, rounded_values, pressure_attrs)}
    )

    # Check if pressure needs to be flipped (should be decreasing)
    if len(pressure_values) > 1 and pressure_values[0] < pressure_values[-1]:
        logger.debug("Flipping pressure coordinate to be decreasing")
        result = result.isel({pressure_coord: slice(None, None, -1)})

    logger.debug("Pressure coordinate fixing complete")
    return result


def process_dataset(
    dataset: xr.Dataset,
    masking_options: MaskingOptions,
    dtype_options: DataTypeOptions,
    grid_info: dict[str, Any] | None = None,
    heaviside_data: dict[str, xr.DataArray] | None = None,
) -> xr.Dataset:
    """
    Apply comprehensive data processing to a dataset.

    This is the main processing function that orchestrates all data transformations.

    Args:
        dataset: Input dataset
        masking_options: Masking configuration
        dtype_options: Data type configuration
        grid_info: Grid information dictionary with keys:
                  'grid_type', 'dlat', 'dlon', 'z_rho', 'z_theta'
        heaviside_data: Dictionary with 'uv' and 't' heaviside arrays if available

    Returns:
        Processed dataset
    """
    logger.info("Starting comprehensive dataset processing")

    result = dataset.copy()

    # --- Heaviside masking ---
    if masking_options.nomask:
        # Caller explicitly wants unmasked data — skip all masking steps.
        logger.info("nomask=True: skipping all masking")
    else:
        # Build the heaviside lookup from caller-supplied data OR from the
        # dataset itself (variables recognised by STASH section/item).
        if heaviside_data is None:
            heaviside_data = _find_heaviside_vars(result)
            # Convert name→array for uniform handling below
            heaviside_data = {
                k: (result[v] if v is not None else None)
                for k, v in heaviside_data.items()
            }

        hv_t = heaviside_data.get("t")
        hv_uv = heaviside_data.get("uv")

        if hv_t is not None or hv_uv is not None:
            logger.info(
                "Applying heaviside masking to pressure-level variables "
                f"(hcrit={masking_options.hcrit})"
            )

            # Track the names of the heaviside arrays so we can drop them later
            hv_var_names: set = set()
            for hv_arr in (hv_t, hv_uv):
                if hv_arr is not None:
                    # Find matching var_name by identity check (handle copy)
                    for vn, da in result.data_vars.items():
                        if da.equals(hv_arr):
                            hv_var_names.add(vn)
                            break

            masked_count = 0
            for var_name, da in list(result.data_vars.items()):
                if var_name in hv_var_names:
                    continue  # Skip heaviside arrays themselves
                if not _is_pressure_level_var(da):
                    continue  # Only mask pressure-level fields

                # Choose theta vs UV heaviside based on the variable's grid type
                if hv_uv is not None and _is_uv_grid_var(da):
                    hv = hv_uv
                elif hv_t is not None:
                    hv = hv_t
                elif hv_uv is not None:
                    hv = hv_uv
                else:
                    continue  # No suitable heaviside available

                try:
                    result[var_name] = apply_heaviside_mask_dask_aware(
                        da, hv, masking_options.hcrit
                    )
                    masked_count += 1
                    logger.debug(f"Applied heaviside mask to '{var_name}'")
                except Exception as exc:
                    logger.warning(
                        f"Could not apply heaviside mask to '{var_name}': {exc}"
                    )

            logger.info(f"Heaviside masking applied to {masked_count} variable(s)")

            # Drop heaviside variables — they are auxiliary masks, not scientific output
            if hv_var_names:
                result = result.drop_vars(list(hv_var_names))
                logger.info(
                    f"Dropped heaviside auxiliary variables: {sorted(hv_var_names)}"
                )

    # Remove residual numpy masked-array masks (e.g. from Iris) when nomask=True
    if masking_options.nomask:
        logger.info("Removing all masks")
        for var_name in result.data_vars:
            result[var_name] = remove_all_masks(result[var_name])

    # Apply data type coercion
    logger.info("Applying data type coercion")
    result = coerce_data_types(result, dtype_options)

    # Set fill values
    logger.info("Setting fill values")
    result = set_fill_values(result)

    # Fix coordinates if grid info is available
    if grid_info:
        logger.info("Fixing coordinates")

        # Fix lat/lon coordinate names
        if all(key in grid_info for key in ["grid_type", "dlat", "dlon"]):
            result = fix_coordinate_names(
                result, grid_info["grid_type"], grid_info["dlat"], grid_info["dlon"]
            )

        # Fix level coordinates
        if "z_rho" in grid_info and "z_theta" in grid_info:
            result = fix_level_coordinates(
                result, grid_info["z_rho"], grid_info["z_theta"]
            )

    # Fix pressure coordinates
    result = fix_pressure_coordinates(result)

    logger.info("Dataset processing complete")
    return result


def is_dask_array(array: Any) -> bool:
    """
    Check if array is a Dask array.

    Args:
        array: Array to check

    Returns:
        True if array is a Dask array
    """
    if not HAS_DASK:
        return False
    return isinstance(array, da.Array)


def preserve_dask_chunks(func):
    """
    Decorator to preserve Dask chunking in array operations.

    Ensures that operations on Dask arrays maintain their chunking structure.
    """

    def wrapper(data_array: xr.DataArray, *args, **kwargs) -> xr.DataArray:
        # If input is chunked, ensure output preserves chunking
        if hasattr(data_array.data, "chunks") and HAS_DASK:
            original_chunks = data_array.chunks
            result = func(data_array, *args, **kwargs)

            # Try to maintain original chunking if possible
            if hasattr(result, "chunk") and result.dims == data_array.dims:
                try:
                    result = result.chunk(original_chunks)
                except Exception as e:
                    logger.debug(f"Could not preserve original chunks: {e}")

            return result
        else:
            return func(data_array, *args, **kwargs)

    return wrapper


@preserve_dask_chunks
def apply_heaviside_mask_dask_aware(
    data_array: xr.DataArray, heaviside_array: xr.DataArray, hcrit: float = 0.5
) -> xr.DataArray:
    """
    Dask-aware version of heaviside masking.

    This version preserves Dask chunking and uses lazy evaluation.
    """
    logger.debug("Applying Dask-aware heaviside masking")

    # Use the standard implementation but ensure Dask compatibility
    return apply_heaviside_mask(data_array, heaviside_array, hcrit)


def optimize_dask_graph(dataset: xr.Dataset) -> xr.Dataset:
    """
    Optimize Dask computation graph for the dataset.

    Args:
        dataset: Dataset with potential Dask arrays

    Returns:
        Dataset with optimized Dask graph
    """
    if not HAS_DASK:
        return dataset

    logger.debug("Optimizing Dask computation graph")

    # Check if any variables use Dask
    has_dask_vars = any(
        hasattr(var.data, "chunks") for var in dataset.data_vars.values()
    )

    if not has_dask_vars:
        return dataset

    # Persist commonly accessed coordinates in memory
    coords_to_persist = []
    for coord_name, coord_array in dataset.coords.items():
        # Persist small coordinate arrays
        if coord_array.nbytes < 1024 * 1024:  # Less than 1MB
            coords_to_persist.append(coord_name)

    if coords_to_persist:
        logger.debug(f"Persisting small coordinates: {coords_to_persist}")
        dataset = dataset.persist()

    return dataset


def estimate_memory_usage(dataset: xr.Dataset) -> dict[str, float]:
    """
    Estimate memory usage for dataset variables.

    Args:
        dataset: Dataset to analyze

    Returns:
        Dictionary with memory estimates in MB per variable
    """
    estimates = {}

    for var_name, data_array in dataset.data_vars.items():
        # Get array size in bytes
        if hasattr(data_array.data, "nbytes"):
            bytes_size = data_array.data.nbytes
        else:
            # Estimate based on shape and dtype
            bytes_size = np.prod(data_array.shape) * data_array.dtype.itemsize

        mb_size = bytes_size / (1024**2)

        # If it's a Dask array, get chunk information
        chunk_info = {}
        if hasattr(data_array.data, "chunks"):
            chunk_shape = tuple(c[0] if c else 0 for c in data_array.data.chunks)
            chunk_bytes = np.prod(chunk_shape) * data_array.dtype.itemsize
            chunk_mb = chunk_bytes / (1024**2)

            chunk_info = {
                "is_dask": True,
                "chunk_shape": chunk_shape,
                "chunk_size_mb": chunk_mb,
                "n_chunks": data_array.data.npartitions
                if hasattr(data_array.data, "npartitions")
                else "unknown",
            }
        else:
            chunk_info = {"is_dask": False}

        estimates[var_name] = {"total_size_mb": mb_size, **chunk_info}

    return estimates

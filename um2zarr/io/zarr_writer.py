"""
Zarr writer with CF-compliant metadata and optimal encoding.

Provides efficient Zarr output with compression, chunking, and storage backend support.
Ensures CF compliance and optimal performance for UM datasets.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

# Zarr and compression imports
try:
    import zarr

    # For Zarr 3.x compatibility, use numcodecs for compression constants
    try:
        from numcodecs import blosc

        BLOSC_SHUFFLE = blosc.SHUFFLE
        BLOSC_NOSHUFFLE = blosc.NOSHUFFLE
    except ImportError:
        # Fallback for older zarr versions
        try:
            from zarr import blosc

            BLOSC_SHUFFLE = blosc.SHUFFLE
            BLOSC_NOSHUFFLE = blosc.NOSHUFFLE
        except ImportError:
            # Use constants directly
            BLOSC_SHUFFLE = 1
            BLOSC_NOSHUFFLE = 0
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    BLOSC_SHUFFLE = 1
    BLOSC_NOSHUFFLE = 0

# Storage backend imports
try:
    import s3fs

    HAS_S3 = True
except ImportError:
    HAS_S3 = False

try:
    import gcsfs

    HAS_GCS = True
except ImportError:
    HAS_GCS = False

# CF validation imports
try:
    import cfchecker

    HAS_CF_CHECKER = True
except ImportError:
    HAS_CF_CHECKER = False

logger = logging.getLogger(__name__)


@dataclass
class EncodingStrategy:
    """Configuration for Zarr encoding strategy."""

    # Compression settings
    compressor: str = "zstd"  # 'zstd', 'blosc', 'lz4', 'gzip'
    compression_level: int = 3  # 1-9, higher = better compression, slower
    shuffle: bool = True  # Byte shuffling for better compression

    # Chunk settings
    auto_chunk: bool = True  # Use ChunkManager for optimal chunking
    chunk_strategy: str = "balanced"  # 'memory', 'balanced', 'storage'

    # Data type optimization
    auto_dtype: bool = True  # Optimize dtypes automatically
    float_precision: str = "preserve"  # 'preserve', 'float32', 'float64'

    # Fill value handling
    auto_fill_values: bool = True  # Set appropriate fill values

    # Storage backend
    storage_backend: str = "local"  # 'local', 's3', 'gcs'


class ZarrWriter:
    """
    Write Xarray datasets to Zarr with optimal encoding and CF compliance.

    Features:
    - Automatic compression and chunking optimization
    - CF convention validation and compliance
    - Multiple storage backend support (local, S3, GCS)
    - Time series append mode
    - Comprehensive error handling and logging
    """

    def __init__(
        self,
        encoding_strategy: EncodingStrategy | None = None,
        validate_cf: bool = True,
    ):
        """
        Initialize Zarr writer.

        Args:
            encoding_strategy: Encoding configuration
            validate_cf: Whether to validate CF compliance
        """
        if not HAS_ZARR:
            raise ImportError(
                "Zarr is required for ZarrWriter. Install with: pip install zarr"
            )

        self.encoding_strategy = encoding_strategy or EncodingStrategy()
        self.validate_cf = validate_cf
        self.logger = logging.getLogger(__name__)

        # Storage backend mapping
        self._storage_backends = {
            "local": self._get_local_mapper,
            "s3": self._get_s3_mapper,
            "gcs": self._get_gcs_mapper,
        }

        # Note: Compression configurations are now handled by _create_compressor method

    def write_dataset(
        self,
        dataset: xr.Dataset,
        store_path: str | Path,
        mode: str = "w",
        append_dim: str | None = None,
        consolidated: bool = True,
        optimize_graph: bool = True,
    ) -> dict[str, Any]:
        """
        Write dataset to Zarr store with optimal encoding.

        Args:
            dataset: Dataset to write
            store_path: Path to Zarr store
            mode: Write mode ('w', 'w-', 'a', 'r+')
            append_dim: Dimension to append along (for time series)
            consolidated: Whether to consolidate metadata
            optimize_graph: Whether to optimize Dask graph before writing

        Returns:
            Dictionary with write statistics and metadata
        """
        logger.info(f"Writing dataset to Zarr store: {store_path}")
        start_time = time.time()

        try:
            # Validate CF compliance if requested
            if self.validate_cf:
                cf_warnings = self._validate_cf_compliance(dataset)
                if cf_warnings:
                    logger.warning(
                        f"CF compliance issues found: {len(cf_warnings)} warnings"
                    )
                    for warning in cf_warnings[:5]:  # Show first 5 warnings
                        logger.warning(f"  {warning}")

            # Prepare dataset for writing
            prepared_dataset = self._prepare_dataset(dataset)

            # Optimize Dask graph if requested
            if optimize_graph:
                prepared_dataset = self._optimize_dataset_graph(prepared_dataset)

            # Create encoding configuration
            encoding = self._create_encoding(prepared_dataset)

            # Get storage mapper
            store_mapper = self._get_storage_mapper(store_path)

            # Write dataset
            write_stats = self._write_to_zarr(
                prepared_dataset, store_mapper, encoding, mode, append_dim, consolidated
            )

            # Calculate final statistics
            write_time = time.time() - start_time
            write_stats.update(
                {
                    "write_time_seconds": write_time,
                    "store_path": str(store_path),
                    "mode": mode,
                    "consolidated": consolidated,
                }
            )

            logger.info(f"Zarr write completed in {write_time:.2f}s")
            logger.info(f"Variables written: {write_stats.get('n_variables', 0)}")
            logger.info(
                f"Compression ratio: {write_stats.get('compression_ratio', 0.0):.2f} "
                f"({write_stats.get('compressed_size_mb', 0.0):.1f} MB compressed)"
            )

            return write_stats

        except Exception as e:
            logger.error(f"Failed to write Zarr store: {e}")
            raise

    def _prepare_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        """Prepare dataset for Zarr writing."""
        logger.debug("Preparing dataset for Zarr writing")

        result = dataset.copy()

        # Apply chunking if needed
        if self.encoding_strategy.auto_chunk:
            result = self._apply_optimal_chunking(result)

        # Optimize data types
        if self.encoding_strategy.auto_dtype:
            result = self._optimize_dtypes(result)

        # Set fill values
        if self.encoding_strategy.auto_fill_values:
            result = self._set_fill_values(result)

        # Ensure CF compliance
        result = self._ensure_cf_compliance(result)

        return result

    def _apply_optimal_chunking(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply optimal chunking using ChunkManager."""
        from ..processing.chunk_manager import ChunkingStrategy, ChunkManager

        # Configure chunking strategy based on encoding strategy
        chunk_strategy = ChunkingStrategy(
            storage_backend=self.encoding_strategy.storage_backend,
            target_chunk_size_mb=self._get_target_chunk_size(),
        )

        chunk_manager = ChunkManager(chunk_strategy)

        # Apply chunking
        if self.encoding_strategy.chunk_strategy == "memory":
            # Optimize for memory usage
            chunked = chunk_manager.apply_chunking(dataset)
        elif self.encoding_strategy.chunk_strategy == "storage":
            # Optimize for storage backend
            chunked = chunk_manager.optimize_for_zarr(
                dataset, self.encoding_strategy.storage_backend
            )
        else:
            # Balanced approach
            chunked = chunk_manager.optimize_for_zarr(
                dataset, self.encoding_strategy.storage_backend
            )

        logger.debug("Applied optimal chunking to dataset")
        return chunked

    def _get_target_chunk_size(self) -> float:
        """Get target chunk size based on strategy."""
        chunk_sizes = {
            "memory": 50.0,  # 50 MB - memory conservative
            "balanced": 100.0,  # 100 MB - balanced
            "storage": 200.0,  # 200 MB - storage optimized
        }
        return chunk_sizes.get(self.encoding_strategy.chunk_strategy, 100.0)

    def _optimize_dtypes(self, dataset: xr.Dataset) -> xr.Dataset:
        """Optimize data types for storage efficiency."""
        logger.debug("Optimizing data types")

        result = dataset.copy()

        for var_name, data_array in result.data_vars.items():
            original_dtype = data_array.dtype

            if self.encoding_strategy.float_precision == "float32":
                if data_array.dtype == np.float64:
                    result[var_name] = data_array.astype(np.float32)
                    logger.debug(f"Converted {var_name} from float64 to float32")

            elif self.encoding_strategy.float_precision == "float64":
                if data_array.dtype == np.float32:
                    result[var_name] = data_array.astype(np.float64)
                    logger.debug(f"Converted {var_name} from float32 to float64")

            # Convert large integers to smaller types if possible.
            # Skip the range check for Dask-backed arrays: calling .min().values
            # or .max().values on a lazy array forces a full computation of the
            # entire array into memory, which can cause OOM on large UM datasets.
            # For eager (in-memory) arrays the check is cheap and safe.
            elif data_array.dtype == np.int64:
                if hasattr(data_array.data, "chunks"):
                    logger.debug(
                        f"Skipping int64→int32 range check for lazy array '{var_name}' "
                        f"to avoid eager compute"
                    )
                else:
                    min_val = data_array.min().values
                    max_val = data_array.max().values
                    if (
                        np.iinfo(np.int32).min <= min_val
                        and max_val <= np.iinfo(np.int32).max
                    ):
                        result[var_name] = data_array.astype(np.int32)
                        logger.debug(f"Converted {var_name} from int64 to int32")

        return result

    def _set_fill_values(self, dataset: xr.Dataset) -> xr.Dataset:
        """Set appropriate fill values for variables - only in missing_value attribute."""
        logger.debug("Setting fill values")

        result = dataset.copy()

        for var_name, data_array in result.data_vars.items():
            dtype = data_array.dtype

            # Determine appropriate fill value
            fill_value = self._get_fill_value_for_dtype(dtype)
            if fill_value is not None:
                # Only set missing_value attribute, not _FillValue (that goes in encoding)
                result[var_name].attrs["missing_value"] = fill_value

        return result

    def _get_fill_value_for_dtype(self, dtype) -> float | int | None:
        """Get appropriate fill value for a data type."""
        if dtype.kind == "f":  # float
            return 1e20
        elif dtype.kind == "i":  # integer
            if dtype == np.int32:
                return -2147483647
            elif dtype == np.int16:
                return -32767
            else:
                return -999
        else:
            return None  # Skip other types

    def _ensure_cf_compliance(self, dataset: xr.Dataset) -> xr.Dataset:
        """Ensure basic CF compliance."""
        logger.debug("Ensuring CF compliance")

        result = dataset.copy()

        # Add global attributes
        if "Conventions" not in result.attrs:
            result.attrs["Conventions"] = "CF-1.8"

        if "title" not in result.attrs:
            result.attrs["title"] = "UM data converted to Zarr"

        if "institution" not in result.attrs:
            result.attrs["institution"] = "Converted with um2zarr"

        if "history" not in result.attrs:
            result.attrs["history"] = (
                f"Converted to Zarr on {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        # Ensure coordinates have proper attributes (but avoid conflicts with xarray encoding)
        for coord_name, coord_array in result.coords.items():
            if coord_name.lower() in ["time"]:
                # Don't set calendar - let xarray handle time encoding
                # Only set basic metadata if missing
                if "standard_name" not in coord_array.attrs:
                    coord_array.attrs["standard_name"] = "time"

            elif coord_name.lower() in ["latitude", "lat"]:
                if "units" not in coord_array.attrs:
                    coord_array.attrs["units"] = "degrees_north"
                if "standard_name" not in coord_array.attrs:
                    coord_array.attrs["standard_name"] = "latitude"

            elif coord_name.lower() in ["longitude", "lon"]:
                if "units" not in coord_array.attrs:
                    coord_array.attrs["units"] = "degrees_east"
                if "standard_name" not in coord_array.attrs:
                    coord_array.attrs["standard_name"] = "longitude"

        return result

    def _optimize_dataset_graph(self, dataset: xr.Dataset) -> xr.Dataset:
        """Optimize Dask graph for the dataset before writing."""
        from ..orchestration.dask_integration import (
            check_graph_size_and_warn,
            optimize_dask_graph,
        )

        logger.debug("Checking and optimizing Dask graph for Zarr writing")

        try:
            # Check graph size and warn if large
            check_graph_size_and_warn(
                dataset,
                operation_name="Zarr writing",
                warn_threshold_mb=5.0,
                error_threshold_mb=25.0,  # Lower threshold for writing operations
            )

            # Apply graph optimizations
            optimized_dataset = optimize_dask_graph(
                dataset, optimize_graph=True, fuse_operations=True
            )

            # Additional optimization: rechunk arrays to reduce fragmentation
            optimized_dataset = self._rechunk_for_zarr_writing(optimized_dataset)

            return optimized_dataset

        except Exception as e:
            logger.warning(
                f"Graph optimization failed, proceeding with original dataset: {e}"
            )
            return dataset

    def _rechunk_for_zarr_writing(self, dataset: xr.Dataset) -> xr.Dataset:
        """Rechunk dataset arrays to reduce task graph fragmentation for Zarr writing."""
        logger.debug("Rechunking dataset arrays for Zarr writing")

        result = dataset.copy()

        for var_name, data_array in result.data_vars.items():
            if hasattr(data_array.data, "chunks"):
                # Get current chunks
                current_chunks = data_array.data.chunks

                # Check if chunks are very small (which creates large graphs)
                total_chunks = sum(
                    len(chunks_along_dim) for chunks_along_dim in current_chunks
                )

                if total_chunks > 100:  # Many chunks indicate potential fragmentation
                    logger.debug(
                        f"Rechunking {var_name}: {total_chunks} chunks detected"
                    )

                    # Create more balanced chunks
                    new_chunks = {}
                    for dim_name, chunks_along_dim in zip(
                        data_array.dims, current_chunks
                    ):
                        if (
                            len(chunks_along_dim) > 10
                        ):  # Many chunks along this dimension
                            # Consolidate into fewer, larger chunks
                            total_size = sum(chunks_along_dim)
                            target_chunks = min(
                                8, len(chunks_along_dim)
                            )  # Target max 8 chunks per dim
                            new_chunk_size = max(1, total_size // target_chunks)
                            new_chunks[dim_name] = new_chunk_size

                    if new_chunks:
                        logger.debug(f"Rechunking {var_name} with: {new_chunks}")
                        result[var_name] = data_array.chunk(new_chunks)

        return result

    def _create_encoding(self, dataset: xr.Dataset) -> dict[str, dict[str, Any]]:
        """Create encoding configuration for Zarr writing."""
        logger.debug(
            f"Creating encoding with {self.encoding_strategy.compressor} compression"
        )

        encoding = {}

        # Create compressor using Zarr 3.x codec API
        compressor = self._create_compressor()

        # Configure encoding for each variable
        for var_name, data_array in dataset.data_vars.items():
            var_encoding = {"compressor": compressor, "dtype": data_array.dtype}

            # Add chunking if array is chunked
            if hasattr(data_array.data, "chunks"):
                # Convert Dask chunks to Zarr chunks
                chunks = tuple(
                    chunk[0] if len(chunk) == 1 else chunk[0]
                    for chunk in data_array.data.chunks
                )
                var_encoding["chunks"] = chunks

            # Add fill value from data type (not from attributes to avoid conflict)
            if self.encoding_strategy.auto_fill_values:
                fill_value = self._get_fill_value_for_dtype(data_array.dtype)
                if fill_value is not None:
                    var_encoding["_FillValue"] = fill_value

            encoding[var_name] = var_encoding

        # Configure coordinate encoding
        for coord_name, coord_array in dataset.coords.items():
            if coord_name not in encoding:  # Don't override data variables
                coord_encoding = {"compressor": compressor, "dtype": coord_array.dtype}

                # Coordinates typically don't need chunking unless very large
                if coord_array.size > 10000:  # Chunk large coordinates
                    coord_encoding["chunks"] = (min(1000, coord_array.size),)

                encoding[coord_name] = coord_encoding

        logger.debug(f"Created encoding for {len(encoding)} variables/coordinates")
        return encoding

    def _create_compressor(self):
        """Create compressor codec for Zarr 3.x.

        BZ2 and LZ4 codec class names vary between Zarr releases, so we probe
        several known names and fall back to zstd when none are found.
        """
        compressor_type = self.encoding_strategy.compressor
        level = self.encoding_strategy.compression_level

        if compressor_type == "zstd":
            return zarr.codecs.ZstdCodec(level=level)

        elif compressor_type == "blosc":
            return zarr.codecs.BloscCodec(
                cname="zstd",
                clevel=level,
                shuffle="shuffle" if self.encoding_strategy.shuffle else "noshuffle",
            )

        elif compressor_type == "gzip":
            return zarr.codecs.GzipCodec(level=level)

        elif compressor_type == "bz2":
            # Class name differs across Zarr releases: Bz2Codec, BZ2Codec, BZ2
            for name in ("Bz2Codec", "BZ2Codec", "BZ2"):
                cls = getattr(zarr.codecs, name, None)
                if cls is not None:
                    try:
                        return cls(level=level)
                    except TypeError:
                        return cls()
            logger.warning(
                "BZ2 codec not available in this Zarr version, falling back to zstd"
            )
            return zarr.codecs.ZstdCodec(level=level)

        elif compressor_type == "lz4":
            # Class name differs across Zarr releases: Lz4Codec, LZ4Codec, LZ4
            for name in ("Lz4Codec", "LZ4Codec", "LZ4"):
                cls = getattr(zarr.codecs, name, None)
                if cls is not None:
                    try:
                        return cls()
                    except Exception:
                        pass
            logger.warning(
                "LZ4 codec not available in this Zarr version, falling back to zstd"
            )
            return zarr.codecs.ZstdCodec(level=level)

        else:
            logger.warning(f"Unknown compressor '{compressor_type}', using zstd")
            return zarr.codecs.ZstdCodec(level=level)

    def _get_storage_mapper(self, store_path: str | Path):
        """Get storage mapper for the specified backend."""
        backend = self.encoding_strategy.storage_backend

        if backend not in self._storage_backends:
            raise ValueError(f"Unsupported storage backend: {backend}")

        return self._storage_backends[backend](store_path)

    def _get_local_mapper(self, store_path: str | Path):
        """Get local filesystem mapper."""
        return str(store_path)

    def _get_s3_mapper(self, store_path: str | Path):
        """Get S3 filesystem mapper."""
        if not HAS_S3:
            raise ImportError(
                "s3fs is required for S3 storage. Install with: pip install s3fs"
            )

        # Parse S3 path
        store_str = str(store_path)
        if not store_str.startswith("s3://"):
            raise ValueError(f"S3 paths must start with s3://, got: {store_str}")

        # Create S3 filesystem
        fs = s3fs.S3FileSystem()
        return fs.get_mapper(store_str)

    def _get_gcs_mapper(self, store_path: str | Path):
        """Get Google Cloud Storage mapper."""
        if not HAS_GCS:
            raise ImportError(
                "gcsfs is required for GCS storage. Install with: pip install gcsfs"
            )

        # Parse GCS path
        store_str = str(store_path)
        if not store_str.startswith("gs://"):
            raise ValueError(f"GCS paths must start with gs://, got: {store_str}")

        # Create GCS filesystem
        fs = gcsfs.GCSFileSystem()
        return fs.get_mapper(store_str)

    def _write_to_zarr(
        self,
        dataset: xr.Dataset,
        store_mapper,
        encoding: dict[str, dict[str, Any]],
        mode: str,
        append_dim: str | None,
        consolidated: bool,
    ) -> dict[str, Any]:
        """Write dataset to Zarr with specified configuration."""
        logger.debug(f"Writing to Zarr with mode='{mode}', consolidated={consolidated}")

        write_kwargs = {
            "store": store_mapper,
            "mode": mode,
            "encoding": encoding,
            "consolidated": consolidated,
        }

        # Handle append mode
        if append_dim and mode == "a":
            write_kwargs["append_dim"] = append_dim

        # Perform the write
        write_start = time.time()

        try:
            dataset.to_zarr(**write_kwargs)
            write_duration = time.time() - write_start

            # Calculate statistics — pass store_mapper so we can read back
            # actual compressed sizes from the written Zarr arrays.
            stats = self._calculate_write_stats(
                dataset, encoding, write_duration, store_mapper
            )

            return stats

        except Exception as e:
            logger.error(f"Zarr write failed: {e}")
            raise

    def _calculate_write_stats(
        self,
        dataset: xr.Dataset,
        encoding: dict[str, dict[str, Any]],
        write_duration: float,
        store_mapper=None,
    ) -> dict[str, Any]:
        """
        Calculate write statistics.

        When *store_mapper* is supplied the method re-opens the Zarr store
        immediately after writing and reads actual ``nbytes_stored`` values,
        producing a real compression ratio.  If re-opening fails (e.g. the
        store is remote and the driver does not expose ``nbytes_stored``),
        the method falls back to a lookup-table estimate and logs a warning.
        """
        n_variables = len(dataset.data_vars)
        n_coordinates = len(dataset.coords)

        total_bytes = sum(
            data_array.nbytes for data_array in dataset.data_vars.values()
        )
        total_mb = total_bytes / (1024**2)

        # --- Attempt to read real compressed sizes from the written store ---
        real_compressed_mb: float | None = None
        if store_mapper is not None and HAS_ZARR:
            try:
                zgroup = zarr.open_group(store_mapper, mode="r")
                compressed_bytes = 0
                for var_name in dataset.data_vars:
                    if var_name in zgroup:
                        arr = zgroup[var_name]
                        compressed_bytes += arr.nbytes_stored
                real_compressed_mb = compressed_bytes / (1024**2)
                logger.debug(
                    f"Real compressed size: {real_compressed_mb:.2f} MB "
                    f"(uncompressed: {total_mb:.2f} MB)"
                )
            except Exception as exc:
                logger.warning(
                    f"Could not read actual compressed sizes from store "
                    f"({type(exc).__name__}: {exc}) — falling back to estimate"
                )

        # --- Fall back to lookup-table estimate when real data is unavailable ---
        if real_compressed_mb is None:
            compression_ratios = {
                "zstd": {"float32": 0.4, "float64": 0.5, "int32": 0.6},
                "blosc": {"float32": 0.3, "float64": 0.4, "int32": 0.5},
                "lz4": {"float32": 0.6, "float64": 0.7, "int32": 0.8},
            }
            compressor_ratios = compression_ratios.get(
                self.encoding_strategy.compressor,
                compression_ratios["zstd"],
            )
            estimated = 0.0
            for data_array in dataset.data_vars.values():
                ratio = compressor_ratios.get(str(data_array.dtype), 0.5)
                estimated += data_array.nbytes / (1024**2) * ratio
            real_compressed_mb = estimated

        compression_ratio = real_compressed_mb / total_mb if total_mb > 0 else 1.0
        throughput_mbps = total_mb / write_duration if write_duration > 0 else 0.0

        return {
            "n_variables": n_variables,
            "n_coordinates": n_coordinates,
            "uncompressed_size_mb": total_mb,
            "compressed_size_mb": real_compressed_mb,
            "compression_ratio": compression_ratio,
            "write_duration_seconds": write_duration,
            "throughput_mbps": throughput_mbps,
            "compressor": self.encoding_strategy.compressor,
            "compression_level": self.encoding_strategy.compression_level,
        }

    def _validate_cf_compliance(self, dataset: xr.Dataset) -> list[str]:
        """Validate CF compliance and return warnings."""
        warnings = []

        # Basic CF checks (simplified)

        # Check for Conventions attribute
        if "Conventions" not in dataset.attrs:
            warnings.append("Missing 'Conventions' global attribute")

        # Check coordinate attributes
        for coord_name, coord_array in dataset.coords.items():
            coord_lower = coord_name.lower()

            if coord_lower in ["time"]:
                if "units" not in coord_array.attrs:
                    warnings.append(f"Time coordinate '{coord_name}' missing units")
                if "calendar" not in coord_array.attrs:
                    warnings.append(f"Time coordinate '{coord_name}' missing calendar")

            elif coord_lower in ["latitude", "lat"]:
                if "units" not in coord_array.attrs:
                    warnings.append(f"Latitude coordinate '{coord_name}' missing units")
                if coord_array.attrs.get("units") not in [
                    "degrees_north",
                    "degree_north",
                    "degree_N",
                    "degrees_N",
                ]:
                    warnings.append(
                        f"Latitude coordinate '{coord_name}' has non-standard units"
                    )

            elif coord_lower in ["longitude", "lon"]:
                if "units" not in coord_array.attrs:
                    warnings.append(
                        f"Longitude coordinate '{coord_name}' missing units"
                    )
                if coord_array.attrs.get("units") not in [
                    "degrees_east",
                    "degree_east",
                    "degree_E",
                    "degrees_E",
                ]:
                    warnings.append(
                        f"Longitude coordinate '{coord_name}' has non-standard units"
                    )

        # Check variable attributes
        for var_name, data_array in dataset.data_vars.items():
            if "units" not in data_array.attrs:
                warnings.append(f"Variable '{var_name}' missing units")
            if (
                "long_name" not in data_array.attrs
                and "standard_name" not in data_array.attrs
            ):
                warnings.append(
                    f"Variable '{var_name}' missing both long_name and standard_name"
                )

        return warnings

    def validate_store(self, store_path: str | Path) -> dict[str, Any]:
        """
        Validate an existing Zarr store.

        Args:
            store_path: Path to Zarr store to validate

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating Zarr store: {store_path}")

        try:
            # Open the store
            store_mapper = self._get_storage_mapper(store_path)
            ds = xr.open_zarr(store_mapper)

            # Basic validation
            validation_results = {
                "store_path": str(store_path),
                "valid": True,
                "n_variables": len(ds.data_vars),
                "n_coordinates": len(ds.coords),
                "dimensions": dict(ds.sizes),
                "global_attributes": dict(ds.attrs),
                "warnings": [],
                "errors": [],
            }

            # CF compliance check
            if self.validate_cf:
                cf_warnings = self._validate_cf_compliance(ds)
                validation_results["cf_warnings"] = cf_warnings
                validation_results["cf_compliant"] = len(cf_warnings) == 0

            # Check compression and encoding
            encoding_info = {}
            for var_name in ds.data_vars:
                var_encoding = ds[var_name].encoding
                encoding_info[var_name] = {
                    "dtype": str(ds[var_name].dtype),
                    "chunks": var_encoding.get("chunks"),
                    "compressor": str(var_encoding.get("compressor", "none")),
                    "fill_value": var_encoding.get("_FillValue"),
                }

            validation_results["encoding_info"] = encoding_info

            logger.info(
                f"Store validation completed: {validation_results['n_variables']} variables"
            )
            return validation_results

        except Exception as e:
            logger.error(f"Store validation failed: {e}")
            return {"store_path": str(store_path), "valid": False, "error": str(e)}

    def validate_append_schema(
        self,
        store_path: str | Path,
        new_dataset: xr.Dataset,
        append_dim: str = "time",
    ) -> dict[str, Any]:
        """
        Check whether *new_dataset* is compatible with an existing Zarr store for
        appending along *append_dim*.

        Validates:
        - Variable list matches (same names in both datasets)
        - Data types match per variable
        - Chunk shapes along non-append dimensions match

        Returns a dict with keys ``compatible`` (bool), ``errors`` (list of str),
        and ``warnings`` (list of str).  Callers should abort the append if
        ``compatible`` is False.
        """
        result: dict[str, Any] = {"compatible": True, "errors": [], "warnings": []}

        if not Path(store_path).exists():
            # Store does not yet exist — any schema is valid (first write)
            result["warnings"].append(
                f"Store {store_path} does not exist; schema check skipped (first write)."
            )
            return result

        try:
            store_mapper = self._get_storage_mapper(store_path)
            existing = xr.open_zarr(store_mapper, consolidated=None)
        except Exception as exc:
            result["compatible"] = False
            result["errors"].append(
                f"Cannot open existing store for schema check: {exc}"
            )
            return result

        # ---- variable presence ----
        existing_vars = set(existing.data_vars)
        new_vars = set(new_dataset.data_vars)
        missing_in_new = existing_vars - new_vars
        extra_in_new = new_vars - existing_vars
        if missing_in_new:
            result["errors"].append(
                f"Variables present in store but missing from new data: {sorted(missing_in_new)}"
            )
            result["compatible"] = False
        if extra_in_new:
            result["warnings"].append(
                f"New variables not in existing store (will be ignored or cause error): "
                f"{sorted(extra_in_new)}"
            )

        # ---- dtype and chunk shape per variable ----
        for var in existing_vars & new_vars:
            existing_da = existing[var]
            new_da = new_dataset[var]

            # dtype check
            if existing_da.dtype != new_da.dtype:
                result["errors"].append(
                    f"Variable '{var}': dtype mismatch — store has {existing_da.dtype}, "
                    f"new data has {new_da.dtype}"
                )
                result["compatible"] = False

            # chunk shape on non-append dims
            existing_chunks = existing_da.encoding.get("chunks") or {}
            new_chunks = getattr(new_da, "chunks", {})
            for dim in existing_da.dims:
                if dim == append_dim:
                    continue
                e_chunk = (
                    existing_chunks.get(dim)
                    if isinstance(existing_chunks, dict)
                    else None
                )
                n_chunk = (
                    new_chunks.get(dim, (None,))[0]
                    if isinstance(new_chunks, dict)
                    else None
                )
                if e_chunk and n_chunk and e_chunk != n_chunk:
                    result["warnings"].append(
                        f"Variable '{var}': chunk size on dim '{dim}' differs "
                        f"(store={e_chunk}, new={n_chunk}).  This may cause performance issues."
                    )

        if result["errors"]:
            logger.error(
                f"Append schema validation FAILED with {len(result['errors'])} error(s): "
                + "; ".join(result["errors"])
            )
        elif result["warnings"]:
            logger.warning(
                f"Append schema validation passed with {len(result['warnings'])} warning(s)"
            )
        else:
            logger.info("Append schema validation passed.")

        return result

    def get_store_info(self, store_path: str | Path) -> dict[str, Any]:
        """
        Get information about a Zarr store.

        Args:
            store_path: Path to Zarr store

        Returns:
            Dictionary with store information
        """
        try:
            store_mapper = self._get_storage_mapper(store_path)

            # Open as Zarr group to get low-level info
            zarr_group = zarr.open_group(store_mapper, mode="r")

            # Get array information
            arrays_info = {}
            for name, array in zarr_group.arrays():
                # Handle Zarr 3.x compressor deprecation
                try:
                    # Try new Zarr 3.x API first
                    if hasattr(array, "compressors") and array.compressors:
                        compressor_info = (
                            str(array.compressors[0]) if array.compressors else None
                        )
                    else:
                        compressor_info = None
                except AttributeError:
                    # Fall back to deprecated API for older versions
                    try:
                        compressor_info = (
                            str(array.compressor)
                            if hasattr(array, "compressor") and array.compressor
                            else None
                        )
                    except Exception:
                        compressor_info = None

                arrays_info[name] = {
                    "shape": array.shape,
                    "dtype": str(array.dtype),
                    "chunks": array.chunks,
                    "compressor": compressor_info,
                    "size_bytes": array.nbytes,
                    "size_mb": array.nbytes / (1024**2),
                }

            total_size = sum(info["size_bytes"] for info in arrays_info.values())

            return {
                "store_path": str(store_path),
                "arrays": arrays_info,
                "total_arrays": len(arrays_info),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024**2),
                "zarr_version": zarr.__version__,
            }

        except Exception as e:
            logger.error(f"Failed to get store info: {e}")
            return {"store_path": str(store_path), "error": str(e)}


# Convenience functions


def write_um_dataset_to_zarr(
    dataset: xr.Dataset,
    output_path: str | Path,
    compression: str = "zstd",
    compression_level: int = 3,
    storage_backend: str = "local",
    validate_cf: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to write UM dataset to Zarr with optimal settings.

    Args:
        dataset: UM dataset to write
        output_path: Output Zarr store path
        compression: Compression algorithm ('zstd', 'blosc', 'lz4')
        compression_level: Compression level (1-9)
        storage_backend: Storage backend ('local', 's3', 'gcs')
        validate_cf: Whether to validate CF compliance

    Returns:
        Write statistics
    """
    # Configure encoding strategy for UM data
    encoding_strategy = EncodingStrategy(
        compressor=compression,
        compression_level=compression_level,
        storage_backend=storage_backend,
        auto_chunk=True,
        auto_dtype=True,
        auto_fill_values=True,
        chunk_strategy="balanced",
    )

    # Create writer and write dataset
    writer = ZarrWriter(encoding_strategy, validate_cf=validate_cf)
    return writer.write_dataset(dataset, output_path)


def append_to_zarr_time_series(
    dataset: xr.Dataset,
    store_path: str | Path,
    time_dim: str = "time",
    validate_cf: bool = False,
) -> dict[str, Any]:
    """
    Append dataset to existing Zarr time series.

    Args:
        dataset: Dataset to append
        store_path: Existing Zarr store path
        time_dim: Time dimension name
        validate_cf: Whether to validate CF compliance

    Returns:
        Append statistics
    """
    encoding_strategy = EncodingStrategy(
        auto_chunk=True,
        auto_dtype=False,  # Don't modify existing dtypes
        auto_fill_values=False,  # Don't modify existing fill values
    )

    writer = ZarrWriter(encoding_strategy, validate_cf=validate_cf)
    return writer.write_dataset(dataset, store_path, mode="a", append_dim=time_dim)

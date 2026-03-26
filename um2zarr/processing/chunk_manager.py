"""
Chunk management for optimal Dask and Zarr performance.

Provides intelligent chunking strategies based on data dimensions,
target storage backends, and memory constraints.
"""

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

# Try to import dask_setup for intelligent chunking recommendations
try:
    import dask_setup
    from dask.distributed import get_client

    HAS_DASK_SETUP = True
except ImportError:
    HAS_DASK_SETUP = False
    dask_setup = None
    get_client = None

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStrategy:
    """Configuration for chunking strategy."""

    # Target chunk size constraints
    target_chunk_size_mb: float = 100.0  # Target chunk size in MB
    max_chunk_size_mb: float = 200.0  # Maximum chunk size in MB
    min_chunk_size_mb: float = 10.0  # Minimum chunk size in MB

    # Dimension preferences
    time_chunk_size: int | None = 1  # Prefer small time chunks for time series
    preserve_spatial_coherence: bool = (
        True  # Keep spatial dimensions together when possible
    )

    # Storage backend optimization
    storage_backend: str = "local"  # 'local', 's3', 'gcs'

    # Memory constraints
    available_memory_gb: float | None = None  # Available memory for processing


class ChunkManager:
    """
    Manage chunking strategies for optimal Zarr and Dask performance.

    Provides intelligent chunking based on:
    - Data dimensions and patterns
    - Target storage backend characteristics
    - Available memory constraints
    - Zarr performance best practices
    """

    def __init__(self, strategy: ChunkingStrategy | None = None):
        """
        Initialize chunk manager.

        Args:
            strategy: Chunking strategy configuration
        """
        self.strategy = strategy or ChunkingStrategy()
        self.logger = logging.getLogger(__name__)

        # Storage backend characteristics
        self._storage_characteristics = {
            "local": {
                "optimal_chunk_mb": (50, 150),  # (min, max) optimal range
                "prefers_fewer_files": False,
                "latency_sensitive": False,
            },
            "s3": {
                "optimal_chunk_mb": (100, 300),  # Larger chunks for S3
                "prefers_fewer_files": True,  # S3 prefers fewer larger files
                "latency_sensitive": True,  # Network latency considerations
            },
            "gcs": {
                "optimal_chunk_mb": (100, 250),
                "prefers_fewer_files": True,
                "latency_sensitive": True,
            },
        }

    def calculate_optimal_chunks(
        self,
        dataset: xr.Dataset,
        preserve_dimensions: list[str] | None = None,
        use_dask_setup: bool = True,
    ) -> dict[str, int]:
        """
        Calculate optimal chunk sizes for dataset dimensions.

        Uses dask_setup for intelligent recommendations when available,
        falls back to manual calculation otherwise.

        Args:
            dataset: Input dataset to analyze
            preserve_dimensions: Dimensions that should not be chunked
            use_dask_setup: Whether to try using dask_setup for recommendations

        Returns:
            Dictionary mapping dimension names to chunk sizes

        Examples:
            >>> import xarray as xr
            >>> import numpy as np
            >>>
            >>> # Create test dataset
            >>> ds = xr.Dataset({
            ...     'temperature': (['time', 'lev', 'lat', 'lon'],
            ...                    np.random.random((100, 50, 200, 300)))
            ... })
            >>>
            >>> manager = ChunkManager()
            >>> chunks = manager.calculate_optimal_chunks(ds)
            >>> print(chunks)
            {'time': 1, 'lev': 50, 'lat': 200, 'lon': 300}
        """
        logger.info("Calculating optimal chunk sizes")
        logger.debug(f"Dataset dimensions: {dict(dataset.sizes)}")

        # Try dask_setup first if available and requested
        if use_dask_setup and HAS_DASK_SETUP:
            try:
                chunks = self._calculate_chunks_with_dask_setup(
                    dataset, preserve_dimensions
                )
                if chunks:
                    logger.info(f"Using dask_setup recommendations: {chunks}")
                    return chunks
            except Exception as e:
                logger.warning(
                    f"dask_setup chunking failed, falling back to manual calculation: {e}"
                )

        # Fall back to manual calculation
        preserve_dimensions = preserve_dimensions or []
        chunks = self._calculate_chunks_manually(dataset, preserve_dimensions)
        logger.info(f"Calculated chunks (manual): {chunks}")
        return chunks

    def _calculate_chunks_with_dask_setup(
        self, dataset: xr.Dataset, preserve_dimensions: list[str] | None = None
    ) -> dict[str, int] | None:
        """Calculate chunks using dask_setup recommendations."""

        try:
            # Get active Dask client if available
            client = None
            try:
                client = get_client()
                logger.debug("Using active Dask client for dask_setup recommendations")
            except (RuntimeError, ValueError):
                logger.debug(
                    "No active Dask client, dask_setup will use system defaults"
                )

            # Determine workload type based on storage backend
            workload_type_map = {
                "local": "mixed",
                "s3": "io",
                "gcs": "io",
            }
            workload_type = workload_type_map.get(
                self.strategy.storage_backend, "mixed"
            )

            # Get target chunk size from our strategy
            target_chunk_mb = (
                self.strategy.min_chunk_size_mb,
                self.strategy.max_chunk_size_mb,
            )

            logger.debug(
                f"dask_setup parameters: workload_type={workload_type}, "
                f"target_chunk_mb={target_chunk_mb}"
            )

            # Get recommendations from dask_setup
            chunks = dask_setup.recommend_chunks(
                dataset,
                client=client,
                workload_type=workload_type,
                target_chunk_mb=target_chunk_mb,
                verbose=False,
            )

            if not chunks:
                logger.warning("dask_setup returned empty chunk recommendations")
                return None

            # Apply preserve_dimensions constraint
            if preserve_dimensions:
                for dim in preserve_dimensions:
                    if dim in chunks and dim in dataset.sizes:
                        chunks[dim] = dataset.sizes[dim]
                        logger.debug(f"Preserved dimension {dim}: {chunks[dim]}")

            return chunks

        except Exception as e:
            logger.error(f"Error in dask_setup chunking: {e}")
            return None

    def _calculate_chunks_manually(
        self, dataset: xr.Dataset, preserve_dimensions: list[str] | None = None
    ) -> dict[str, int]:
        """Calculate chunks using manual algorithm (original implementation)."""

        chunks = {}

        # Get storage backend characteristics
        backend_config = self._storage_characteristics.get(
            self.strategy.storage_backend,
            self._storage_characteristics["local"],  # Fallback to local settings
        )
        if self.strategy.storage_backend not in self._storage_characteristics:
            logger.warning(
                f"Unknown storage backend '{self.strategy.storage_backend}', using 'local' settings"
            )
        target_mb = np.mean(backend_config["optimal_chunk_mb"])

        # Analyze dataset characteristics
        total_variables = len(dataset.data_vars)
        largest_var = max(dataset.data_vars.values(), key=lambda x: x.nbytes)

        # Calculate bytes per data point for the largest variable
        bytes_per_point = largest_var.dtype.itemsize
        target_points = int((target_mb * 1024**2) / bytes_per_point)

        logger.debug(f"Target chunk size: {target_mb:.1f} MB")
        logger.debug(f"Bytes per point: {bytes_per_point}")
        logger.debug(f"Target points per chunk: {target_points:,}")

        # Dimension-specific chunking strategies
        for dim_name, dim_size in dataset.sizes.items():
            if preserve_dimensions and dim_name in preserve_dimensions:
                # Don't chunk preserved dimensions
                chunks[dim_name] = dim_size
                logger.debug(f"Preserving dimension {dim_name}: {dim_size}")
                continue

            chunk_size = self._calculate_dimension_chunk_size(
                dim_name, dim_size, target_points, dataset
            )
            chunks[dim_name] = chunk_size
            logger.debug(f"Dimension {dim_name}: {chunk_size} (of {dim_size})")

        # Validate and adjust chunks
        chunks = self._validate_and_adjust_chunks(chunks, dataset)

        return chunks

    def _calculate_dimension_chunk_size(
        self, dim_name: str, dim_size: int, target_points: int, dataset: xr.Dataset
    ) -> int:
        """Calculate chunk size for a specific dimension."""

        # Time dimension strategy
        if "time" in dim_name.lower():
            if self.strategy.time_chunk_size is not None:
                return min(self.strategy.time_chunk_size, dim_size)
            else:
                # Small time chunks for time series analysis
                return min(10, dim_size)

        # Vertical level dimension strategy
        elif any(
            lev_name in dim_name.lower() for lev_name in ["lev", "level", "model_level"]
        ):
            # Keep all levels together if reasonable
            if dim_size <= 100:  # Typical model level count
                return dim_size
            else:
                # For very deep models, chunk by blocks
                return min(50, dim_size)

        # Spatial dimensions (lat, lon, y, x)
        elif any(
            spatial_name in dim_name.lower()
            for spatial_name in ["lat", "lon", "x", "y"]
        ):
            if self.strategy.preserve_spatial_coherence:
                # Try to keep spatial dimensions together
                spatial_dims = [
                    d
                    for d in dataset.sizes
                    if any(s in d.lower() for s in ["lat", "lon", "x", "y"])
                ]

                if len(spatial_dims) == 2:  # lat/lon or x/y pair
                    # Calculate chunk size to fit target while keeping both dimensions
                    other_spatial = [d for d in spatial_dims if d != dim_name][0]
                    other_size = dataset.sizes[other_spatial]

                    # Target points for this spatial grid
                    spatial_target = int(np.sqrt(target_points))
                    chunk_size = min(spatial_target, dim_size)

                    # Ensure it's reasonable
                    chunk_size = max(
                        chunk_size, min(64, dim_size)
                    )  # At least 64 points
                    return chunk_size

            # Default spatial chunking
            return min(max(int(np.sqrt(target_points)), 64), dim_size)

        # Default strategy for other dimensions
        else:
            # Aim for chunks that fit target size
            return min(max(target_points // 100, 1), dim_size)

    def _validate_and_adjust_chunks(
        self, chunks: dict[str, int], dataset: xr.Dataset
    ) -> dict[str, int]:
        """Validate chunk configuration and make adjustments."""

        # Calculate actual chunk size in MB
        largest_var = max(dataset.data_vars.values(), key=lambda x: x.nbytes)
        bytes_per_point = largest_var.dtype.itemsize

        chunk_points = np.prod(list(chunks.values()))
        chunk_size_mb = (chunk_points * bytes_per_point) / (1024**2)

        logger.debug(f"Calculated chunk size: {chunk_size_mb:.2f} MB")

        # Adjust if chunk size is outside acceptable range
        if chunk_size_mb > self.strategy.max_chunk_size_mb:
            logger.warning(
                f"Chunk size {chunk_size_mb:.2f} MB exceeds maximum, adjusting"
            )
            chunks = self._reduce_chunk_sizes(chunks, dataset, chunk_size_mb)

        elif chunk_size_mb < self.strategy.min_chunk_size_mb:
            logger.warning(
                f"Chunk size {chunk_size_mb:.2f} MB below minimum, adjusting"
            )
            chunks = self._increase_chunk_sizes(chunks, dataset, chunk_size_mb)

        return chunks

    def _reduce_chunk_sizes(
        self, chunks: dict[str, int], dataset: xr.Dataset, current_mb: float
    ) -> dict[str, int]:
        """Reduce chunk sizes when they're too large."""

        reduction_factor = np.sqrt(self.strategy.target_chunk_size_mb / current_mb)
        adjusted_chunks = chunks.copy()

        # Reduce largest chunkable dimensions first
        chunkable_dims = [
            (dim, size)
            for dim, size in chunks.items()
            if size > 1 and "time" not in dim.lower()
        ]
        chunkable_dims.sort(key=lambda x: x[1], reverse=True)

        for dim, size in chunkable_dims:
            new_size = max(1, int(size * reduction_factor))
            if new_size < size:
                adjusted_chunks[dim] = new_size
                logger.debug(f"Reduced {dim} chunk from {size} to {new_size}")

                # Check if we're now in acceptable range
                chunk_points = np.prod(list(adjusted_chunks.values()))
                bytes_per_point = max(
                    dataset.data_vars.values(), key=lambda x: x.nbytes
                ).dtype.itemsize
                new_mb = (chunk_points * bytes_per_point) / (1024**2)

                if new_mb <= self.strategy.max_chunk_size_mb:
                    break

        return adjusted_chunks

    def _increase_chunk_sizes(
        self, chunks: dict[str, int], dataset: xr.Dataset, current_mb: float
    ) -> dict[str, int]:
        """Increase chunk sizes when they're too small."""

        increase_factor = np.sqrt(self.strategy.target_chunk_size_mb / current_mb)
        adjusted_chunks = chunks.copy()

        # Increase dimensions that have room to grow
        for dim, chunk_size in chunks.items():
            dim_size = dataset.sizes[dim]
            if chunk_size < dim_size:
                new_size = min(dim_size, int(chunk_size * increase_factor))
                adjusted_chunks[dim] = new_size
                logger.debug(f"Increased {dim} chunk from {chunk_size} to {new_size}")

        return adjusted_chunks

    def apply_chunking(
        self, dataset: xr.Dataset, chunk_dict: dict[str, int] | None = None
    ) -> xr.Dataset:
        """
        Apply chunking to dataset using Dask arrays.

        Args:
            dataset: Dataset to chunk
            chunk_dict: Explicit chunk configuration, or None to calculate optimal

        Returns:
            Chunked dataset with Dask arrays
        """
        logger.info("Applying chunking to dataset")

        if chunk_dict is None:
            chunk_dict = self.calculate_optimal_chunks(dataset)

        # Filter chunk_dict to only include dimensions present in dataset
        valid_chunks = {
            dim: size for dim, size in chunk_dict.items() if dim in dataset.sizes
        }

        logger.debug(f"Applying chunks: {valid_chunks}")

        # Chunk the dataset
        chunked_dataset = dataset.chunk(valid_chunks)

        # Log chunking results
        self._log_chunking_results(chunked_dataset)

        return chunked_dataset

    def _log_chunking_results(self, dataset: xr.Dataset) -> None:
        """Log information about the chunked dataset."""

        for var_name, data_array in dataset.data_vars.items():
            if hasattr(data_array.data, "chunks"):
                chunks = data_array.data.chunks
                chunk_shape = tuple(c[0] if c else 0 for c in chunks)
                num_chunks = np.prod([len(c) for c in chunks])

                # Calculate chunk size in MB
                chunk_points = np.prod(chunk_shape)
                chunk_mb = (chunk_points * data_array.dtype.itemsize) / (1024**2)

                logger.info(
                    f"Variable '{var_name}': {num_chunks} chunks of shape {chunk_shape} "
                    f"({chunk_mb:.2f} MB each)"
                )

    def optimize_for_zarr(
        self, dataset: xr.Dataset, target_store_type: str = "local"
    ) -> xr.Dataset:
        """
        Optimize dataset chunking specifically for Zarr storage.

        Args:
            dataset: Dataset to optimize
            target_store_type: Target storage type ('local', 's3', 'gcs')

        Returns:
            Optimally chunked dataset for Zarr
        """
        logger.info(f"Optimizing chunking for Zarr storage on {target_store_type}")

        # Adjust strategy for storage type
        original_backend = self.strategy.storage_backend
        self.strategy.storage_backend = target_store_type

        try:
            # Try dask_setup I/O recommendations first for Zarr
            chunk_dict = None
            if HAS_DASK_SETUP:
                try:
                    chunk_dict = self._get_zarr_io_chunks(dataset, target_store_type)
                    if chunk_dict:
                        logger.info(
                            f"Using dask_setup I/O recommendations for Zarr: {chunk_dict}"
                        )
                except Exception as e:
                    logger.warning(f"dask_setup I/O recommendations failed: {e}")

            # Fall back to normal calculation if dask_setup failed or unavailable
            if not chunk_dict:
                chunk_dict = self.calculate_optimal_chunks(dataset)

            # Apply additional Zarr-specific optimizations
            chunk_dict = self._apply_zarr_optimizations(chunk_dict, dataset)

            # Apply chunking
            chunked_dataset = self.apply_chunking(dataset, chunk_dict)

            return chunked_dataset

        finally:
            # Restore original strategy
            self.strategy.storage_backend = original_backend

    def _apply_zarr_optimizations(
        self, chunks: dict[str, int], dataset: xr.Dataset
    ) -> dict[str, int]:
        """Apply Zarr-specific chunk optimizations."""

        optimized = chunks.copy()

        # Zarr works best with regularly sized chunks
        # Round chunk sizes to nice values
        for dim, size in chunks.items():
            if size > 1:
                # Round to nearest "nice" value
                nice_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                nice_size = min(nice_values, key=lambda x: abs(x - size))
                nice_size = min(nice_size, dataset.sizes[dim])  # Don't exceed dimension
                optimized[dim] = nice_size

                if nice_size != size:
                    logger.debug(f"Rounded {dim} chunk from {size} to {nice_size}")

        return optimized

    def _get_zarr_io_chunks(
        self, dataset: xr.Dataset, target_store_type: str = "local"
    ) -> dict[str, int] | None:
        """Get I/O-optimized chunks for Zarr using dask_setup."""

        try:
            # Get active Dask client if available
            client = None
            try:
                client = get_client()
            except (RuntimeError, ValueError):
                pass

            # Map our storage backend to dask_setup storage location
            storage_location_map = {
                "local": "local",
                "s3": "cloud",
                "gcs": "cloud",
            }
            storage_location = storage_location_map.get(target_store_type, "local")

            # Get target chunk size from our strategy
            target_chunk_mb = (
                self.strategy.min_chunk_size_mb,
                self.strategy.max_chunk_size_mb,
            )

            logger.debug(
                f"dask_setup I/O parameters: format_hint=zarr, "
                f"storage_location={storage_location}, target_chunk_mb={target_chunk_mb}"
            )

            # Get I/O recommendations from dask_setup
            chunks = dask_setup.recommend_io_chunks(
                dataset,
                client=client,
                format_hint="zarr",
                access_pattern="auto",
                target_chunk_mb=target_chunk_mb,
                storage_location=storage_location,
                verbose=False,
            )

            if not chunks:
                logger.warning("dask_setup returned empty I/O chunk recommendations")
                return None

            return chunks

        except Exception as e:
            logger.error(f"Error in dask_setup I/O chunking: {e}")
            return None

    def get_memory_usage_estimate(
        self, dataset: xr.Dataset, chunk_dict: dict[str, int] | None = None
    ) -> dict[str, float]:
        """
        Estimate memory usage for chunked dataset processing.

        Args:
            dataset: Dataset to analyze
            chunk_dict: Chunk configuration to analyze

        Returns:
            Dictionary with memory usage estimates in MB
        """
        if chunk_dict is None:
            chunk_dict = self.calculate_optimal_chunks(dataset)

        estimates = {}

        # Calculate per-chunk memory usage
        chunk_points = np.prod(list(chunk_dict.values()))

        for var_name, data_array in dataset.data_vars.items():
            bytes_per_point = data_array.dtype.itemsize
            chunk_mb = (chunk_points * bytes_per_point) / (1024**2)

            # Estimate total memory for processing (assuming some buffer)
            num_chunks = np.prod(
                [
                    dataset.sizes[dim] // chunk_dict[dim] + 1
                    for dim in chunk_dict.keys()
                    if dim in dataset.sizes
                ]
            )

            estimates[var_name] = {
                "chunk_size_mb": chunk_mb,
                "num_chunks": num_chunks,
                "max_memory_mb": chunk_mb
                * min(num_chunks, 10),  # Assume max 10 chunks in memory
            }

        # Overall estimates
        total_chunk_mb = sum(est["chunk_size_mb"] for est in estimates.values())
        max_memory_mb = sum(est["max_memory_mb"] for est in estimates.values())

        estimates["_summary"] = {
            "total_chunk_size_mb": total_chunk_mb,
            "estimated_max_memory_mb": max_memory_mb,
        }

        return estimates


def create_time_series_chunks(
    datasets: list[xr.Dataset], time_dim: str = "time"
) -> dict[str, int]:
    """
    Create optimal chunks for time series data spanning multiple files.

    Args:
        datasets: List of datasets that will be concatenated
        time_dim: Name of time dimension

    Returns:
        Optimal chunk configuration for time series
    """
    logger.info(f"Creating time series chunks for {len(datasets)} datasets")

    if not datasets:
        return {}

    # Analyze the first dataset for spatial dimensions
    sample_ds = datasets[0]
    chunk_manager = ChunkManager(ChunkingStrategy(time_chunk_size=1))

    # Calculate spatial chunks (excluding time)
    spatial_chunks = chunk_manager.calculate_optimal_chunks(
        sample_ds, preserve_dimensions=[time_dim]
    )

    # Set time chunk to 1 for time series processing
    spatial_chunks[time_dim] = 1

    logger.info(f"Time series chunks: {spatial_chunks}")
    return spatial_chunks


def estimate_zarr_compression_ratio(
    dataset: xr.Dataset, compression: str = "zstd"
) -> dict[str, float]:
    """
    Estimate compression ratios for Zarr storage.

    Args:
        dataset: Dataset to analyze
        compression: Compression algorithm

    Returns:
        Estimated compression ratios per variable
    """
    # Typical compression ratios for scientific data
    compression_ratios = {
        "zstd": {"float32": 0.4, "float64": 0.5, "int32": 0.6, "int16": 0.7},
        "lz4": {"float32": 0.6, "float64": 0.7, "int32": 0.8, "int16": 0.9},
        "blosc": {"float32": 0.3, "float64": 0.4, "int32": 0.5, "int16": 0.6},
    }

    ratios = compression_ratios.get(compression, compression_ratios["zstd"])

    estimates = {}
    for var_name, data_array in dataset.data_vars.items():
        dtype_str = str(data_array.dtype)
        ratio = ratios.get(dtype_str, 0.5)  # Default to 50% compression
        estimates[var_name] = ratio

    return estimates

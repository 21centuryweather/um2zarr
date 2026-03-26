"""
Dask cluster integration and scheduler management.

Provides utilities for setting up Dask clusters, managing workers,
and orchestrating parallel UM processing workflows.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Dask imports
import dask
import psutil
import xarray as xr
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster, as_completed

# Try to import dask_setup for HPC optimization
try:
    import dask_setup

    HAS_DASK_SETUP = True
except ImportError:
    HAS_DASK_SETUP = False
    dask_setup = None

logger = logging.getLogger(__name__)


@dataclass
class DaskConfig:
    """Configuration for Dask cluster setup with dask_setup integration."""

    # Scheduler configuration (for external clusters)
    scheduler_file: str | None = None
    scheduler_address: str | None = None

    # dask_setup parameters (when available)
    workload_type: str = "mixed"  # "cpu", "io", "mixed"
    max_workers: int | None = None
    reserve_mem_gb: float = 2.0
    max_mem_gb: float | None = None
    dashboard: bool = True
    adaptive: bool = True
    min_workers: int | None = None
    profile: str | None = None
    suggest_chunks: bool = False

    # Legacy parameters (for backward compatibility when dask_setup not available)
    n_workers: int = 4
    threads_per_worker: int = 2
    memory_limit_gb: float | None = None

    # Processing configuration
    max_concurrent_tasks: int = 10
    timeout_seconds: int = 3600  # 1 hour timeout for tasks

    # Resource management (legacy)
    adaptive_scaling: bool = True
    max_workers_legacy: int = 8


class DaskClusterManager:
    """
    Manage Dask clusters for UM processing.

    Supports both local clusters and external cluster connections.
    Provides intelligent resource management and monitoring.
    """

    def __init__(self, config: DaskConfig | None = None):
        """
        Initialize Dask cluster manager.

        Args:
            config: Dask configuration options
        """
        self.config = config or DaskConfig()
        self.client: Client | None = None
        self.cluster: LocalCluster | None = None
        self.dask_local_dir: str | None = None  # Set by dask_setup
        self.logger = logging.getLogger(__name__)

    def setup_cluster(self) -> Client:
        """
        Set up Dask cluster and return client.

        Uses dask_setup for HPC optimization when available,
        falls back to manual LocalCluster setup otherwise.

        Returns:
            Configured Dask client
        """
        logger.info("Setting up Dask cluster")

        # Try to connect to external cluster first
        if self.config.scheduler_file or self.config.scheduler_address:
            self.client = self._connect_to_external_cluster()
        else:
            # Create local cluster - prefer dask_setup if available
            if HAS_DASK_SETUP:
                self.client = self._create_cluster_with_dask_setup()
            else:
                logger.info(
                    "dask_setup not available, falling back to manual LocalCluster setup"
                )
                self.client = self._create_local_cluster()

        # Configure Dask settings (if using legacy setup)
        if (
            not HAS_DASK_SETUP
            or self.config.scheduler_file
            or self.config.scheduler_address
        ):
            self._configure_dask_settings()

        # Log cluster information
        self._log_cluster_info()

        return self.client

    def _connect_to_external_cluster(self) -> Client:
        """Connect to external Dask cluster."""

        if self.config.scheduler_file:
            logger.info(
                f"Connecting to Dask cluster via scheduler file: {self.config.scheduler_file}"
            )

            # Verify scheduler file exists
            scheduler_path = Path(self.config.scheduler_file)
            if not scheduler_path.exists():
                raise FileNotFoundError(f"Scheduler file not found: {scheduler_path}")

            client = Client(
                scheduler_file=self.config.scheduler_file,
                timeout=self.config.timeout_seconds,
            )

        elif self.config.scheduler_address:
            logger.info(
                f"Connecting to Dask cluster at: {self.config.scheduler_address}"
            )
            client = Client(
                self.config.scheduler_address, timeout=self.config.timeout_seconds
            )
        else:
            raise ValueError(
                "External cluster specified but no scheduler file or address provided"
            )

        logger.info("Successfully connected to external Dask cluster")
        return client

    def _create_cluster_with_dask_setup(self) -> Client:
        """Create local Dask cluster using dask_setup for HPC optimization."""
        logger.info("Creating Dask cluster using dask_setup")

        try:
            # Prepare dask_setup parameters
            dask_setup_kwargs = {
                "workload_type": self.config.workload_type,
                "max_workers": self.config.max_workers,
                "reserve_mem_gb": self.config.reserve_mem_gb,
                "max_mem_gb": self.config.max_mem_gb,
                "dashboard": self.config.dashboard,
                "adaptive": self.config.adaptive,
                "min_workers": self.config.min_workers,
                "profile": self.config.profile,
                "suggest_chunks": self.config.suggest_chunks,
            }

            # Remove None values to use dask_setup defaults
            dask_setup_kwargs = {
                k: v for k, v in dask_setup_kwargs.items() if v is not None
            }

            logger.info(f"dask_setup parameters: {dask_setup_kwargs}")

            # Create cluster using dask_setup
            client, cluster, dask_local_dir = dask_setup.setup_dask_client(
                **dask_setup_kwargs
            )

            # Store cluster and local dir info
            self.cluster = cluster
            self.dask_local_dir = dask_local_dir

            logger.info("dask_setup cluster created successfully")
            logger.info(f"Dask local directory: {dask_local_dir}")
            logger.info(f"Dashboard: {client.dashboard_link}")

            return client

        except Exception as e:
            logger.error(f"Failed to create cluster with dask_setup: {e}")
            logger.info("Falling back to manual LocalCluster setup")
            return self._create_local_cluster()

    def _create_local_cluster(self) -> Client:
        """Create local Dask cluster."""
        logger.info("Creating local Dask cluster")

        # Determine resource allocation
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Set defaults based on system resources
        n_workers = min(self.config.n_workers, cpu_count)
        threads_per_worker = min(
            self.config.threads_per_worker, max(1, cpu_count // n_workers)
        )

        # Memory limit per worker
        if self.config.memory_limit_gb:
            memory_limit = f"{self.config.memory_limit_gb / n_workers:.1f}GB"
        else:
            # Use 80% of available memory
            memory_limit = f"{0.8 * memory_gb / n_workers:.1f}GB"

        logger.info(
            f"Local cluster: {n_workers} workers, {threads_per_worker} threads/worker, "
            f"{memory_limit} memory/worker"
        )

        # Create local cluster
        cluster_kwargs = {
            "n_workers": n_workers,
            "threads_per_worker": threads_per_worker,
            "memory_limit": memory_limit,
            "silence_logs": False,  # Keep logs visible
            "dashboard_address": ":8787",
        }

        # Add adaptive scaling if requested
        if self.config.adaptive_scaling:
            cluster_kwargs.update(
                {
                    "n_workers": 0,  # Start with 0 workers for adaptive
                }
            )

        self.cluster = LocalCluster(**cluster_kwargs)

        # Set up adaptive scaling
        if self.config.adaptive_scaling:
            # Guard against None / 0 minimum — behaviour of LocalCluster.adapt(minimum=None)
            # is version-dependent and has caused silent no-op adaptive scaling in some
            # Dask releases.  Always pass an explicit integer >= 1.
            min_w = max(1, self.config.min_workers or 1)
            self.cluster.adapt(
                minimum=min_w,
                maximum=self.config.max_workers,
                wait_count=10,  # Wait 10 seconds before scaling
            )
            logger.info(f"Adaptive scaling: {min_w}-{self.config.max_workers} workers")

        # Create client
        client = Client(self.cluster)

        logger.info(
            f"Local Dask cluster created with dashboard at: {client.dashboard_link}"
        )
        return client

    def _configure_dask_settings(self):
        """Configure global Dask settings."""

        # Set chunk size and caching settings
        dask.config.set(
            {
                "array.chunk-size": "128MB",
                "array.slicing.split_large_chunks": True,
                "distributed.worker.memory.target": 0.85,
                "distributed.worker.memory.spill": 0.90,
                "distributed.worker.memory.pause": 0.95,
                "distributed.worker.memory.terminate": 0.98,
                "distributed.scheduler.allowed-failures": 3,
                "distributed.scheduler.work-stealing": True,
            }
        )

        logger.debug("Dask configuration updated")

    def _log_cluster_info(self):
        """Log information about the cluster."""

        if not self.client:
            return

        try:
            # Get cluster information
            info = self.client.scheduler_info()
            workers = info.get("workers", {})

            logger.info("Dask cluster ready:")
            logger.info(f"  Scheduler: {info.get('address', 'Unknown')}")
            logger.info(f"  Workers: {len(workers)}")

            # Log worker details
            total_cores = 0
            total_memory_gb = 0

            for worker_id, worker_info in workers.items():
                cores = worker_info.get("ncores", 0)
                memory_bytes = worker_info.get("memory_limit", 0)
                memory_gb = memory_bytes / (1024**3)

                total_cores += cores
                total_memory_gb += memory_gb

                logger.debug(
                    f"  Worker {worker_id}: {cores} cores, {memory_gb:.1f}GB memory"
                )

            logger.info(
                f"  Total resources: {total_cores} cores, {total_memory_gb:.1f}GB memory"
            )
            logger.info(f"  Dashboard: {self.client.dashboard_link}")

        except Exception as e:
            logger.warning(f"Could not retrieve cluster info: {e}")

    def shutdown_cluster(self):
        """Shutdown Dask cluster and cleanup resources."""
        logger.info("Shutting down Dask cluster")

        try:
            if self.client:
                self.client.close()
                self.client = None

            if self.cluster:
                self.cluster.close()
                self.cluster = None

            logger.info("Dask cluster shutdown complete")

        except Exception as e:
            logger.warning(f"Error during cluster shutdown: {e}")

    def get_cluster_status(self) -> dict[str, Any]:
        """
        Get current cluster status and resource utilization.

        Returns:
            Dictionary with cluster status information
        """
        if not self.client:
            return {"status": "no_cluster"}

        try:
            # Get scheduler info
            info = self.client.scheduler_info()
            workers = info.get("workers", {})

            # Calculate resource utilization
            total_cores = sum(w.get("ncores", 0) for w in workers.values())
            total_memory = sum(w.get("memory_limit", 0) for w in workers.values())
            used_memory = sum(w.get("memory", 0) for w in workers.values())

            # Get task information
            tasks = info.get("tasks", {})
            running_tasks = len(
                [t for t in tasks.values() if t.get("state") == "processing"]
            )

            status = {
                "status": "active",
                "scheduler_address": info.get("address"),
                "n_workers": len(workers),
                "total_cores": total_cores,
                "total_memory_gb": total_memory / (1024**3),
                "used_memory_gb": used_memory / (1024**3),
                "memory_utilization": used_memory / total_memory
                if total_memory > 0
                else 0,
                "running_tasks": running_tasks,
                "dashboard_link": self.client.dashboard_link,
                "dask_local_dir": self.dask_local_dir,  # None when not using dask_setup
            }

            return status

        except Exception as e:
            logger.error(f"Error getting cluster status: {e}")
            return {"status": "error", "error": str(e)}


class LazyDatasetAssembler:
    """
    Assemble multiple UM files into time series using lazy evaluation.

    Uses Dask delayed operations to build time series without loading
    data into memory until needed.
    """

    def __init__(self, client: Client | None = None):
        """
        Initialize lazy dataset assembler.

        Args:
            client: Dask client for distributed processing
        """
        self.client = client
        self.logger = logging.getLogger(__name__)

    @delayed
    def load_um_file_lazy(
        self, file_path: Path, processing_config: dict[str, Any]
    ) -> xr.Dataset:
        """
        Lazy UM file loading function for Dask delayed execution.

        Args:
            file_path: Path to UM file
            processing_config: Configuration for processing

        Returns:
            Processed dataset (delayed)
        """
        from ..io.um_reader_simple import UMFileReader
        from ..processing.data_processing import (
            DataTypeOptions,
            MaskingOptions,
            process_dataset,
        )
        from ..processing.iris_converter import IrisToXarrayConverter
        from ..processing.stash_metadata import StashMetadataManager

        logger.info(f"Loading UM file (lazy): {file_path}")

        try:
            # Load UM file
            reader = UMFileReader()
            cubes, grid_metadata = reader.load_file(file_path)

            # Filter cubes if needed
            if processing_config.get("include_list") or processing_config.get(
                "exclude_list"
            ):
                cubes = reader.filter_cubes_by_stash(
                    cubes,
                    processing_config.get("include_list"),
                    processing_config.get("exclude_list"),
                )

            # Convert to Xarray
            stash_manager = StashMetadataManager()
            converter = IrisToXarrayConverter(stash_manager)
            dataset = converter.convert_cubes_to_dataset(
                cubes,
                grid_metadata,
                simple_names=processing_config.get("simple_names", False),
            )

            # Apply processing
            masking_opts = MaskingOptions(
                nomask=processing_config.get("nomask", False),
                hcrit=processing_config.get("hcrit", 0.5),
            )

            dtype_opts = DataTypeOptions(
                use64bit=processing_config.get("use64bit", False)
            )

            grid_info = {
                "grid_type": grid_metadata.grid_type,
                "dlat": grid_metadata.dlat,
                "dlon": grid_metadata.dlon,
                "z_rho": grid_metadata.z_rho,
                "z_theta": grid_metadata.z_theta,
            }

            processed_dataset = process_dataset(
                dataset, masking_opts, dtype_opts, grid_info
            )

            logger.info(f"Successfully processed UM file: {file_path}")
            return processed_dataset

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            raise

    def create_time_series(
        self,
        file_paths: list[Path],
        processing_config: dict[str, Any],
        time_dim: str = "time",
    ) -> xr.Dataset:
        """
        Create time series dataset from multiple UM files.

        Each file is loaded as a Dask delayed task. All delayed tasks are then
        computed in parallel via dask.compute() before being concatenated along
        the time dimension.  Individual xarray variables inside the resulting
        datasets remain Dask-backed (lazy) if Iris returns lazy arrays.

        Args:
            file_paths: List of UM file paths
            processing_config: Processing configuration
            time_dim: Time dimension name for concatenation

        Returns:
            Concatenated time series dataset
        """
        if not file_paths:
            raise ValueError("No files provided for time series creation")

        logger.info(f"Creating time series from {len(file_paths)} files")

        # Build one delayed task per file
        delayed_datasets = [
            self.load_um_file_lazy(file_path, processing_config)
            for file_path in file_paths
        ]

        logger.info(f"Created {len(delayed_datasets)} delayed loading tasks")

        if len(delayed_datasets) == 1:
            return delayed_datasets[0].compute()

        # Compute all delayed tasks in parallel rather than one by one.
        # dask.compute() submits the whole graph at once so the scheduler
        # can overlap I/O and processing across files.
        import dask

        datasets = list(dask.compute(*delayed_datasets))

        time_series = xr.concat(datasets, dim=time_dim, combine_attrs="drop_conflicts")
        logger.info(f"Created time series dataset with shape: {dict(time_series.dims)}")
        return time_series

    def create_chunked_time_series(
        self,
        file_paths: list[Path],
        processing_config: dict[str, Any],
        chunk_config: dict[str, int] | None = None,
        time_dim: str = "time",
    ) -> xr.Dataset:
        """
        Create chunked time series with optimal Dask chunking.

        Args:
            file_paths: List of UM file paths
            processing_config: Processing configuration
            chunk_config: Explicit chunk configuration
            time_dim: Time dimension name

        Returns:
            Optimally chunked time series dataset
        """
        from ..processing.chunk_manager import ChunkManager, create_time_series_chunks

        logger.info(f"Creating chunked time series from {len(file_paths)} files")

        # Load first file to determine structure
        first_delayed = self.load_um_file_lazy(file_paths[0], processing_config)
        first_ds = first_delayed.compute()

        # Calculate optimal chunks if not provided
        if chunk_config is None:
            chunk_config = create_time_series_chunks([first_ds], time_dim)

        # Apply chunking to first dataset
        chunk_manager = ChunkManager()
        first_chunked = chunk_manager.apply_chunking(first_ds, chunk_config)

        # Create delayed loading tasks for remaining files
        if len(file_paths) > 1:
            remaining_delayed = []
            for file_path in file_paths[1:]:
                delayed_ds = self.load_um_file_lazy(file_path, processing_config)
                # Apply chunking to each dataset
                chunked_delayed = delayed_ds.map_blocks(
                    lambda ds: chunk_manager.apply_chunking(ds, chunk_config),
                    dtype=first_chunked.dtype,
                )
                remaining_delayed.append(chunked_delayed)

            # Concatenate all datasets
            all_datasets = [first_chunked] + [ds.compute() for ds in remaining_delayed]
            time_series = xr.concat(all_datasets, dim=time_dim)
        else:
            time_series = first_chunked

        logger.info(f"Created chunked time series: {dict(time_series.dims)}")
        return time_series


def create_processing_graph(
    file_paths: list[Path],
    processing_config: dict[str, Any],
    output_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Create Dask computation graph for UM file processing.

    Args:
        file_paths: List of UM files to process
        processing_config: Processing configuration
        output_config: Output configuration

    Returns:
        Dask computation graph
    """
    from dask.delayed import delayed

    logger.info(f"Creating processing graph for {len(file_paths)} files")

    # Create lazy dataset assembler
    assembler = LazyDatasetAssembler()

    # Create delayed tasks for each file
    processing_tasks = {}

    for i, file_path in enumerate(file_paths):
        task_name = f"process_file_{i}"

        # Create processing task
        task = assembler.load_um_file_lazy(file_path, processing_config)
        processing_tasks[task_name] = task

    # If creating time series, add concatenation task
    if output_config.get("create_time_series", False):
        time_series_task = delayed(
            lambda *datasets: xr.concat(
                list(datasets), dim=output_config.get("time_dim", "time")
            )
        )(*processing_tasks.values())

        processing_tasks["time_series"] = time_series_task

    logger.info(f"Created computation graph with {len(processing_tasks)} tasks")
    return processing_tasks


def monitor_dask_progress(futures: List, update_interval: int = 10) -> List:
    """
    Monitor progress of Dask futures with logging.

    Args:
        futures: List of Dask futures to monitor
        update_interval: Progress update interval in seconds

    Returns:
        List of completed results
    """
    logger.info(f"Monitoring {len(futures)} Dask tasks")

    results = []
    completed_count = 0
    start_time = time.time()

    try:
        # Monitor completion using as_completed
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1

                # Log progress
                elapsed = time.time() - start_time
                percent_complete = (completed_count / len(futures)) * 100

                logger.info(
                    f"Progress: {completed_count}/{len(futures)} tasks completed "
                    f"({percent_complete:.1f}%) - {elapsed:.1f}s elapsed"
                )

            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)  # Placeholder for failed task
                completed_count += 1

        total_time = time.time() - start_time
        successful_tasks = len([r for r in results if r is not None])

        logger.info(f"All tasks completed in {total_time:.1f}s")
        logger.info(
            f"Success rate: {successful_tasks}/{len(futures)} "
            f"({(successful_tasks / len(futures)) * 100:.1f}%)"
        )

        return results

    except KeyboardInterrupt:
        logger.warning("Progress monitoring interrupted by user")
        # Cancel remaining futures
        for future in futures:
            if not future.done():
                future.cancel()
        raise


def estimate_graph_size(graph_or_future) -> float:
    """
    Estimate the size of a Dask computation graph in MB.

    Uses ``dask.base.tokenize`` and graph key introspection rather than
    ``pickle``, which can fail silently on complex objects (e.g. Zarr mappers).
    Falls back to a key-count heuristic when the graph cannot be inspected.

    Args:
        graph_or_future: Any Dask collection (Array, Dataset, Delayed) or raw
            HighLevelGraph / dict-of-tasks.

    Returns:
        Estimated graph size in MB (0.0 on error).
    """
    try:
        # Resolve to a HighLevelGraph / mapping of tasks
        if hasattr(graph_or_future, "__dask_graph__"):
            graph = graph_or_future.__dask_graph__()
        elif hasattr(graph_or_future, "dask"):
            graph = graph_or_future.dask
        else:
            graph = graph_or_future

        # Primary: count task keys — a reliable proxy for graph complexity
        # Each task key is a (name, *indices) tuple of ~100 bytes on average
        _BYTES_PER_KEY = 120
        if hasattr(graph, "__len__"):
            n_keys = len(graph)
        elif hasattr(graph, "layers"):
            # HighLevelGraph: sum keys across all layers
            n_keys = sum(len(layer) for layer in graph.layers.values())
        else:
            n_keys = 0

        if n_keys > 0:
            size_mb = (n_keys * _BYTES_PER_KEY) / (1024**2)
            logger.debug(f"Graph size estimate: {n_keys} keys → {size_mb:.2f} MB")
            return size_mb

        # Secondary: tokenize the graph object to get a consistent size proxy
        try:
            import dask.base as _db

            token = _db.tokenize(graph)
            # Token is a 32-char hex string; use it as a probe rather than a size
            # Fall through to the heuristic below
        except Exception:
            pass

        return 0.0

    except Exception as exc:
        logger.warning(f"Could not estimate graph size: {exc}")
        return 0.0


def split_dataset_into_batches(dataset, max_graph_size_mb: float = 50.0):
    """
    Split a Dask-backed xarray Dataset into sub-graphs that each stay under
    *max_graph_size_mb*.

    Splits are made along the *time* dimension (or the first dimension found
    on data variables if no time dimension exists).  Each element of the
    returned list is a Dataset slice that can be computed independently.

    Args:
        dataset:            xr.Dataset backed by Dask arrays.
        max_graph_size_mb:  Target maximum graph size per batch in MB.

    Returns:
        List of xr.Dataset slices.  Returns ``[dataset]`` (a single-element
        list) if splitting is not possible or not necessary.
    """
    current_size = estimate_graph_size(dataset)
    if current_size <= max_graph_size_mb or current_size == 0.0:
        return [dataset]

    # Find a splittable dimension
    split_dim = None
    for candidate in ("time", "t", "forecast_reference_time"):
        if candidate in dataset.dims:
            split_dim = candidate
            break
    if split_dim is None and dataset.dims:
        split_dim = next(iter(dataset.dims))

    if split_dim is None or dataset.sizes.get(split_dim, 0) <= 1:
        logger.warning(
            "Cannot split dataset further — graph size remains above threshold."
        )
        return [dataset]

    n = dataset.sizes[split_dim]
    # Heuristic: estimate how many steps fit under the budget
    n_batches = max(2, int(current_size / max_graph_size_mb) + 1)
    step = max(1, n // n_batches)

    slices = []
    for start in range(0, n, step):
        end = min(start + step, n)
        slices.append(dataset.isel({split_dim: slice(start, end)}))

    logger.info(
        f"Split dataset ({n} steps along '{split_dim}') into {len(slices)} batches "
        f"to stay under {max_graph_size_mb} MB graph-size limit"
    )
    return slices


def check_graph_size_and_warn(
    graph_or_future,
    operation_name: str = "computation",
    warn_threshold_mb: float = 10.0,
    error_threshold_mb: float = 50.0,
    auto_split: bool = False,
) -> bool:
    """
    Check Dask graph size and warn or raise error if too large.

    Args:
        graph_or_future:    Dask graph or delayed object to check.
        operation_name:     Name of the operation for logging.
        warn_threshold_mb:  Size threshold for warnings (MB).
        error_threshold_mb: Size threshold for errors / auto-split trigger (MB).
        auto_split:         If True, log a suggestion to split rather than
                            raising a RuntimeError (the caller is responsible
                            for actually splitting).

    Returns:
        True if graph size is acceptable, False otherwise.
    """
    size_mb = estimate_graph_size(graph_or_future)

    if size_mb > error_threshold_mb:
        logger.error(
            f"Dask graph for {operation_name} is extremely large: {size_mb:.2f} MB"
        )
        if auto_split:
            logger.warning(
                "auto_split is enabled — caller should use split_dataset_into_batches()."
            )
            return False
        else:
            raise RuntimeError(
                f"Dask graph too large: {size_mb:.2f} MB exceeds limit of "
                f"{error_threshold_mb} MB.  Pass --max-graph-size-mb to adjust."
            )

    elif size_mb > warn_threshold_mb:
        logger.warning(
            f"Large Dask graph detected for {operation_name}: {size_mb:.2f} MB"
        )
        logger.warning(
            "Consider increasing chunk sizes, using fewer workers, "
            "or processing files individually."
        )
        return True

    else:
        logger.debug(
            f"Dask graph size for {operation_name}: {size_mb:.2f} MB (acceptable)"
        )
        return True


def optimize_dask_graph(
    graph_or_dataset, optimize_graph: bool = True, fuse_operations: bool = True
) -> any:
    """
    Optimize a Dask graph or dataset to reduce task overhead.

    Args:
        graph_or_dataset: Dask object to optimize
        optimize_graph: Whether to apply graph optimization
        fuse_operations: Whether to fuse compatible operations

    Returns:
        Optimized Dask object
    """
    import dask

    try:
        if not (optimize_graph or fuse_operations):
            return graph_or_dataset

        original_size = estimate_graph_size(graph_or_dataset)
        logger.debug(f"Optimizing Dask graph (original size: {original_size:.2f} MB)")

        result = graph_or_dataset

        # Apply Dask's built-in optimizations
        if optimize_graph:
            if hasattr(result, "__dask_optimize__"):
                result = result.__dask_optimize__()
            elif hasattr(result, "dask"):
                # For arrays/datasets with .dask attribute
                optimized_graph = dask.optimize.optimize(result.dask)
                if hasattr(result, "_rebuild"):
                    result = result._rebuild(optimized_graph)

        # Apply operation fusion where possible
        if fuse_operations and hasattr(result, "data"):
            # For xarray objects with Dask arrays
            if hasattr(result.data, "chunks"):
                # Rechunk to consolidate small chunks
                result = result.chunk(result.data.chunks)

        optimized_size = estimate_graph_size(result)
        reduction_pct = (
            ((original_size - optimized_size) / original_size) * 100
            if original_size > 0
            else 0
        )

        logger.debug(
            f"Graph optimization complete: {original_size:.2f} MB -> {optimized_size:.2f} MB "
            f"({reduction_pct:.1f}% reduction)"
        )

        return result

    except Exception as e:
        logger.warning(f"Graph optimization failed, returning original: {e}")
        return graph_or_dataset


# Context manager for automatic cluster management
class DaskCluster:
    """Context manager for automatic Dask cluster setup and teardown."""

    def __init__(self, config: DaskConfig | None = None):
        """
        Initialize context manager.

        Args:
            config: Dask configuration
        """
        self.manager = DaskClusterManager(config)
        self.client: Client | None = None

    def __enter__(self) -> Client:
        """Set up cluster on context entry."""
        self.client = self.manager.setup_cluster()
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Tear down cluster on context exit."""
        self.manager.shutdown_cluster()
        return False  # Don't suppress exceptions

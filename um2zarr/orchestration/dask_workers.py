"""
Standalone functions for Dask-based parallel UM-to-Zarr conversion.

These functions are designed to be serializable and efficient for Dask workers,
without requiring the full ConversionOrchestrator.
"""

import logging
import time
from pathlib import Path
from typing import Any

# Configure module logger
logger = logging.getLogger(__name__)


def convert_file_standalone(
    input_path: str,
    output_path: str,
    stash_codes: list[int] | None = None,
    compression: str = "zstd",
    compression_level: int = 3,
    use_simple_names: bool = False,
) -> dict[str, Any]:
    """
    Standalone function to convert a single UM file to Zarr.

    This function doesn't depend on the ConversionOrchestrator and is
    designed to be efficiently serialized and executed by Dask workers.

    Args:
        input_path: Path to input UM file
        output_path: Path for output Zarr store
        stash_codes: Optional list of STASH codes to include
        compression: Compression algorithm
        compression_level: Compression level (1-9)
        use_simple_names: Use simple variable names instead of CF-compliant

    Returns:
        Dictionary with conversion statistics
    """
    try:
        # Lazy imports - these modules are loaded only when the function runs on the worker
        # This reduces the serialization overhead since we don't serialize these imports
        # Apply NumPy compatibility shim before importing mule-dependent modules
        from ..utils.numpy_compat import ensure_numpy_compatibility

        ensure_numpy_compatibility()

        from ..io.um_reader import UMFileReader
        from ..io.zarr_writer import EncodingStrategy, ZarrWriter
        from ..processing.iris_converter import IrisToXarrayConverter
        from ..processing.stash_metadata import StashMetadataManager

        # Convert string paths to Path objects
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)

        start_time = time.time()
        logger.info(f"Converting {input_path} → {output_path}")

        # Step 1: Load UM file
        logger.info("Step 1/5: Loading UM file...")
        um_reader = UMFileReader()
        cubes, grid_metadata = um_reader.load_file(input_path_obj)
        logger.info(f"Loaded {len(cubes)} cubes")

        # Step 2: Filter cubes by STASH codes if specified
        if stash_codes:
            logger.info(f"Step 2/5: Filtering by STASH codes {stash_codes}...")
            cubes = um_reader.filter_cubes_by_stash(cubes, include_list=stash_codes)
            logger.info(f"Filtered to {len(cubes)} cubes")
        else:
            logger.info("Step 2/5: No STASH filtering requested")

        if not cubes:
            raise ValueError("No cubes remain after filtering")

        # Step 3: Convert to xarray dataset
        logger.info("Step 3/5: Converting to xarray dataset...")
        stash_manager = StashMetadataManager()
        iris_converter = IrisToXarrayConverter(stash_manager)
        conversion_result = iris_converter.convert_cubes_to_dataset(
            cubes, grid_metadata, simple_names=use_simple_names
        )
        dataset = conversion_result.dataset
        logger.info(f"Converted {conversion_result.variables_converted} variables")

        # Step 4: Process dataset (optimize dtypes, coordinates, etc.)
        logger.info("Step 4/5: Processing dataset...")

        # Simple processing (like in DataProcessingPipeline class)
        processed_dataset = dataset.copy()

        # Add some basic CF compliance if missing
        if "Conventions" not in processed_dataset.attrs:
            processed_dataset.attrs["Conventions"] = "CF-1.8"

        processing_warnings = []

        # Step 5: Write to Zarr
        logger.info("Step 5/5: Writing to Zarr...")

        # Check for graph size issues before writing
        try:
            from ..orchestration.dask_integration import check_graph_size_and_warn

            check_graph_size_and_warn(
                processed_dataset,
                operation_name=f"writing {input_path_obj.name}",
                warn_threshold_mb=5.0,
                error_threshold_mb=20.0,
            )
        except ImportError:
            logger.debug("Graph size checking not available")

        encoding_strategy = EncodingStrategy(
            compressor=compression,
            compression_level=compression_level,
            auto_chunk=True,
            chunk_strategy="balanced",
        )
        zarr_writer = ZarrWriter(encoding_strategy)
        # Graph optimization is enabled by default in write_dataset
        write_stats = zarr_writer.write_dataset(
            processed_dataset, output_path_obj, optimize_graph=True
        )

        # Calculate final statistics
        total_time = time.time() - start_time

        stats = {
            "input_file": str(input_path),
            "output_store": str(output_path),
            "total_time_seconds": total_time,
            "cubes_loaded": len(cubes),
            "variables_converted": conversion_result.variables_converted,
            "variables_written": write_stats.get("n_variables", 0),
            "compression_ratio": write_stats.get("compression_ratio", 0.0),
            "zarr_write_time": write_stats.get("write_time_seconds", 0.0),
            "conversion_warnings": conversion_result.conversion_warnings,
            "processing_warnings": processing_warnings,
        }

        logger.info(f"✅ Conversion completed in {total_time:.1f}s")
        logger.info(
            f"   Variables: {stats['variables_converted']} → {stats['variables_written']}"
        )
        logger.info(f"   Compression ratio: {stats['compression_ratio']:.2f}")

        return stats

    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback

        error_details = traceback.format_exc()
        return {
            "input_file": str(input_path),
            "output_store": str(output_path),
            "error": str(e),
            "error_details": error_details,
        }


def convert_batch_with_optimization(
    file_paths: list[str],
    output_dir: str,
    stash_codes: list[int] | None = None,
    compression: str = "zstd",
    compression_level: int = 3,
    use_simple_names: bool = False,
    max_concurrent: int = 4,
) -> list[dict[str, Any]]:
    """
    Convert multiple UM files to Zarr with graph optimization for batch processing.

    This function processes files in smaller batches to avoid creating
    excessively large Dask task graphs that cause memory issues.

    Args:
        file_paths: List of input UM file paths
        output_dir: Directory for output Zarr stores
        stash_codes: Optional list of STASH codes to include
        compression: Compression algorithm
        compression_level: Compression level (1-9)
        use_simple_names: Use simple variable names
        max_concurrent: Maximum number of concurrent conversions

    Returns:
        List of conversion result dictionaries
    """
    from pathlib import Path

    from dask.delayed import delayed
    from dask.distributed import Client

    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {len(file_paths)} files in optimized batch mode")
    logger.info(f"Maximum concurrent conversions: {max_concurrent}")

    # Create delayed tasks for each file
    futures = []
    for input_path in file_paths:
        input_path_obj = Path(input_path)
        output_filename = input_path_obj.stem + ".zarr"
        output_path = output_dir_obj / output_filename

        # Create delayed task
        task = delayed(convert_file_standalone)(
            input_path=input_path,
            output_path=str(output_path),
            stash_codes=stash_codes,
            compression=compression,
            compression_level=compression_level,
            use_simple_names=use_simple_names,
        )
        futures.append(task)

    # Process in batches to avoid large graphs
    batch_size = min(max_concurrent, len(futures))
    results = []

    try:
        # Get client if available
        client = Client.current()

        for i in range(0, len(futures), batch_size):
            batch_futures = futures[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(futures) + batch_size - 1) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_futures)} files)"
            )

            # Check graph size for this batch before submission
            try:
                from ..orchestration.dask_integration import check_graph_size_and_warn

                for j, task in enumerate(batch_futures):
                    check_graph_size_and_warn(
                        task,
                        operation_name=f"batch {batch_num}, file {j + 1}",
                        warn_threshold_mb=3.0,
                        error_threshold_mb=15.0,
                    )
            except ImportError:
                logger.debug("Graph size checking not available for batch")

            # Submit batch and collect results
            batch_results = client.compute(batch_futures)
            completed_results = client.gather(batch_results)
            results.extend(completed_results)

            logger.info(f"Completed batch {batch_num}/{total_batches}")

    except RuntimeError:
        # No Dask client available, compute sequentially
        logger.warning("No Dask client available, computing sequentially")
        for future in futures:
            result = future.compute()
            results.append(result)

    # Log summary statistics
    successful_conversions = sum(1 for r in results if "error" not in r)
    failed_conversions = len(results) - successful_conversions

    logger.info("Batch processing complete:")
    logger.info(f"  Successful: {successful_conversions}/{len(results)}")
    logger.info(f"  Failed: {failed_conversions}/{len(results)}")

    return results

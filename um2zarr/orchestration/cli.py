"""
CLI entry point and conversion orchestration for um2zarr.

Contains the ConversionOrchestrator class and the main() Click command that
powers the `um2zarr` command-line tool installed by pip.

Configuration can be supplied via CLI flags, a YAML config file (--config),
or a mix of both — CLI flags always take precedence over config file values.
"""

import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import click
import yaml

# Core imports
from ..core.data_structures import ProcessingConfig, ProcessingResult
from ..io import UMFileReader
from ..processing.iris_converter import IrisToXarrayConverter
from ..processing.stash_metadata import StashMetadataManager
from .checkpoint import CheckpointManager

# Optional Dask imports
try:
    from .dask_integration import DaskClusterManager, DaskConfig

    HAS_DASK_MANAGER = True
except ImportError:
    HAS_DASK_MANAGER = False

    class DaskClusterManager:  # type: ignore[no-redef]
        def __init__(self, config: Any) -> None:
            pass

        def get_cluster(self) -> None:
            return None

    class DaskConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass


# Optional Zarr imports
try:
    from ..io import EncodingStrategy, ZarrWriter

    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

try:
    import dask

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("um2zarr")


# ---------------------------------------------------------------------------
# Processing classes (unchanged by CLI refactor)
# ---------------------------------------------------------------------------


class DataProcessingPipeline:
    """
    Wraps um2zarr.processing.data_processing.process_dataset for use in
    ConversionOrchestrator.

    Applies the full processing chain: mask removal, data-type coercion,
    fill-value assignment, and coordinate standardisation (lat/lon, model
    levels, pressure levels).
    """

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        self.config = config or ProcessingConfig()

    def process_dataset(
        self,
        dataset: Any,
        grid_metadata: Any | None = None,
    ) -> ProcessingResult:
        """
        Run the full processing pipeline on *dataset*.

        Args:
            dataset: Xarray dataset returned by IrisToXarrayConverter.
            grid_metadata: Optional GridMetadata instance.  When provided,
                coordinate-fixing steps (lat/lon names, rho/theta levels,
                pressure conventions) are applied.

        Returns:
            ProcessingResult containing the processed dataset.
        """
        from ..processing.data_processing import (
            DataTypeOptions,
            MaskingOptions,
        )
        from ..processing.data_processing import (
            process_dataset as _process_dataset,
        )

        start_time = time.time()

        masking_opts = MaskingOptions(
            nomask=getattr(self.config, "nomask", False),
            hcrit=getattr(self.config, "hcrit", 0.5),
        )
        dtype_opts = DataTypeOptions(use64bit=getattr(self.config, "use64bit", False))

        grid_info: dict[str, Any] | None = None
        if grid_metadata is not None:
            grid_info = {
                "grid_type": grid_metadata.grid_type,
                "dlat": grid_metadata.dlat,
                "dlon": grid_metadata.dlon,
                "z_rho": grid_metadata.z_rho,
                "z_theta": grid_metadata.z_theta,
            }

        processed_dataset = _process_dataset(
            dataset, masking_opts, dtype_opts, grid_info
        )

        cmor_warnings: list[str] = []
        if getattr(self.config, "cmor", False):
            try:
                from ..processing.cmor_processor import CMORConfig, CMORProcessor

                cmor_cfg = CMORConfig(
                    activity_id=getattr(self.config, "cmor_activity_id", "CMIP"),
                    experiment_id=getattr(self.config, "cmor_experiment_id", ""),
                    source_id=getattr(self.config, "cmor_source_id", ""),
                    variant_label=getattr(
                        self.config, "cmor_variant_label", "r1i1p1f1"
                    ),
                    institution_id=getattr(self.config, "cmor_institution_id", ""),
                    calendar=getattr(self.config, "cmor_calendar", ""),
                    drop_unmapped=getattr(self.config, "cmor_drop_unmapped", False),
                )
                cmor_proc = CMORProcessor(cmor_cfg)
                processed_dataset, cmor_report = cmor_proc.process(processed_dataset)
                cmor_warnings = cmor_report.get("warnings", []) + cmor_report.get(
                    "errors", []
                )
                logger.info(
                    f"CMOR processing complete — "
                    f"renamed: {len(cmor_report.get('renamed', []))}, "
                    f"converted_units: {len(cmor_report.get('converted_units', []))}, "
                    f"unmapped: {len(cmor_report.get('unmapped', []))}"
                )
            except Exception as exc:
                cmor_warnings.append(f"CMOR processing failed: {exc}")
                logger.error(f"CMOR processing failed: {exc}")

        return ProcessingResult(
            dataset=processed_dataset,
            processing_time=time.time() - start_time,
            memory_usage={},
            warnings=cmor_warnings,
        )


class ConversionOrchestrator:
    """Orchestrates the complete UM to Zarr conversion workflow."""

    def __init__(self, config: ProcessingConfig | None = None) -> None:
        self.config = config or ProcessingConfig()
        self.setup_components()

    def setup_components(self) -> None:
        """Initialize all processing components."""
        logger.info("Setting up conversion components...")

        try:
            from ..utils.numpy_compat import ensure_numpy_compatibility

            ensure_numpy_compatibility()
        except Exception as e:
            logger.debug(f"NumPy compatibility shim not applied: {e}")

        self.um_reader = UMFileReader()
        self.stash_manager = StashMetadataManager()
        self.iris_converter = IrisToXarrayConverter(self.stash_manager)
        self.data_processor = DataProcessingPipeline(self.config)

        if HAS_ZARR:
            encoding_strategy = EncodingStrategy(
                compressor=self.config.compression,
                compression_level=self.config.compression_level,
                auto_chunk=True,
                chunk_strategy=self.config.chunk_strategy,
            )
            self.zarr_writer = ZarrWriter(encoding_strategy)
        else:
            logger.error("Zarr writer not available - cannot proceed with conversion")
            raise ImportError(
                "Zarr is required for conversion. Install with: pip install zarr"
            )

        if self.config.use_dask and HAS_DASK and HAS_DASK_MANAGER:
            memory_limit_gb = None
            if self.config.memory_limit:
                memory_str = self.config.memory_limit.upper()
                if memory_str.endswith("GB"):
                    memory_limit_gb = float(memory_str[:-2])
                elif memory_str.endswith("MB"):
                    memory_limit_gb = float(memory_str[:-2]) / 1024
                else:
                    memory_limit_gb = float(memory_str)

            dask_config = DaskConfig(
                # External / HPC cluster connection
                scheduler_file=getattr(self.config, "scheduler_file", None),
                scheduler_address=getattr(self.config, "scheduler_address", None),
                # dask_setup parameters
                workload_type=self.config.workload_type,
                max_workers=self.config.n_workers
                if self.config.n_workers > 0
                else None,
                min_workers=max(1, getattr(self.config, "min_workers", 1) or 1),
                reserve_mem_gb=getattr(self.config, "reserve_memory_gb", 2.0),
                max_mem_gb=memory_limit_gb,
                dashboard=True,
                adaptive=self.config.adaptive_scaling,
                suggest_chunks=getattr(self.config, "suggest_chunks", False),
                # Legacy / fallback parameters
                n_workers=self.config.n_workers,
                memory_limit_gb=memory_limit_gb,
                threads_per_worker=self.config.threads_per_worker,
            )
            self.dask_manager = DaskClusterManager(dask_config)
        else:
            self.dask_manager = None

        logger.info("All components initialized successfully")

    # ------------------------------------------------------------------
    # Catalogue helper
    # ------------------------------------------------------------------

    def _maybe_write_catalogue(
        self,
        store_path: Path,
        dataset: Any = None,
        write_stats: dict[str, Any] | None = None,
        source_files: list[Path] | None = None,
    ) -> None:
        """
        If the config requests catalogue generation, write it now.

        Silently skips if ``config.catalogue_path`` is falsy.  Errors are
        logged as warnings rather than raising, so a catalogue failure never
        aborts the conversion itself.
        """
        cat_path = getattr(self.config, "catalogue_path", None)
        if not cat_path:
            return

        cat_format = getattr(self.config, "catalogue_format", "intake")

        try:
            from ..io.catalogue_writer import write_catalogue

            written = write_catalogue(
                store_path=store_path,
                dataset=dataset,
                write_stats=write_stats,
                source_files=source_files,
                catalogue_path=cat_path,
                format=cat_format,
            )
            for p in written:
                logger.info(f"Catalogue written: {p}")
        except Exception as exc:
            logger.warning(f"Catalogue generation failed (non-fatal): {exc}")

    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
        stash_codes: list[int] | None = None,
    ) -> dict[str, Any]:
        """Convert a single UM file to Zarr."""
        start_time = time.time()
        logger.info(f"Converting {input_path} → {output_path}")

        try:
            logger.info("Step 1/5: Loading UM file...")
            cubes, grid_metadata = self.um_reader.load_file(input_path)
            logger.info(f"Loaded {len(cubes)} cubes")

            if stash_codes:
                logger.info(f"Step 2/5: Filtering by STASH codes {stash_codes}...")
                cubes = self.um_reader.filter_cubes_by_stash(
                    cubes, include_list=stash_codes
                )
                logger.info(f"Filtered to {len(cubes)} cubes")
            else:
                logger.info("Step 2/5: No STASH filtering requested")

            if not cubes:
                raise ValueError("No cubes remain after filtering")

            logger.info("Step 3/5: Converting to xarray dataset...")
            conversion_result = self.iris_converter.convert_cubes_to_dataset(
                cubes,
                grid_metadata,
                simple_names=self.config.simple_names,
            )
            dataset = conversion_result.dataset
            logger.info(f"Converted {conversion_result.variables_converted} variables")

            logger.info("Step 4/5: Processing dataset...")
            processing_result = self.data_processor.process_dataset(
                dataset, grid_metadata=grid_metadata
            )
            dataset = processing_result.dataset

            logger.info("Step 5/5: Writing to Zarr...")
            write_stats = self.zarr_writer.write_dataset(dataset, output_path)

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
                "processing_warnings": processing_result.warnings,
            }
            logger.info(f"Conversion completed in {total_time:.1f}s")
            logger.info(
                f"   Variables: {stats['variables_converted']} → {stats['variables_written']}"
            )
            logger.info(f"   Compression ratio: {stats['compression_ratio']:.2f}")

            self._maybe_write_catalogue(
                store_path=output_path,
                dataset=dataset,
                write_stats=write_stats,
                source_files=[input_path],
            )
            return stats

        except Exception as e:
            logger.error(f"❌ Conversion failed: {e}")
            raise

    def convert_file_append(
        self,
        input_path: Path,
        output_path: Path,
        append_dim: str = "time",
        stash_codes: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Convert a UM file and *append* the result to an existing Zarr store.

        Before appending, the existing store is validated for schema compatibility
        (variable list, dtypes, chunk shapes along non-append dimensions).
        Raises ``ValueError`` if the schemas are incompatible.
        """
        start_time = time.time()
        logger.info(f"Appending {input_path} → {output_path} (dim={append_dim})")

        try:
            cubes, grid_metadata = self.um_reader.load_file(input_path)
            if stash_codes:
                cubes = self.um_reader.filter_cubes_by_stash(
                    cubes, include_list=stash_codes
                )
            if not cubes:
                raise ValueError("No cubes remain after filtering")

            conversion_result = self.iris_converter.convert_cubes_to_dataset(
                cubes,
                grid_metadata,
                simple_names=self.config.simple_names,
            )
            dataset = conversion_result.dataset

            processing_result = self.data_processor.process_dataset(
                dataset, grid_metadata=grid_metadata
            )
            dataset = processing_result.dataset

            # Schema validation before appending
            schema_check = self.zarr_writer.validate_append_schema(
                output_path, dataset, append_dim=append_dim
            )
            if not schema_check.get("compatible", True):
                raise ValueError(
                    f"Append schema validation failed: {schema_check.get('errors')}"
                )

            write_stats = self.zarr_writer.write_dataset(
                dataset, output_path, mode="a", append_dim=append_dim
            )

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
                "mode": "append",
                "append_dim": append_dim,
            }
            logger.info(f"Append completed in {total_time:.1f}s")

            self._maybe_write_catalogue(
                store_path=output_path,
                dataset=dataset,
                write_stats=write_stats,
                source_files=[input_path],
            )
            return stats

        except Exception as exc:
            logger.error(f"Append conversion failed: {exc}")
            raise

    def convert_ensemble(
        self,
        member_paths: list[Path],
        output_path: Path,
        ensemble_dim: str = "realization",
        member_id_pattern: str | None = None,
        stash_codes: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Convert multiple UM ensemble members and concatenate along an ensemble
        dimension before writing a single Zarr store.

        Parameters
        ----------
        member_paths:
            Ordered list of UM files — one per ensemble member.
        output_path:
            Destination Zarr store.
        ensemble_dim:
            Name of the new ensemble dimension (CF convention: ``realization``).
        member_id_pattern:
            Optional regex applied to each file name to extract the member ID.
            The first capture group is used as the coordinate value.  If not
            supplied, integer indices 0, 1, 2, … are used.
        stash_codes:
            Optional STASH code filter applied to every member file.
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray is required for ensemble conversion.") from None

        start_time = time.time()
        n_members = len(member_paths)
        logger.info(f"Converting {n_members} ensemble members → {output_path}")

        member_datasets = []
        member_ids = []

        for idx, path in enumerate(member_paths):
            logger.info(f"  Member {idx + 1}/{n_members}: {path.name}")
            cubes, grid_metadata = self.um_reader.load_file(path)
            if stash_codes:
                cubes = self.um_reader.filter_cubes_by_stash(
                    cubes, include_list=stash_codes
                )
            if not cubes:
                logger.warning(f"  No cubes after filtering — skipping {path.name}")
                continue

            conv = self.iris_converter.convert_cubes_to_dataset(
                cubes,
                grid_metadata,
                simple_names=self.config.simple_names,
            )
            proc = self.data_processor.process_dataset(
                conv.dataset, grid_metadata=grid_metadata
            )
            member_datasets.append(proc.dataset)

            # Extract member ID
            if member_id_pattern:
                m = re.search(member_id_pattern, path.name)
                member_ids.append(m.group(1) if m else idx)
            else:
                member_ids.append(idx)

        if not member_datasets:
            raise ValueError("No member datasets could be loaded")

        logger.info(
            f"Concatenating {len(member_datasets)} members along '{ensemble_dim}'"
        )
        ensemble_ds = xr.concat(
            member_datasets,
            dim=xr.DataArray(member_ids, dims=[ensemble_dim], name=ensemble_dim),
        )

        write_stats = self.zarr_writer.write_dataset(ensemble_ds, output_path)
        total_time = time.time() - start_time

        stats = {
            "output_store": str(output_path),
            "total_time_seconds": total_time,
            "n_members": len(member_datasets),
            "ensemble_dim": ensemble_dim,
            "member_ids": member_ids,
            "variables_written": write_stats.get("n_variables", 0),
        }
        logger.info(
            f"Ensemble conversion done in {total_time:.1f}s — "
            f"{len(member_datasets)} members, {stats['variables_written']} variables"
        )

        self._maybe_write_catalogue(
            store_path=output_path,
            dataset=ensemble_ds,
            write_stats=write_stats,
            source_files=member_paths,
        )
        return stats

    def convert_batch(
        self,
        input_paths: list[Path],
        output_dir: Path,
        stash_codes: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert multiple UM files to individual Zarr stores."""
        logger.info(f"Starting batch conversion of {len(input_paths)} files")
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.dask_manager and len(input_paths) > 1:
            logger.info("Using Dask for parallel processing...")
            results = self._convert_batch_with_dask(
                input_paths, output_dir, stash_codes
            )
        else:
            logger.info("Using sequential processing...")
            results = self._convert_batch_sequential(
                input_paths, output_dir, stash_codes
            )

        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        logger.info(
            f"Batch complete — successful: {len(successful)}, failed: {len(failed)}"
        )

        if failed:
            logger.warning("Failed conversions:")
            for result in failed:
                logger.warning(f"   {result['input_file']}: {result['error']}")

        # Write a single catalogue covering all successful stores
        cat_path = getattr(self.config, "catalogue_path", None)
        if cat_path and successful:
            try:
                from ..io.catalogue_writer import IntakeCatalogueWriter

                cat_format = getattr(self.config, "catalogue_format", "intake")
                cat_writer = IntakeCatalogueWriter(
                    catalogue_dir=cat_path
                    if Path(cat_path).suffix == ""
                    else Path(cat_path).parent
                )
                for r in successful:
                    cat_writer.add_store(
                        store_path=r["output_store"],
                        source_files=[r["input_file"]],
                        write_stats=r,
                    )
                written = cat_writer.write(format=cat_format)
                for p in written:
                    logger.info(f"Batch catalogue written: {p}")
            except Exception as exc:
                logger.warning(f"Batch catalogue generation failed (non-fatal): {exc}")

        return results

    def _convert_batch_sequential(
        self,
        input_paths: list[Path],
        output_dir: Path,
        stash_codes: list[int] | None,
    ) -> list[dict[str, Any]]:
        on_error = getattr(self.config, "on_error", "skip")
        append_mode = getattr(self.config, "append", False)
        append_dim = getattr(self.config, "append_dim", "time")
        resume = getattr(self.config, "resume", False)

        checkpoint = CheckpointManager(output_dir)
        if resume:
            n_skip = len([p for p in input_paths if checkpoint.is_complete(p)])
            if n_skip:
                logger.info(f"Checkpoint: skipping {n_skip} already-completed file(s)")

        results: list[dict[str, Any]] = []
        for i, input_path in enumerate(input_paths, 1):
            if resume and checkpoint.is_complete(input_path):
                logger.info(
                    f"[{i}/{len(input_paths)}] Skipping (already done): {input_path.name}"
                )
                continue

            logger.info(f"[{i}/{len(input_paths)}] Processing: {input_path.name}")
            output_path = output_dir / (input_path.stem + ".zarr")

            _retry_left = 1 if on_error == "retry" else 0
            while True:
                try:
                    if append_mode:
                        stats = self.convert_file_append(
                            input_path,
                            output_path,
                            append_dim=append_dim,
                            stash_codes=stash_codes,
                        )
                    else:
                        stats = self.convert_file(input_path, output_path, stash_codes)
                    results.append(stats)
                    checkpoint.mark_complete(input_path, stats)
                    break
                except Exception as exc:
                    if _retry_left > 0:
                        _retry_left -= 1
                        logger.warning(f"Retrying {input_path.name} after error: {exc}")
                        continue
                    err_record = {
                        "input_file": str(input_path),
                        "output_store": str(output_path),
                        "error": str(exc),
                    }
                    results.append(err_record)
                    checkpoint.mark_failed(input_path, str(exc))
                    logger.error(f"Failed to convert {input_path}: {exc}")
                    if on_error == "abort":
                        logger.error("on_error=abort — stopping batch immediately")
                        return results
                    break  # on_error == 'skip' or 'retry' exhausted

        summary = checkpoint.summary()
        logger.info(
            f"Checkpoint summary — completed: {summary['completed']}, "
            f"failed: {summary['failed']}"
        )
        return results

    def _convert_batch_with_dask(
        self,
        input_paths: list[Path],
        output_dir: Path,
        stash_codes: list[int] | None,
    ) -> list[dict[str, Any]]:
        logger.info("Setting up Dask cluster...")
        from .dask_integration import DaskCluster
        from .dask_workers import convert_file_standalone

        on_error = getattr(self.config, "on_error", "skip")
        resume = getattr(self.config, "resume", False)
        checkpoint = CheckpointManager(output_dir)

        # Filter out already-completed files when resuming
        pending_paths = input_paths
        if resume:
            pending_paths = [p for p in input_paths if not checkpoint.is_complete(p)]
            n_skip = len(input_paths) - len(pending_paths)
            if n_skip:
                logger.info(f"Checkpoint: skipping {n_skip} already-completed file(s)")

        if not pending_paths:
            logger.info("All files already completed — nothing to do")
            return []

        with DaskCluster(self.dask_manager.config) as client:
            import dask.bag as db

            tasks = [
                {
                    "input_path": str(p),
                    "output_path": str(output_dir / (p.stem + ".zarr")),
                    "stash_codes": stash_codes,
                    "compression": self.config.compression,
                    "compression_level": self.config.compression_level,
                    "use_simple_names": self.config.simple_names,
                }
                for p in pending_paths
            ]
            logger.info(f"Created {len(tasks)} conversion tasks for Dask workers")
            results: list[dict[str, Any]] = list(
                db.from_sequence(tasks)
                .map(lambda t: convert_file_standalone(**t))
                .compute()
            )

        # Post-process results through checkpoint
        final: list[dict[str, Any]] = []
        for r in results:
            if "error" in r:
                checkpoint.mark_failed(Path(r["input_file"]), r["error"])
                if on_error == "abort":
                    logger.error(
                        f"on_error=abort — stopping after first Dask worker failure: "
                        f"{r['input_file']}"
                    )
                    final.append(r)
                    break
            else:
                checkpoint.mark_complete(Path(r["input_file"]), r)
            final.append(r)

        summary = checkpoint.summary()
        logger.info(
            f"Checkpoint summary — completed: {summary['completed']}, "
            f"failed: {summary['failed']}"
        )
        return final

    def list_files(self, input_path: Path, pattern: str = "*.pp") -> list[Path]:
        """Find UM files to process."""
        if input_path.is_file():
            return [input_path]
        elif input_path.is_dir():
            files = sorted(input_path.glob(pattern))
            logger.info(
                f"Found {len(files)} files matching '{pattern}' in {input_path}"
            )
            return files
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_yaml_config(ctx: click.Context, param: click.Parameter, value: Any) -> Any:
    """Eager callback: load YAML file into ctx.default_map so CLI flags still win."""
    if not value:
        return value
    with open(value) as fh:
        data = yaml.safe_load(fh) or {}
    # Merge on top of any existing default_map (e.g. from env vars)
    ctx.default_map = {**(ctx.default_map or {}), **data}
    return value


def _build_config_dict(
    *,
    compression: str,
    compression_level: int,
    dask: bool,
    workers: int,
    memory_limit: str,
    threads_per_worker: int,
    chunk_strategy: str,
    validate_cf: bool,
    simple_names: bool,
    nomask: bool,
    hcrit: float,
    workload_type: str,
    adaptive: bool,
    min_workers: int,
    scheduler_file: str | None,
    scheduler_address: str | None,
    reserve_memory_gb: float,
    suggest_chunks: bool,
    # Phase 2 additions
    append: bool,
    append_dim: str,
    resume: bool,
    on_error: str,
    ensemble_dim: str | None,
    ensemble_member_pattern: str | None,
    max_graph_size_mb: float,
    catalogue_path: str | None,
    catalogue_format: str,
    cmor: bool,
    cmor_activity_id: str,
    cmor_experiment_id: str,
    cmor_source_id: str,
    cmor_variant_label: str,
    cmor_institution_id: str,
    cmor_calendar: str,
    cmor_drop_unmapped: bool,
) -> dict[str, Any]:
    """Return a plain dict matching the YAML config schema."""
    cfg: dict[str, Any] = {
        "compression": compression,
        "compression_level": compression_level,
        "dask": dask,
        "workers": workers,
        "memory_limit": memory_limit,
        "threads_per_worker": threads_per_worker,
        "workload_type": workload_type,
        "adaptive": adaptive,
        "min_workers": min_workers,
        "reserve_memory_gb": reserve_memory_gb,
        "suggest_chunks": suggest_chunks,
        "chunk_strategy": chunk_strategy,
        "validate_cf": validate_cf,
        "simple_names": simple_names,
        "nomask": nomask,
        "hcrit": hcrit,
        # Phase 2
        "append": append,
        "append_dim": append_dim,
        "resume": resume,
        "on_error": on_error,
        "max_graph_size_mb": max_graph_size_mb,
    }
    # Only include optional cluster connection if set (avoids cluttering saved configs)
    if scheduler_file is not None:
        cfg["scheduler_file"] = scheduler_file
    if scheduler_address is not None:
        cfg["scheduler_address"] = scheduler_address
    if ensemble_dim is not None:
        cfg["ensemble_dim"] = ensemble_dim
    if ensemble_member_pattern is not None:
        cfg["ensemble_member_pattern"] = ensemble_member_pattern
    if catalogue_path is not None:
        cfg["catalogue_path"] = catalogue_path
    cfg["catalogue_format"] = catalogue_format
    cfg["cmor"] = cmor
    if cmor:
        cfg["cmor_activity_id"] = cmor_activity_id
        cfg["cmor_experiment_id"] = cmor_experiment_id
        cfg["cmor_source_id"] = cmor_source_id
        cfg["cmor_variant_label"] = cmor_variant_label
        cfg["cmor_institution_id"] = cmor_institution_id
        cfg["cmor_calendar"] = cmor_calendar
        cfg["cmor_drop_unmapped"] = cmor_drop_unmapped
    return cfg


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else (logging.ERROR if quiet else logging.INFO)
    logging.getLogger().setLevel(level)
    for name in [
        "um2zarr",
        "um2zarr.io",
        "um2zarr.processing",
        "um2zarr.orchestration",
    ]:
        logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
# ---- config file (eager so it populates default_map before other options) ----
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    is_eager=True,
    expose_value=False,
    callback=_load_yaml_config,
    help="YAML config file.  CLI flags override values set here.",
)
@click.option(
    "--save-config",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the effective configuration to this YAML file and exit.",
)
# ---- compression ----
@click.option(
    "--compression",
    type=click.Choice(["zstd", "blosc", "gzip", "bz2", "lz4"]),
    default="zstd",
    show_default=True,
    help="Compression algorithm.",
)
@click.option(
    "--compression-level",
    type=click.IntRange(1, 9),
    default=3,
    show_default=True,
    help="Compression level (1 = fast, 9 = best ratio).",
)
# ---- dask ----
@click.option(
    "--dask/--no-dask",
    default=True,
    show_default=True,
    help="Enable/disable Dask parallel processing.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=4,
    show_default=True,
    help="Number of Dask workers.",
)
@click.option(
    "--memory-limit",
    default="4GB",
    show_default=True,
    help="Memory limit per Dask worker (e.g. 4GB, 512MB).",
)
@click.option(
    "--threads-per-worker",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Threads per Dask worker.",
)
@click.option(
    "--workload-type",
    type=click.Choice(["cpu", "io", "mixed"]),
    default="io",
    show_default=True,
    help="Workload optimisation hint passed to dask_setup.",
)
@click.option(
    "--adaptive/--no-adaptive",
    default=True,
    show_default=True,
    help="Enable/disable adaptive Dask worker scaling.",
)
@click.option(
    "--min-workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Minimum number of live workers when adaptive scaling is enabled.",
)
@click.option(
    "--reserve-memory-gb",
    type=float,
    default=2.0,
    show_default=True,
    help="GB of RAM to reserve per node (passed to dask_setup).",
)
@click.option(
    "--suggest-chunks/--no-suggest-chunks",
    default=False,
    show_default=True,
    help="Ask dask_setup to recommend chunk sizes for this workload.",
)
# ---- HPC / external cluster ----
@click.option(
    "--scheduler-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a Dask scheduler JSON file (for PBS/SLURM-managed clusters).",
)
@click.option(
    "--scheduler-address",
    default=None,
    help="Address of a running Dask scheduler, e.g. tcp://10.0.0.1:8786.",
)
# ---- processing ----
@click.option(
    "--chunk-strategy",
    type=click.Choice(["memory", "balanced", "storage"]),
    default="balanced",
    show_default=True,
    help="Zarr chunk strategy.",
)
@click.option(
    "--validate-cf/--no-validate-cf",
    default=True,
    show_default=True,
    help="Validate CF convention compliance before writing.",
)
@click.option(
    "--nomask",
    is_flag=True,
    default=False,
    help="Disable all heaviside masking (pass pressure-level data through unmasked).",
)
@click.option(
    "--hcrit",
    type=click.FloatRange(0.0, 1.0, min_open=True, max_open=True),
    default=0.5,
    show_default=True,
    help="Heaviside critical threshold: grid points where heaviside <= hcrit are masked.",
)
@click.option(
    "--simple-names/--no-simple-names",
    default=False,
    show_default=True,
    help="Use STASH-code variable names (fld_s01i001) instead of CF names.",
)
# ---- filtering ----
@click.option(
    "--stash",
    default=None,
    help="Comma-separated STASH codes to include, e.g. '4,2,3,409'.",
)
@click.option(
    "--pattern",
    default="*.pp",
    show_default=True,
    help="Glob pattern for discovering UM files when INPUT is a directory.",
)
# ---- append / resume (Phase 2.1) ----
@click.option(
    "--append/--no-append",
    default=False,
    show_default=True,
    help="Open the output Zarr store in append mode instead of overwriting it.",
)
@click.option(
    "--append-dim",
    default="time",
    show_default=True,
    help="Dimension along which to append when --append is set.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help=(
        "In batch mode, skip files already recorded in the checkpoint as "
        "successfully processed.  Implies --append for the output store."
    ),
)
# ---- error recovery (Phase 2.5) ----
@click.option(
    "--on-error",
    type=click.Choice(["skip", "abort", "retry"]),
    default="skip",
    show_default=True,
    help=(
        "Per-file error policy in batch mode: 'skip' logs and continues, "
        "'abort' stops immediately, 'retry' retries once before skipping."
    ),
)
# ---- ensemble support (Phase 2.4) ----
@click.option(
    "--ensemble-dim",
    default=None,
    help=(
        "Concatenate all INPUT files along this ensemble dimension "
        "(e.g. 'realization') instead of writing separate Zarr stores."
    ),
)
@click.option(
    "--ensemble-member-pattern",
    default=None,
    help=(
        "Regex applied to each input filename to extract the member ID.  "
        "The first capture group becomes the coordinate value.  "
        "Example: 'r(\\d+)i1p1f1'."
    ),
)
# ---- graph size (Phase 2.6) ----
@click.option(
    "--max-graph-size-mb",
    type=float,
    default=50.0,
    show_default=True,
    help=(
        "Maximum Dask computation graph size in MB.  Graphs exceeding this "
        "limit are automatically split into smaller sub-graphs."
    ),
)
# ---- intake catalogue (Phase 3.1) ----
@click.option(
    "--write-catalogue",
    type=click.Path(path_type=Path),
    default=None,
    metavar="PATH",
    help=(
        "Write an intake catalogue after conversion.  "
        "PATH can be a directory (uses default filename) or a full file path. "
        "Catalogue includes variable list, coordinate ranges, compression info, "
        "and provenance for every Zarr store produced."
    ),
)
@click.option(
    "--catalogue-format",
    type=click.Choice(["intake", "esm", "json"]),
    default="intake",
    show_default=True,
    help=(
        "Catalogue format: 'intake' (YAML, requires intake-xarray), "
        "'esm' (JSON+CSV, requires intake-esm), "
        "'json' (plain provenance JSON, no intake required)."
    ),
)
# ---- CMOR / CMIP6 output (Phase 3.3) ----
@click.option(
    "--cmor/--no-cmor",
    default=False,
    show_default=True,
    help=(
        "Apply CMIP6/CMOR post-processing: rename variables to CMIP6 short "
        "names, convert units, add required global attributes, and standardise "
        "the time axis calendar."
    ),
)
@click.option(
    "--cmor-activity-id",
    default="CMIP",
    show_default=True,
    help="CMIP6 activity_id global attribute (e.g. 'CMIP', 'ScenarioMIP').",
)
@click.option(
    "--cmor-experiment-id",
    default="",
    show_default=False,
    help="CMIP6 experiment_id global attribute (e.g. 'historical', 'ssp585').",
)
@click.option(
    "--cmor-source-id",
    default="",
    show_default=False,
    help="CMIP6 source_id global attribute (e.g. 'ACCESS-CM2', 'UKESM1-0-LL').",
)
@click.option(
    "--cmor-variant-label",
    default="r1i1p1f1",
    show_default=True,
    help="CMIP6 variant_label global attribute (e.g. 'r1i1p1f1').",
)
@click.option(
    "--cmor-institution-id",
    default="",
    show_default=False,
    help="CMIP6 institution_id global attribute (e.g. 'CSIRO-ARCCSS').",
)
@click.option(
    "--cmor-calendar",
    default="",
    show_default=False,
    help=(
        "Target calendar for time axis encoding.  "
        "One of: '360_day', '365_day', 'proleptic_gregorian', or '' to preserve "
        "the source calendar."
    ),
)
@click.option(
    "--cmor-drop-unmapped/--no-cmor-drop-unmapped",
    default=False,
    show_default=True,
    help=(
        "Drop variables that have no CMIP6 name mapping.  "
        "Default is to keep them with their original names and emit a warning."
    ),
)
# ---- run control ----
@click.option(
    "--dry-run", is_flag=True, help="List files that would be processed, then exit."
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors.")
@click.option(
    "--output-stats",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write conversion statistics to this JSON file.",
)
def main(
    input: Path,
    output: Path,
    save_config: Path | None,
    compression: str,
    compression_level: int,
    dask: bool,
    workers: int,
    memory_limit: str,
    threads_per_worker: int,
    workload_type: str,
    adaptive: bool,
    min_workers: int,
    reserve_memory_gb: float,
    suggest_chunks: bool,
    scheduler_file: str | None,
    scheduler_address: str | None,
    chunk_strategy: str,
    validate_cf: bool,
    simple_names: bool,
    nomask: bool,
    hcrit: float,
    stash: str | None,
    pattern: str,
    # Phase 2 additions
    append: bool,
    append_dim: str,
    resume: bool,
    on_error: str,
    ensemble_dim: str | None,
    ensemble_member_pattern: str | None,
    max_graph_size_mb: float,
    write_catalogue: Path | None,
    catalogue_format: str,
    cmor: bool,
    cmor_activity_id: str,
    cmor_experiment_id: str,
    cmor_source_id: str,
    cmor_variant_label: str,
    cmor_institution_id: str,
    cmor_calendar: str,
    cmor_drop_unmapped: bool,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
    output_stats: Path | None,
) -> None:
    """Convert Unified Model fieldsfiles to cloud-optimised Zarr stores.

    \b
    INPUT   UM file or directory of UM files.
    OUTPUT  Zarr store path (single file) or output directory (batch).

    \b
    Examples
    --------
      # Single file, defaults
      um2zarr input.pp output.zarr

      # Batch from directory, I/O-optimised, 8 workers
      um2zarr /data/um/ /data/zarr/ --workload-type io --workers 8

      # Load from config file, override one value on the fly
      um2zarr input.pp output.zarr --config job.yaml --compression-level 6

      # Save current flags as a reusable config template
      um2zarr input.pp output.zarr --workers 16 --save-config job.yaml
    """
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive.")

    configure_logging(verbose, quiet)

    # --save-config: serialise effective config and exit
    if save_config:
        cfg_dict = _build_config_dict(
            compression=compression,
            compression_level=compression_level,
            dask=dask,
            workers=workers,
            memory_limit=memory_limit,
            threads_per_worker=threads_per_worker,
            workload_type=workload_type,
            adaptive=adaptive,
            min_workers=min_workers,
            reserve_memory_gb=reserve_memory_gb,
            suggest_chunks=suggest_chunks,
            scheduler_file=scheduler_file,
            scheduler_address=scheduler_address,
            chunk_strategy=chunk_strategy,
            validate_cf=validate_cf,
            simple_names=simple_names,
            nomask=nomask,
            hcrit=hcrit,
            append=append,
            append_dim=append_dim,
            resume=resume,
            on_error=on_error,
            ensemble_dim=ensemble_dim,
            ensemble_member_pattern=ensemble_member_pattern,
            max_graph_size_mb=max_graph_size_mb,
            catalogue_path=str(write_catalogue) if write_catalogue else None,
            catalogue_format=catalogue_format,
            cmor=cmor,
            cmor_activity_id=cmor_activity_id,
            cmor_experiment_id=cmor_experiment_id,
            cmor_source_id=cmor_source_id,
            cmor_variant_label=cmor_variant_label,
            cmor_institution_id=cmor_institution_id,
            cmor_calendar=cmor_calendar,
            cmor_drop_unmapped=cmor_drop_unmapped,
        )
        save_config.parent.mkdir(parents=True, exist_ok=True)
        with open(save_config, "w") as fh:
            yaml.dump(cfg_dict, fh, default_flow_style=False, sort_keys=True)
        click.echo(f"Config written to {save_config}")
        return

    # Parse STASH codes
    stash_codes: list[int] | None = None
    if stash:
        try:
            stash_codes = [int(s.strip()) for s in stash.split(",")]
        except ValueError as e:
            raise click.BadParameter(
                "STASH codes must be comma-separated integers, e.g. '4,2,3'",
                param_hint="--stash",
            ) from e

    if not HAS_ZARR:
        raise click.ClickException(
            "Zarr library not found. Install with: pip install zarr"
        )

    use_dask = dask and HAS_DASK and HAS_DASK_MANAGER
    if dask and not use_dask:
        missing = []
        if not HAS_DASK:
            missing.append("dask")
        if not HAS_DASK_MANAGER:
            missing.append(
                "um2zarr.orchestration.dask_integration "
                "(check dask[distributed] is installed)"
            )
        logger.warning(
            f"Dask dependencies missing ({', '.join(missing)}) "
            f"— falling back to sequential processing"
        )

    config = ProcessingConfig(
        compression=compression,
        compression_level=compression_level,
        use_dask=use_dask,
        n_workers=workers,
        memory_limit=memory_limit,
        threads_per_worker=threads_per_worker,
        chunk_strategy=chunk_strategy,
        validate_cf=validate_cf,
        simple_names=simple_names,
        nomask=nomask,
        hcrit=hcrit,
        workload_type=workload_type,
        adaptive_scaling=adaptive,
        min_workers=min_workers,
        scheduler_file=scheduler_file,
        scheduler_address=scheduler_address,
        reserve_memory_gb=reserve_memory_gb,
        suggest_chunks=suggest_chunks,
        # Phase 2
        append=append,
        append_dim=append_dim,
        resume=resume,
        on_error=on_error,
        ensemble_dim=ensemble_dim,
        ensemble_member_pattern=ensemble_member_pattern,
        max_graph_size_mb=max_graph_size_mb,
        catalogue_path=str(write_catalogue) if write_catalogue else None,
        catalogue_format=catalogue_format,
        cmor=cmor,
        cmor_activity_id=cmor_activity_id,
        cmor_experiment_id=cmor_experiment_id,
        cmor_source_id=cmor_source_id,
        cmor_variant_label=cmor_variant_label,
        cmor_institution_id=cmor_institution_id,
        cmor_calendar=cmor_calendar,
        cmor_drop_unmapped=cmor_drop_unmapped,
    )

    logger.info("Initializing UM to Zarr converter...")
    orchestrator = ConversionOrchestrator(config)

    input_files = orchestrator.list_files(input, pattern)
    if not input_files:
        raise click.ClickException(
            f"No files found matching pattern '{pattern}' in {input}"
        )

    if dry_run:
        click.echo(f"DRY RUN — {len(input_files)} file(s) would be processed:")
        for f in input_files:
            click.echo(f"  {f}")
        return

    start_time = time.time()

    if ensemble_dim:
        logger.info(f"Ensemble conversion mode — dim='{ensemble_dim}'")
        ens_result = orchestrator.convert_ensemble(
            member_paths=input_files,
            output_path=output,
            ensemble_dim=ensemble_dim,
            member_id_pattern=ensemble_member_pattern,
            stash_codes=stash_codes,
        )
        results = [ens_result]
    elif len(input_files) == 1:
        if append:
            logger.info("Single file append mode")
            results = [
                orchestrator.convert_file_append(
                    input_files[0],
                    output,
                    append_dim=append_dim,
                    stash_codes=stash_codes,
                )
            ]
        else:
            logger.info("Single file conversion mode")
            results = [orchestrator.convert_file(input_files[0], output, stash_codes)]
    else:
        logger.info("Batch conversion mode")
        results = orchestrator.convert_batch(input_files, output, stash_codes)

    total_time = time.time() - start_time
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    click.echo(
        f"\n🎉 Done in {total_time:.1f}s — "
        f"{len(successful)} succeeded, {len(failed)} failed."
    )

    if output_stats:
        stats_data = {
            "total_time_seconds": total_time,
            "files_processed": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "results": results,
        }
        output_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(output_stats, "w") as fh:
            json.dump(stats_data, fh, indent=2, default=str)
        logger.info(f"Statistics written to {output_stats}")

    sys.exit(1 if failed else 0)


@click.command(
    name="um2zarr-rechunk",
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 100},
)
@click.argument("store", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--target",
    type=click.Choice(["timeseries", "map", "profile"]),
    default="map",
    show_default=True,
    help=(
        "Chunk layout preset: 'timeseries' maximises the time axis, "
        "'map' maximises horizontal extent per time step, "
        "'profile' maximises vertical resolution per point."
    ),
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Destination Zarr store.  Defaults to "
        "<store>.rechunked.zarr in the same directory."
    ),
)
@click.option(
    "--max-mem",
    default="2GB",
    show_default=True,
    help="Memory budget for the rechunker staging step (e.g. '4GB', '512MB').",
)
@click.option(
    "--time-chunk",
    type=click.IntRange(min=1),
    default=24,
    show_default=True,
    help="Time steps per chunk (timeseries preset).",
)
@click.option(
    "--spatial-chunk",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
    help="Grid points per horizontal chunk edge (map preset).",
)
@click.option(
    "--level-chunk",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Vertical levels per chunk (profile preset).",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Delete and replace the output store if it already exists.",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all output except errors.")
def rechunk_cmd(
    store: Path,
    target: str,
    output: Path | None,
    max_mem: str,
    time_chunk: int,
    spatial_chunk: int,
    level_chunk: int,
    overwrite: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Rechunk an existing um2zarr Zarr store for a different access pattern.

    \b
    STORE   Path to the Zarr store to rechunk (must already exist).

    \b
    Examples
    --------
      # Rechunk for fast horizontal map plots
      um2zarr-rechunk output.zarr --target map

      # Rechunk for point-extraction time series
      um2zarr-rechunk output.zarr --target timeseries --output ts.zarr

      # Large store — give the rechunker 8 GB of staging memory
      um2zarr-rechunk output.zarr --target profile --max-mem 8GB
    """
    if verbose and quiet:
        raise click.UsageError("--verbose and --quiet are mutually exclusive.")
    configure_logging(verbose, quiet)

    from ..processing.rechunker import RechunkTarget, rechunk_store

    click.echo(f"Rechunking {store} with preset '{target}'...")
    result = rechunk_store(
        store_path=store,
        target=RechunkTarget(target),
        output_path=output,
        max_mem=max_mem,
        time_chunk=time_chunk,
        spatial_chunk=spatial_chunk,
        level_chunk=level_chunk,
        overwrite=overwrite,
    )

    if result.get("success"):
        click.echo(
            f"Done — rechunked store written to: {result['output_path']} "
            f"(backend: {result.get('backend', 'unknown')})"
        )
    else:
        raise click.ClickException(
            f"Rechunking failed: {result.get('error', 'unknown error')}"
        )


if __name__ == "__main__":
    main()

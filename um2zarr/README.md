# um2zarr Module Reference

This document provides detailed information about each file in the `um2zarr/` directory, including their purpose, key classes and functions, dependencies, and role in the overall architecture.

## Package Structure Overview

```text
um2zarr/
├── __init__.py                 # Main package exports
├── core/                       # Core data structures and configuration
│   ├── __init__.py
│   └── data_structures.py
├── io/                         # Input/Output layer
│   ├── __init__.py
│   ├── um_reader.py
│   ├── um_reader_simple.py
│   └── zarr_writer.py
├── orchestration/              # Workflow orchestration and CLI
│   ├── __init__.py
│   ├── cli.py
│   ├── dask_integration.py
│   └── dask_workers.py
├── processing/                 # Data processing and transformation
│   ├── __init__.py
│   ├── stash_metadata.py
│   ├── iris_converter.py
│   ├── chunk_manager.py
│   ├── coordinate_processor.py
│   ├── dtype_optimizer.py
│   ├── masking_engine.py
│   └── data_processing.py
└── utils/                      # Utility functions
    ├── __init__.py
    └── numpy_compat.py
```

## Detailed File Documentation

### Main Package

#### `__init__.py`

**Purpose**: Main package initialization and public API definition

**Key Exports**:

- `IrisToXarrayConverter`: Core conversion functionality
- `StashMetadataManager`: STASH code to CF metadata mapping
- `UMFileReader`: UM file input interface
- `GridMetadata`: UM grid metadata container

**Dependencies**: Core processing modules (iris_converter, stash_metadata, um_reader, data_structures)

**Role**: Defines the public API that users interact with when using um2zarr as a library

---

### Core Module (`core/`)

#### `core/__init__.py`

**Purpose**: Core module initialization (currently empty)

#### `core/data_structures.py`

**Purpose**: Fundamental data types and configuration for the entire pipeline

**Key Classes**:

- `GridMetadata`: UM grid information (grid type, spacing, vertical levels)
- `StashMetadata`: CF-compliant metadata for STASH codes  
- `ProcessingConfig`: Pipeline configuration (compression, workers, chunking)
- `ConversionResult`: Iris→Xarray conversion results and statistics
- `ProcessingResult`: Dataset processing results and metadata

**Key Exceptions**:

- `UM2ZarrError`: Base exception class
- `UMFileError`, `GridMetadataError`, `ConversionError`, `ProcessingError`, `ZarrWriteError`, `ValidationError`: Specific error types

**Dependencies**: numpy, dataclasses, typing, pathlib

**Role**: Foundation layer providing type-safe data structures used throughout the pipeline. Includes validation logic and error handling.

---

### I/O Layer (`io/`)

#### `io/__init__.py`

**Purpose**: I/O layer initialization with fallback dependency handling

**Key Features**:

- Automatic fallback from full UM reader (requires Mule) to simple reader
- Conditional imports based on available dependencies
- Compatibility handling for numpy 2.x issues

**Dependencies**: Conditional imports for Mule, Zarr

#### `io/um_reader.py`

**Purpose**: Full-featured UM file reader using Iris and Mule for complete metadata extraction

**Key Classes**:

- `UMFileReader`: Primary UM file reading interface

**Key Methods**:

- `load_file()`: Load UM file and extract all metadata
- `extract_grid_metadata()`: Extract grid information using Mule
- `filter_cubes_by_stash()`: Filter cubes by STASH codes
- `get_cube_summary()`: Debugging/logging information

**Dependencies**: iris, mule, numpy compatibility layer

**Role**: Provides the most complete UM file reading capability with full metadata extraction. Handles UM-specific format details and grid information.

#### `io/um_reader_simple.py`  

**Purpose**: Simplified UM file reader that works without Mule (fallback implementation)

**Key Classes**:

- `UMFileReader`: Simplified reader interface (same API as full reader)

**Key Methods**:

- `load_file()`: Load UM file using only Iris
- `extract_grid_metadata_from_cubes()`: Extract basic grid info from cubes
- `filter_cubes_by_stash()`: STASH code filtering
- `get_cube_summary()`: Debugging information

**Dependencies**: iris only (no Mule required)

**Role**: Backup implementation for environments where Mule is not available. Provides basic functionality with reduced metadata extraction capabilities.

#### `io/zarr_writer.py`

**Purpose**: High-performance Zarr output with CF compliance and multiple storage backends

**Key Classes**:

- `EncodingStrategy`: Configuration for compression and chunking
- `ZarrWriter`: Main Zarr writing interface

**Key Methods**:

- `write_dataset()`: Write Xarray dataset to Zarr with optimization
- `_create_encoding()`: Generate optimal encoding configuration
- `_get_storage_mapper()`: Storage backend abstraction (local, S3, GCS)

**Key Features**:

- Advanced compression algorithms (zstd, blosc, lz4, gzip, bz2)
- CF compliance validation
- Cloud storage integration (S3, Google Cloud)
- Time series append functionality
- Automatic chunking optimization

**Dependencies**: zarr, xarray, numcodecs, optional storage backends (s3fs, gcsfs)

**Role**: Final output stage providing cloud-optimized Zarr format with maximum compression and performance.

---

### Orchestration Layer (`orchestration/`)

#### `orchestration/__init__.py`

**Purpose**: Orchestration layer exports

**Key Exports**: Dask integration components for parallel processing

#### `orchestration/cli.py`

**Purpose**: CLI entry point for installed um2zarr package

**Key Functions**:

- `main()`: Entry point that imports and runs convert_to_zarr.py

**Role**: Bridge between pip-installed package and main CLI script

#### `orchestration/dask_integration.py`

**Purpose**: Dask cluster management and HPC optimization

**Key Classes**:

- `DaskConfig`: Dask cluster configuration with dask_setup integration
- `DaskClusterManager`: Intelligent cluster setup and management

**Key Features**:

- HPC-optimized cluster setup via dask_setup integration
- Automatic resource detection and optimal worker topology
- Support for external clusters (PBS, SLURM)
- Adaptive scaling and memory management
- Workload-aware optimization (CPU/I/O/mixed)

**Key Methods**:

- `setup_cluster()`: Create optimized Dask cluster
- `_create_cluster_with_dask_setup()`: HPC-optimized cluster creation
- `_create_local_cluster()`: Fallback local cluster

**Dependencies**: dask, distributed, optional dask_setup, psutil

**Role**: Provides high-performance parallel processing with automatic resource optimization for HPC environments.

#### `orchestration/dask_workers.py`

**Purpose**: Standalone functions for Dask-based parallel processing

**Key Functions**:

- `convert_file_standalone()`: Complete file conversion for Dask workers
- `convert_batch_with_optimization()`: Batch processing with graph optimization

**Key Features**:

- Serializable functions for distributed processing
- Graph size monitoring to prevent memory issues
- Lazy imports to reduce serialization overhead
- Complete error handling and statistics

**Dependencies**: Lazy imports of all processing modules

**Role**: Worker functions that execute the actual UM→Zarr conversion in parallel, designed for optimal Dask serialization and performance.

---

### Processing Layer (`processing/`)

#### `processing/__init__.py`

**Purpose**: Processing core module exports

**Key Exports**: All processing components for data transformation pipeline

#### `processing/stash_metadata.py`

**Purpose**: STASH code to CF-compliant metadata mapping

**Key Classes**:

- `StashMetadataManager`: Main metadata management interface

**Key Methods**:

- `get_metadata()`: Retrieve CF metadata for STASH code
- `get_variable_name()`: Generate CF-compliant variable names
- `get_cf_attributes()`: Extract CF attributes dictionary
- `_load_legacy_mappings()`: Load comprehensive STASH mappings

**Key Features**:

- Comprehensive STASH code database from legacy stashvar_cmip6.py
- CF-compliant variable naming
- Custom mapping overlay support
- Statistics tracking for coverage analysis

**Dependencies**: Legacy STASH mappings, core data structures

**Role**: Critical metadata bridge converting UM-specific STASH codes to modern CF-compliant variable names and attributes.

#### `processing/iris_converter.py`

**Purpose**: Convert Iris cubes to Xarray datasets preserving all metadata

**Key Classes**:

- `IrisToXarrayConverter`: Primary conversion interface

**Key Methods**:

- `convert_cubes_to_dataset()`: Convert multiple cubes to single dataset
- `convert_cube_to_dataarray()`: Convert individual cube to DataArray
- `_generate_variable_name()`: Create appropriate variable names
- `_create_global_attributes()`: Generate dataset-level attributes

**Key Features**:

- Preserves lazy evaluation with Dask integration
- Handles coordinate conflicts with override strategies
- Maintains all UM-specific metadata
- CF compliance enforcement

**Dependencies**: iris, xarray, numpy, STASH metadata manager

**Role**: Core transformation engine that bridges UM-specific Iris format with general-purpose Xarray format while preserving all scientific metadata.

#### `processing/chunk_manager.py`

**Purpose**: Intelligent chunking strategies for optimal Dask and Zarr performance

**Key Classes**:

- `ChunkingStrategy`: Chunking configuration
- `ChunkManager`: Main chunking optimization interface

**Key Methods**:

- `calculate_optimal_chunks()`: Determine optimal chunk sizes
- `_calculate_chunks_with_dask_setup()`: HPC-optimized chunking via dask_setup
- `_calculate_chunks_manually()`: Fallback manual calculation

**Key Features**:

- Workload-aware optimization (CPU/I/O/mixed workloads)
- Storage backend optimization (local vs cloud)
- Memory constraint awareness
- dask_setup integration for HPC environments
- Zarr performance best practices

**Dependencies**: xarray, dask, optional dask_setup

**Role**: Performance optimization layer that determines optimal data chunking strategies based on workload type, storage backend, and available resources.

#### `processing/coordinate_processor.py`

**Purpose**: CF-compliant coordinate processing (stub implementation)

**Key Classes**:

- `CoordinateProcessor`: Coordinate standardization interface

**Role**: Placeholder for coordinate standardization and CF compliance fixes.

#### `processing/dtype_optimizer.py`

**Purpose**: Data type optimization (stub implementation)

**Key Classes**:

- `DtypeOptimizer`: Data type optimization interface  

**Role**: Placeholder for automatic data type optimization to reduce storage size while preserving precision.

#### `processing/masking_engine.py`

**Purpose**: Data masking operations (stub implementation)

**Key Classes**:

- `MaskingEngine`: Masking interface

**Role**: Placeholder for pressure level masking and other data masking operations.

#### `processing/data_processing.py`

**Purpose**: Data processing utilities reproducing legacy um2netcdf4.py behavior

**Key Classes**:

- `MaskingOptions`: Configuration for masking operations
- `DataTypeOptions`: Configuration for data type handling

**Key Functions**:

- `apply_heaviside_mask()`: Pressure level masking using heaviside functions
- `remove_all_masks()`: Remove all masks from data arrays
- `coerce_data_types()`: Apply data type coercion following legacy logic

**Key Features**:

- Heaviside function masking for pressure levels
- Legacy um2netcdf4.py behavior reproduction
- Configurable masking and type coercion
- Dask array support

**Dependencies**: numpy, xarray, pandas, netCDF4, optional dask

**Role**: Provides specific data processing operations that reproduce legacy conversion behavior while using modern Xarray operations.

---

### Utilities (`utils/`)

#### `utils/__init__.py`

**Purpose**: Utility module exports

**Key Exports**: NumPy compatibility functions

#### `utils/numpy_compat.py`

**Purpose**: NumPy compatibility layer for legacy dependencies

**Key Functions**:

- `ensure_numpy_compatibility()`: Add compatibility shims for deprecated NumPy functions
- `check_numpy_mule_compatibility()`: Verify NumPy/Mule compatibility

**Key Features**:

- Automatic numpy.product → numpy.prod mapping for NumPy 2.x
- Compatibility detection and warnings
- Support for legacy libraries like Mule

**Dependencies**: numpy, logging

**Role**: Critical compatibility layer that allows the package to work with both legacy dependencies (Mule) and modern NumPy versions, addressing breaking changes in NumPy 2.x.

---

## Architecture Integration

### Data Flow Through Modules

1. **Entry Point**: `__init__.py` or CLI via `orchestration/cli.py`
2. **Configuration**: `core/data_structures.py` provides type-safe configuration
3. **File Reading**: `io/um_reader*.py` loads UM files with metadata
4. **Metadata Processing**: `processing/stash_metadata.py` maps STASH codes to CF names
5. **Format Conversion**: `processing/iris_converter.py` converts Iris cubes to Xarray
6. **Data Processing**: `processing/data_processing.py` applies masking and type optimization
7. **Performance Optimization**: `processing/chunk_manager.py` optimizes chunking
8. **Parallel Execution**: `orchestration/dask_*` manages parallel processing
9. **Output Writing**: `io/zarr_writer.py` writes optimized Zarr stores

### Key Design Patterns

- **Layered Architecture**: Clear separation between I/O, processing, and orchestration
- **Fallback Systems**: Graceful degradation when optional dependencies unavailable
- **Type Safety**: Extensive use of dataclasses and type hints
- **Error Handling**: Comprehensive exception hierarchy with specific error types
- **Performance Focus**: HPC optimization through dask_setup integration
- **Cloud-Ready**: Native support for multiple storage backends
- **Legacy Compatibility**: Careful handling of deprecated dependencies

This modular design ensures maintainability, testability, and extensibility while providing high-performance UM-to-Zarr conversion with modern cloud-computing capabilities.

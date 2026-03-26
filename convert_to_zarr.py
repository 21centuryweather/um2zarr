#!/usr/bin/env python3
"""
UM to Zarr Conversion CLI — compatibility shim.

All logic lives inside the package at um2zarr/orchestration/cli.py, which is
also the entry point used by the `um2zarr` command installed by pip.

This file is kept so that `python convert_to_zarr.py` still works for users
who invoke the script directly from the repository root.
"""

from um2zarr.orchestration.cli import (  # noqa: F401
    ConversionOrchestrator,
    DataProcessingPipeline,
    configure_logging,
    main,
)

if __name__ == "__main__":
    main()

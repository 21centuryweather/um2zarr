"""
Checkpoint management for resumable batch conversions.

Writes a JSON sidecar file (.um2zarr_progress.json) alongside the output
directory so that interrupted batch jobs can skip already-processed files
on re-invocation.

Usage::

    mgr = CheckpointManager(output_dir)
    for f in files:
        if mgr.is_complete(f):
            continue
        try:
            stats = convert(f)
            mgr.mark_complete(f, stats)
        except Exception as exc:
            mgr.mark_failed(f, str(exc))
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = ".um2zarr_progress.json"


class CheckpointManager:
    """
    Track which input files have been successfully processed in a batch job.

    The checkpoint is written to ``<output_dir>/.um2zarr_progress.json``.
    An existing file from a previous (interrupted) run is loaded
    automatically so that ``is_complete()`` reflects prior work.
    """

    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = output_dir / CHECKPOINT_FILENAME
        self._data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict[str, Any]:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path) as fh:
                    data = json.load(fh)
                n_done = len(data.get("completed", {}))
                n_fail = len(data.get("failed", {}))
                logger.info(
                    f"Checkpoint loaded: {n_done} completed, {n_fail} previously failed"
                )
                return data
            except Exception as exc:
                logger.warning(
                    f"Could not read checkpoint file ({exc}) — starting fresh"
                )
        return {"completed": {}, "failed": {}}

    def _save(self) -> None:
        try:
            with open(self.checkpoint_path, "w") as fh:
                json.dump(self._data, fh, indent=2, default=str)
        except Exception as exc:
            logger.warning(f"Could not write checkpoint: {exc}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_complete(self, input_path: Path) -> bool:
        """Return True if *input_path* was successfully processed in a prior run."""
        return str(input_path) in self._data["completed"]

    def mark_complete(
        self,
        input_path: Path,
        stats: dict[str, Any] | None = None,
    ) -> None:
        """Record *input_path* as successfully processed and persist the checkpoint."""
        self._data["completed"][str(input_path)] = {
            "timestamp": time.time(),
            "stats": stats or {},
        }
        # Remove from failed if it was there
        self._data["failed"].pop(str(input_path), None)
        self._save()

    def mark_failed(self, input_path: Path, error: str) -> None:
        """Record *input_path* as failed and persist the checkpoint."""
        self._data["failed"][str(input_path)] = {
            "timestamp": time.time(),
            "error": error,
        }
        self._save()

    @property
    def completed_files(self) -> set[str]:
        """Set of absolute path strings that have been successfully processed."""
        return set(self._data["completed"].keys())

    @property
    def failed_files(self) -> set[str]:
        """Set of absolute path strings whose last attempt failed."""
        return set(self._data["failed"].keys())

    def summary(self) -> dict[str, int]:
        """Return a dict with ``completed`` and ``failed`` counts."""
        return {
            "completed": len(self._data["completed"]),
            "failed": len(self._data["failed"]),
        }

    def reset_failed(self) -> int:
        """Clear the failed list so those files will be retried.  Returns count cleared."""
        n = len(self._data["failed"])
        self._data["failed"] = {}
        self._save()
        return n

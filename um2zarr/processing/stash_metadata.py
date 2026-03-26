"""
STASH metadata management for UM fields.

Provides mapping from STASH codes to CF-compliant metadata including
variable names, units, standard names, and long descriptions.

This implementation uses the comprehensive STASH mappings from the legacy
stashvar_cmip6.py but wraps them in a modern, efficient interface.
"""

import logging
import warnings
from typing import Any

from ..core.data_structures import StashMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supplemental STASH-to-CF mappings
# ---------------------------------------------------------------------------
# These entries cover priority variables not yet present in the legacy
# stashvar_cmip6.py tables.  Each key is the integer STASH code
# (section * 1000 + item); the value is a StashMetadata instance.
#
# Sections covered:
#   Section 0  — Prognostic fields on theta/rho levels (items 1–3)
#   Section 1  — Shortwave and longwave radiation diagnostics
#   Section 3  — Boundary-layer diagnostics
#   Section 30 — Heaviside functions (already used internally, exposed here)
#
# Conventions: ``positive`` is stored in the ``unique_name`` field with the
# prefix "positive:" to stay within the existing dataclass schema until a
# future refactor adds dedicated fields.


def _sm(
    long_name: str,
    cf_name: str,
    units: str,
    standard_name: str = "",
    unique_name: str = "",
) -> StashMetadata:
    return StashMetadata(
        long_name=long_name,
        cf_name=cf_name,
        units=units,
        standard_name=standard_name,
        unique_name=unique_name,
    )


_SUPPLEMENTAL_STASH: dict[int, StashMetadata] = {
    # --- Section 0: prognostic fields (theta/rho levels) ---
    1: _sm(
        "Potential temperature",
        "theta",
        "K",
        standard_name="air_potential_temperature",
    ),
    2: _sm(
        "Specific humidity",
        "q",
        "1",
        standard_name="specific_humidity",
    ),
    3: _sm(
        "Wind component u",
        "u",
        "m s-1",
        standard_name="eastward_wind",
    ),
    # --- Section 1: radiation diagnostics ---
    1201: _sm(
        "Net downward shortwave flux at surface",
        "rsds",
        "W m-2",
        standard_name="surface_downwelling_shortwave_flux_in_air",
        unique_name="positive:down",
    ),
    1202: _sm(
        "Net downward longwave flux at surface",
        "rlds",
        "W m-2",
        standard_name="surface_downwelling_longwave_flux_in_air",
        unique_name="positive:down",
    ),
    1207: _sm(
        "Outgoing shortwave radiation at top of atmosphere",
        "rsut",
        "W m-2",
        standard_name="toa_outgoing_shortwave_flux",
        unique_name="positive:up",
    ),
    1208: _sm(
        "Outgoing longwave radiation at top of atmosphere",
        "rlut",
        "W m-2",
        standard_name="toa_outgoing_longwave_flux",
        unique_name="positive:up",
    ),
    1235: _sm(
        "Total downward shortwave flux at surface",
        "rsdscs",
        "W m-2",
        standard_name="surface_downwelling_shortwave_flux_in_air",
    ),
    # --- Section 3: boundary-layer diagnostics ---
    3217: _sm(
        "Surface sensible heat flux",
        "hfss",
        "W m-2",
        standard_name="surface_upward_sensible_heat_flux",
        unique_name="positive:up",
    ),
    3234: _sm(
        "Surface latent heat flux",
        "hfls",
        "W m-2",
        standard_name="surface_upward_latent_heat_flux",
        unique_name="positive:up",
    ),
    3209: _sm(
        "Boundary layer height",
        "zblh",
        "m",
        standard_name="atmosphere_boundary_layer_thickness",
    ),
    3225: _sm(
        "Surface friction velocity",
        "ustar",
        "m s-1",
        standard_name="atmosphere_boundary_layer_thickness",  # proxy standard name
    ),
    # --- Section 30: heaviside functions ---
    30301: _sm(
        "Heaviside function on theta levels",
        "heaviside_t",
        "1",
    ),
    30302: _sm(
        "Heaviside function on UV levels",
        "heaviside_uv",
        "1",
    ),
}


class StashMetadataManager:
    """Manage STASH code to CF metadata mappings."""

    def __init__(
        self,
        custom_mappings: dict[int, StashMetadata] | None = None,
        load_legacy_mappings: bool = True,
        warn_on_missing: bool = True,
    ):
        """
        Initialize STASH metadata manager.

        Args:
            custom_mappings: Optional custom STASH mappings to overlay on top
                of the built-in tables.  Keys are integer STASH codes.
            load_legacy_mappings: Whether to load the comprehensive legacy
                stashvar_cmip6.py mappings.
            warn_on_missing: If True, emit a Python warning (via the standard
                ``warnings`` module) the first time a STASH code without any
                mapping is encountered.  This makes CI checks easy — convert
                warnings to errors with ``-W error``.
        """
        self.warn_on_missing = warn_on_missing
        self.mappings: dict[int, StashMetadata] = {}
        self._load_stats: dict[str, Any] = {
            "total_loaded": 0,
            "with_cf_names": 0,
            "with_standard_names": 0,
            "missing_mappings": set(),
        }

        # Load supplemental built-in mappings first (lower priority)
        self.mappings.update(_SUPPLEMENTAL_STASH)

        if load_legacy_mappings:
            self._load_legacy_mappings()

        if custom_mappings:
            self.mappings.update(custom_mappings)
            logger.info(f"Added {len(custom_mappings)} custom STASH mappings")

    def _load_legacy_mappings(self) -> None:
        """Load mappings from legacy stashvar_cmip6.py format."""
        try:
            # Import the legacy STASH variable mappings
            import sys
            from pathlib import Path

            # Add the legacy directory to the path
            legacy_path = Path(__file__).parent.parent.parent / "legacy"
            sys.path.insert(0, str(legacy_path))

            # Import the legacy STASH mappings
            from stashvar_cmip6 import atm_stashvar

            # Remove the legacy path from sys.path
            sys.path.remove(str(legacy_path))

            for itemcode, legacy_data in atm_stashvar.items():
                try:
                    # legacy_data format: [long_name, cf_name, units, standard_name, unique_name]
                    if len(legacy_data) >= 4:
                        long_name = legacy_data[0].strip() if legacy_data[0] else None
                        cf_name = legacy_data[1].strip() if legacy_data[1] else None
                        units = legacy_data[2].strip() if legacy_data[2] else None
                        standard_name = (
                            legacy_data[3].strip() if legacy_data[3] else None
                        )
                        unique_name = (
                            legacy_data[4].strip()
                            if len(legacy_data) > 4 and legacy_data[4]
                            else None
                        )

                        # Create StashMetadata object
                        metadata = StashMetadata(
                            long_name=long_name,
                            cf_name=cf_name,
                            units=units,
                            standard_name=standard_name,
                            unique_name=unique_name,
                        )

                        self.mappings[itemcode] = metadata

                        # Update statistics
                        self._load_stats["total_loaded"] += 1
                        if cf_name:
                            self._load_stats["with_cf_names"] += 1
                        if standard_name:
                            self._load_stats["with_standard_names"] += 1

                except Exception as e:
                    logger.warning(f"Failed to load STASH mapping for {itemcode}: {e}")
                    continue

            logger.info(f"Loaded {self._load_stats['total_loaded']} STASH mappings")
            logger.info(f"Variables with CF names: {self._load_stats['with_cf_names']}")
            logger.info(
                f"Variables with standard names: {self._load_stats['with_standard_names']}"
            )

        except ImportError as e:
            logger.warning(f"Could not load legacy STASH mappings: {e}")
            logger.info(
                "Proceeding with empty mappings - only custom mappings will be available"
            )
        except Exception as e:
            logger.error(f"Error loading legacy STASH mappings: {e}")

    def get_metadata(self, stash_code: int) -> StashMetadata | None:
        """
        Get CF metadata for STASH code.

        Args:
            stash_code: STASH code in format (section * 1000 + item)

        Returns:
            StashMetadata object if found, None otherwise
        """
        metadata = self.mappings.get(stash_code)
        if not metadata:
            is_new_miss = stash_code not in self._load_stats["missing_mappings"]
            self._load_stats["missing_mappings"].add(stash_code)
            logger.debug(f"No metadata found for STASH code {stash_code}")
            if self.warn_on_missing and is_new_miss:
                section = stash_code // 1000
                item = stash_code % 1000
                warnings.warn(
                    f"STASH code {stash_code} (s{section:02d}i{item:03d}) has no CF mapping. "
                    "Output will use the fallback 'fld_sXXiYYY' name with no units, "
                    "which violates CF conventions.  Add an entry to "
                    "_SUPPLEMENTAL_STASH in stash_metadata.py or pass custom_mappings.",
                    stacklevel=2,
                )
        return metadata

    def get_variable_name(
        self,
        stash_code: int,
        simple: bool = False,
        has_max: bool = False,
        has_min: bool = False,
    ) -> str | None:
        """
        Generate appropriate variable name for STASH code.

        Args:
            stash_code: STASH code in format (section * 1000 + item)
            simple: If True, return simple name like "fld_s01i002"
            has_max: If True, append "_max" suffix
            has_min: If True, append "_min" suffix

        Returns:
            Variable name string if possible, None otherwise
        """
        if simple:
            section = stash_code // 1000
            item = stash_code % 1000
            return f"fld_s{section:02d}i{item:03d}"

        metadata = self.get_metadata(stash_code)
        if metadata and metadata.cf_name:
            var_name = metadata.cf_name

            # Add statistical suffixes
            if has_max:
                var_name += "_max"
            if has_min:
                var_name += "_min"

            return self._clean_variable_name(var_name)

        return None

    def get_cf_attributes(self, stash_code: int) -> dict[str, Any]:
        """
        Get CF-compliant attributes for a STASH code.

        Args:
            stash_code: STASH code in format (section * 1000 + item)

        Returns:
            Dictionary of CF attributes
        """
        metadata = self.get_metadata(stash_code)
        if not metadata:
            return {}

        attrs = {}

        if metadata.long_name:
            attrs["long_name"] = metadata.long_name
        if metadata.units:
            attrs["units"] = metadata.units
        if metadata.standard_name:
            attrs["standard_name"] = metadata.standard_name

        return attrs

    def _clean_variable_name(self, name: str) -> str:
        """
        Clean variable name to be CF-compliant.

        Args:
            name: Raw variable name

        Returns:
            Cleaned variable name
        """
        if not name:
            return ""

        # Replace spaces and hyphens with underscores
        cleaned = name.replace(" ", "_").replace("-", "_")

        # Remove non-alphanumeric characters except underscores
        cleaned = "".join(c for c in cleaned if c.isalnum() or c == "_")

        # Ensure it starts with a letter
        if cleaned and not cleaned[0].isalpha():
            cleaned = "var_" + cleaned

        # Convert to lowercase for consistency
        return cleaned.lower()

    def search_by_name(self, search_term: str) -> list[int]:
        """
        Search for STASH codes by variable or long name.

        Args:
            search_term: Term to search for (case insensitive)

        Returns:
            List of matching STASH codes
        """
        search_term = search_term.lower()
        matches = []

        for stash_code, metadata in self.mappings.items():
            if (
                (search_term in metadata.cf_name.lower() if metadata.cf_name else False)
                or (
                    search_term in metadata.long_name.lower()
                    if metadata.long_name
                    else False
                )
                or (
                    search_term in metadata.standard_name.lower()
                    if metadata.standard_name
                    else False
                )
            ):
                matches.append(stash_code)

        return matches

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about loaded STASH mappings.

        Returns:
            Dictionary of statistics
        """
        stats = self._load_stats.copy()
        stats["missing_mappings"] = list(stats["missing_mappings"])
        stats["coverage_cf_names"] = (
            stats["with_cf_names"] / max(stats["total_loaded"], 1)
        ) * 100
        stats["coverage_standard_names"] = (
            stats["with_standard_names"] / max(stats["total_loaded"], 1)
        ) * 100
        return stats

    def get_known_stash_codes(self) -> set[int]:
        """
        Get set of all known STASH codes.

        Returns:
            Set of STASH codes that have mappings
        """
        return set(self.mappings.keys())

    def validate_stash_code(self, stash_code: int) -> bool:
        """
        Check if a STASH code has a known mapping.

        Args:
            stash_code: STASH code to validate

        Returns:
            True if mapping exists, False otherwise
        """
        return stash_code in self.mappings

    def get_stash_info(self, stash_code: int) -> dict[str, Any]:
        """
        Get comprehensive information about a STASH code.

        Args:
            stash_code: STASH code to get info for

        Returns:
            Dictionary with all available information
        """
        section = stash_code // 1000
        item = stash_code % 1000

        info = {
            "stash_code": stash_code,
            "section": section,
            "item": item,
            "stash_string": f"{section:02d}i{item:03d}",
            "has_mapping": stash_code in self.mappings,
        }

        if info["has_mapping"]:
            metadata = self.mappings[stash_code]
            info.update(
                {
                    "long_name": metadata.long_name,
                    "cf_name": metadata.cf_name,
                    "units": metadata.units,
                    "standard_name": metadata.standard_name,
                    "unique_name": metadata.unique_name,
                    "has_cf_name": bool(metadata.cf_name),
                    "has_standard_name": bool(metadata.standard_name),
                    "has_units": bool(metadata.units),
                }
            )

        return info

    def report_missing(self) -> list[tuple[int, int, int]]:
        """
        Return a sorted list of STASH codes seen during this session that have
        no CF mapping.

        Each element is a ``(stash_code, section, item)`` tuple.  The list is
        empty if all codes encountered had mappings.

        This method is suitable for use in CI: run the conversion on a
        representative dataset and then call ``report_missing()`` to detect
        new STASH codes before they silently produce non-CF-compliant output.

        Returns:
            Sorted list of ``(stash_code, section, item)`` tuples.
        """
        result: list[tuple[int, int, int]] = []
        for code in sorted(self._load_stats["missing_mappings"]):
            result.append((code, code // 1000, code % 1000))
        return result

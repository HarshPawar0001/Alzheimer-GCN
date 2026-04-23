"""Utility classes and functions for logging metrics to CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class CSVLogger:
    """Simple CSV logger that appends metric dictionaries to a CSV file.

    The logger creates the directory if it does not exist and writes a header
    row on first use based on the keys of the first logged record.
    """

    def __init__(self, log_path: Path) -> None:
        """Initialize the CSVLogger.

        Args:
            log_path: Full path (including filename) of the CSV log file.
        """
        self.log_path: Path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written: bool = self.log_path.is_file() and self._has_content()

    def _has_content(self) -> bool:
        """Check whether the CSV file already has any content."""
        try:
            return self.log_path.stat().st_size > 0
        except FileNotFoundError:
            return False

    def log(self, record: Dict[str, Any]) -> None:
        """Append a single record (row) to the CSV file.

        Args:
            record: Dictionary mapping column names to values.
        """
        try:
            write_header: bool = not self._header_written
            with self.log_path.open(mode="a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(record.keys()))
                if write_header:
                    writer.writeheader()
                    self._header_written = True
                writer.writerow(record)
        except OSError as exc:
            raise RuntimeError(f"Failed to write log record to '{self.log_path}'.") from exc


def create_csv_logger(log_dir: str, filename: str) -> CSVLogger:
    """Create a CSVLogger from a log directory and filename.

    This helper exists so that callers can pass the directory string taken
    directly from the configuration file.

    Args:
        log_dir: Directory where log files should be stored (from config).
        filename: Name of the CSV file inside the log directory.

    Returns:
        An initialized CSVLogger instance.
    """
    log_path: Path = Path(log_dir) / filename
    return CSVLogger(log_path=log_path)


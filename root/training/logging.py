"""Set up logging and append metrics to CSV."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import IO, Iterable


def setup_logger(
    name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a logger that writes to stdout and (optionally) to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class CSVMetricLogger:
    """Append metrics to a CSV with fixed fields."""

    def __init__(self, path: str | Path, fields: Iterable[str]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fields = list(fields)
        is_new = not path.exists() or path.stat().st_size == 0
        self._fh: IO = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._fh, fieldnames=self.fields, extrasaction="ignore",
        )
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

#!/usr/bin/env python3
"""
run.py — Minimal MLOps-style batch job for a trading-signal pipeline.

Mirrors the shape of a production trading-signal service:
    1. Load + validate a YAML config (seed, window, version)
    2. Load + validate an OHLCV CSV dataset (requires a 'close' column)
    3. Compute a rolling mean on 'close' using the configured window
    4. Generate a binary signal: 1 if close > rolling_mean else 0
    5. Compute metrics (rows_processed, signal_rate, latency_ms)
    6. Write structured metrics JSON + detailed logs (success AND failure)

Usage:
    python run.py --input data.csv --config config.yaml \
        --output metrics.json --log-file run.log

Exit codes:
    0 -> success
    1 -> failure (a metrics JSON with status="error" is still written)

Design notes:
    - Determinism: the configured `seed` is applied via np.random.seed()
      before any processing. The pipeline itself introduces no other
      source of randomness, so repeated runs on the same input/config
      produce byte-identical output aside from `latency_ms`.
    - Rolling-mean edge case: the first (window - 1) rows have no full
      window of history. `rolling_mean` is NaN for those rows and they
      are excluded from `signal` / `signal_rate`, but are still counted
      in `rows_processed` for observability (see compute_signal()).
    - Error taxonomy: every anticipated failure mode (missing/empty
      file, malformed CSV, missing column, malformed config) raises a
      `ValueError` with a human-readable message. `main()` catches the
      broader `Exception` as a safety net, but the specific messages
      always come from a `ValueError` raised deliberately below —
      nothing here relies on catching an unexpected stack trace.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

REQUIRED_CONFIG_FIELDS: Tuple[str, ...] = ("seed", "window", "version")
REQUIRED_DATA_COLUMN = "close"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments. No path is hardcoded — all four are required."""
    parser = argparse.ArgumentParser(
        description="Batch job: rolling-mean trading signal generator."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV (OHLCV data).")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", required=True, help="Path to write metrics JSON.")
    parser.add_argument("--log-file", required=True, help="Path to write the run log.")
    return parser.parse_args(argv)


def setup_logging(log_file: str) -> logging.Logger:
    """Configure a logger that writes to both `log_file` and stdout."""
    logger = logging.getLogger("mlops_task")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicate handlers if main() is re-invoked (e.g. in tests)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_and_validate_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Parse the YAML config and validate required fields/types.

    Raises:
        ValueError: file missing, unparseable YAML, wrong top-level
            structure, missing required field, or a field with the
            wrong type (e.g. `window` not a positive int).
    """
    if not os.path.isfile(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    if not isinstance(config, dict):
        raise ValueError("Invalid config structure: expected a top-level YAML mapping.")

    missing = [field for field in REQUIRED_CONFIG_FIELDS if field not in config]
    if missing:
        raise ValueError(f"Config missing required field(s): {', '.join(missing)}")

    if not isinstance(config["seed"], int):
        raise ValueError("Config field 'seed' must be an integer.")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError("Config field 'window' must be a positive integer.")
    if not isinstance(config["version"], str):
        raise ValueError("Config field 'version' must be a string.")

    logger.info(
        "Config loaded + validated (seed=%s, window=%s, version=%s)",
        config["seed"], config["window"], config["version"],
    )
    return config


def load_and_validate_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load the CSV and validate it is readable, non-empty, and has 'close'.

    Raises:
        ValueError: file missing, empty, unparseable, no rows, missing
            the required 'close' column, or 'close' not coercible to
            numeric.
    """
    if not os.path.isfile(input_path):
        raise ValueError(f"Input file not found: {input_path}")

    if os.path.getsize(input_path) == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file has no parsable data: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {e}")

    if df.empty:
        raise ValueError("Input dataset contains no rows.")

    if REQUIRED_DATA_COLUMN not in df.columns:
        raise ValueError(f"Required column '{REQUIRED_DATA_COLUMN}' not found in input dataset.")

    if not pd.api.types.is_numeric_dtype(df[REQUIRED_DATA_COLUMN]):
        df[REQUIRED_DATA_COLUMN] = pd.to_numeric(df[REQUIRED_DATA_COLUMN], errors="coerce")
        if df[REQUIRED_DATA_COLUMN].isna().all():
            raise ValueError(f"Column '{REQUIRED_DATA_COLUMN}' contains no valid numeric values.")

    logger.info("Rows loaded: %d", len(df))
    return df


def compute_signal(
    df: pd.DataFrame, window: int, logger: logging.Logger
) -> Tuple[pd.DataFrame, float]:
    """Compute the rolling mean on 'close' and derive a binary signal.

    signal[i] = 1 if close[i] > rolling_mean[i] else 0, for rows with a
    full window of history. The first (window - 1) rows have no full
    window, so rolling_mean is NaN there and signal is left NaN too —
    those rows are excluded from signal_rate but still counted in
    rows_processed.

    Raises:
        ValueError: if `window` exceeds the number of rows available,
            which would make signal_rate undefined for every row.
    """
    if window > len(df):
        raise ValueError(
            f"Config 'window' ({window}) exceeds dataset size ({len(df)} rows); "
            "no row would have a full rolling window."
        )

    logger.info("Processing step: computing rolling mean (window=%d)", window)
    df["rolling_mean"] = df[REQUIRED_DATA_COLUMN].rolling(window=window, min_periods=window).mean()

    logger.info("Processing step: generating binary signal (close > rolling_mean)")
    df["signal"] = np.where(df[REQUIRED_DATA_COLUMN] > df["rolling_mean"], 1, 0)
    # Rows without a rolling mean (first window-1 rows) have no valid signal.
    df.loc[df["rolling_mean"].isna(), "signal"] = np.nan

    valid_signals = df["signal"].dropna()
    signal_rate = float(valid_signals.mean()) if len(valid_signals) > 0 else 0.0
    return df, signal_rate


def write_metrics(output_path: str, metrics: Dict[str, Any], logger: logging.Logger) -> None:
    """Write the metrics dict as JSON. Called in both success and error paths."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to %s", output_path)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_file)
    start_time = time.time()

    logger.info("Job start")

    try:
        config = load_and_validate_config(args.config, logger)
        seed = config["seed"]
        window = config["window"]
        version = config["version"]

        np.random.seed(seed)
        logger.info("Random seed set to %d for deterministic run", seed)

        df = load_and_validate_dataset(args.input, logger)
        df, signal_rate = compute_signal(df, window, logger)

        rows_processed = int(len(df))
        latency_ms = int(round((time.time() - start_time) * 1000))

        metrics = {
            "version": version,
            "rows_processed": rows_processed,
            "metric": "signal_rate",
            "value": round(signal_rate, 4),
            "latency_ms": latency_ms,
            "seed": seed,
            "status": "success",
        }

        logger.info(
            "Metrics summary: rows_processed=%d, signal_rate=%.4f, latency_ms=%d",
            rows_processed, signal_rate, latency_ms,
        )

        write_metrics(args.output, metrics, logger)
        logger.info("Job end | status=success")

        print(json.dumps(metrics, indent=2))
        return 0

    except Exception as e:
        logger.exception("Job failed with an exception")
        error_metrics = {
            "version": "unknown",
            "status": "error",
            "error_message": str(e),
        }
        try:
            write_metrics(args.output, error_metrics, logger)
        except Exception:
            logger.exception("Failed to write error metrics file")

        logger.info("Job end | status=error")
        print(json.dumps(error_metrics, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())

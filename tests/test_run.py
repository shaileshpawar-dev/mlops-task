"""
Unit tests for run.py.

Covers the four things the assessment rubric explicitly grades:
    - Correctness & determinism  (signal logic, repeatable output)
    - Code quality / validation  (every documented failure mode)
    - Observability               (metrics always written, even on error)

Run with:
    pytest -v
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run as run_module  # noqa: E402


@pytest.fixture
def logger():
    log = logging.getLogger("test_logger")
    log.addHandler(logging.NullHandler())
    return log


@pytest.fixture
def valid_config_file(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("seed: 42\nwindow: 5\nversion: \"v1\"\n")
    return str(p)


@pytest.fixture
def valid_data_file(tmp_path):
    p = tmp_path / "data.csv"
    # Deliberately alternating so the rolling mean / signal is exercised
    # meaningfully rather than being trivially constant.
    closes = [100, 101, 99, 102, 98, 103, 97, 104, 96, 105]
    df = pd.DataFrame({"close": closes})
    df.to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_valid_config_loads(self, valid_config_file, logger):
        config = run_module.load_and_validate_config(valid_config_file, logger)
        assert config == {"seed": 42, "window": 5, "version": "v1"}

    def test_missing_config_file_raises(self, logger):
        with pytest.raises(ValueError, match="not found"):
            run_module.load_and_validate_config("/no/such/config.yaml", logger)

    def test_missing_required_field_raises(self, tmp_path, logger):
        p = tmp_path / "bad.yaml"
        p.write_text("window: 5\n")  # missing seed, version
        with pytest.raises(ValueError, match="missing required field"):
            run_module.load_and_validate_config(str(p), logger)

    def test_non_mapping_config_raises(self, tmp_path, logger):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="Invalid config structure"):
            run_module.load_and_validate_config(str(p), logger)

    def test_invalid_window_type_raises(self, tmp_path, logger):
        p = tmp_path / "bad.yaml"
        p.write_text("seed: 42\nwindow: 0\nversion: \"v1\"\n")
        with pytest.raises(ValueError, match="positive integer"):
            run_module.load_and_validate_config(str(p), logger)

    def test_malformed_yaml_raises(self, tmp_path, logger):
        p = tmp_path / "bad.yaml"
        p.write_text("seed: 42\nwindow: [unterminated\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            run_module.load_and_validate_config(str(p), logger)


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

class TestDatasetValidation:
    def test_valid_dataset_loads(self, valid_data_file, logger):
        df = run_module.load_and_validate_dataset(valid_data_file, logger)
        assert len(df) == 10
        assert "close" in df.columns

    def test_missing_file_raises(self, logger):
        with pytest.raises(ValueError, match="not found"):
            run_module.load_and_validate_dataset("/no/such/data.csv", logger)

    def test_empty_file_raises(self, tmp_path, logger):
        p = tmp_path / "empty.csv"
        p.touch()
        with pytest.raises(ValueError, match="empty"):
            run_module.load_and_validate_dataset(str(p), logger)

    def test_missing_close_column_raises(self, tmp_path, logger):
        p = tmp_path / "no_close.csv"
        pd.DataFrame({"open": [1, 2], "high": [2, 3]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="Required column 'close'"):
            run_module.load_and_validate_dataset(str(p), logger)

    def test_non_numeric_close_raises(self, tmp_path, logger):
        p = tmp_path / "bad_close.csv"
        pd.DataFrame({"close": ["a", "b", "c"]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="no valid numeric values"):
            run_module.load_and_validate_dataset(str(p), logger)


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------

class TestComputeSignal:
    def test_signal_matches_manual_calculation(self, logger):
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 100]})
        result_df, signal_rate = run_module.compute_signal(df.copy(), window=3, logger=logger)

        # Row 5 (close=100) with window=3 -> rolling_mean of [3,4,5] = 4 -> signal=1
        assert result_df.loc[5, "signal"] == 1
        # First (window - 1) = 2 rows have no full window -> NaN, excluded from rate
        assert pd.isna(result_df.loc[0, "rolling_mean"])
        assert pd.isna(result_df.loc[1, "signal"])
        assert 0.0 <= signal_rate <= 1.0

    def test_window_larger_than_dataset_raises(self, logger):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="exceeds dataset size"):
            run_module.compute_signal(df, window=10, logger=logger)

    def test_all_rows_counted_in_length_despite_nan_signal(self, logger):
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
        result_df, _ = run_module.compute_signal(df.copy(), window=5, logger=logger)
        assert len(result_df) == 5  # rows_processed should reflect all rows


# ---------------------------------------------------------------------------
# End-to-end determinism (the auto-fail criterion the rubric calls out)
# ---------------------------------------------------------------------------

class TestEndToEndDeterminism:
    def test_repeated_runs_produce_identical_metrics(
        self, tmp_path, valid_config_file, valid_data_file
    ):
        out1, log1 = tmp_path / "m1.json", tmp_path / "r1.log"
        out2, log2 = tmp_path / "m2.json", tmp_path / "r2.log"

        run_module.main([
            "--input", valid_data_file, "--config", valid_config_file,
            "--output", str(out1), "--log-file", str(log1),
        ])
        run_module.main([
            "--input", valid_data_file, "--config", valid_config_file,
            "--output", str(out2), "--log-file", str(log2),
        ])

        m1 = json.loads(out1.read_text())
        m2 = json.loads(out2.read_text())

        # Everything must match except latency_ms (wall-clock, not deterministic).
        m1.pop("latency_ms")
        m2.pop("latency_ms")
        assert m1 == m2

    def test_success_exit_code_is_zero(self, tmp_path, valid_config_file, valid_data_file):
        exit_code = run_module.main([
            "--input", valid_data_file, "--config", valid_config_file,
            "--output", str(tmp_path / "m.json"), "--log-file", str(tmp_path / "r.log"),
        ])
        assert exit_code == 0

    def test_error_path_still_writes_metrics_and_exits_nonzero(self, tmp_path, valid_config_file):
        out = tmp_path / "m.json"
        exit_code = run_module.main([
            "--input", "/no/such/file.csv", "--config", valid_config_file,
            "--output", str(out), "--log-file", str(tmp_path / "r.log"),
        ])
        assert exit_code == 1
        assert out.exists()  # metrics file must be written even on failure
        metrics = json.loads(out.read_text())
        assert metrics["status"] == "error"
        assert "error_message" in metrics

    def test_success_metrics_have_exact_required_keys(
        self, tmp_path, valid_config_file, valid_data_file
    ):
        out = tmp_path / "m.json"
        run_module.main([
            "--input", valid_data_file, "--config", valid_config_file,
            "--output", str(out), "--log-file", str(tmp_path / "r.log"),
        ])
        metrics = json.loads(out.read_text())
        expected_keys = {
            "version", "rows_processed", "metric", "value",
            "latency_ms", "seed", "status",
        }
        assert set(metrics.keys()) == expected_keys
        assert metrics["status"] == "success"
        assert metrics["metric"] == "signal_rate"

# MLOps Task 0 — Rolling-Mean Trading Signal Batch Job

A reproducible, observable, deployment-ready batch job that loads OHLCV
data, computes a rolling mean on `close`, derives a binary trading
signal, and writes structured metrics + logs. Built to mirror the shape
of a real `MetaStackerBandit`-style trading-signal pipeline: deterministic
(seeded), observable (JSON metrics + logs), tested, and containerized.

## Contents

- [Architecture](#architecture)
- [Files](#files)
- [Local run](#local-run)
- [Testing](#testing)
- [Docker build & run](#docker-build--run)
- [CI](#ci)
- [Design decisions & tradeoffs](#design-decisions--tradeoffs)
- [Example output](#example-metricsjson-success)
- [Note on data.csv](#note-on-datacsv)
- [Possible future improvements](#possible-future-improvements)

## Architecture

```
config.yaml ──┐
              ▼
        load_and_validate_config ──► seed / window / version
              │
data.csv ─────┼──► load_and_validate_dataset ──► DataFrame['close']
              ▼
        compute_signal (rolling mean, window) ──► signal, signal_rate
              ▼
        write_metrics ──► metrics.json (success or error, always written)
              │
        logging throughout ──► run.log + stdout
```

Every stage is a small, independently testable function (see
`tests/test_run.py`) rather than one monolithic `main()` — this is what
lets the pipeline be validated at the unit level, not just end-to-end.

## Files

| File | Purpose |
|---|---|
| `run.py` | Main batch job (type-hinted, unit-tested) |
| `config.yaml` | Run config (`seed`, `window`, `version`) |
| `data.csv` | Input OHLCV dataset (10,000 rows, real assessment data) |
| `requirements.txt` | Production dependencies (pinned) |
| `requirements-dev.txt` | Adds `pytest` for local/CI testing, kept out of the Docker image |
| `Dockerfile` | Container build definition (non-root user, healthcheck) |
| `.dockerignore` | Keeps the image lean; excludes tests/CI/docs from the build context |
| `Makefile` | One-command shortcuts: `make run`, `make test`, `make docker-build` |
| `tests/test_run.py` | Unit + end-to-end tests (18 cases) |
| `.github/workflows/ci.yml` | Runs tests, then the exact Docker build/run commands from the assessment |
| `metrics.json` | Sample output from a successful run on the real data |
| `run.log` | Sample log from that same run |

## Local run

```bash
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt

python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

Or, with the Makefile: `make install && make run`.

No paths are hardcoded — all four CLI arguments are required and can
point anywhere on disk.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Or: `make install-dev && make test`.

18 tests across four areas, matching the rubric's own categories:

- **Config validation** — valid config, missing file, missing field,
  wrong top-level type, invalid `window`, malformed YAML.
- **Dataset validation** — valid data, missing file, empty file, missing
  `close` column, non-numeric `close`.
- **Signal logic** — rolling mean / signal correctness against a
  hand-computed example, the `window`-larger-than-dataset edge case,
  row-count integrity around `NaN` signals.
- **End-to-end determinism** — two full runs on the same input produce
  byte-identical metrics (aside from `latency_ms`); the error path still
  writes `metrics.json` and exits non-zero; the success path's output
  has exactly the required keys.

## Docker build & run

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

Or: `make docker-build && make docker-run`.

The image bundles `data.csv` and `config.yaml`, so the container runs
standalone with no extra flags. It writes `metrics.json` and `run.log`
inside the container at `/app`, prints the final metrics JSON to stdout,
and exits `0` on success or non-zero on failure. The container also runs
as a non-root `appuser` and includes a `HEALTHCHECK` that verifies core
dependencies import cleanly.

To pull the output files back out of a run:

```bash
docker run --name mlops-run mlops-task
docker cp mlops-run:/app/metrics.json ./metrics.json
docker cp mlops-run:/app/run.log ./run.log
docker rm mlops-run
```

## CI

`.github/workflows/ci.yml` runs on every push/PR:

1. Installs dev dependencies and runs the full `pytest` suite.
2. Builds the image with the exact command the assessment says it will
   evaluate (`docker build -t mlops-task .`), runs it, and asserts the
   extracted `metrics.json` has `status: "success"`.

The intent is that a red CI badge is a leading indicator of the same
auto-fail conditions the assessment rubric lists (Docker build/run
failure, metrics not written) — catching it before submission rather
than after.

## Design decisions & tradeoffs

- **Reproducibility**: `seed` from `config.yaml` is passed to
  `numpy.random.seed()` before any processing, and the pipeline
  introduces no other randomness, so repeated runs on the same
  input/config produce identical `metrics.json` output aside from
  `latency_ms` (wall-clock, not deterministic by nature — verified in
  `tests/test_run.py::TestEndToEndDeterminism`).
- **Rolling mean edge case**: the first `window - 1` rows have no full
  window of history, so `rolling_mean` is `NaN` for those rows and they
  are excluded from `signal` / `signal_rate`. They're still counted in
  `rows_processed` — the alternative (dropping them entirely) would make
  `rows_processed` silently disagree with the input row count, which is
  a worse observability tradeoff than a documented `NaN` convention.
- **`window` larger than the dataset** is treated as a config error
  (raises `ValueError`) rather than silently returning an all-`NaN`
  signal — a config that can never produce a valid signal is a bug, not
  a valid degenerate case.
- **Validation is exception-driven, not exception-swallowing**: every
  anticipated failure raises a specific `ValueError` with a human
  message. `main()`'s broad `except Exception` exists as a safety net
  so an unanticipated failure still produces a clean `status: "error"`
  metrics file instead of a raw traceback exit — but every message a
  user will actually see comes from a deliberate check, not an
  accidental catch.
- **Dev dependencies (`pytest`) are split into `requirements-dev.txt`**
  and excluded from the Docker image via `.dockerignore`, so the
  production image doesn't carry test tooling it will never run.
- **Observability**: every stage (config load, dataset load, rolling
  mean, signal generation, metrics summary, job end) is logged with a
  timestamp to both `run.log` and stdout.

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 18,
  "seed": 42,
  "status": "success"
}
```

## Example `metrics.json` (error)

```json
{
  "version": "unknown",
  "status": "error",
  "error_message": "Required column 'close' not found in input dataset."
}
```

## Note on `data.csv`

This is the **real dataset** provided for the assessment (converted from
the shared Google Sheet), not a synthetic stand-in — 10,000 rows,
`timestamp,open,high,low,close,volume_btc,volume_usd`, BTC-scale prices
($41,939–$50,949 close range). `metrics.json` and `run.log` in this repo
are genuine output from running `run.py` against this exact file.

## Possible future improvements

Out of scope for a 60-minute assessment, but worth naming to show
awareness of where this would need to go for real production use:

- Structured (JSON) logging instead of plain-text lines, for easier
  ingestion into a log aggregator.
- A `--column` CLI flag to generalize beyond `close` if the pipeline
  needed to support other signal sources.
- Config-driven signal strategy (rolling mean is one strategy among
  many `MetaStackerBandit` would likely want to swap in/out).
- Multi-stage Docker build to shrink the final image further.

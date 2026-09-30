# Architecture

[Back to project overview](../README.md)

## Application flow

1. FastAPI serves `web/index.html` and the assets in `web/static/`.
2. The browser checks `/api/session` for sign-in and model availability.
3. Google OpenID Connect establishes a signed session; profiles are stored in SQLite.
4. Authenticated uploads to `/api/analyze` are checked for filename extension and size, then decoded for inference.
5. The inference module resamples to 16 kHz mono, trims or pads to four seconds, extracts features, and runs the dual-head LCNN.
6. The binary head supplies a synthetic-speech score. Source attribution is withheld unless the binary decision and independent source-confidence threshold both qualify.
7. Result metadata is stored in SQLite and exposed through `/api/history`. The temporary audio file is removed after inference.

## Component boundaries

| Component | Responsibility |
| --- | --- |
| `web/app.py` | Routes, OAuth, sessions, uploads, and SQLite persistence |
| `web/inference.py` | Audio preprocessing, network architecture, checkpoint loading, and predictions |
| `web/static/` | Browser-side state, uploads, history, and styling |
| `web/model/` | Configuration example and locally supplied checkpoint |
| `notebooks/` | Original experimental training and evaluation workflow |

The service loads its checkpoint once per process. Missing model assets leave the interface available and analysis disabled. The root `.env` is resolved relative to the application location; optional `MODEL_DIR` and `DATABASE_PATH` overrides are filesystem paths, with relative overrides resolved from the working directory. The default database is `web/voiceproof.db`.

## Current limitations

- The notebook calls the features LFCC, but its implementation uses a mel filter bank before the DCT. The inference module preserves this preprocessing for checkpoint compatibility.
- Only the first four seconds of each clip contribute to inference.
- Source classification covers the configured training classes; it cannot establish the identity of an unseen generator.
- Thresholds are configurable defaults, not evidence of calibration. Deployment requires independent evaluation.
- History returns the latest 20 results; older records are not automatically deleted.
- SQLite and synchronous inference are intended for a small prototype. A production service needs explicit retention controls, rate limits, and an appropriate database and inference execution strategy.

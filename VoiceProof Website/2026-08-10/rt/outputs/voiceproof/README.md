# VoiceProof

A full-stack web wrapper for the LFCC + dual-head LCNN deepfake-audio notebook you supplied. It gives each user a Google account, a private analysis history, audio upload and validation, a calibrated real/synthetic verdict, and cautious source-attribution reporting.

## What is ready

- Modern responsive interface with drag-and-drop audio uploads.
- Google OpenID Connect sign-in and signed sessions.
- SQLite tables for users and their last 20 result records. Audio data is never stored: it is decoded from a temporary file and then removed.
- The notebook's `extract_lfcc()` preprocessing and `LCNN` network translated into an API-ready inference module.
- Attribution only appears if the clip is predicted synthetic *and* passes an independent source-confidence threshold.

## Essential model requirement

The provided folder contains only `Deepfake_AI_Detection_and_Source_Attribution.ipynb`; it does **not** contain `best_lcnn.pth`, the `all_models` class mapping, calibration data, or final test metrics. A website cannot truthfully provide accuracy without those artifacts.

Before enabling public use, add these two private files:

```text
model/best_lcnn.pth        # model weights emitted by notebook training
model/model_config.json    # exact class ordering and thresholds
```

Create `model/model_config.json` from `model/model_config.example.json`. Its `class_labels` must be exactly the notebook's `all_models`, in the same sorted order, and `real_label` must match the label used for real speech. Never guess or rearrange it; the source-head output indexes depend on that order. The service withholds source attribution if the source head itself predicts the real-speech class.

Export the weight safely at the end of training (the original notebook needs the `model.module` guard to also work on a single GPU):

```python
state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
torch.save(state, "best_lcnn.pth")
json.dump({
    "class_labels": all_models,
    "real_label": "real",
    "fake_threshold": 0.50,
    "source_threshold": 0.60,
}, open("model_config.json", "w"), indent=2)
```

Those threshold numbers are starter defaults only. Tune them on a held-out validation/calibration split, then report the resulting false-positive/false-negative rates, EER, binary F1/AUROC, and per-source precision/recall before making an accuracy claim. Retrain with non-overlapping speakers, codecs, languages, microphones, and unseen generator versions to reduce data leakage and domain overfitting.

## Run locally

Requires Python 3.11+ and FFmpeg (Librosa/SoundFile must be able to decode the formats you accept).

```bash
cd /Users/yamin/Documents/Codex/2026-08-10/rt/outputs/voiceproof
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp model/model_config.example.json model/model_config.json
uvicorn app:app --reload
```

For Google sign-in, create a Google OAuth **Web application** client. Add `http://localhost:8000/auth/callback` as an authorized redirect URI for local work, and add your public HTTPS callback (`https://your-domain/auth/callback`) for deployment. Put its client ID and client secret in `.env`. The app intentionally rejects sign-in until those credentials are configured.

Open `http://localhost:8000` rather than opening the old `templates/index.html` file directly. The application now also includes a self-contained `index.html` that renders correctly in a direct file preview; API features need the FastAPI server.

## Deploy securely

- Use HTTPS and a long, unique `SESSION_SECRET`; remove `DEV_MODE=1` in production.
- Use PostgreSQL rather than local SQLite once multiple instances/users are involved. Store only user profile data and result metadata; keep audio out of backups and logs.
- Put the API behind upload-rate limiting, malware/content scanning, monitoring, and a job queue if inference can take more than a few seconds.
- Restrict Google OAuth redirect URIs exactly. Use a privacy policy and a data-retention/deletion mechanism before opening access broadly.
- Make inference model assets read-only and deploy a tested model version plus calibration report together.

## Project structure

```text
app.py                 FastAPI routes, OAuth, session and SQLite history
inference.py           Notebook-equivalent LFCC + LCNN inference
templates/index.html   Interface shell
static/                Responsive design and interactions
model/                 Private model checkpoint and configuration
```

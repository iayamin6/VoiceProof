# Setup and deployment

[Back to project overview](../README.md)

## Essential model requirement

The repository includes the research notebook and web application, but no deployable checkpoint or matching class-map configuration. Supply those artifacts and evaluate the model on representative held-out audio before making deployment accuracy claims.

Before enabling public use, add these two private files:

```text
web/model/best_lcnn.pth        # model weights emitted by notebook training
web/model/model_config.json    # exact class ordering and thresholds
```

Create `web/model/model_config.json` from `web/model/model_config.example.json`. Its `class_labels` must be exactly the notebook's `all_models`, in the same sorted order, and `real_label` must match the label used for real speech. Never guess or rearrange it; the source-head output indexes depend on that order. The service withholds source attribution if the source head itself predicts the real-speech class.

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

Use Python 3.11 or 3.12 and FFmpeg (Librosa/SoundFile must be able to decode the formats you accept).

```bash
git clone https://github.com/iayamin6/VoiceProof.git
cd VoiceProof
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
cp web/model/model_config.example.json web/model/model_config.json
uvicorn web.app:app --reload
```

For Google sign-in, create a Google OAuth **Web application** client. Add `http://localhost:8000/auth/callback` as an authorized redirect URI for local work, and add your public HTTPS callback (`https://your-domain/auth/callback`) for deployment. Put its client ID and client secret in `.env`. The app intentionally rejects sign-in until those credentials are configured.

Open `http://localhost:8000`. The interface is served from `web/index.html`; authentication and analysis require the server.

## Deploy securely

- Use HTTPS and a long, unique `SESSION_SECRET`; remove `DEV_MODE=1` in production.
- Use PostgreSQL rather than local SQLite once multiple instances/users are involved. Store only user profile data and result metadata; keep audio out of backups and logs.
- Put the API behind upload-rate limiting, malware/content scanning, monitoring, and a job queue if inference can take more than a few seconds.
- Restrict Google OAuth redirect URIs exactly. Use a privacy policy and a data-retention/deletion mechanism before opening access broadly.
- Make inference model assets read-only and deploy a tested model version plus calibration report together.

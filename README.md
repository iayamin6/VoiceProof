# VoiceProof

Audio deepfake screening and source attribution with a dual-head Light CNN, paired with a FastAPI web application.

VoiceProof explores two questions: **does an audio clip contain synthetic-speech signals, and which known generator class is the closest match?** The repository includes the training and evaluation notebook and a web interface for signed-in users to upload audio and review analysis history.

## At a glance

- **Machine learning:** 16 kHz mono audio, four-second inputs, cepstral features, and a PyTorch LCNN with binary and source-classification heads.
- **Application:** FastAPI, Google OpenID Connect, signed sessions, SQLite result history, and a plain HTML/CSS/JavaScript interface.
- **Analysis:** WAV, MP3, M4A, FLAC, and OGG uploads up to 25 MB; source attribution is shown only when confidence thresholds are met.
- **Audio handling:** uploads are processed through temporary files; the database stores profile and result metadata, not audio.

**Status:** research prototype. Trained weights, the exact class mapping, and validated deployment thresholds are not included. The interface can run without model assets, but audio analysis remains unavailable until they are configured. Predictions are screening signals, not proof of origin or authorship.

## Repository guide

```text
VoiceProof/
├── web/                    # Web application and inference
│   ├── app.py              # HTTP routes, authentication, and history
│   ├── inference.py        # Feature extraction and dual-head LCNN
│   ├── index.html          # Interface entry point
│   ├── static/             # Styles and browser interactions
│   └── model/              # Example configuration; private weights go here
├── notebooks/              # Original training and evaluation research
├── docs/                   # Setup, deployment, and architecture notes
├── tests/                  # Application and inference smoke checks
├── .env.example            # Local configuration template
└── requirements.txt        # Web application dependencies
```

## Run locally

Use Python 3.11 or 3.12 and install FFmpeg for audio decoding.

```bash
git clone https://github.com/iayamin6/VoiceProof.git
cd VoiceProof
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn web.app:app --reload
```

Open [localhost:8000](http://localhost:8000). Set a unique `SESSION_SECRET` and Google OAuth credentials in `.env` to enable sign-in. Model setup and deployment instructions are in the [setup guide](docs/setup.md).

## Explore the project

- [Training and evaluation notebook](notebooks/Deepfake_AI_Detection_and_Source_Attribution.ipynb): data discovery, feature extraction, training, evaluation, and Grad-CAM experiments.
- [Research notes](notebooks/README.md): notebook environment and reproducibility considerations.
- [Architecture](docs/architecture.md): request flow, component responsibilities, and current limitations.
- [Setup and deployment](docs/setup.md): authentication, checkpoint export, and model configuration.

## Development checks

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Checks cover page and static asset serving, startup without a checkpoint, authentication guards, model-path configuration, and the inference network's output shapes. They do not establish model accuracy or test live Google sign-in.

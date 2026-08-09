"""Inference utilities for the LFCC / LCNN model from the supplied notebook.

The model is intentionally loaded once per server process.  No result is
returned unless a real checkpoint and its accompanying class-map are present.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import dct

SAMPLE_RATE = 16_000
DURATION_SECONDS = 4.0
N_LFCC = 60
N_FFT = 512
HOP_LENGTH = 160
MAX_LEN = int(SAMPLE_RATE * DURATION_SECONDS / HOP_LENGTH) + 1


class MFM(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = torch.chunk(x, 2, dim=1)
        return torch.max(first, second)


class LCNN(nn.Module):
    """The dual-head Light CNN architecture used by the original notebook."""
    def __init__(self, n_classes: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 128, kernel_size=1), MFM(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 192, kernel_size=1), MFM(),
            nn.Conv2d(96, 128, kernel_size=3, padding=1), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=1), MFM(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1), MFM(), nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 256, kernel_size=1), MFM(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.shared_fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 512), MFM(), nn.Dropout(0.5))
        self.binary_head = nn.Linear(256, 1)
        self.source_head = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared_fc(self.pool(self.backbone(x)))
        return self.binary_head(shared).squeeze(1), self.source_head(shared)


def extract_lfcc(audio_path: str | Path) -> np.ndarray:
    """Mirror the notebook preprocessing exactly: mono, 16 kHz, first 4 sec."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    target_samples = int(SAMPLE_RATE * DURATION_SECONDS)
    audio = audio[:target_samples]
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))

    power = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
    filter_bank = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_LFCC, htk=True, norm=None)
    log_linear = librosa.power_to_db(filter_bank @ power + 1e-9)
    lfcc = dct(log_linear, type=2, axis=0, norm="ortho")[:N_LFCC]
    lfcc = lfcc[:, :MAX_LEN]
    if lfcc.shape[1] < MAX_LEN:
        lfcc = np.pad(lfcc, ((0, 0), (0, MAX_LEN - lfcc.shape[1])))
    return ((lfcc - lfcc.mean()) / (lfcc.std() + 1e-9)).astype(np.float32)


@dataclass
class LoadedModel:
    model: LCNN
    labels: list[str]
    real_label: str
    fake_threshold: float
    source_threshold: float
    device: torch.device


def _model_dir() -> Path:
    return Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))


def load_model() -> LoadedModel:
    model_dir = _model_dir()
    weights_path = model_dir / "best_lcnn.pth"
    config_path = model_dir / "model_config.json"
    if not weights_path.exists() or not config_path.exists():
        raise RuntimeError(
            "Model assets are missing. Add model/best_lcnn.pth and model/model_config.json before analysing audio."
        )
    config = json.loads(config_path.read_text())
    labels = config.get("class_labels")
    if not isinstance(labels, list) or not labels:
        raise RuntimeError("model_config.json must contain a non-empty class_labels list.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LCNN(len(labels)).to(device)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    # Accept either a standard checkpoint or DataParallel keys exported by a notebook.
    checkpoint = {k.removeprefix("module."): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    return LoadedModel(
        model=model,
        labels=labels,
        real_label=str(config.get("real_label", "real")),
        fake_threshold=float(config.get("fake_threshold", 0.5)),
        source_threshold=float(config.get("source_threshold", 0.60)),
        device=device,
    )


def analyze_audio(audio_path: str | Path, loaded: LoadedModel) -> dict:
    features = extract_lfcc(audio_path)
    input_tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0).to(loaded.device)
    with torch.inference_mode():
        binary_logit, source_logits = loaded.model(input_tensor)
        fake_probability = torch.sigmoid(binary_logit).item()
        source_probabilities = F.softmax(source_logits, dim=1)[0]
        source_confidence, source_index = torch.max(source_probabilities, dim=0)

    predicted_fake = fake_probability >= loaded.fake_threshold
    source_confidence = source_confidence.item()
    # Attribution is meaningful only for clips classified as synthetic and above
    # a validation-set-calibrated source confidence threshold.
    raw_source = loaded.labels[source_index.item()]
    source = raw_source if (
        predicted_fake
        and raw_source != loaded.real_label
        and source_confidence >= loaded.source_threshold
    ) else None
    alternatives = [
        {"name": loaded.labels[i], "probability": round(prob.item(), 4)}
        for i, prob in sorted(enumerate(source_probabilities), key=lambda item: item[1], reverse=True)
        if loaded.labels[i] != loaded.real_label
    ]
    return {
        "verdict": "synthetic" if predicted_fake else "likely_human",
        "fake_probability": round(fake_probability, 4),
        "threshold": loaded.fake_threshold,
        "source": source,
        "source_confidence": round(source_confidence, 4) if source else None,
        "source_threshold": loaded.source_threshold,
        "alternatives": alternatives[:3] if predicted_fake else [],
        "notice": "This is a screening signal, not proof of origin or authorship.",
    }

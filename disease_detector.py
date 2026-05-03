from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt, resample, welch

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


FS = 360
SEGMENT_LEN = FS * 10
CLASSES = ["ARR", "CHF", "NSR"]
CLASS_DESCRIPTIONS = {
    "ARR": {
        "name": "Arytmia",
        "description": "Nieregularny rytm serca. Serce bije zbyt szybko, zbyt wolno lub nieregularnie.",
        "severity": "warning",
        "color": "#ff9f1c",
    },
    "CHF": {
        "name": "Zastoinowa niewydolnosc serca",
        "description": "Serce nie pompuje krwi wystarczajaco wydajnie. Wymaga konsultacji kardiologicznej.",
        "severity": "danger",
        "color": "#ef476f",
    },
    "NSR": {
        "name": "Normalny rytm zatokowy",
        "description": "Sygnal EKG w normie. Nie wykryto patologii.",
        "severity": "success",
        "color": "#2a9d8f",
    },
}


if nn is not None:
    class SpectralCNN(nn.Module):
        def __init__(self, input_dim: int, n_classes: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 8, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(self.conv(x.unsqueeze(1)))


@dataclass(slots=True)
class LoadedModel:
    name: str
    kind: str
    model: Any


def bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = FS,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * fs
    if fs <= 0 or highcut >= nyq:
        raise ValueError("Invalid sampling rate for bandpass filtering.")
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)


def extract_features(segment: np.ndarray, fs: float = FS) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32).reshape(-1)
    freqs, psd = welch(segment, fs=fs, nperseg=min(512, len(segment)))
    total_power = np.trapezoid(psd, freqs) + 1e-10
    features: list[float] = []
    for lo, hi in ((0.04, 0.15), (0.15, 0.4), (0.5, 5.0), (5.0, 15.0)):
        idx = (freqs >= lo) & (freqs <= hi)
        band_power = np.trapezoid(psd[idx], freqs[idx]) if np.any(idx) else 0.0
        features.append(float(band_power / total_power))
    features.extend((float(np.mean(psd)), float(np.std(psd))))
    fft_vals = np.abs(fft(segment))[:30]
    fft_vals = fft_vals / (np.max(fft_vals) + 1e-10)
    features.extend(float(value) for value in fft_vals)
    return np.asarray(features, dtype=np.float32)


class DiseaseDetector:
    def __init__(self, models_dir: str | Path | None = None) -> None:
        self.models_dir = Path(models_dir) if models_dir is not None else Path("models")
        self.models: list[LoadedModel] = []
        self.scaler: Any | None = None

    def load_models(self, models_dir: str | Path | None = None) -> None:
        target_dir = Path(models_dir) if models_dir is not None else self.models_dir
        self.models_dir = target_dir
        self.models = []
        self.scaler = None

        if not target_dir.exists():
            return

        scaler_path = target_dir / "scaler.pkl"
        if scaler_path.exists():
            if joblib is None:
                raise RuntimeError("joblib is required to load scaler.pkl.")
            self.scaler = joblib.load(scaler_path)

        self._load_sklearn_model(target_dir / "rf_model.pkl", "Random Forest")
        self._load_sklearn_model(target_dir / "svm_model.pkl", "SVM")
        self._load_cnn_model(target_dir / "cnn_model.pth")

    def predict(self, signal: np.ndarray, fs: float = FS) -> dict[str, Any]:
        if not self.models:
            raise RuntimeError("No trained models are available in the models directory.")

        prepared_signal = self._prepare_signal(signal, fs)
        segments = self._segment_signal(prepared_signal)
        votes = {label: 0 for label in CLASSES}
        probabilities_accumulator = np.zeros(len(CLASSES), dtype=np.float64)
        inference_count = 0
        models_used: list[str] = []

        for loaded_model in self.models:
            models_used.append(loaded_model.name)
            for segment in segments:
                features = extract_features(segment, fs=FS)
                probabilities = self._predict_probabilities(loaded_model, features)
                predicted_index = int(np.argmax(probabilities))
                votes[CLASSES[predicted_index]] += 1
                probabilities_accumulator += probabilities
                inference_count += 1

        if inference_count == 0:
            raise RuntimeError("No model predictions could be computed.")

        averaged_probabilities = probabilities_accumulator / inference_count
        predicted_index = int(np.argmax(averaged_probabilities))
        predicted_class = CLASSES[predicted_index]
        return {
            "predicted_class": predicted_class,
            "confidence": float(averaged_probabilities[predicted_index]),
            "votes": votes,
            "probabilities": {
                label: float(averaged_probabilities[index])
                for index, label in enumerate(CLASSES)
            },
            "models_used": models_used,
            "n_segments": len(segments),
            "class_info": CLASS_DESCRIPTIONS[predicted_class],
        }

    def _load_sklearn_model(self, path: Path, name: str) -> None:
        if not path.exists():
            return
        if joblib is None:
            raise RuntimeError(f"joblib is required to load {path.name}.")
        self.models.append(
            LoadedModel(name=name, kind="sklearn", model=joblib.load(path))
        )

    def _load_cnn_model(self, path: Path) -> None:
        if not path.exists():
            return
        if torch is None or nn is None:
            raise RuntimeError("torch is required to load cnn_model.pth.")
        checkpoint = torch.load(path, map_location="cpu")
        model = SpectralCNN(
            input_dim=int(checkpoint.get("input_dim", 36)),
            n_classes=int(checkpoint.get("n_classes", len(CLASSES))),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.models.append(LoadedModel(name="CNN", kind="cnn", model=model))

    def _prepare_signal(self, signal: np.ndarray, fs: float) -> np.ndarray:
        prepared_signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        if prepared_signal.size == 0:
            raise ValueError("ECG signal is empty.")
        if fs <= 0:
            raise ValueError("Sampling rate must be positive.")
        if fs != FS:
            target_len = int(round(len(prepared_signal) * FS / fs))
            prepared_signal = resample(prepared_signal, max(target_len, 1)).astype(
                np.float32
            )
        return prepared_signal

    def _segment_signal(self, signal: np.ndarray) -> list[np.ndarray]:
        if signal.size < SEGMENT_LEN:
            padded = np.pad(signal, (0, SEGMENT_LEN - signal.size), mode="constant")
            return [bandpass_filter(padded, fs=FS).astype(np.float32)]

        segments: list[np.ndarray] = []
        for start_idx in range(0, len(signal) - SEGMENT_LEN + 1, SEGMENT_LEN):
            segments.append(
                bandpass_filter(signal[start_idx : start_idx + SEGMENT_LEN], fs=FS).astype(
                    np.float32
                )
            )
        return segments

    def _predict_probabilities(
        self, loaded_model: LoadedModel, features: np.ndarray
    ) -> np.ndarray:
        if loaded_model.kind == "cnn":
            return self._predict_cnn_probabilities(loaded_model.model, features)
        return self._predict_sklearn_probabilities(loaded_model.model, features)

    def _predict_sklearn_probabilities(
        self, model: Any, features: np.ndarray
    ) -> np.ndarray:
        input_features = features.reshape(1, -1)
        if self.scaler is not None:
            input_features = self.scaler.transform(input_features)
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(input_features)[0], dtype=np.float64)
            return self._align_probabilities(probabilities, getattr(model, "classes_", None))
        if hasattr(model, "decision_function"):
            decision = np.asarray(model.decision_function(input_features), dtype=np.float64)
            if decision.ndim == 1:
                decision = decision.reshape(1, -1)
            logits = decision[0] - np.max(decision[0])
            exp_logits = np.exp(logits)
            probabilities = exp_logits / np.sum(exp_logits)
            return self._align_probabilities(probabilities, getattr(model, "classes_", None))
        predicted = np.asarray(model.predict(input_features)).reshape(-1)
        probabilities = np.zeros(len(CLASSES), dtype=np.float64)
        probabilities[int(predicted[0])] = 1.0
        return probabilities

    def _predict_cnn_probabilities(self, model: Any, features: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("torch is required for CNN inference.")
        input_features = features.reshape(1, -1)
        if self.scaler is not None:
            input_features = self.scaler.transform(input_features)
        with torch.no_grad():
            tensor = torch.tensor(input_features, dtype=torch.float32)
            probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
        return self._align_probabilities(np.asarray(probabilities, dtype=np.float64), CLASSES)

    def _align_probabilities(
        self, probabilities: np.ndarray, model_classes: Any | None
    ) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if model_classes is None:
            if probabilities.shape[0] != len(CLASSES):
                raise RuntimeError("Model output dimension does not match expected classes.")
            return probabilities

        aligned = np.zeros(len(CLASSES), dtype=np.float64)
        for source_index, source_class in enumerate(model_classes):
            class_name = (
                CLASSES[int(source_class)]
                if isinstance(source_class, (np.integer, int))
                else str(source_class)
            )
            if class_name in CLASSES:
                aligned[CLASSES.index(class_name)] = probabilities[source_index]
        if aligned.sum() <= 0:
            raise RuntimeError("Model returned invalid class probabilities.")
        return aligned / aligned.sum()

from __future__ import annotations
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis as kurt
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder, StandardScaler


class WindowCNN(nn.Module):
    """
    1D CNN on a feature vector from FFT+PSD+time-domain.
    """

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(1)
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.input_bn(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.gap(self.block3(x))
        return self.fc(x)


# =========================
# 2. FEATURE EXTRACTION
# =========================
_BP_BANDS = [(0.04, 0.15), (0.15, 0.4), (0.5, 5), (5, 15), (15, 40)]


def extract_features_batch(windows_2d: np.ndarray, fs: float) -> np.ndarray:
    """
    Vectorized feature extraction for a batch of windows.
    - windows_2d: (N, window_len) float32 array
    - Returns: (N, n_features) float32 array
    """
    N, window_len = windows_2d.shape

    # --- FFT (batch) ---
    fft_c = np.fft.rfft(windows_2d, axis=1)
    mag_all = np.abs(fft_c)
    mag60 = mag_all[:, :60].copy()
    mag60 /= mag60.max(axis=1, keepdims=True) + 1e-10
    phase60 = np.angle(fft_c[:, :60])

    # --- Welch PSD (loop, as welch is not vectorized) ---
    freqs, _ = welch(windows_2d[0], fs=fs, nperseg=512)
    psds = np.empty((N, len(freqs)), dtype=np.float32)
    for i, w in enumerate(windows_2d):
        _, psds[i] = welch(w, fs=fs, nperseg=512)

    total_power = np.trapezoid(psds, freqs, axis=1) + 1e-10
    band_feats = []
    for lo, hi in _BP_BANDS:
        idx = (freqs >= lo) & (freqs <= hi)
        bp = np.trapezoid(psds[:, idx], freqs[idx], axis=1)
        band_feats.append(bp / total_power)
    band_feats = np.stack(band_feats, axis=1)

    dom_freq = freqs[np.argmax(psds, axis=1)]
    psd_mean = psds.mean(axis=1)
    psd_std = psds.std(axis=1)
    psd_max = psds.max(axis=1)

    # --- Time domain (vectorized) ---
    td_mean = windows_2d.mean(axis=1)
    td_std = windows_2d.std(axis=1)
    td_rms = np.sqrt((windows_2d**2).mean(axis=1))
    td_skew = skew(windows_2d, axis=1).astype(np.float32)
    td_kurt = kurt(windows_2d, axis=1).astype(np.float32)
    zcr = (np.diff(np.sign(windows_2d), axis=1) != 0).sum(axis=1) / window_len

    features = np.concatenate(
        [
            band_feats,
            psd_mean[:, None],
            psd_std[:, None],
            psd_max[:, None],
            dom_freq[:, None],
            mag60,
            phase60,
            td_mean[:, None],
            td_std[:, None],
            td_skew[:, None],
            td_kurt[:, None],
            td_rms[:, None],
            zcr[:, None],
        ],
        axis=1,
    )

    return features.astype(np.float32)


# =========================
# 3. MODEL & DATA MANAGEMENT
# =========================
@dataclass(slots=True)
class MLModel:
    net: WindowCNN
    scaler: StandardScaler
    le: LabelEncoder
    device: torch.device
    fs: float
    window_len: int
    step_len: int
    classes: list[str]


def load_model(model_dir: Path) -> MLModel:
    """Loads all model artifacts from the specified directory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights_path = model_dir / "window_cnn_balanced_weights.pt"
    sklearn_path = model_dir / "window_cnn_balanced_sklearn.pkl"
    meta_path = model_dir / "window_cnn_balanced_meta.json"

    if not all([weights_path.exists(), sklearn_path.exists(), meta_path.exists()]):
        raise FileNotFoundError(f"One or more model files are missing in {model_dir}")

    ckpt = torch.load(weights_path, map_location=device)

    with open(sklearn_path, "rb") as f:
        sk = pickle.load(f)

    with open(meta_path) as f:
        meta = json.load(f)

    net = WindowCNN(
        input_dim=ckpt["model_config"]["input_dim"],
        n_classes=ckpt["model_config"]["n_classes"],
    ).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    return MLModel(
        net=net,
        scaler=sk["scaler"],
        le=sk["le"],
        device=device,
        fs=meta["fs"],
        window_len=meta["window_len"],
        step_len=meta["step_len"],
        classes=meta["classes"],
    )


# =========================
# 4. PREDICTION PIPELINE
# =========================
def predict_signal(
    signal_1d: np.ndarray, model: MLModel, return_windows: bool = False
) -> dict:
    """
    Runs the full prediction pipeline on a raw 1D signal.
    """
    sig = np.nan_to_num(signal_1d.astype(np.float64))

    try:
        nyq = 0.5 * model.fs
        b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype="band")
        sig = filtfilt(b, a, sig)
    except Exception:
        pass  # Ignore filtering errors for short signals

    starts = np.arange(0, len(sig) - model.window_len + 1, model.step_len)
    if len(starts) == 0:
        raise ValueError(
            f"Signal is too short. It must have at least {model.window_len} samples, "
            f"but has {len(sig)}. ({model.window_len / model.fs:.1f}s at {model.fs}Hz)"
        )

    windows = np.stack([sig[s : s + model.window_len] for s in starts]).astype(
        np.float32
    )
    feats = extract_features_batch(windows, fs=model.fs)
    feats_sc = model.scaler.transform(feats)

    with torch.no_grad():
        logits = model.net(torch.tensor(feats_sc).to(model.device))
        probs = torch.softmax(logits, 1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()

    pred_labels = model.le.inverse_transform(preds)
    counts = Counter(pred_labels)
    total = len(pred_labels)
    majority_class, _ = counts.most_common(1)[0]

    # Calculate weighted probability for the majority class
    majority_indices = np.where(pred_labels == majority_class)[0]
    majority_class_int = model.le.transform([majority_class])[0]
    majority_probs = probs[majority_indices, majority_class_int]
    majority_prob_avg = (
        float(np.mean(majority_probs)) if len(majority_probs) > 0 else 0.0
    )

    result = {
        "majority_class": majority_class,
        "majority_prob": round(majority_prob_avg, 3),
        "class_distribution": {
            k: round(v / total * 100, 1) for k, v in sorted(counts.items())
        },
        "n_windows": total,
    }

    if return_windows:
        result["window_results"] = [
            {
                "start_s": float(s / model.fs),
                "end_s": float((s + model.window_len) / model.fs),
                "class": pred_labels[i],
                "probs": dict(zip(model.classes, probs[i].round(3).tolist())),
            }
            for i, s in enumerate(starts)
        ]

    return result

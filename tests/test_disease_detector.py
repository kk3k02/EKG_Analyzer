from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disease_detector import DiseaseDetector, extract_features


def test_extract_features() -> None:
    segment = np.random.randn(3600).astype(np.float32)
    features = extract_features(segment)
    assert features.shape == (36,), f"Oczekiwano 36 cech, got {features.shape}"


def test_detector_no_models() -> None:
    detector = DiseaseDetector()
    detector.load_models("nonexistent_dir")
    assert len(detector.models) == 0

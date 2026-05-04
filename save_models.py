from __future__ import annotations

import os

import joblib
import torch


def save_models(rf, svm, scaler, cnn, x_train_sc, classes) -> None:
    """Run after notebook training to export inference artifacts to models/."""
    os.makedirs("models", exist_ok=True)

    joblib.dump(rf, "models/rf_model.pkl")
    joblib.dump(svm, "models/svm_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    torch.save(
        {
            "model_state_dict": cnn.state_dict(),
            "input_dim": x_train_sc.shape[1],
            "n_classes": len(classes),
            "classes": classes.tolist(),
        },
        "models/cnn_model.pth",
    )

    print("Modele zapisane w katalogu models/")

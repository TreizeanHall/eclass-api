from __future__ import annotations

import os
from pathlib import Path
import joblib
import pandas as pd

from ml.preprocess import preprocess_df

# Env override:
#   MODEL_PATH=/mounted/artifacts/latest/inference_model.joblib
DEFAULT_MODEL_PATH = Path("artifacts") / "latest" / "inference_model.joblib"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

_MODEL = None
_MODEL_MTIME = None


def _get_model():
    global _MODEL, _MODEL_MTIME

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Run POST /train first, or set MODEL_PATH to the correct location."
        )

    mtime = MODEL_PATH.stat().st_mtime
    if _MODEL is None or _MODEL_MTIME != mtime:
        _MODEL = joblib.load(MODEL_PATH)
        _MODEL_MTIME = mtime

    return _MODEL


def predict_one(subject: str = "", description: str = "") -> dict:
    """
    Preprocess subject+description into modeltext, then load the model and predict.
    MODEL_PATH can be overridden via env var MODEL_PATH.
    """
    one = pd.DataFrame([{"subject": subject, "description": description}])
    one = preprocess_df(one)

    if len(one) == 0:
        return {"error": "Empty text after preprocessing. Provide subject and/or description."}

    modeltext = one["modeltext"].iloc[0]
    model = _get_model()

    pred = model.predict([modeltext])[0]
    out = {"pred_label": str(pred)}

    # Confidence if calibrated
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([modeltext])[0]
        out["conf_max_proba"] = float(proba.max())

        # Optional: top-3 probabilities (super useful for debugging/UI)
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            top_idx = proba.argsort()[-3:][::-1]
            out["top3"] = [{"label": str(classes[i]), "proba": float(proba[i])} for i in top_idx]

    return out

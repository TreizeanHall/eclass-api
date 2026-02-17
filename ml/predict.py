from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from ml.preprocess import preprocess_df

# Always load the "latest" trained model
MODEL_PATH = Path("artifacts") / "latest" / "inference_model.joblib"


def predict_one(subject: str = "", description: str = "") -> dict:
    """
    Preprocess subject+description into modeltext, then load the latest trained model and predict.
    """
    # 1) Preprocess exactly like training
    one = pd.DataFrame([{"subject": subject, "description": description}])
    one = preprocess_df(one)

    if len(one) == 0:
        return {"error": "Empty text after preprocessing. Provide subject and/or description."}

    modeltext = one["modeltext"].iloc[0]

    # 2) Load model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run POST /train first.")

    model = joblib.load(MODEL_PATH)

    # 3) Predict
    pred = model.predict([modeltext])[0]
    out = {"pred_label": str(pred)}

    # 4) Confidence if calibrated
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([modeltext])[0]
        out["conf_max_proba"] = float(proba.max())

    return out

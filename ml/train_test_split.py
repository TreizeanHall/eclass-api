import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer,
    f1_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from ml.tds_fetch import fetch_training_df
from ml.preprocess import preprocess_df

# ----------------------------
# Helpers
# ----------------------------
def _safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-]+", "_", str(s).strip())
    return s[:max_len] if len(s) > max_len else s


def post_predict_override(text: str, pred_label: str):
    """
    Review-only suggestions (does NOT change predictions).
    Only suggests when model predicted 'Other'.
    """
    if str(pred_label).strip().lower() != "other":
        return None, None

    t = (text or "").lower()

    certificate_patterns = [
        r"\bcoi\b",
        r"\bcertificate of insurance\b",
        r"\badditional insured\b",
        r"\bcertificate\b",
        r"\bholder\b",
    ]
    if any(re.search(p, t) for p in certificate_patterns):
        return "Certificate", "certificate_patterns"

    cancel_patterns = [
        r"\bcancel(lation|led|ing)?\b",
        r"\bcancellation request\b",
        r"\bcompleted cancellation\b",
        r"\bpolicy cancellation\b",
        r"\bplease cancel\b",
        r"\bflat cancel\b",
        r"\beffective cancel\b",
    ]
    if any(re.search(p, t) for p in cancel_patterns):
        return "Cancel Policy", "cancel_patterns"

    mortgage_patterns = [
        r"\bmortgage\b",
        r"\bmortgage company\b",
        r"\bmortgagee\b",
        r"\blender\b",
        r"\bloss payee\b",
        r"\besrow\b",
        r"\bescrow\b",
        r"\b1st mortgage\b",
        r"\bsecond mortgage\b",
    ]
    if any(re.search(p, t) for p in mortgage_patterns):
        return "Mortgage-Related", "mortgage_patterns"

    update_policy_patterns = [
        r"\bchange coverage\b",
        r"\bcoverage change\b",
        r"\bupdate coverage\b",
        r"\badd coverage\b",
        r"\bremove coverage\b",
        r"\bincrease\b.*\blimit\b",
        r"\bdecrease\b.*\blimit\b",
        r"\bendorsement\b",
        r"\bbind coverages\b",
        r"\bbind\b.*\bcoverage\b",
    ]
    if any(re.search(p, t) for p in update_policy_patterns):
        return "Update Policy", "update_policy_patterns"

    proof_docs_patterns = [
        r"\bdocusign\b",
        r"\bsigning\b",
        r"\battached\b.*\b(signed|document|documents|doc|pdf)\b",
        r"\bdeclarations?\b",
        r"\bdec page\b",
        r"\bpolicy (forms?|documents?)\b",
        r"\bdriver exclusion\b",
    ]
    if any(re.search(p, t) for p in proof_docs_patterns):
        return "Proof of Insurance/Documents", "proof_docs_patterns"

    return None, None


def top_ngrams_per_label_chi2(
    tfidf_word: TfidfVectorizer,
    X_text: pd.Series,
    y_encoded: np.ndarray,
    le: LabelEncoder,
    out_dir: Path,
    top_n: int = 20,
    save_plots: bool = True,
):
    X_vec = tfidf_word.transform(X_text)

    feature_names = tfidf_word.get_feature_names_out()
    rows = []

    plots_dir = out_dir / "label_ngrams"
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    for c in np.unique(y_encoded):
        label_name = le.inverse_transform([c])[0]
        y_bin = (y_encoded == c).astype(int)

        scores, _ = chi2(X_vec, y_bin)
        top_idx = np.argsort(scores)[-top_n:][::-1]
        top_terms = [(feature_names[i], float(scores[i])) for i in top_idx if scores[i] > 0]

        for term, sc in top_terms:
            rows.append({"label_name": label_name, "ngram": term, "chi2_score": sc})

        if save_plots:
            terms_plot = [t for t, _ in top_terms][::-1]
            scores_plot = [sc for _, sc in top_terms][::-1]

            plt.figure(figsize=(12, 8))
            plt.barh(terms_plot, scores_plot)
            plt.title(f"Top {top_n} n-grams for '{label_name}' (chi2 vs rest)")
            plt.xlabel("Chi-square score")
            plt.tight_layout()
            plt.savefig(plots_dir / f"{_safe_filename(label_name)}_top_{top_n}.png", dpi=200)
            plt.close()

    pd.DataFrame(rows).sort_values(["label_name", "chi2_score"], ascending=[True, False]).to_csv(
        out_dir / f"top_{top_n}_ngrams_per_label_chi2.csv", index=False
    )


# ----------------------------
# MAIN TRAIN FUNCTION (API calls this)
# ----------------------------
def train_model(
    use_calibration: bool = True,
    best_c_grid=None,
    artifacts_root: str = "artifacts",
):
    """
    API-safe training entrypoint:
      - pulls from Dataverse TDS via fetch_training_df()
      - expects columns: subject, body, label
      - trains model
      - saves artifacts under artifacts/<RUN_ID>/
      - returns JSON-safe summary
    """
    if best_c_grid is None:
        best_c_grid = [3]

    # 1) Fetch from SQL (no import-time execution elsewhere)
    df_raw = fetch_training_df(...)
    df = preprocess_df(df_raw)
    X = df["modeltext"]
    y_raw = df["casetype"].fillna("Other").astype(str)   # or your label column

    # 2) Build model text (TEMP placeholder)
    # You said you’ll share preprocessing code after — for now:
    df["modeltext"] = (df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)).str.strip()

    X = df["modeltext"].astype(str)
    y_raw = df["label"].fillna("Other").astype(str)

    # 3) Run folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(artifacts_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 5) Holdout split
    row_ids = df.index.to_numpy()
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, row_ids,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # CV folds (for calibration)
    min_class = pd.Series(y_train).value_counts().min()
    n_splits = max(3, min(5, int(min_class)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 6) Word + char TFIDF
    word_tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
    char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)

    features = FeatureUnion([
        ("word", word_tfidf),
        ("char", char_tfidf),
    ])

    base_svc = LinearSVC(class_weight="balanced", random_state=42, max_iter=20000, dual=False)

    pipe = Pipeline([
        ("features", features),
        ("clf", base_svc),
    ])

    scorer = make_scorer(f1_score, average="macro")
    param_grid = {"clf__C": best_c_grid}

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scorer,
        cv=skf,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True
    )

    gs.fit(X_train, y_train)
    best_c = gs.best_params_["clf__C"]
    best_cv = float(gs.best_score_)

    pipe_svc = gs.best_estimator_

    # 7) Optional calibration
    if use_calibration:
        tuned_svc = LinearSVC(C=best_c, class_weight="balanced", random_state=42, max_iter=20000)
        pipe_final = Pipeline([
            ("features", features),
            ("clf", CalibratedClassifierCV(tuned_svc, cv=skf, method="sigmoid")),
        ])
        pipe_final.fit(X_train, y_train)
    else:
        pipe_final = pipe_svc

    # 8) Chi2 n-grams (word only)
    fitted_word = pipe_final.named_steps["features"].transformer_list[0][1]
    top_ngrams_per_label_chi2(fitted_word, X_train, y_train, le, out_dir, top_n=20, save_plots=True)

    # 9) Holdout predict
    X_test_s = pd.Series(X_test).reset_index(drop=True)
    y_pred = pipe_final.predict(X_test_s)

    labels_present = sorted(set(y_test))
    target_names = [le.inverse_transform([i])[0] for i in labels_present]

    report = classification_report(
        y_test, y_pred,
        labels=labels_present,
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=90, values_format="d")
    ax.set_title(f"Confusion Matrix (Holdout) - {'Calibrated' if use_calibration else 'LinearSVC'}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_holdout.png", dpi=200)
    plt.close()

    # Review suggestions (only when pred == Other)
    true_labels = le.inverse_transform(y_test)
    pred_labels = le.inverse_transform(y_pred)

    suggested = [post_predict_override(txt, pred) for txt, pred in zip(X_test_s.values, pred_labels)]
    suggested_labels = [s for s, _ in suggested]
    suggested_reasons = [r for _, r in suggested]

    test_results_df = pd.DataFrame({
        "row_id": id_test,
        "text": X_test_s.values,
        "true_label": true_labels,
        "pred_label": pred_labels,
        "review_suggested_label": suggested_labels,
        "review_reason": suggested_reasons
    })
    test_results_df.to_csv(out_dir / "holdout_test_results.csv", index=False)

    # Save artifacts
    joblib.dump(pipe_svc, out_dir / "linearsvc_bestC_uncalibrated.joblib")
    joblib.dump(pipe_final, out_dir / "inference_model.joblib")
    pd.DataFrame({"label_id": range(len(le.classes_)), "label_name": le.classes_}).to_csv(out_dir / "label_map.csv", index=False)
    latest_dir = Path("artifacts") / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_final, latest_dir / "inference_model.joblib")


    params = {
        "run_id": run_id,
        "rows": int(len(df)),
        "labels": list(le.classes_),
        "best_C": float(best_c),
        "best_cv_macro_f1": best_cv,
        "calibrated": bool(use_calibration),
        "artifacts_dir": str(out_dir),
        "files": {
            "inference_model": "inference_model.joblib",
            "uncalibrated_model": "linearsvc_bestC_uncalibrated.joblib",
            "label_map": "label_map.csv",
            "holdout_results": "holdout_test_results.csv",
            "confusion_matrix": "confusion_matrix_holdout.png",
        }
    }
    with open(out_dir / "train_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Return API-safe summary
    return {
        "status": "trained",
        "run_id": run_id,
        "rows": int(len(df)),
        "best_C": float(best_c),
        "best_cv_macro_f1": best_cv,
        "calibrated": bool(use_calibration),
        "labels": list(le.classes_),
        "classification_report": report,
        "artifacts_dir": str(out_dir),
    }

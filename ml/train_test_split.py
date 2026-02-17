from datetime import datetime
from pathlib import Path
import json
import re
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joblib

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
# MAIN TRAIN FUNCTION (API call)
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
      - optionally includes: incidentid, ticketnumber
      - trains model
      - saves artifacts under artifacts/<RUN_ID>/ and artifacts/latest/
      - returns JSON-safe summary
    """
    if best_c_grid is None:
        best_c_grid = [3]

    # 1) Fetch from SQL
    df_raw = fetch_training_df()

    # Normalize columns
    df_raw.columns = [c.lower() for c in df_raw.columns]

    required = {"subject", "body", "label"}
    if not required.issubset(df_raw.columns):
        raise RuntimeError(f"TDS fetch must return {required}. Got: {list(df_raw.columns)}")

    trace_cols = [c for c in ["incidentid", "ticketnumber"] if c in df_raw.columns]

    # Ensure strings for trace cols (Dataverse GUIDs can be special types)
    for c in trace_cols:
        df_raw[c] = df_raw[c].astype(str)

    # 2) Preprocess (should preserve extra columns)
    df = preprocess_df(df_raw)

    # If preprocess_df drops columns, reattach trace cols from raw by index alignment
    for c in trace_cols:
        if c not in df.columns and c in df_raw.columns and len(df_raw) == len(df):
            df[c] = df_raw[c].values

    # Ensure modeltext exists (build if preprocess didn't create it)
    if "modeltext" not in df.columns:
        df["modeltext"] = (
            df["subject"].fillna("").astype(str) + " " + df["body"].fillna("").astype(str)
        ).str.strip()

    # Clean labels
    y_raw = df["label"].fillna("Other").astype(str)
    X = df["modeltext"].fillna("").astype(str)

    # Drop empties after preprocessing (safety)
    keep_mask = X.str.strip().ne("")
    df = df.loc[keep_mask].copy()
    X = X.loc[keep_mask]
    y_raw = y_raw.loc[keep_mask]

    # 3) Run folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(artifacts_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4) Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 5) Holdout split (keep trace info)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # CV folds
    min_class = pd.Series(y_train).value_counts().min()
    n_splits = 3 if min_class >= 3 else 2
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


    # 6) TFIDF features
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
        tuned_svc = LinearSVC(C=best_c, class_weight="balanced", random_state=42, max_iter=20000, dual=False)
        pipe_final = Pipeline([
            ("features", features),
            ("clf", CalibratedClassifierCV(tuned_svc, cv=skf, method="sigmoid")),
        ])
        pipe_final.fit(X_train, y_train)
    else:
        pipe_final = pipe_svc

    # 7.5) Chi2 n-grams diagnostics (word TF-IDF only)
    fitted_word = pipe_final.named_steps["features"].transformer_list[0][1]
    top_ngrams_per_label_chi2(
        tfidf_word=fitted_word,
        X_text=pd.Series(X_train),
        y_encoded=np.array(y_train),
        le=le,
        out_dir=out_dir,
        top_n=20,
        save_plots=True,
    )

    
    # 8) Holdout predict + reports
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

    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=90, values_format="d")
    ax.set_title(f"Confusion Matrix (Holdout) - {'Calibrated' if use_calibration else 'LinearSVC'}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_holdout.png", dpi=200)
    plt.close()

    # Decode labels for export
    true_labels = le.inverse_transform(y_test)
    pred_labels = le.inverse_transform(y_pred)

    # 9) Holdout results export WITH incidentid/ticketnumber if present
    df_test_export = df_test.reset_index(drop=True).copy()
    results_df = pd.DataFrame({
        "text": X_test_s.values,
        "true_label": true_labels,
        "pred_label": pred_labels,
    })

    # Attach trace cols
    for c in trace_cols:
        results_df[c] = df_test_export[c].astype(str).values

    # Attach subject/body for easier review
    if "subject" in df_test_export.columns:
        results_df["subject"] = df_test_export["subject"].astype(str).values
    if "body" in df_test_export.columns:
        results_df["body"] = df_test_export["body"].astype(str).values

    results_df.to_csv(out_dir / "holdout_test_results.csv", index=False)

    # 10) Save artifacts
    joblib.dump(pipe_svc, out_dir / "linearsvc_bestC_uncalibrated.joblib")
    joblib.dump(pipe_final, out_dir / "inference_model.joblib")

    pd.DataFrame({"label_id": range(len(le.classes_)), "label_name": le.classes_}).to_csv(
        out_dir / "label_map.csv", index=False
    )

    latest_dir = Path(artifacts_root) / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_final, latest_dir / "inference_model.joblib")
    pd.DataFrame({"label_id": range(len(le.classes_)), "label_name": le.classes_}).to_csv(
        latest_dir / "label_map.csv", index=False
    )

    params = {
        "run_id": run_id,
        "rows": int(len(df)),
        "labels": list(le.classes_),
        "best_C": float(best_c),
        "best_cv_macro_f1": best_cv,
        "calibrated": bool(use_calibration),
        "artifacts_dir": str(out_dir),
        "trace_cols": trace_cols,
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
        "trace_cols": trace_cols,
    }

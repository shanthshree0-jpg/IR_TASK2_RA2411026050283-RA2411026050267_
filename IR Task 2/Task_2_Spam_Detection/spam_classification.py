"""
Information Retrieval CLA2 — Task 2: SMS Spam Detection
Comparative Text Classification Using TF-IDF and Traditional Machine Learning

This module implements a complete, reproducible experimental pipeline for SMS
spam detection. It performs:
1. Robust dataset loading and data-quality validation.
2. Defensible duplicate-handling and class-distribution analysis.
3. TF-IDF feature extraction (unigrams + bigrams) fitted strictly on training data.
4. Training and evaluation of four machine learning classifiers:
   - Decision Tree
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (LinearSVC)
   - Multinomial Naive Bayes
5. Metric computation (Accuracy, Spam Precision, Spam Recall, Spam F1-Score).
6. Confusion matrix generation and comparative visualization.
7. What-if parameter sensitivity experiment (varying K in KNN: 3, 5, 7, 9).
8. Evidence-based model recommendation and trade-off analysis.

Dataset Source:
UCI Machine Learning Repository: SMS Spam Collection
DOI: https://doi.org/10.24432/C5VG8H
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


# ==============================================================================
# 1. Directory and Path Management
# ==============================================================================

def get_default_directories() -> Tuple[Path, Path, Path]:
    """
    Determine default paths for project root, data directory, and outputs directory.
    Supports running from either the project root or the parent repository directory.
    """
    script_dir = Path(__file__).resolve().parent

    # If running inside Task_2_Spam_Detection
    if script_dir.name == "Task_2_Spam_Detection":
        project_root = script_dir
    elif (script_dir / "Task_2_Spam_Detection").is_dir():
        project_root = script_dir / "Task_2_Spam_Detection"
    else:
        project_root = script_dir

    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"

    return project_root, data_dir, outputs_dir


def ensure_output_directories(outputs_dir: Path) -> Dict[str, Path]:
    """Create all required output subdirectories if they do not exist."""
    subdirs = {
        "root": outputs_dir,
        "classification_reports": outputs_dir / "classification_reports",
        "confusion_matrices": outputs_dir / "confusion_matrices",
        "what_if_experiment": outputs_dir / "what_if_experiment",
    }
    for directory in subdirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return subdirs


# ==============================================================================
# 2. Dataset Discovery, Loading, and Validation
# ==============================================================================

def locate_dataset_file(explicit_path: Optional[str] = None, data_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Locate the SMS Spam Collection dataset file.
    Searches explicit path, followed by common candidate names in the data directory.
    """
    if explicit_path:
        p = Path(explicit_path)
        if p.is_file():
            return p
        print(f"[WARNING] Specified dataset path '{explicit_path}' was not found.")

    if data_dir and data_dir.is_dir():
        candidates = [
            "SMSSpamCollection",
            "SMSSpamCollection.txt",
            "smsspamcollection",
            "SMSSpamCollection.tsv",
            "SMSSpamCollection.csv",
            "sms_spam_collection.txt",
        ]
        for name in candidates:
            target = data_dir / name
            if target.is_file() and target.stat().st_size > 0:
                return target

        # Check for any non-hidden file in data_dir
        for file in data_dir.iterdir():
            if file.is_file() and not file.name.startswith(".") and not file.name.endswith(".zip"):
                return file

    return None


def print_missing_dataset_help(data_dir: Path) -> None:
    """Print student-friendly instructions for obtaining and placing the dataset."""
    banner = "=" * 78
    print(f"\n{banner}")
    print(" [ERROR] SMS Spam Collection Dataset File Not Found")
    print(f"{banner}")
    print(" The required dataset could not be located in the data folder:")
    print(f" Expected location: {data_dir.resolve() / 'SMSSpamCollection'}")
    print("\n HOW TO FIX THIS IN 3 EASY STEPS:")
    print(" 1. Visit the official UCI Machine Learning Repository:")
    print("    https://doi.org/10.24432/C5VG8H")
    print("    (or search 'UCI SMS Spam Collection')")
    print(" 2. Download the archive ('smsspamcollection.zip').")
    print(" 3. Extract the file named 'SMSSpamCollection' (no extension) directly into:")
    print(f"    {data_dir.resolve()}")
    print("\n Once placed, re-run this script:")
    print("    python spam_classification.py")
    print(f"{banner}\n")


def load_and_validate_dataset(
    file_path: Path,
    remove_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and validate the tab-separated SMS Spam Collection file.
    
    Robust parsing:
    - Reads line-by-line using UTF-8 (with latin-1 fallback if encoding issues occur).
    - Splits strictly on the first tab character (tab-separated: label \\t message).
    - Validates recognized labels ('ham', 'spam') and non-empty message bodies.
    - Records malformed or skipped lines with reasons.
    - Inspects and defensibly handles exact duplicates to prevent train-test data leakage.
    """
    total_lines = 0
    valid_records: List[Tuple[str, str]] = []
    skipped_records: List[Tuple[int, str]] = []

    # Attempt UTF-8 first, fallback to latin-1
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1"]
    lines = None
    chosen_encoding = None

    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                lines = f.readlines()
            chosen_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if lines is None:
        raise ValueError(f"Failed to decode '{file_path}' with supported encodings: {encodings_to_try}")

    for line_idx, raw_line in enumerate(lines, start=1):
        total_lines += 1
        line = raw_line.strip("\r\n")

        # Skip purely blank lines
        if not line.strip():
            skipped_records.append((line_idx, "Blank line"))
            continue

        parts = line.split("\t", 1)
        if len(parts) < 2:
            skipped_records.append((line_idx, f"Missing tab separator. Line preview: '{line[:40]}'"))
            continue

        raw_label, message = parts[0].strip().lower(), parts[1].strip()

        if raw_label not in ("ham", "spam"):
            skipped_records.append((line_idx, f"Invalid label '{raw_label}'. Expected 'ham' or 'spam'."))
            continue

        if not message:
            skipped_records.append((line_idx, "Empty message body."))
            continue

        valid_records.append((raw_label, message))

    # Construct initial DataFrame
    df_raw = pd.DataFrame(valid_records, columns=["label", "message"])
    raw_valid_count = len(df_raw)
    raw_label_counts = df_raw["label"].value_counts().to_dict()

    # Duplicate analysis
    exact_duplicates_count = int(df_raw.duplicated(subset=["message"]).sum())
    
    # Check for label conflict on identical messages
    grouped_labels = df_raw.groupby("message")["label"].nunique()
    conflicting_messages_count = int((grouped_labels > 1).sum())

    if remove_duplicates:
        # Keep first occurrence of unique message content to prevent train/test data leakage
        df = df_raw.drop_duplicates(subset=["message"], keep="first").reset_index(drop=True)
        dedup_reason = (
            "Exact duplicate messages were removed to prevent data leakage between train "
            "and test sets. Duplicate SMS broadcasts in real-world spam data artificially "
            "inflate held-out evaluation metrics if present across splits."
        )
    else:
        df = df_raw.copy()
        dedup_reason = "Duplicates retained as requested by configuration."

    final_label_counts = df["label"].value_counts().to_dict()

    metadata: Dict[str, Any] = {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "file_size_bytes": file_path.stat().st_size,
        "encoding_used": chosen_encoding,
        "total_lines_read": total_lines,
        "valid_rows_parsed": raw_valid_count,
        "skipped_rows_count": len(skipped_records),
        "skipped_records_sample": skipped_records[:10],
        "raw_label_counts": raw_label_counts,
        "exact_duplicates_count": exact_duplicates_count,
        "conflicting_messages_count": conflicting_messages_count,
        "remove_duplicates_applied": remove_duplicates,
        "dedup_reason": dedup_reason,
        "retained_rows_count": len(df),
        "final_label_counts": final_label_counts,
        "missing_values_count": int(df.isnull().sum().sum()),
    }

    return df, metadata


def save_dataset_summary_files(metadata: Dict[str, Any], outputs_dir: Path) -> None:
    """Save dataset summary text and class distribution CSV."""
    # 1. Dataset Summary Text
    summary_txt_path = outputs_dir / "dataset_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(" INFORMATION RETRIEVAL CLA2 — TASK 2: DATASET SUMMARY & AUDIT\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Dataset File Name          : {metadata['file_name']}\n")
        f.write(f"Full File Path             : {metadata['file_path']}\n")
        f.write(f"File Size                  : {metadata['file_size_bytes']:,} bytes\n")
        f.write(f"Encoding Detected          : {metadata['encoding_used']}\n")
        f.write(f"Total Lines Processed      : {metadata['total_lines_read']:,}\n")
        f.write(f"Valid Records Parsed       : {metadata['valid_rows_parsed']:,}\n")
        f.write(f"Skipped / Malformed Lines  : {metadata['skipped_rows_count']:,}\n")
        f.write(f"Missing Values in Columns  : {metadata['missing_values_count']}\n")
        f.write(f"Raw Ham Count              : {metadata['raw_label_counts'].get('ham', 0):,}\n")
        f.write(f"Raw Spam Count             : {metadata['raw_label_counts'].get('spam', 0):,}\n")
        f.write(f"Exact Duplicate Messages   : {metadata['exact_duplicates_count']:,}\n")
        f.write(f"Conflicting Label Messages : {metadata['conflicting_messages_count']}\n")
        f.write(f"Deduplication Applied      : {metadata['remove_duplicates_applied']}\n")
        f.write(f"Retained Messages Count    : {metadata['retained_rows_count']:,}\n")
        f.write(f"Retained Ham Count         : {metadata['final_label_counts'].get('ham', 0):,}\n")
        f.write(f"Retained Spam Count        : {metadata['final_label_counts'].get('spam', 0):,}\n\n")
        f.write("DUPLICATE-HANDLING STRATEGY & RATIONALE:\n")
        f.write(f"{metadata['dedup_reason']}\n\n")
        
        if metadata["skipped_rows_count"] > 0:
            f.write("SAMPLE OF SKIPPED / MALFORMED ROWS:\n")
            for line_no, reason in metadata["skipped_records_sample"]:
                f.write(f" - Line {line_no}: {reason}\n")
        else:
            f.write("DATA INTEGRITY CHECK:\n")
            f.write(" No malformed or unparseable rows detected. All lines adhered to standard format.\n")

    # 2. Class Distribution CSV
    dist_csv_path = outputs_dir / "dataset_class_distribution.csv"
    tot = metadata["retained_rows_count"]
    ham_cnt = metadata["final_label_counts"].get("ham", 0)
    spam_cnt = metadata["final_label_counts"].get("spam", 0)
    
    dist_df = pd.DataFrame([
        {
            "Class": "ham (legitimate)",
            "Count": ham_cnt,
            "Percentage": round((ham_cnt / tot) * 100, 2) if tot > 0 else 0.0,
            "Description": "Legitimate personal and transactional SMS"
        },
        {
            "Class": "spam (unsolicited)",
            "Count": spam_cnt,
            "Percentage": round((spam_cnt / tot) * 100, 2) if tot > 0 else 0.0,
            "Description": "Unsolicited promotional, prize, or phishing SMS"
        },
        {
            "Class": "Total",
            "Count": tot,
            "Percentage": 100.0,
            "Description": "Retained unique SMS messages evaluated"
        }
    ])
    dist_df.to_csv(dist_csv_path, index=False)


# ==============================================================================
# 3. Text Preprocessing and TF-IDF Representation
# ==============================================================================

def build_custom_stop_words() -> List[str]:
    """
    Build a refined English stop words list suited for SMS spam detection.
    
    Rationale:
    Standard English stop words (like scikit-learn's built-in list) aggressively
    strip polarity markers (e.g. 'not', 'no', 'nor') and commercial terms that 
    may be informative. We preserve polarity words and ensure critical domain
    terms are retained.
    """
    base_stop_words = set(ENGLISH_STOP_WORDS)
    # Words with potential discriminative value or negation meaning in SMS:
    preserve_words = {
        "not", "no", "nor", "call", "free", "win", "won", "urgent",
        "claim", "prize", "cash", "txt", "text", "stop", "reply"
    }
    refined_words = sorted(list(base_stop_words - preserve_words))
    return refined_words


def fit_tfidf_and_extract_features(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
   outputs_dir: Path,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: Optional[int] = 5000,
) -> Tuple[TfidfVectorizer, Any, Any, Dict[str, Any]]:
    """
    Fit TF-IDF vectorizer strictly on training data to prevent data leakage,
    then transform both train and test partitions.
    
    Settings:
    - lowercase=True: Normalizes case variations ('FREE' vs 'free').
    - stop_words: Refined English stop words preserving polarity/domain markers.
    - ngram_range=(1, 2): Captures unigrams ('urgent') and bigrams ('cash prize').
    - min_df=2: Removes singleton noise/typos appearing only once in the corpus.
    - max_features: Limits vocabulary to top discriminative terms, mitigating curse of dimensionality.
    - sublinear_tf=True: Uses 1 + log(tf) scaling to temper repeating spam words.
    """
    custom_stop_words = build_custom_stop_words()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=custom_stop_words,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )

    # FIT ONLY ON TRAINING SET
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # TRANSFORM TEST SET
    X_test_tfidf = vectorizer.transform(X_test)

    feature_names = np.array(vectorizer.get_feature_names_out())
    num_features = len(feature_names)

    # Compute mean TF-IDF weights across training set to sample top features
    mean_tfidf = np.asarray(X_train_tfidf.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[::-1][:20]
    top_features_sample = [
        (feature_names[i], round(float(mean_tfidf[i]), 5)) for i in top_indices
    ]

    # Save feature summary text
    feature_summary_path = outputs_dir / "feature_summary.txt"
    with open(feature_summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(" INFORMATION RETRIEVAL CLA2 — TASK 2: TF-IDF FEATURE SUMMARY\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Training Documents Count     : {X_train.shape[0]:,}\n")
        f.write(f"Testing Documents Count      : {X_test.shape[0]:,}\n")
        f.write(f"N-Gram Range                 : {ngram_range} (Unigrams + Bigrams)\n")
        f.write(f"Minimum Document Frequency   : {min_df}\n")
        f.write(f"Maximum Features Ceiling     : {max_features}\n")
        f.write(f"Actual Vocabulary Size       : {num_features:,} features\n")
        f.write(f"Sublinear TF Scaling         : True (1 + log(TF))\n")
        f.write(f"Normalization                : L2 Euclidean Norm\n\n")
        f.write("TF-IDF WEIGHTING CONCEPT & INFORMATION RETRIEVAL FOUNDATION:\n")
        f.write(
            "TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a word\n"
            "or bigram is to a message within the corpus. It combines two factors:\n"
            "1. Term Frequency (TF): Accounts for how many times a word occurs in a specific SMS.\n"
            "2. Inverse Document Frequency (IDF): Decreases the weight of terms that appear in\n"
            "   almost every message (e.g. 'ok', 'good'), while heavily boosting words that appear\n"
            "   frequently only in specific contexts (e.g. 'claim', 'urgent', 'txt', 'prize').\n"
            "This transformation yields a sparse vector representation where discriminative spam\n"
            "indicators stand out distinctly from everyday interpersonal dialogue.\n\n"
        )
        f.write("SAMPLE OF TOP LEARNED TF-IDF FEATURES (CORPUS-WIDE AVERAGE WEIGHT):\n")
        for rank, (feat, weight) in enumerate(top_features_sample, start=1):
            f.write(f" {rank:2d}. {feat:<20} (mean TF-IDF weight: {weight:.5f})\n")

    # Generate Preprocessing Examples CSV (5 ham, 5 spam from training set)
    analyzer = vectorizer.build_analyzer()
    sample_rows = []
    
    # Pick 5 ham and 5 spam samples
    ham_samples = X_train[y_train == "ham"].head(5)
    spam_samples = X_train[y_train == "spam"].head(5)
    combined_samples = pd.concat([ham_samples, spam_samples])

    for idx, raw_msg in combined_samples.items():
        label = y_train.loc[idx]
        tokens = analyzer(raw_msg)
        tokens_preview = " | ".join(tokens[:10]) if tokens else "(No vocabulary tokens matched)"
        sample_rows.append({
            "Index": idx,
            "Label": label,
            "Original_Message": raw_msg,
            "Extracted_Tokens_Sample": tokens_preview,
            "Total_Extracted_Tokens": len(tokens),
        })

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(outputs_dir / "preprocessing_examples.csv", index=False)

    feature_meta = {
        "num_features": num_features,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_features": max_features,
        "top_features_sample": top_features_sample,
    }

    return vectorizer, X_train_tfidf, X_test_tfidf, feature_meta


# ==============================================================================
# 4. Model Training, Evaluation, and Diagnostic Reporting
# ==============================================================================

def initialize_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Initialize the four benchmark classifiers with documented baseline settings:
    1. Decision Tree: Interpretable non-linear rule builder. Balanced class weights.
    2. K-Nearest Neighbors: Instance-based lazy learner using cosine similarity.
    3. Support Vector Machine (LinearSVC): High-dimensional margin maximizer.
    4. Multinomial Naive Bayes: Fast probabilistic model based on word counts.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            random_state=random_state,
            class_weight="balanced",
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            metric="cosine",  # Cosine metric is standard in text IR with TF-IDF
            weights="uniform",
        ),
        "Support Vector Machine": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=random_state,
            dual="auto",  # scikit-learn standard for compatibility
            max_iter=3000,
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0,  # Standard Laplace add-one smoothing
            fit_prior=True,
        ),
    }


def evaluate_single_model(
    name: str,
    model: Any,
    X_train: Any,
    y_train: pd.Series,
    X_test: Any,
    y_test: pd.Series,
    outputs_dir: Path,
) -> Dict[str, Any]:
    """
    Train and evaluate a classifier on identical training and test splits.
    Measures training time, inference time, accuracy, precision, recall, and F1.
    Saves individual classification report and confusion matrix plot.
    """
    # Measure Training Time
    t_train_start = time.perf_counter()
    model.fit(X_train, y_train)
    t_train = time.perf_counter() - t_train_start

    # Measure Prediction / Inference Time
    t_pred_start = time.perf_counter()
    y_pred = model.predict(X_test)
    t_pred = time.perf_counter() - t_pred_start

    # Calculate Evaluation Metrics (Positive class = 'spam')
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)

    # Generate Full Classification Report
    target_names = ["ham", "spam"]
    report_str = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    # Confusion Matrix: [[TN, FP], [FN, TP]] where ham=0, spam=1
    cm = confusion_matrix(y_test, y_pred, labels=target_names)
    tn, fp, fn, tp = cm.ravel()

    # Save Classification Report
    slug = name.lower().replace(" ", "_").replace("-", "_")
    report_file = outputs_dir / "classification_reports" / f"{slug}_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(f" CLASSIFICATION REPORT: {name.upper()}\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Training Time   : {t_train:.4f} seconds\n")
        f.write(f"Prediction Time : {t_pred:.4f} seconds\n")
        f.write(f"Overall Accuracy: {acc * 100:.2f}%\n")
        f.write(f"Spam Precision  : {prec:.4f}\n")
        f.write(f"Spam Recall     : {rec:.4f}\n")
        f.write(f"Spam F1-Score   : {f1:.4f}\n\n")
        f.write("CONFUSION MATRIX BREAKDOWN:\n")
        f.write(f" True Negatives  (Ham correctly identified as Ham)   : {tn:,}\n")
        f.write(f" False Positives (Ham incorrectly flagged as Spam)   : {fp:,}\n")
        f.write(f" False Negatives (Spam missed, marked as Ham)        : {fn:,}\n")
        f.write(f" True Positives  (Spam correctly identified as Spam) : {tp:,}\n\n")
        f.write("DETAILED PER-CLASS METRICS:\n")
        f.write(report_str)

    # Plot and Save Confusion Matrix
    cm_plot_path = outputs_dir / "confusion_matrices" / f"{slug}_confusion_matrix.png"
    plot_styled_confusion_matrix(cm, name, acc, prec, rec, f1, cm_plot_path)

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Spam_Precision": round(prec, 4),
        "Spam_Recall": round(rec, 4),
        "Spam_F1_Score": round(f1, 4),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "Train_Time_Sec": round(t_train, 4),
        "Predict_Time_Sec": round(t_pred, 4),
    }


def plot_styled_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    save_path: Path
) -> None:
    """Render and save an annotated publication-quality confusion matrix image."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    
    # Colormap and matrix display
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    classes = ["Ham (Legit)", "Spam"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)

    tn, fp, fn, tp = cm.ravel()
    annotations = [
        [f"TN\n{tn:,}\n(Legit Ham)", f"FP\n{fp:,}\n(False Alarm)"],
        [f"FN\n{fn:,}\n(Missed Spam)", f"TP\n{tp:,}\n(Caught Spam)"],
    ]

    for i in range(2):
        for j in range(2):
            cell_val = cm[i, j]
            # Use white text for dark cells, dark text for light cells
            text_color = "white" if cell_val > (cm.max() / 2) else "black"
            ax.text(
                j, i, annotations[i][j],
                ha="center", va="center",
                color=text_color, fontsize=10, fontweight="bold"
            )

    plt.title(
        f"Confusion Matrix: {model_name}\n"
        f"Acc: {acc*100:.1f}% | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}",
        fontsize=12, fontweight="bold", pad=15
    )
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("Actual True Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 5. Comparative Visualization and Analysis
# ==============================================================================

def plot_model_comparison(comparison_df: pd.DataFrame, save_path: Path) -> None:
    """
    Generate a grouped horizontal or vertical bar chart comparing Accuracy,
    Spam Precision, Spam Recall, and Spam F1 across all four algorithms.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    models = comparison_df["Model"].tolist()
    metrics = ["Accuracy", "Spam_Precision", "Spam_Recall", "Spam_F1_Score"]
    metric_labels = ["Accuracy", "Spam Precision", "Spam Recall", "Spam F1-Score"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    x = np.arange(len(models))
    width = 0.20

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = comparison_df[metric].tolist()
        rects = ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color, alpha=0.9)
        
        # Add numeric labels on top of each bar
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", rotation=0
            )

    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Performance Comparison of Four Classifiers on SMS Spam Detection\n"
        "(Stratified 80/20 Test Set Evaluation)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 6. What-If Experiment: KNN Parameter Sensitivity
# ==============================================================================

def run_what_if_knn_experiment(
    X_train: Any,
    y_train: pd.Series,
    X_test: Any,
    y_test: pd.Series,
    outputs_dir: Path,
    k_values: List[int] = [3, 5, 7, 9],
) -> pd.DataFrame:
    """
    Perform a controlled what-if experiment by systematically varying the number
    of neighbors (K) in KNeighborsClassifier while keeping dataset, train/test split,
    and TF-IDF features strictly identical.
    """
    records = []
    n_train_samples = X_train.shape[0]

    for k in k_values:
        if k > n_train_samples:
            print(f"[WARNING] K={k} exceeds training sample count {n_train_samples}. Skipping.")
            continue

        t0 = time.perf_counter()
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = knn.predict(X_test)
        t_pred = time.perf_counter() - t1

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label="spam", zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label="spam", zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label="spam", zero_division=0)

        records.append({
            "K_Neighbors": k,
            "Accuracy": round(acc, 4),
            "Spam_Precision": round(prec, 4),
            "Spam_Recall": round(rec, 4),
            "Spam_F1_Score": round(f1, 4),
            "Train_Time_Sec": round(t_fit, 4),
            "Predict_Time_Sec": round(t_pred, 4),
        })

    what_if_df = pd.DataFrame(records)
    csv_path = outputs_dir / "what_if_experiment" / "knn_k_comparison.csv"
    what_if_df.to_csv(csv_path, index=False)

    # Plot K vs Performance Metrics
    chart_path = outputs_dir / "what_if_experiment" / "knn_k_comparison.png"
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    ks = what_if_df["K_Neighbors"]
    ax.plot(ks, what_if_df["Accuracy"], marker="o", linewidth=2, label="Accuracy", color="#2b5c8f")
    ax.plot(ks, what_if_df["Spam_Precision"], marker="s", linewidth=2, label="Spam Precision", color="#d95f02")
    ax.plot(ks, what_if_df["Spam_Recall"], marker="^", linewidth=2, label="Spam Recall", color="#7570b3")
    ax.plot(ks, what_if_df["Spam_F1_Score"], marker="D", linewidth=2.5, label="Spam F1-Score", color="#1b9e77")

    for _, row in what_if_df.iterrows():
        ax.annotate(f"{row['Spam_F1_Score']:.3f}", (row['K_Neighbors'], row['Spam_F1_Score']),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Number of Nearest Neighbors (K)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("What-If Experiment: Impact of Neighbor Count (K) on KNN Performance", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(ks)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close(fig)

    return what_if_df


# ==============================================================================
# 7. Final Observations and Recommendation Generation
# ==============================================================================

def generate_final_observations(
    meta: Dict[str, Any],
    feat_meta: Dict[str, Any],
    comp_df: pd.DataFrame,
    what_if_df: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    """Generate final_observations.txt synthesizing all empirical results."""
    # Identify top models
    best_f1_row = comp_df.loc[comp_df["Spam_F1_Score"].idxmax()]
    best_acc_row = comp_df.loc[comp_df["Accuracy"].idxmax()]
    best_rec_row = comp_df.loc[comp_df["Spam_Recall"].idxmax()]
    best_prec_row = comp_df.loc[comp_df["Spam_Precision"].idxmax()]

    obs_path = outputs_dir / "final_observations.txt"
    with open(obs_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write(" INFORMATION RETRIEVAL CLA2 — TASK 2: FINAL OBSERVATIONS & RECOMMENDATION\n")
        f.write("=" * 78 + "\n\n")

        f.write("1. DATASET & AUDIT SUMMARY\n")
        f.write(f" - Raw Parsed Messages : {meta['valid_rows_parsed']:,}\n")
        f.write(f" - Exact Duplicates    : {meta['exact_duplicates_count']:,}\n")
        f.write(f" - Retained Corpus Size: {meta['retained_rows_count']:,} messages\n")
        ham_c = meta['final_label_counts'].get('ham', 0)
        spam_c = meta['final_label_counts'].get('spam', 0)
        tot = meta['retained_rows_count']
        f.write(f" - Class Distribution  : {ham_c:,} Ham ({ham_c/tot*100:.1f}%) | {spam_c:,} Spam ({spam_c/tot*100:.1f}%)\n")
        f.write(f" - Split Details       : 80% Train ({feat_meta['n_train']:,}) / 20% Test ({feat_meta['n_test']:,}), Stratified\n\n")

        f.write("2. PREPROCESSING & TF-IDF CONFIGURATION\n")
        f.write(f" - Representation      : TF-IDF with sublinear scaling (1 + log(TF))\n")
        f.write(f" - N-Gram Range        : {feat_meta['ngram_range']} (Unigrams + Bigrams)\n")
        f.write(f" - Vocabulary Size     : {feat_meta['num_features']:,} features extracted\n")
        f.write(f" - Stop Words Policy   : Refined English stop words preserving polarity/domain markers\n\n")

        f.write("3. EMPIRICAL MODEL COMPARISON (TEST SET RESULTS)\n")
        header = f" {'Model':<24} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Inference':<10}"
        f.write(header + "\n")
        f.write(" " + "-" * len(header) + "\n")
        for _, r in comp_df.iterrows():
            f.write(
                f" {r['Model']:<24} | {r['Accuracy']*100:>8.2f}% | {r['Spam_Precision']:>9.4f} | "
                f"{r['Spam_Recall']:>9.4f} | {r['Spam_F1_Score']:>9.4f} | {r['Predict_Time_Sec']:>8.4f}s\n"
            )
        f.write("\n")

        f.write("4. BEST PERFORMING MODELS ACCORDING TO METRICS\n")
        f.write(f" - Best Spam F1-Score : {best_f1_row['Model']} ({best_f1_row['Spam_F1_Score']:.4f})\n")
        f.write(f" - Best Accuracy      : {best_acc_row['Model']} ({best_acc_row['Accuracy']*100:.2f}%)\n")
        f.write(f" - Best Spam Recall   : {best_rec_row['Model']} ({best_rec_row['Spam_Recall']:.4f})\n")
        f.write(f" - Best Spam Precision: {best_prec_row['Model']} ({best_prec_row['Spam_Precision']:.4f})\n\n")

        f.write("5. WHAT-IF EXPERIMENT FINDINGS (KNN NEIGHBOR SENSITIVITY)\n")
        f.write(" Systematically varying K across 3, 5, 7, and 9 revealed:\n")
        for _, kr in what_if_df.iterrows():
            f.write(
                f" - K={int(kr['K_Neighbors'])}: Accuracy={kr['Accuracy']*100:.2f}%, "
                f"Spam Precision={kr['Spam_Precision']:.4f}, Spam Recall={kr['Spam_Recall']:.4f}, "
                f"Spam F1={kr['Spam_F1_Score']:.4f}\n"
            )
        f.write(
            " Analysis: In highly imbalanced text datasets (~87% ham), increasing K expands the\n"
            " neighborhood to include predominantly majority-class (ham) messages. While precision\n"
            " often stays high, spam recall declines for higher K because minority spam points are\n"
            " easily outvoted by surrounding legitimate vectors in sparse high-dimensional space.\n\n"
        )

        f.write("6. PRACTICAL TRADE-OFFS & ALGORITHMIC BEHAVIOR IN TEXT CLASSIFICATION\n")
        f.write(
            " a. High Accuracy vs. Spam Recall Paradox:\n"
            "    Because 87% of the corpus is legitimate (ham), a naive dummy classifier predicting\n"
            "    'ham' for every single message would achieve ~87% accuracy while completely failing\n"
            "    at spam detection (0% recall, 0% F1). Thus, Spam F1-Score and Spam Recall are the\n"
            "    primary evaluation metrics for this task.\n\n"
            " b. False Positives (Ham classified as Spam) vs False Negatives (Spam classified as Ham):\n"
            "    - False Positive Cost: An urgent legitimate SMS (e.g. bank OTP, flight alert, or family\n"
            "      emergency) is blocked or diverted. This damages user trust severely.\n"
            "    - False Negative Cost: An unsolicited advertisement, phishing link, or scam reaches the\n"
            "      inbox, exposing the recipient to fraud.\n"
            "    In production systems, a high-precision threshold is prioritized to guarantee minimal\n"
            "    false alarms while maintaining satisfactory recall.\n\n"
            " c. High-Dimensional TF-IDF Representation:\n"
            "    - Support Vector Machine (LinearSVC): Excels in text classification because high-dimensional\n"
            "      sparse text representations are frequently linearly separable. L2 regularization\n"
            "      and margin maximization prevent overfitting.\n"
            "    - Multinomial Naive Bayes: Exceptionally fast, low-parameter, and robust with word count\n"
            "      frequencies. It typically achieves outstanding precision with near-zero false alarms.\n"
            "    - Decision Trees: Suffer from greedy, axis-aligned splits over thousands of sparse\n"
            "      dimensions, often leading to sub-optimal generalization compared to margin classifiers.\n"
            "    - KNN: Computationally expensive at inference time (O(N_train * D) per query), and\n"
            "      distances in sparse TF-IDF space suffer from dimensionality degradation.\n\n"
        )

        f.write("7. EVIDENCE-BASED FINAL RECOMMENDATION\n")
        f.write(
            f" Based on the measured experimental evidence on the held-out test partition,\n"
            f" {best_f1_row['Model']} is the recommended classifier for production SMS spam detection.\n"
            f" It achieved the optimal balance of Spam Recall ({best_f1_row['Spam_Recall']:.4f}) and\n"
            f" Spam Precision ({best_f1_row['Spam_Precision']:.4f}), yielding the highest F1-Score\n"
            f" ({best_f1_row['Spam_F1_Score']:.4f}) and robust generalization under class imbalance.\n"
        )
        f.write("=" * 78 + "\n")


# ==============================================================================
# 8. Command-Line Interface and Main Pipeline Execution
# ==============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for configurable pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Information Retrieval CLA2 - Task 2: SMS Spam Detection via TF-IDF & ML",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to the SMSSpamCollection file. If omitted, searches default locations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files and plots.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of the dataset reserved for held-out testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible train-test splitting and model initialization.",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Retain exact duplicate messages (default: False, duplicates are removed).",
    )
    return parser.parse_args()


def main() -> int:
    """Main pipeline execution routine."""
    args = parse_arguments()
    default_root, default_data_dir, default_outputs_dir = get_default_directories()

    output_dir = Path(args.output_dir) if args.output_dir else default_outputs_dir
    subdirs = ensure_output_directories(output_dir)

    print("=" * 78)
    print(" IR CLA2 -- TASK 2: SMS SPAM DETECTION USING TF-IDF & CLASSIFICATION")
    print("=" * 78)

    # 1. Locate Dataset
    dataset_file = locate_dataset_file(args.data_path, default_data_dir)
    if not dataset_file:
        print_missing_dataset_help(default_data_dir)
        return 1

    print(f"\n[1/7] Loading and validating dataset from: {dataset_file}")
    remove_dups = not args.keep_duplicates
    try:
        df, meta = load_and_validate_dataset(dataset_file, remove_duplicates=remove_dups)
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not read dataset: {e}")
        return 1

    print(f"      - Lines read: {meta['total_lines_read']:,}")
    print(f"      - Valid parsed rows: {meta['valid_rows_parsed']:,}")
    print(f"      - Exact duplicates: {meta['exact_duplicates_count']:,}")
    print(f"      - Retained rows: {meta['retained_rows_count']:,}")
    print(f"      - Class balance: {meta['final_label_counts']}")

    # Save dataset audit files
    save_dataset_summary_files(meta, output_dir)
    print(f"      - Saved audit to: {output_dir / 'dataset_summary.txt'}")

    # 2. Stratified Train-Test Split
    print(f"\n[2/7] Splitting data ({1 - args.test_size:.0%} train / {args.test_size:.0%} test, stratified)...")
    X = df["message"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    print(f"      - Training set : {X_train.shape[0]:,} messages")
    print(f"      - Test set     : {X_test.shape[0]:,} messages")

    # 3. TF-IDF Extraction (Strictly on Train Split)
    print("\n[3/7] Fitting TF-IDF Vectorizer on training data...")
    vectorizer, X_train_tfidf, X_test_tfidf, feat_meta = fit_tfidf_and_extract_features(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        outputs_dir=output_dir,
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
    )
    print(f"      - Learned features: {feat_meta['num_features']:,} (unigrams + bigrams)")
    print(f"      - Saved feature summary: {output_dir / 'feature_summary.txt'}")
    print(f"      - Saved preprocessing examples: {output_dir / 'preprocessing_examples.csv'}")

    # 4. Model Training & Evaluation
    print("\n[4/7] Training and evaluating the four baseline classifiers...")
    models = initialize_classifiers(random_state=args.random_state)
    comparison_records = []

    for name, clf in models.items():
        print(f"      -> Training {name}...")
        res = evaluate_single_model(
            name=name,
            model=clf,
            X_train=X_train_tfidf,
            y_train=y_train,
            X_test=X_test_tfidf,
            y_test=y_test,
            outputs_dir=output_dir,
        )
        comparison_records.append(res)
        print(
            f"         Accuracy: {res['Accuracy']*100:.2f}% | "
            f"Spam Precision: {res['Spam_Precision']:.4f} | "
            f"Spam Recall: {res['Spam_Recall']:.4f} | "
            f"Spam F1: {res['Spam_F1_Score']:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_records)
    comp_csv_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comp_csv_path, index=False)
    print(f"      - Saved metrics table: {comp_csv_path}")

    # 5. Comparative Visualization
    print("\n[5/7] Generating comparative performance charts...")
    comp_chart_path = output_dir / "model_comparison.png"
    plot_model_comparison(comparison_df, comp_chart_path)
    print(f"      - Saved chart: {comp_chart_path}")

    # 6. What-If Parameter Sensitivity Experiment
    print("\n[6/7] Running What-If Experiment on KNN (varying K: 3, 5, 7, 9)...")
    what_if_df = run_what_if_knn_experiment(
        X_train=X_train_tfidf,
        y_train=y_train,
        X_test=X_test_tfidf,
        y_test=y_test,
        outputs_dir=output_dir,
        k_values=[3, 5, 7, 9],
    )
    print(f"      - Saved What-If CSV & chart to: {output_dir / 'what_if_experiment'}")

    # 7. Final Observations & Recommendation
    print("\n[7/7] Synthesizing final observations and justified recommendations...")
    generate_final_observations(
        meta=meta,
        feat_meta=feat_meta,
        comp_df=comparison_df,
        what_if_df=what_if_df,
        outputs_dir=output_dir,
    )
    print(f"      - Saved final observations: {output_dir / 'final_observations.txt'}")

    print("\n" + "=" * 78)
    print(" [SUCCESS] Pipeline execution finished successfully!")
    print(f" All reports, metrics, and visual plots generated in: {output_dir.resolve()}")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

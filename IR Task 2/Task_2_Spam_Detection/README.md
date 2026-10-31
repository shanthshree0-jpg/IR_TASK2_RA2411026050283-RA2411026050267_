# Information Retrieval CLA2 — Task 2: SMS Spam Detection
## Comparative Text Classification Using TF-IDF and Traditional Machine Learning

**Course:** Information Retrieval (CLA2)  
**Academic Task:** Task 2 — Text Classification and Information Retrieval  
**Platform:** Windows 10/11 & Visual Studio Code  
**Python Version:** Python 3.11+  

---

## 1. Project Overview & Problem Statement

Short Message Service (SMS) remains one of the most immediate and trusted personal communication channels. However, due to low delivery costs and high open rates, it is heavily targeted by spammers, commercial advertisers, and phishing fraudsters. 

Unlike standard email or long documents, SMS messages are:
- Strictly length-constrained (typically 160 characters or fewer).
- Rich in informal internet slang, abbreviations (`u`, `dun`, `wif`), and phonetic shortcuts.
- Prone to deliberate obfuscations (e.g., substituting digits for letters, irregular capitalization).
- Highly imbalanced in real-world distribution (legitimate messages vastly outnumber spam).

### Objective
This project builds an end-to-end, reproducible Machine Learning pipeline for SMS spam classification rooted in fundamental **Information Retrieval (IR)** concepts. Using the standard **Term Frequency-Inverse Document Frequency (TF-IDF)** vector space model, the project trains, evaluates, and rigorously compares four distinct traditional classification algorithms:
1. **Decision Tree Classifier**
2. **K-Nearest Neighbors (KNN)**
3. **Support Vector Machine (LinearSVC)**
4. **Multinomial Naive Bayes**

The system executes on a held-out test split, visualizes comparative performance and confusion matrices, conducts a parameter sensitivity "What-If" experiment on KNN ($K \in \{3, 5, 7, 9\}$), and generates diagnostic audit reports.

---

## 2. Dataset Source & Expected Format

The project utilizes the benchmark **SMS Spam Collection** dataset hosted on the UCI Machine Learning Repository.

- **Official Repository Page:** [UCI SMS Spam Collection (DOI: 10.24432/C5VG8H)](https://doi.org/10.24432/C5VG8H)
- **Dataset Structure:** A single text file containing one SMS record per line.
- **Delimiter:** Tab-separated (`\t`), where:
  - Column 1: Binary class label (`ham` for legitimate, `spam` for unsolicited/scam).
  - Column 2: Raw SMS text message.
- **Expected Filename:** `SMSSpamCollection` (extensionless) placed in the `data/` folder.

```tsv
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121...
```

> **Data Integrity & Duplicate Handling:**  
> The raw dataset contains several identical repeated messages (e.g., identical spam broadcasts or common replies like *"Ok"*). If retained, duplicate messages can leak across both the training and testing sets, artificially inflating evaluation metrics. By default, `spam_classification.py` detects and removes exact message duplicates (keeping the first occurrence) while fully logging the before-and-after audit counts in `outputs/dataset_summary.txt`.

---

## 3. Project Directory Structure

```text
Task_2_Spam_Detection/
│
├── data/
│   └── SMSSpamCollection                 # Manually downloaded dataset from UCI
│
├── outputs/
│   ├── dataset_summary.txt               # Dataset audit, row counts, duplicate statistics
│   ├── dataset_class_distribution.csv    # Class breakdown (Ham vs Spam percentages)
│   ├── preprocessing_examples.csv        # Sample messages before and after tokenization
│   ├── feature_summary.txt               # TF-IDF vocabulary metrics & top features
│   ├── model_comparison.csv              # Accuracy, Precision, Recall, F1, and runtimes
│   ├── model_comparison.png              # Grouped comparative bar chart
│   ├── final_observations.txt            # Synthesized findings, analysis, and recommendations
│   │
│   ├── classification_reports/           # Full per-class precision/recall reports
│   │   ├── decision_tree_report.txt
│   │   ├── k_nearest_neighbors_report.txt
│   │   ├── support_vector_machine_report.txt
│   │   └── multinomial_naive_bayes_report.txt
│   │
│   ├── confusion_matrices/               # Visual confusion matrices with TP/FP/TN/FN annotations
│   │   ├── decision_tree_confusion_matrix.png
│   │   ├── k_nearest_neighbors_confusion_matrix.png
│   │   ├── support_vector_machine_confusion_matrix.png
│   │   └── multinomial_naive_bayes_confusion_matrix.png
│   │
│   └── what_if_experiment/              # KNN sensitivity experiment (K = 3, 5, 7, 9)
│       ├── knn_k_comparison.csv
│       └── knn_k_comparison.png
│
├── spam_classification.py                # Main end-to-end Python script
├── requirements.txt                      # Project package dependencies
├── README.md                             # Comprehensive execution & technical documentation
└── .gitignore                            # Version control exclusions
```

---

## 4. Text Preprocessing & TF-IDF Representation

### Preprocessing Strategy
1. **Case Normalization:** Converts all characters to lowercase to prevent vocabulary fragmentation (e.g., `"FREE"`, `"Free"`, and `"free"` map to the same token).
2. **Refined Stop Word Filtering:**  
   Standard English stop-word lists aggressively delete polarity markers and commercial cues. This pipeline starts with scikit-learn's standard stop words but explicitly **preserves** words critical to SMS spam identification:
   - Polarity indicators: `not`, `no`, `nor`
   - Spam/commercial triggers: `free`, `win`, `won`, `urgent`, `claim`, `prize`, `cash`, `txt`, `text`, `call`, `reply`, `stop`
3. **N-Gram Tokenization ($1, 2$):** Extracts both unigrams (`"urgent"`) and bigrams (`"cash prize"`, `"free entry"`, `"reply yes"`), capturing contextual word pairs.
4. **Frequency Thresholding (`min_df=2`):** Terms appearing only once across the entire corpus (e.g., misspelled words, one-off telephone numbers) are discarded to reduce sparsity and prevent overfitting.
5. **Sublinear TF Scaling:** Uses $1 + \log(\text{TF})$ scaling to dampen the effect of repetitive words within a single message.
6. **L2 Normalization:** Normalizes document vector lengths onto a unit hypersphere, making vector magnitudes invariant to message length.

### Prevention of Data Leakage
The `TfidfVectorizer` is fitted **strictly on the training partition** (`X_train`). The held-out test partition (`X_test`) is transformed using the learned training vocabulary and IDF weights. Preprocessing parameters are never estimated on the combined corpus.

---

## 5. Benchmark Classification Algorithms

| Model | Underlying Paradigm | Why It Is Evaluated |
| :--- | :--- | :--- |
| **Decision Tree** | Hierarchical axis-aligned recursive partitioning (Gini impurity) | Highly interpretable decision boundaries; tests how well single token splits partition sparse text spaces. |
| **K-Nearest Neighbors (KNN)** | Instance-based / memory-based lazy learner using Cosine similarity | Classifies an SMS by majority vote of its $K$ most similar messages in TF-IDF space. |
| **Support Vector Machine (LinearSVC)** | Maximum-margin linear hyperplane separation with L2 regularization | Industry standard for high-dimensional sparse text vectors; resistant to overfitting. |
| **Multinomial Naive Bayes** | Probabilistic generative model with conditional word independence | Classic text classification benchmark; handles sparse word counts with Laplace smoothing. |

---

## 6. Experimental Methodology & Evaluation Metrics

### Split Setup
- **Training Set:** 80%
- **Held-Out Test Set:** 20%
- **Stratification:** Stratified by class to preserve the natural ham/spam class ratio across both splits.
- **Random Seed:** Fixed at `42` for exact reproducibility.

### Evaluation Metrics
Because the dataset exhibits heavy class imbalance (~87% Ham vs. ~13% Spam), **Accuracy is deceptive**: a naive dummy baseline predicting `ham` for every message achieves ~87% accuracy while failing 100% of spam detections.

Therefore, `spam` is explicitly designated as the **positive class** ($1$):
- **Spam Precision ($\frac{TP}{TP + FP}$):** The proportion of messages flagged as spam that were truly spam. High precision means minimal false alarms (legitimate messages are not accidentally lost or blocked).
- **Spam Recall ($\frac{TP}{TP + FN}$):** The proportion of actual spam messages caught. High recall means few spam messages slip into the inbox.
- **Spam F1-Score ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$):** The harmonic mean, providing the single most reliable metric for model selection under class imbalance.
- **Confusion Matrix:** Evaluates exact counts of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
- **Latency Tracking:** Measures precise wall-clock training time and per-batch test inference time.

---

## 7. What-If Experiment: KNN Neighbor Sensitivity

An empirical parameter sensitivity experiment evaluates the impact of the neighbor parameter $K$ in `KNeighborsClassifier`:
- Tested values: $K \in \{3, 5, 7, 9\}$
- Fixed conditions: Identical dataset, train-test split, and TF-IDF feature matrix.
- Metrics recorded: Accuracy, Spam Precision, Spam Recall, Spam F1-Score, and inference time.
- Output artifacts: Saved to `outputs/what_if_experiment/knn_k_comparison.csv` and plotted in `knn_k_comparison.png`.

---

## 8. Step-by-Step Installation & Execution Guide

### Prerequisites
- Windows 10 or 11
- Python 3.10+ (Python 3.11 recommended)
- Visual Studio Code

### Step 1: Clone or Navigate to the Workspace
Open VS Code and navigate to the project directory:
```powershell
cd "d:\Users\shreeshanth\Desktop\IR Task 2\Task_2_Spam_Detection"
```

### Step 2: Set Up a Python Virtual Environment

**Using Windows PowerShell:**
```powershell
# Create virtual environment
py -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an `Execution_Policies` restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating).*

**Using Command Prompt (cmd.exe):**
```cmd
py -m venv .venv
.venv\Scripts\activate.bat
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Download and Place the Dataset
1. Visit the official UCI repository: [https://doi.org/10.24432/C5VG8H](https://doi.org/10.24432/C5VG8H)
2. Click **Download** to obtain `smsspamcollection.zip`.
3. Extract `SMSSpamCollection` (the extensionless plain text file) into the project's `data/` folder:
   ```text
   Task_2_Spam_Detection/data/SMSSpamCollection
   ```

### Step 5: Execute the Pipeline
Run the main script:
```powershell
python spam_classification.py
```

### Optional Command-Line Arguments
```powershell
# Specify a custom dataset path
python spam_classification.py --data-path "path/to/SMSSpamCollection"

# Adjust test split size (e.g., 25% test)
python spam_classification.py --test-size 0.25

# Change random seed
python spam_classification.py --random-state 123

# Keep exact duplicate messages instead of deduplicating
python spam_classification.py --keep-duplicates
```

---

## 9. Generated Outputs & How to Interpret Them

Once executed on the real dataset, the script automatically populates the `outputs/` directory:

1. **`outputs/dataset_summary.txt`:** Full audit report showing total rows read, encoding detected, valid rows, skipped lines, exact duplicate counts, and retained rows.
2. **`outputs/dataset_class_distribution.csv`:** Clean CSV table showing counts and percentages of legitimate (`ham`) versus spam messages.
3. **`outputs/preprocessing_examples.csv`:** Side-by-side inspection of 10 sample messages showing original text and extracted tokens.
4. **`outputs/feature_summary.txt`:** Explains vocabulary size, n-gram configuration, and lists top corpus features by mean TF-IDF weight.
5. **`outputs/model_comparison.csv`:** Summary table with Accuracy, Spam Precision, Spam Recall, Spam F1-Score, and inference times for all four models.
6. **`outputs/model_comparison.png`:** High-resolution bar chart comparing all models side-by-side.
7. **`outputs/classification_reports/*.txt`:** Per-model precision, recall, and F1-score for both `ham` and `spam` classes.
8. **`outputs/confusion_matrices/*.png`:** Heatmap plots clearly delineating:
   - **TN (Top-Left):** Genuine messages delivered safely.
   - **FP (Top-Right):** False alarms (legitimate messages blocked).
   - **FN (Bottom-Left):** Missed spam (spam delivered to user).
   - **TP (Bottom-Right):** Successfully blocked spam.
9. **`outputs/what_if_experiment/`:** CSV table and trend chart showing the impact of $K$ on KNN performance.
10. **`outputs/final_observations.txt`:** Comprehensive academic write-up synthesizing all numerical findings and providing the final model recommendation.

---

## 10. Troubleshooting & Common Issues

| Issue / Error | Cause | Solution |
| :--- | :--- | :--- |
| `[ERROR] SMS Spam Collection Dataset File Not Found` | The file `SMSSpamCollection` is missing from `data/`. | Download `smsspamcollection.zip` from UCI, extract `SMSSpamCollection`, and place it in `data/`. |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy restriction. | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell window and re-activate. |
| `ModuleNotFoundError: No module named 'sklearn'` | Dependencies are not installed in the active environment. | Run `python -m pip install -r requirements.txt` within your activated `.venv`. |
| `UnicodeDecodeError` | Encoding incompatibility in custom edited files. | The script automatically handles UTF-8, UTF-8-SIG, and Latin-1 encodings. Ensure the file has not been corrupted by an external editor. |

---

## 11. Limitations & Future Work

1. **Corpus Age & Modern Spam Evasion:** The UCI SMS Spam Collection reflects SMS patterns from the late 2000s and early 2010s. Modern spam frequently employs zero-width characters, homoglyphs, URL shorteners, and image-based spam that evade pure bag-of-words / n-gram models.
2. **Grammar & Word Order:** TF-IDF captures lexical presence and local bigrams but discards long-range semantic dependencies and grammatical context.
3. **Character-Level Obfuscations:** Typosquatting (e.g., `w!n`, `fr33`) requires character n-gram tokenization or subword embeddings (e.g., FastText or Byte-Pair Encoding) for improved robustness.

---

## 12. Final Experimental Results & Conclusion

The pipeline was executed on the genuine UCI SMS Spam Collection (5,160 retained unique messages after audit). Below are the empirical measurements calculated on the held-out test set (1,032 messages; 904 Ham, 128 Spam):

### Empirical Model Comparison Summary

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Train Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **97.97%** | **0.9147** | **0.9219** | **0.9183** | 0.0173s | 0.0007s |
| **K-Nearest Neighbors ($K=5$)** | 97.38% | 1.0000 | 0.7891 | 0.8821 | 0.0028s | 0.0709s |
| **Multinomial Naive Bayes** | 97.00% | 1.0000 | 0.7578 | 0.8622 | 0.0076s | 0.0003s |
| **Decision Tree** | 96.03% | 0.8321 | 0.8516 | 0.8417 | 0.1589s | 0.0012s |

### What-If Experiment Summary (KNN Neighbor Sensitivity)

| $K$ Neighbors | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Inference Latency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 3$** | 97.48% | 0.9811 | 0.8125 | 0.8889 | 0.0671s |
| **$K = 5$** | 97.38% | 1.0000 | 0.7891 | 0.8821 | 0.0534s |
| **$K = 7$** | 97.09% | 1.0000 | 0.7656 | 0.8673 | 0.0531s |
| **$K = 9$** | 96.61% | 1.0000 | 0.7266 | 0.8416 | 0.0516s |

### Key Academic Findings
1. **Best Model Selection:** **Support Vector Machine (LinearSVC)** is the clear overall winner. It achieved both the highest test Accuracy (**97.97%**) and the highest Spam F1-Score (**0.9183**), successfully detecting **92.19%** of actual spam messages while maintaining very high precision (91.47%).
2. **Zero False Alarm Performers:** **Multinomial Naive Bayes** and **KNN ($K \ge 5$)** achieved a perfect **1.0000 Precision** (0 False Positives out of 904 legitimate test messages). However, this came at the expense of lower Spam Recall (75.78% and 78.91% respectively), letting ~25% of spam slip through.
3. **Imbalance Effect in KNN:** The What-If experiment empirically proved that as $K$ increases from 3 to 9, Spam Recall consistently degrades (from 81.25% down to 72.66%). In high-dimensional sparse TF-IDF space, expanding the neighborhood allows the majority class (87.6% Ham) to outvote minority spam instances.
4. **Computational Efficiency:** Naive Bayes and LinearSVC exhibited sub-millisecond inference latencies per batch, whereas KNN required ~70ms due to instance-based pairwise distance calculations at test time.


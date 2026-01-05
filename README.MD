# Entity Resolution PoC

This repository contains a Proof of Concept (PoC) for an Entity Resolution system designed to identify and link client records across disparate banking systems. It attempts to  address the challenge of matching 50 million records by combining efficient blocking techniques (LSH) with a hybrid Rule-Based + Machine Learning matching engine.

## 🚀 Quick Start (Google Colab)

The easiest way to run this project is via the provided Jupyter Notebook in Google Colab.

1.  **Open `notebooks/prototype.ipynb`**.
2.  Upload the notebook to Google Colab.
3.  Run the cells sequentially. The notebook handles:
    *   Downloading the repository.
    *   Installing dependencies.
    *   Generating synthetic data (1k records).
    *   Preprocessing and Indexing (LSH + SQLite).
    *   Training the XGBoost model.
    *   Running the Evaluation pipeline.
    *   Starting the FastAPI service for real-time resolution.

## 🛠 Local Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd silent-eight-assignment
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Pipeline**:
    ```bash
    # 1. Generate Data
    python -m src.generation
    
    # 2. Preprocess & Index
    python -m src.preprocessing
    
    # 3. Train Model
    python -m src.train_model
    
    # 4. Run Matching & Evaluation
    python -m src.matching
    ```
4.  **Start API**:
    ```bash
    uvicorn src.api:app --reload
    ```

---

## 💡 Task 2: Entity Resolution Approach

### 1. Signals & Features
We utilize a multi-faceted feature set to determine identity:
*   **Strong Identifiers**: National ID (PESEL/SSN), Email, Phone Number.
*   **Fuzzy Identifiers**: First Name, Last Name (Jaro-Winkler similarity), Address (Levenshtein distance).
*   **Temporal**: Date of Birth (Exact match vs Year match).
*   **Derived**: `nid_score` (Damerau-Levenshtein for typos), `dob_match` (Exact), `year_match` (Soft).

### 2. Scaling Strategy 
The project implements a **Blocking** strategy to reduce the search space:
*   **MinHash LSH (Locality Sensitive Hashing)**: Hashes names into buckets. Only records in the same bucket are compared. This handles typos (e.g., "Kowalski" vs "Kovalski").
*   **SQL Exact Blocking**: Rapidly retrieves candidates sharing exact keys (e.g., exact National ID, exact Email, exact Phone).
*   **Result**: We reduce 50M potential comparisons to a manageable ~20-50 candidates per query.

### 3. Hybrid Matching Logic
We combine the interpretability of rules with the power of ML:
1.  **Heuristic Rules (High Precision)**:
    *   *Rule 1*: Strong ID + Strong Name = **Match**.
    *   *Rule 2*: Strong Contact (Email/Phone) + Name + Corroboration (DOB/ID) = **Match**.
    *   *Rule 3*: Exact DOB + Very Strong Name = **Match**.
    *   *Rule 4*: Strong Address + Very Strong Name = **Match**.
    *   *Rule 6*: Strong ID + Year Match + Initials/Weak Name = **Match** (Handles typo cases).
2.  **Machine Learning (XGBoost)**:
    *   Captures non-linear relationships (e.g., weak name match but strong address and partial ID).
    *   *Rule 5*: High ML Probability (>0.80) with Explainability (checks feature contributions).
3.  **Graph Clustering**:
    *   Resolves transitive links (A matches B, B matches C -> A, B, C are one entity).

### 4. Evaluation
*   **Ground Truth**: Generated using `Faker` with controlled noise (typos, transpositions), nulls and "Doppelgangers" (different people with similar attributes) to force the model to learn robust features.
*   **Metrics**: Precision (avoiding false links) and Recall (finding all links).
    *   *Current Performance*: Precision ~ 100%, Recall ~ 99% on 1k synthetic records.
*   **Trade-off**: We prioritize **Precision** (High Cost of False Positive in AML - merging wrong profiles) over Recall. Ambiguous cases are flagged as `review` for human analysts.

---

## 📂 Repository Structure

```
.
├── data/                   # Generated datasets and SQLite DB
├── models/                 # Serialized ML models and LSH indexes
├── notebooks/
│   └── prototype.ipynb     # End-to-End Demo Notebook
├── src/
│   ├── api.py              # FastAPI Service
│   ├── generation.py       # Synthetic Data Generator
│   ├── matching.py         # Core Matching Logic & Evaluation
│   ├── preprocessing.py    # Text Normalization & LSH Indexing
│   ├── settings.py         # Configuration
│   └── train_model.py      # XGBoost Training Script
├── tests/                  # Pytest Unit Tests
├── requirements.txt
├── ARCHITECTURE.MD         # System Design Documentation (Task 4)
└── README.md
```

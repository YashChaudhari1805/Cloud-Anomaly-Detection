# Hybrid Anomaly Detection Pipeline

## 1. Overview

This pipeline implements a dual-model approach to detect system anomalies in structured log data. By combining **Isolation Forest** (distance-based) and a **Deep Autoencoder** (reconstruction-based), the system captures both global outliers and subtle multivariate deviations.

## 2. Technical Architecture

### 2.1 Configuration Management (`AnomalyConfig`)

The pipeline uses a centralized `dataclass` to manage:

* **Synthetic Injection Patterns**: Defines multivariate ranges for 6 anomaly types (e.g., `resource_exhaustion`, `brute_force`).
* **Model Hyperparameters**: Controls Isolation Forest estimators and the Deep Autoencoder's latent space dimensionality.
* **Ensemble Weighting**: Default weights are set to 40% Isolation Forest and 60% Autoencoder to prioritize reconstruction accuracy.

### 2.2 Feature Engineering (`FeatureProcessor`)

Raw logs undergo complex transformations to expose system-level stress:

* **Time-based Features**: Extraction of `hour`, `day_of_week`, and `is_weekend`.
* **System Ratios**: Log-transformed network I/O ratios and hardware pressure indices.
* **Categorical Encoding**: Robust handling of high-cardinality features using `OrdinalEncoder`.
* **Scaling**: Use of `RobustScaler` to ensure the pipeline is resilient to outliers in the training data.

### 2.3 Modeling Suite

* **Isolation Forest**: A non-parametric model that isolates observations by randomly selecting a feature and a split value.
* **Deep Autoencoder**: A PyTorch-based neural network featuring `LayerNorm` and `GELU` activations. It compresses data into an 8-dimensional latent space and calculates anomaly scores based on Mean Squared Error (MSE) reconstruction loss.

---

## 3. Installation & Requirements

The following Python libraries are required:

```bash
pip install numpy pandas torch scikit-learn plotly joblib

```

---

## 4. Pipeline Execution Flow

### Step 1: Initialization

Configure the environment, seeds, and compute device (CUDA supported).

### Step 2: Data Ingestion & Injection

Synthetic anomalies are injected into the dataset based on predefined security patterns to provide ground-truth labels for evaluation.

### Step 3: Training

1. **Isolation Forest**: Trained exclusively on normal traffic profiles.
2. **Autoencoder**: Trained using a `CosineAnnealingLR` scheduler and early stopping to prevent overfitting on uniform distributions.

### Step 4: Ensemble Evaluation

Scores from both models are normalized and blended. A **Precision-Recall (PR) Curve** is used to determine the mathematically optimal F1 threshold.

---

## 5. Model Interpretability

The pipeline includes built-in visualization tools to assist in Root Cause Analysis (RCA):

* **Anomaly Landscape**: A scatter plot comparing the perspectives of the two models across various anomaly types.
* **Rolling Timeline**: Monitors detection rates and score volatility over time.
* **Root Cause Bar Chart**: Analyzes the per-feature reconstruction error to identify which specific metric triggered an anomaly.

---

## 6. Artifacts and Persistence

Upon completion, the pipeline exports the following to the `./anomaly_artifacts` directory:

* `processor.joblib`: The fitted feature scaling and encoding pipeline.
* `if_model.joblib`: The trained Isolation Forest model.
* `ae_weights.pt`: The state dictionary (weights) of the Autoencoder.
* `metrics.json`: Performance results including AUC, F1-Score, Precision, and Recall.

---

## 7. Performance Metrics

Typical results based on synthetic injection:

* **ROC_AUC**: ~0.96+
* **F1-Score**: Optimized via PR-Curve thresholding.
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

# Use non-GUI backend to ensure scripts can save PNG plots in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# Global Settings / Reproducibility
# =============================================================================

# Fix random seed to ensure consistent results across runs for report reproducibility.
SEED = 7016

# Simulate 1,200 inpatient records. This scale is sufficient to demonstrate
# classification, regression, clustering, and association rules.
N = 1200

# Model-validation settings.
CV_FOLDS = 5            # k-fold cross-validation inside the training partition
TARGET_RECALL = 0.80    # design requirement for 24-hour deterioration screening
N_BOOTSTRAP = 2000      # bootstrap resamples for test-set ROC-AUC confidence intervals

# Directory to save all analysis outputs uniformly.
OUT = Path("analysis_outputs")
OUT.mkdir(parents=True, exist_ok=True)

# Colors used consistently across all figures.
BLUE = "#4C78A8"
ORANGE = "#F58518"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert linear risk scores into probabilities between 0 and 1."""
    return 1 / (1 + np.exp(-x))


# =============================================================================
# Report Sections 1-2: Introduction and Problem Definition
# =============================================================================

# Problem: Predict whether an inpatient will experience clinical deterioration within the next 24 hours.
#
# Objectives:
#   O1. Clustering: Segment admissions into interpretable physiological profiles.
#   O2. Classification: Flag deterioration_24h with a target sensitivity of 0.80.
#   O3. Regression: Estimate length_of_stay_days for bed-capacity planning.
#   O4. Association rules: Identify clinical combinations associated with deterioration.
#
# The data simulation function below constructs a data environment for a medical monitoring system:
#   - bedside sensors: heart rate, blood pressure, respiratory rate, oxygen saturation, temperature
#   - laboratory system: lactate, white blood cells, creatinine, glucose
#   - ward operations: ward type, nursing alerts, mobility score


def simulate_patient_monitoring_data(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """
    Simulate inpatient monitoring data.
    """
    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1. Demographics
    # -----------------------------
    age = np.clip(rng.normal(62, 16, n), 18, 92).round(0)
    sex = rng.choice(["Female", "Male"], size=n, p=[0.48, 0.52])

    # -----------------------------
    # 2. Ward context
    # -----------------------------
    # ICU patients have higher overall risk, high dependency wards are moderate, general wards are lower.
    ward = rng.choice(["General ward", "High dependency", "ICU"], size=n, p=[0.53, 0.27, 0.20])
    ward_risk = pd.Series(ward).map(
        {"General ward": -0.35, "High dependency": 0.35, "ICU": 0.85}
    ).to_numpy()

    # -----------------------------
    # 3. Clinical history: Comorbidities
    # -----------------------------
    # Older patients generally have a higher number of comorbidities.
    comorbidity_count = rng.poisson(np.clip((age - 35) / 28, 0.2, 2.4)).clip(0, 6)
    diabetes = rng.binomial(
        1,
        sigmoid(-2.0 + 0.035 * (age - 50) + 0.25 * comorbidity_count),
    )
    chronic_kidney_disease = rng.binomial(
        1,
        sigmoid(-2.6 + 0.04 * (age - 55) + 0.3 * comorbidity_count),
    )

    # -----------------------------
    # 4. Latent severity
    # -----------------------------
    latent_severity = (
        rng.normal(0, 0.9, n)
        + 0.025 * (age - 60)
        + 0.22 * comorbidity_count
        + 0.55 * diabetes
        + 0.65 * chronic_kidney_disease
        + ward_risk
    )

    # -----------------------------
    # 5. Vital signs
    # -----------------------------
    # Higher severity typically increases heart rate, respiratory rate, and temperature,
    # while blood pressure and oxygen saturation may drop.
    avg_hr = np.clip(rng.normal(82 + 8.5 * latent_severity, 10, n), 48, 155)
    max_hr = np.clip(avg_hr + rng.gamma(3.0, 4.2, n), 60, 185)
    min_sbp = np.clip(rng.normal(121 - 7.8 * latent_severity, 12, n), 65, 178)
    max_rr = np.clip(rng.normal(19 + 2.8 * latent_severity, 3.2, n), 10, 38)
    min_spo2 = np.clip(rng.normal(96.5 - 2.1 * latent_severity, 2.5, n), 78, 100)
    max_temp = np.clip(rng.normal(37.2 + 0.34 * latent_severity, 0.55, n), 35.2, 40.6)

    # -----------------------------
    # 6. Laboratory measurements
    # -----------------------------
    lactate = np.clip(rng.lognormal(0.28 + 0.24 * latent_severity, 0.33, n), 0.4, 7.5)
    wbc = np.clip(rng.normal(8.4 + 1.25 * latent_severity, 2.2, n), 2.0, 24.0)
    creatinine = np.clip(
        rng.lognormal(
            -0.08 + 0.18 * latent_severity + 0.55 * chronic_kidney_disease,
            0.28,
            n,
        ),
        0.35,
        5.8,
    )
    glucose = np.clip(rng.normal(118 + 24 * diabetes + 8 * latent_severity, 28, n), 58, 340)

    # -----------------------------
    # 7. Operational variables
    # -----------------------------
    nursing_alerts = rng.poisson(np.clip(0.45 + 0.35 * latent_severity, 0.05, 4.2))
    mobility_score = np.clip(np.round(rng.normal(4.1 - 0.43 * latent_severity, 0.9, n)), 1, 5)

    # -----------------------------
    # 8. Classification target: 24-hour deterioration
    # -----------------------------
    # Low oxygen, high lactate, high RR, low BP, and frequent nursing alerts increase the probability of deterioration.
    risk_logit = (
        -2.45
        + 0.62 * latent_severity
        + 0.42 * (lactate > 2.2)
        + 0.38 * (min_spo2 < 92)
        + 0.35 * (max_rr > 24)
        + 0.31 * (min_sbp < 95)
        + 0.17 * nursing_alerts
        + 0.25 * chronic_kidney_disease
    )
    deterioration_24h = rng.binomial(1, sigmoid(risk_logit))

    # -----------------------------
    # 9. Regression target: Length of stay
    # -----------------------------
    # Deterioration, ICU admission, elevated lactate, and high comorbidity counts typically extend length of stay.
    los_days = np.clip(
        2.1
        + 0.9 * latent_severity
        + 2.1 * deterioration_24h
        + 0.22 * comorbidity_count
        + 0.8 * (ward == "ICU")
        + 0.35 * lactate
        + rng.normal(0, 1.2, n),
        0.4,
        19.5,
    )

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "ward_type": ward,
            "comorbidity_count": comorbidity_count,
            "diabetes": diabetes,
            "chronic_kidney_disease": chronic_kidney_disease,
            "avg_hr": avg_hr.round(1),
            "max_hr": max_hr.round(1),
            "min_sbp": min_sbp.round(1),
            "max_rr": max_rr.round(1),
            "min_spo2": min_spo2.round(1),
            "max_temp": max_temp.round(1),
            "lactate": lactate.round(2),
            "wbc": wbc.round(1),
            "creatinine": creatinine.round(2),
            "glucose": glucose.round(1),
            "nursing_alerts": nursing_alerts,
            "mobility_score": mobility_score.astype(int),
            "deterioration_24h": deterioration_24h,
            "length_of_stay_days": los_days.round(1),
        }
    )

    # Introduce common issues found in real-world clinical data:
    #   - Missing values in some laboratory or sensor readings (missing completely at random)
    #   - A few anomalous extreme values
    missing_rates = {
        "lactate": 0.08,
        "creatinine": 0.05,
        "wbc": 0.04,
        "glucose": 0.04,
        "min_spo2": 0.03,
        "min_sbp": 0.03,
    }
    for col, rate in missing_rates.items():
        df.loc[rng.random(n) < rate, col] = np.nan

    spike_hr = rng.choice(n, size=8, replace=False)
    df.loc[spike_hr, "max_hr"] *= rng.uniform(1.25, 1.45, 8)
    spike_glucose = rng.choice(n, size=6, replace=False)
    df.loc[spike_glucose, "glucose"] *= rng.uniform(1.35, 1.7, 6)

    # Record which rows received injected spikes so the outlier handling can be audited.
    df.attrs["injected_spikes"] = {"max_hr": spike_hr.tolist(), "glucose": spike_glucose.tolist()}
    return df


# =============================================================================
# Report Section 3: Data Description & EDA
# =============================================================================


def winsorize_frame(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Outlier handling via winsorization for the descriptive analyses
    (EDA, clustering, association rules), which use the full dataset.

    Implementation: Applies 1st/99th percentile winsorization to cap extreme sensor or lab readings.

    Rationale:
        Extreme values in medical data can represent genuinely high-risk cases or sensor noise.
        Winsorization restricts overly extreme values without removing clinical records entirely.
    """
    out = df.copy()
    bounds = {}

    for col in columns:
        lo, hi = out[col].quantile([0.01, 0.99])
        out[col] = out[col].clip(lo, hi)
        bounds[col] = {"p01": float(lo), "p99": float(hi)}

    return out, bounds


class PercentileClipper(BaseEstimator, TransformerMixin):
    """
    Winsorization step for the supervised pipelines.

    The 1st/99th percentile bounds are learned in fit(), i.e. on the training data only,
    and then applied unchanged to validation and test data. This prevents information
    from the test distribution leaking into preprocessing.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """
    Construct a unified preprocessing pipeline. Every step is fitted on training data only.

    Numerical Variables:
        1. Median imputation: Fill missing values with the training median.
        2. PercentileClipper: Cap values at the training 1st/99th percentiles.
        3. StandardScaler: Standardize data, suitable for linear models like Logistic/Ridge.

    Categorical Variables:
        1. Most frequent imputation: Fill missing values with the mode.
        2. OneHotEncoder: Transform into dummy variables readable by models.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", PercentileClipper()),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ]
    )


def describe_by_outcome(raw: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Descriptive statistics by 24-hour outcome on the raw data (missing values excluded).

    The standardized mean difference (SMD) = (mean_det - mean_no) / pooled SD
    expresses the separation between groups on a common scale.
    """
    rows = []
    for col in columns:
        no = raw.loc[raw["deterioration_24h"] == 0, col].dropna()
        yes = raw.loc[raw["deterioration_24h"] == 1, col].dropna()
        pooled_sd = np.sqrt((no.var(ddof=1) + yes.var(ddof=1)) / 2)
        rows.append(
            {
                "variable": col,
                "no_deterioration_mean": float(no.mean()),
                "no_deterioration_sd": float(no.std(ddof=1)),
                "deterioration_mean": float(yes.mean()),
                "deterioration_sd": float(yes.std(ddof=1)),
                "smd": float((yes.mean() - no.mean()) / pooled_sd),
            }
        )
    return pd.DataFrame(rows)


def plot_eda(df: pd.DataFrame) -> None:
    """
    Generate EDA charts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.countplot(data=df, x="deterioration_24h", ax=axes[0], color=BLUE)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No deterioration", "Deterioration"])
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Encounters")
    axes[0].set_title("24-hour deterioration outcome")

    sns.histplot(
        data=df,
        x="length_of_stay_days",
        hue="deterioration_24h",
        bins=26,
        ax=axes[1],
        palette=[BLUE, "#E45756"],
    )
    axes[1].set_xlabel("Length of stay (days)")
    axes[1].set_ylabel("Encounters")
    axes[1].set_title("Length of stay by outcome")

    fig.tight_layout()
    fig.savefig(OUT / "eda_outcomes.png", dpi=220)
    plt.close(fig)


# =============================================================================
# Report Section 4: Proposed AI-based Solution Concept
# =============================================================================

# Unified Workflow:
#   1. Data ingestion: Load patient monitoring data.
#   2. Preprocessing: Handle missing values, outliers, standardization, and encoding (fitted on training data).
#   3. Analytics engine: Perform classification, regression, clustering, and association rules.
#   4. Clinical interface: Output risk scores, LOS forecasts, cluster labels, and rule explanations.
#   5. Monitoring: Conduct calibration, fairness checks, and threshold reviews.
#
# Validation design for the supervised models:
#   - A stratified 25% test partition is held out and never used for tuning or threshold selection.
#   - Hyperparameters are selected by 5-fold cross-validation inside the 75% training partition.
#   - The decision threshold is selected on out-of-fold training predictions to meet TARGET_RECALL.
#   - Test-set ROC-AUC is reported with bootstrap 95% confidence intervals.
#
# At the code level, the main() function executes the complete pipeline in this sequence.


# =============================================================================
# Report Section 5.1: Patient Clustering
# =============================================================================


def run_clustering(df: pd.DataFrame, numeric: list[str]) -> tuple[dict, pd.DataFrame]:
    """
    K-means clustering: Discover patient risk phenotypes.
    Segment patients into low, medium, and high-risk cohorts using K-means.
        Note: K-means relies on distance, so differing feature scales will bias the distance calculation.
    """
    scaled = StandardScaler().fit_transform(df[numeric])

    silhouette = {}
    inertia = {}

    # Compare candidate cluster counts from k=2 to k=6.
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=30, random_state=SEED)
        labels = model.fit_predict(scaled)
        silhouette[str(k)] = float(silhouette_score(scaled, labels))
        inertia[str(k)] = float(model.inertia_)

    # Select k=3: the inertia elbow occurs at k=3, and three clusters give the clearest
    # clinical interpretation (lower- / intermediate- / higher-severity profiles).
    selected_k = 3
    final_model = KMeans(n_clusters=selected_k, n_init=50, random_state=SEED)

    df = df.copy()
    df["cluster"] = final_model.fit_predict(scaled)

    # Generate a clinical profile for each cluster for report tables and heatmaps.
    cluster_profile = (
        df.groupby("cluster")
        .agg(
            n=("age", "size"),
            age=("age", "mean"),
            lactate=("lactate", "mean"),
            min_spo2=("min_spo2", "mean"),
            max_rr=("max_rr", "mean"),
            min_sbp=("min_sbp", "mean"),
            deterioration_rate=("deterioration_24h", "mean"),
            length_of_stay=("length_of_stay_days", "mean"),
        )
        .reset_index()
    )

    results = {
        "selected_k": selected_k,
        "silhouette": silhouette,
        "inertia": inertia,
        "profile": cluster_profile.to_dict(orient="records"),
    }
    return results, cluster_profile


def plot_clustering(cluster_profile: pd.DataFrame, silhouette: dict, inertia: dict) -> None:
    """Generate the K-means selection plot (inertia and silhouette) and the cluster heatmap."""
    x_values = list(map(int, silhouette.keys()))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(x_values, [inertia[str(k)] for k in x_values], marker="o", color=BLUE)
    axes[0].set_ylabel("Inertia (within-cluster SS)")
    axes[0].set_title("Elbow criterion")
    axes[1].plot(x_values, [silhouette[str(k)] for k in x_values], marker="o", color=BLUE)
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette criterion")
    for ax in axes:
        ax.set_xlabel("Number of clusters (k)")
        ax.set_xticks(x_values)
        ax.axvline(3, color="#777777", linestyle=":", linewidth=1.2)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "kmeans_selection.png", dpi=220)
    plt.close(fig)

    # Standardize cluster means across different scales to facilitate direct comparison in a single heatmap.
    profile_features = ["age", "lactate", "min_spo2", "max_rr", "min_sbp", "deterioration_rate"]
    display_profile = cluster_profile.set_index("cluster")[profile_features].copy()
    display_profile = (display_profile - display_profile.mean()) / display_profile.std(ddof=0)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    sns.heatmap(
        display_profile,
        cmap="vlag",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Cluster clinical profile, standardized means")
    ax.set_xlabel("Profile feature")
    ax.set_ylabel("Cluster")
    fig.tight_layout()
    fig.savefig(OUT / "cluster_profile_heatmap.png", dpi=220)
    plt.close(fig)


# =============================================================================
# Report Section 5.2: Deterioration Classification
# =============================================================================


def classification_metrics(y_true, y_prob, threshold: float) -> dict:
    """
    Calculate classification evaluation metrics at a given decision threshold.
    ROC-AUC and PR-AUC (average precision) are threshold-independent.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
        "flagged": int(y_pred.sum()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def select_threshold(y_true, oof_prob, target_recall: float = TARGET_RECALL) -> tuple[float, pd.DataFrame]:
    """
    Select the decision threshold on out-of-fold (OOF) training predictions.

    Rule: the HIGHEST threshold whose OOF recall is still >= target_recall.
    A higher threshold raises fewer alerts, so this rule meets the sensitivity
    requirement with the smallest alert volume. The test set is never used here.
    """
    y_true = np.asarray(y_true)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    rows = []
    for t in grid:
        pred = (oof_prob >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "recall": float(recall_score(y_true, pred)),
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "alert_rate": float(pred.mean()),
            }
        )
    curve = pd.DataFrame(rows)
    eligible = curve[curve["recall"] >= target_recall]
    threshold = float(eligible["threshold"].max())
    return threshold, curve


def tune_classifier(pipeline: Pipeline, grid: dict, X_train, y_train, cv) -> dict:
    """
    Tune hyperparameters with k-fold cross-validation on the training partition only,
    then derive out-of-fold probabilities for threshold selection.
    """
    search = GridSearchCV(
        pipeline,
        grid,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        refit="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    i = search.best_index_

    oof_prob = cross_val_predict(
        clone(search.best_estimator_), X_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold, curve = select_threshold(y_train, oof_prob)
    at_t = curve.loc[curve["threshold"] == threshold].iloc[0]

    return {
        "model": search.best_estimator_,
        "best_params": {k.split("__")[1]: v for k, v in search.best_params_.items()},
        "cv_roc_auc_mean": float(search.cv_results_["mean_test_roc_auc"][i]),
        "cv_roc_auc_sd": float(search.cv_results_["std_test_roc_auc"][i]),
        "cv_pr_auc_mean": float(search.cv_results_["mean_test_pr_auc"][i]),
        "cv_pr_auc_sd": float(search.cv_results_["std_test_pr_auc"][i]),
        "threshold": threshold,
        "oof_recall_at_threshold": float(at_t["recall"]),
        "oof_precision_at_threshold": float(at_t["precision"]),
        "oof_alert_rate_at_threshold": float(at_t["alert_rate"]),
        "threshold_curve": curve,
    }


def bootstrap_auc(y_true, prob_a, prob_b, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Bootstrap 95% confidence intervals for two test-set ROC-AUCs and their difference (b - a)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    auc_a, auc_b, diff = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue
        a = roc_auc_score(y_true[idx], prob_a[idx])
        b = roc_auc_score(y_true[idx], prob_b[idx])
        auc_a.append(a)
        auc_b.append(b)
        diff.append(b - a)

    def ci(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {"auc_a_ci": ci(auc_a), "auc_b_ci": ci(auc_b), "diff_b_minus_a_ci": ci(diff)}


def run_classification(X: pd.DataFrame, y: pd.Series, numeric: list[str], categorical: list[str]) -> dict:
    """
    Train two classification models:
        1. Logistic Regression - interpretable linear baseline
        2. Random Forest Classifier - captures non-linear patterns and interactions
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=SEED,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    # Logistic Regression:
    #   - class_weight="balanced" accounts for the low proportion of positive (deterioration) events
    #   - C (inverse regularization strength) is tuned by cross-validation
    logistic = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        ]
    )
    logistic_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

    # Random Forest:
    #   - 360 trees; max_depth and min_samples_leaf are tuned to control overfitting
    random_forest = Pipeline(
        [
            ("prep", make_preprocessor(numeric, categorical)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=360,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                ),
            ),
        ]
    )
    random_forest_grid = {"clf__max_depth": [4, 7, 10], "clf__min_samples_leaf": [4, 8, 16]}

    tuned = {
        "logistic_regression": tune_classifier(logistic, logistic_grid, X_train, y_train, cv),
        "random_forest": tune_classifier(random_forest, random_forest_grid, X_train, y_train, cv),
    }

    results, probs, curves = {}, {}, {}
    for name, t in tuned.items():
        prob = t["model"].predict_proba(X_test)[:, 1]
        probs[name] = prob
        curves[name] = t["threshold_curve"]
        results[name] = {
            "best_params": t["best_params"],
            "cv_roc_auc_mean": t["cv_roc_auc_mean"],
            "cv_roc_auc_sd": t["cv_roc_auc_sd"],
            "cv_pr_auc_mean": t["cv_pr_auc_mean"],
            "cv_pr_auc_sd": t["cv_pr_auc_sd"],
            "oof_recall_at_threshold": t["oof_recall_at_threshold"],
            "oof_precision_at_threshold": t["oof_precision_at_threshold"],
            "oof_alert_rate_at_threshold": t["oof_alert_rate_at_threshold"],
            "test": classification_metrics(y_test, prob, t["threshold"]),
        }

    boot = bootstrap_auc(y_test, probs["logistic_regression"], probs["random_forest"])
    results["bootstrap"] = {
        "logistic_regression_auc_ci": boot["auc_a_ci"],
        "random_forest_auc_ci": boot["auc_b_ci"],
        "rf_minus_lr_auc_ci": boot["diff_b_minus_a_ci"],
        "n_resamples": N_BOOTSTRAP,
    }
    results["test_size"] = int(len(y_test))
    results["test_positives"] = int(y_test.sum())

    models = {name: t["model"] for name, t in tuned.items()}
    return {
        "results": results,
        "models": models,
        "X_test": X_test,
        "y_test": y_test,
        "probs": probs,
        "curves": curves,
    }


def plot_threshold_analysis(curves: dict, results: dict) -> None:
    """Out-of-fold recall and precision across thresholds, with the selected threshold marked."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
    for ax, (name, title) in zip(
        axes, [("logistic_regression", "Logistic regression"), ("random_forest", "Random forest")]
    ):
        curve = curves[name]
        t = results[name]["test"]["threshold"]
        ax.plot(curve["threshold"], curve["recall"], color=BLUE, linewidth=2, label="Recall (OOF)")
        ax.plot(curve["threshold"], curve["precision"], color=ORANGE, linewidth=2, linestyle="--",
                label="Precision (OOF)")
        ax.axhline(TARGET_RECALL, color="#777777", linestyle=":", linewidth=1.2, label="Target recall 0.80")
        ax.axvline(t, color="#333333", linewidth=1)
        ax.text(t + 0.015, 0.05, f"selected = {t:.2f}", fontsize=9, color="#333333")
        ax.set_title(title)
        ax.set_xlabel("Decision threshold")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Out-of-fold score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "threshold_analysis.png", dpi=220)
    plt.close(fig)


def plot_classification(y_test, probs: dict, results: dict) -> None:
    """ROC curves for both models and confusion matrices at each model's selected threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})

    for name, label, color, style in [
        ("logistic_regression", "Logistic regression", BLUE, "-"),
        ("random_forest", "Random forest", ORANGE, "--"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        auc = results[name]["test"]["roc_auc"]
        axes[0].plot(fpr, tpr, label=f"{label} AUC={auc:.3f}", color=color, linewidth=2, linestyle=style)

    axes[0].plot([0, 1], [0, 1], linestyle=":", color="#777777")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Test-set ROC curves")
    axes[0].legend(loc="lower right", fontsize=8)

    for ax, (name, label) in zip(
        axes[1:], [("logistic_regression", "Logistic regression"), ("random_forest", "Random forest")]
    ):
        m = results[name]["test"]
        sns.heatmap(np.array(m["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"fontsize": 12})
        ax.set_title(f"{label} (threshold {m['threshold']:.2f})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["No", "Yes"], rotation=0)

    fig.tight_layout()
    fig.savefig(OUT / "classification_performance.png", dpi=220)
    plt.close(fig)


def compute_permutation_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Permutation importance on the held-out test set (drop in ROC-AUC when one input is shuffled).
    Unlike impurity-based importance, it is not biased toward continuous variables, and each
    original input (including the one-hot encoded ward type) is permuted as a whole.
    """
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=SEED, n_jobs=-1
    )
    return pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_sd": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


FEATURE_LABELS = {
    "age": "Age",
    "comorbidity_count": "Comorbidity count",
    "avg_hr": "Average heart rate",
    "max_hr": "Maximum heart rate",
    "min_sbp": "Minimum systolic BP",
    "max_rr": "Maximum respiratory rate",
    "min_spo2": "Minimum SpO$_2$",
    "max_temp": "Maximum temperature",
    "lactate": "Lactate",
    "wbc": "White blood cells",
    "creatinine": "Creatinine",
    "glucose": "Glucose",
    "nursing_alerts": "Nursing alerts",
    "mobility_score": "Mobility score",
    "sex": "Sex",
    "ward_type": "Ward type",
    "diabetes": "Diabetes",
    "chronic_kidney_disease": "Chronic kidney disease",
}


def plot_feature_importance(importances: pd.DataFrame) -> None:
    """Plot the 10 most important inputs with their variability across permutations."""
    top_imp = importances.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(
        [FEATURE_LABELS.get(f, f) for f in top_imp["feature"]],
        top_imp["importance_mean"],
        xerr=top_imp["importance_sd"],
        color=BLUE,
        error_kw={"ecolor": "#555555", "capsize": 2, "linewidth": 1},
    )
    ax.set_xlabel("Mean decrease in test ROC-AUC when permuted (±1 SD, 30 repeats)")
    ax.set_title("Random forest permutation importance")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=220)
    plt.close(fig)


# =============================================================================
# Report Section 5.3: Length-of-Stay Regression
# =============================================================================


def run_regression(X: pd.DataFrame, y_los: pd.Series, numeric: list[str], categorical: list[str]) -> dict:
    """
    Train models to predict Length of Stay (LOS).

    Target Variable:
        length_of_stay_days

    Models:
        1. Ridge Linear Regression (alpha tuned)
        2. Random Forest Regression (depth and leaf size tuned)

    The preferred model is chosen by cross-validated RMSE on the training partition,
    not by test-set performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_los,
        test_size=0.25,
        random_state=SEED,
    )
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    candidates = {
        # Ridge Regression: linear regression with L2 regularization; interpretable for resource planning.
        "ridge_linear_regression": (
            Pipeline([("prep", make_preprocessor(numeric, categorical)), ("reg", Ridge())]),
            {"reg__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        # Random Forest Regression: tests whether a non-linear model improves performance.
        "random_forest_regression": (
            Pipeline(
                [
                    ("prep", make_preprocessor(numeric, categorical)),
                    ("reg", RandomForestRegressor(n_estimators=260, random_state=SEED, n_jobs=-1)),
                ]
            ),
            {"reg__max_depth": [4, 7, 10], "reg__min_samples_leaf": [4, 7, 14]},
        ),
    }

    results, predictions = {}, {}
    for name, (pipeline, grid) in candidates.items():
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv,
            scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
            refit="rmse",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        i = search.best_index_
        pred = search.best_estimator_.predict(X_test)
        predictions[name] = pred

        # MAE and RMSE are expressed in days for direct operational interpretation.
        results[name] = {
            "best_params": {k.split("__")[1]: v for k, v in search.best_params_.items()},
            "cv_rmse_mean": float(-search.cv_results_["mean_test_rmse"][i]),
            "cv_rmse_sd": float(search.cv_results_["std_test_rmse"][i]),
            "cv_mae_mean": float(-search.cv_results_["mean_test_mae"][i]),
            "cv_mae_sd": float(search.cv_results_["std_test_mae"][i]),
            "cv_r2_mean": float(search.cv_results_["mean_test_r2"][i]),
            "cv_r2_sd": float(search.cv_results_["std_test_r2"][i]),
            "test": {
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "r2": float(r2_score(y_test, pred)),
                # Ward-level view: total predicted vs observed bed-days in the test partition.
                "total_bed_days_observed": float(y_test.sum()),
                "total_bed_days_predicted": float(pred.sum()),
            },
        }

    preferred = min(results, key=lambda k: results[k]["cv_rmse_mean"])
    results["preferred_model"] = preferred
    results["test_size"] = int(len(y_test))
    return {"results": results, "y_test": y_test, "predictions": predictions}


def plot_regression(y_test: pd.Series, pred: np.ndarray, title: str) -> None:
    """Predicted versus observed LOS on the held-out test partition."""
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(y_test, pred, s=16, alpha=0.6, color=BLUE, edgecolor="white", linewidth=0.4)
    lim = [0, float(max(y_test.max(), pred.max())) + 0.5]
    ax.plot(lim, lim, color="#777777", linestyle=":", linewidth=1.2, label="Perfect prediction")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Observed LOS (days)")
    ax.set_ylabel("Predicted LOS (days)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "regression_predicted_vs_observed.png", dpi=220)
    plt.close(fig)


# =============================================================================
# Report Section 5.4: Association-Rule Analysis
# =============================================================================


def mine_association_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mine clinical association rules:
        IF clinical condition X THEN deterioration_24h

    Metrics:
        support: The proportion of cases where condition X and deterioration co-occur.
        confidence: The probability of deterioration given that condition X is present.
        lift: confidence / baseline deterioration rate

    lift > 1:
        Indicates that patients matching this condition subset have a higher likelihood of deterioration
        compared to the general baseline.
    """
    item_df = pd.DataFrame(
        {
            "Age >=75": df["age"] >= 75,
            "Comorbidity count >=3": df["comorbidity_count"] >= 3,
            "ICU admission": df["ward_type"] == "ICU",
            "High dependency ward": df["ward_type"] == "High dependency",
            "SpO2 <92%": df["min_spo2"] < 92,
            "Respiratory rate >24": df["max_rr"] > 24,
            "Max heart rate >=115": df["max_hr"] >= 115,
            "Systolic BP <95": df["min_sbp"] < 95,
            "Lactate >=2.2": df["lactate"] >= 2.2,
            "Creatinine >=1.5": df["creatinine"] >= 1.5,
            "WBC >=12": df["wbc"] >= 12,
            "Temperature >=38.3": df["max_temp"] >= 38.3,
            "Glucose >=180": df["glucose"] >= 180,
            "Nursing alerts >=3": df["nursing_alerts"] >= 3,
            "Mobility score <=2": df["mobility_score"] <= 2,
        }
    ).fillna(False)

    outcome = df["deterioration_24h"].astype(bool)
    baseline_deterioration_rate = outcome.mean()
    rows = []

    # Generate 1-item, 2-item, and 3-item antecedent combinations (15 + 105 + 455 = 575 candidates).
    for size in [1, 2, 3]:
        for antecedent in itertools.combinations(item_df.columns, size):
            mask = item_df[list(antecedent)].all(axis=1)
            support_x = mask.mean()

            # Filter out rare antecedents (support < 0.05) to avoid spurious rules.
            if support_x < 0.05:
                continue

            support_xy = (mask & outcome).mean()
            confidence = support_xy / support_x if support_x else 0
            lift = confidence / baseline_deterioration_rate if baseline_deterioration_rate else 0

            # Retain rules that satisfy support, confidence, and lift thresholds simultaneously.
            if support_xy >= 0.035 and confidence >= 0.35 and lift >= 1.20:
                rows.append(
                    {
                        "antecedent": " + ".join(antecedent),
                        "support": support_xy,
                        "confidence": confidence,
                        "lift": lift,
                    }
                )

    return pd.DataFrame(rows).sort_values(["lift", "confidence", "support"], ascending=False)


def plot_association_rules(rules: pd.DataFrame) -> None:
    """Plot the association rules with the highest lift."""
    top_rules = rules.head(8).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(top_rules["antecedent"], top_rules["lift"], color="#B279A2")
    ax.set_xlabel("Lift")
    ax.set_title("Association rules leading to deterioration")
    fig.tight_layout()
    fig.savefig(OUT / "association_rules_lift.png", dpi=220)
    plt.close(fig)


# =============================================================================
# Report Sections 5.5-5.6: Business Intelligence, Limitations, and Risk Controls
# =============================================================================

# Model outputs support typical Business Intelligence (BI) scenarios:
#   - Ward risk dashboard: Driven by the classification risk scores and the selected threshold.
#   - Cluster census: Categorized via K-means cluster labels.
#   - LOS forecast: Enabled via the regression output module (total bed-days per ward).
#   - Rule alerts: Implemented via association rule triggers.
#
# Key Limitations:
#   - This dataset consists entirely of simulated data and cannot be deployed directly in clinical practice.
#   - Real-world deployment demands rigorous external validation, calibration monitoring,
#     fairness metrics assessments, privacy reviews, and human-in-the-loop oversight.


# =============================================================================
# Main End-to-End Pipeline
# =============================================================================


def main() -> None:
    """Execute the complete end-to-end analytics pipeline in report sequence."""

    # -------------------------------------------------------------------------
    # Global font setting: Times New Roman, with metric-compatible fallbacks
    # (Liberation Serif / TeX Gyre Termes) on systems where it is not installed.
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=0.95)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"]
    plt.rcParams["mathtext.fontset"] = "stix"

    # Numerical variables as specified in the data dictionary.
    numeric = [
        "age",
        "comorbidity_count",
        "avg_hr",
        "max_hr",
        "min_sbp",
        "max_rr",
        "min_spo2",
        "max_temp",
        "lactate",
        "wbc",
        "creatinine",
        "glucose",
        "nursing_alerts",
        "mobility_score",
    ]

    # Categorical variables as specified in the data dictionary.
    categorical = ["sex", "ward_type", "diabetes", "chronic_kidney_disease"]

    # -------------------------------------------------------------------------
    # Step 1. Simulate dataset
    # -------------------------------------------------------------------------
    raw = simulate_patient_monitoring_data()
    raw.to_csv(OUT / "simulated_patient_monitoring_raw.csv", index=False)

    # Descriptive statistics by outcome (raw data, missing values excluded).
    descriptive = describe_by_outcome(raw, numeric + ["length_of_stay_days"])
    descriptive.to_csv(OUT / "descriptive_by_outcome.csv", index=False)

    # -------------------------------------------------------------------------
    # Step 2. Data preprocessing for the descriptive analyses
    #         (EDA, clustering and association rules use the full dataset)
    # -------------------------------------------------------------------------
    clean = raw.copy()

    # Impute missing values for numerical variables using the median.
    for col in numeric:
        clean[col] = clean[col].fillna(clean[col].median())

    # Apply 1%/99% winsorization to numerical variables.
    clean_before_clip = clean.copy()
    clean, winsor_bounds = winsorize_frame(clean, numeric)
    clipped_counts = {col: int((clean_before_clip[col] != clean[col]).sum()) for col in numeric}
    spikes = raw.attrs["injected_spikes"]
    spikes_capped = {
        col: {
            "injected": len(idx),
            "capped_by_clipping": int((clean_before_clip.loc[idx, col] != clean.loc[idx, col]).sum()),
        }
        for col, idx in spikes.items()
    }
    clean.to_csv(OUT / "simulated_patient_monitoring_preprocessed.csv", index=False)

    # Generate EDA charts.
    plot_eda(clean)

    # -------------------------------------------------------------------------
    # Step 3. Clustering
    # -------------------------------------------------------------------------
    clustering, cluster_profile = run_clustering(clean, numeric)
    cluster_profile.to_csv(OUT / "cluster_profile.csv", index=False)
    plot_clustering(cluster_profile, clustering["silhouette"], clustering["inertia"])

    # -------------------------------------------------------------------------
    # Step 4. Classification
    #         Supervised models start from the RAW data: imputation and clipping are
    #         learned inside each pipeline from training data only (no test leakage).
    # -------------------------------------------------------------------------
    X = raw[numeric + categorical]
    y_class = raw["deterioration_24h"]
    y_reg = raw["length_of_stay_days"]

    clf = run_classification(X, y_class, numeric, categorical)
    classification = clf["results"]
    plot_threshold_analysis(clf["curves"], classification)
    plot_classification(clf["y_test"], clf["probs"], classification)
    pd.concat(
        {name: curve for name, curve in clf["curves"].items()}, names=["model", "row"]
    ).to_csv(OUT / "threshold_analysis_oof.csv")

    # Random Forest permutation importance for model explanation.
    feature_importance = compute_permutation_importance(
        clf["models"]["random_forest"], clf["X_test"], clf["y_test"]
    )
    feature_importance.to_csv(OUT / "permutation_importance.csv", index=False)
    plot_feature_importance(feature_importance)

    # -------------------------------------------------------------------------
    # Step 5. Regression
    # -------------------------------------------------------------------------
    reg = run_regression(X, y_reg, numeric, categorical)
    regression = reg["results"]
    preferred = regression["preferred_model"]
    plot_regression(
        reg["y_test"],
        reg["predictions"][preferred],
        "Ridge regression: test-set LOS" if preferred.startswith("ridge") else "Random forest: test-set LOS",
    )

    # -------------------------------------------------------------------------
    # Step 6. Association rule mining
    # -------------------------------------------------------------------------
    association_rules = mine_association_rules(clean)
    association_rules.head(20).to_csv(OUT / "association_rules_top20.csv", index=False)
    plot_association_rules(association_rules)

    # -------------------------------------------------------------------------
    # Step 7. Save all results for report tables
    # -------------------------------------------------------------------------
    metrics = {
        "dataset": {
            "n": int(len(raw)),
            "deterioration_rate": float(raw["deterioration_24h"].mean()),
            "mean_los_days": float(raw["length_of_stay_days"].mean()),
            "missingness_percent": {
                col: float(raw[col].isna().mean() * 100)
                for col in raw.columns
                if raw[col].isna().any()
            },
            "winsor_bounds": winsor_bounds,
            "values_clipped_full_dataset": clipped_counts,
            "injected_spikes": spikes_capped,
        },
        "descriptive_by_outcome": descriptive.to_dict(orient="records"),
        "clustering": clustering,
        "classification": classification,
        "regression": regression,
        "association_rules": association_rules.head(10).to_dict(orient="records"),
        "permutation_importance": feature_importance.head(10).to_dict(orient="records"),
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Analysis completed. Outputs saved in:", OUT.resolve())


if __name__ == "__main__":
    main()

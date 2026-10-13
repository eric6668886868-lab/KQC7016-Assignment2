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
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# Report Section: Global Settings / Reproducibility
# =============================================================================

# Fix random seed to ensure consistent results across runs for report reproducibility.
SEED = 7016

# Simulate 1,200 inpatient records. This scale is sufficient to demonstrate 
# classification, regression, clustering, and association rules.
N = 1200

# Directory to save all analysis outputs uniformly.
OUT = Path("analysis_outputs")
OUT.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Convert linear risk scores into probabilities between 0 and 1."""
    return 1 / (1 + np.exp(-x))


# =============================================================================
# Report Section 1-2: Introduction and Problem Definition
# =============================================================================

# Problem: Predict whether an inpatient will experience clinical deterioration within the next 24 hours.
#
# Tasks:
#   1. Classification: Predict deterioration_24h (binary classification).
#   2. Regression: Predict length_of_stay_days (continuous variable).
#   3. Clustering: Discover low-risk, medium-risk, and high-risk patient phenotypes.
#   4. Association rules: Identify clinical combinations that frequently lead to deterioration.
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
    #   - Missing values in some laboratory or sensor readings
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

    df.loc[rng.choice(n, size=8, replace=False), "max_hr"] *= rng.uniform(1.25, 1.45, 8)
    df.loc[rng.choice(n, size=6, replace=False), "glucose"] *= rng.uniform(1.35, 1.7, 6)
    return df


# =============================================================================
# Report Section 3: Data Description, Data Understanding, and Preprocessing
# =============================================================================


def winsorize_frame(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Outlier handling via winsorization.

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


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """
    Construct a unified preprocessing pipeline.

    Numerical Variables:
        1. Median imputation: Fill missing values with the median.
        2. StandardScaler: Standardize data, suitable for distance/linear models like K-means and Logistic/Ridge.

    Categorical Variables:
        1. Most frequent imputation: Fill missing values with the mode.
        2. OneHotEncoder: Transform into dummy variables readable by models.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
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


def plot_eda(df: pd.DataFrame) -> None:
    """
    Generate EDA charts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.countplot(data=df, x="deterioration_24h", ax=axes[0], color="#4C78A8")
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
        palette=["#4C78A8", "#E45756"],
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
#   2. Preprocessing: Handle missing values, outliers, standardization, and encoding.
#   3. Analytics engine: Perform classification, regression, clustering, and association rules.
#   4. Clinical interface: Output risk scores, LOS forecasts, cluster labels, and rule explanations.
#   5. Monitoring: Conduct calibration, fairness checks, and threshold reviews.
#
# At the code level, the main() function executes the complete pipeline in this sequence.


# =============================================================================
# Report Section 5.1: Clustering Results
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

    # Select k=3 as three clusters provide the clearest clinical interpretation:
    #   Low-risk / Moderate-risk / High-risk patient clusters.
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


def plot_clustering(cluster_profile: pd.DataFrame, silhouette: dict) -> None:
    """Generate two plots for clustering: the silhouette curve and the cluster heatmap."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x_values = list(map(int, silhouette.keys()))
    ax.plot(x_values, [silhouette[str(k)] for k in x_values], marker="o", color="#4C78A8")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("K-means model selection")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "kmeans_silhouette.png", dpi=220)
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
# Report Section 5.2: Classification Results
# =============================================================================


def classification_metrics(y_true, y_prob, threshold=0.40) -> dict:
    """
    Calculate classification evaluation metrics.

    Rationale for threshold=0.40:
        Medical early warning systems place greater emphasis on Recall, as missing a genuinely 
        deteriorating patient carries severe consequences. A lower threshold improves recall, 
        albeit at the cost of potentially increasing false positives.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def run_classification(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
) -> tuple[dict, Pipeline, Pipeline, pd.Series, np.ndarray, np.ndarray]:
    """
    Train two classification models:
        1. Logistic Regression
        2. Random Forest Classifier
    Logistic Regression offers high interpretability, while Random Forest excels at capturing non-linear patterns.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=SEED,
    )

    # Logistic Regression:
    #   - Well-suited for binary classification
    #   - class_weight="balanced" accounts for the low proportion of positive (deterioration) events
    logistic_model = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=SEED)),
        ]
    )

    # Random Forest:
    #   - Ensemble of multiple decision trees
    #   - max_depth and min_samples_leaf restrict model complexity to mitigate overfitting
    random_forest_model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=360,
                    max_depth=7,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)
    random_forest_model.fit(X_train, y_train)

    logit_prob = logistic_model.predict_proba(X_test)[:, 1]
    rf_prob = random_forest_model.predict_proba(X_test)[:, 1]

    metrics = {
        "logistic_regression": classification_metrics(y_test, logit_prob),
        "random_forest": classification_metrics(y_test, rf_prob),
    }
    return metrics, logistic_model, random_forest_model, y_test, logit_prob, rf_prob


def plot_classification(y_test, logit_prob, rf_prob, metrics: dict) -> None:
    """
    Generate classification evaluation plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    for name, prob, color in [
        ("Logistic regression", logit_prob, "#4C78A8"),
        ("Random forest", rf_prob, "#F58518"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        axes[0].plot(fpr, tpr, label=f"{name} AUC={auc:.2f}", color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Classification ROC curves")
    axes[0].legend(loc="lower right", fontsize=8)

    sns.heatmap(
        np.array(metrics["random_forest"]["confusion_matrix"]),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Random forest confusion matrix")
    axes[1].set_xticks([0.5, 1.5])
    axes[1].set_xticklabels(["No", "Yes"])
    axes[1].set_yticks([0.5, 1.5])
    axes[1].set_yticklabels(["No", "Yes"], rotation=0)

    fig.tight_layout()
    fig.savefig(OUT / "classification_performance.png", dpi=220)
    plt.close(fig)


def extract_feature_importance(rf_model: Pipeline) -> pd.DataFrame:
    """
    Extract Random Forest feature importances.
        Explains which variables contribute most heavily to predicting 24-hour deterioration.
    """
    feature_names = rf_model.named_steps["prep"].get_feature_names_out()
    importances = pd.DataFrame(
        {
            "feature": [name.replace("num__", "").replace("cat__", "") for name in feature_names],
            "importance": rf_model.named_steps["clf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importances


def plot_feature_importance(importances: pd.DataFrame) -> None:
    """Plot the top 10 most important features."""
    top_imp = importances.head(10).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(top_imp["feature"], top_imp["importance"], color="#4C78A8")
    ax.set_xlabel("Random forest importance")
    ax.set_title("Top predictors of 24-hour deterioration")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=220)
    plt.close(fig)


# =============================================================================
# Report Section 5.3: Regression Results
# =============================================================================


def run_regression(X: pd.DataFrame, y_los: pd.Series, preprocessor: ColumnTransformer) -> dict:
    """
    Train models to predict Length of Stay (LOS).

    Target Variable:
        length_of_stay_days

    Models:
        1. Ridge Linear Regression
        2. Random Forest Regression
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_los,
        test_size=0.25,
        random_state=SEED,
    )

    # Ridge Regression:
    #   Linear Regression with L2 regularization; highly suited for interpretable resource planning models.
    ridge_model = Pipeline(
        [
            ("prep", preprocessor),
            ("reg", Ridge(alpha=1.0)),
        ]
    )

    # Random Forest Regression:
    #   Serves as a baseline to evaluate if non-linear models noticeably enhance predictive performance.
    rf_regression_model = Pipeline(
        [
            ("prep", preprocessor),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=260,
                    max_depth=7,
                    min_samples_leaf=7,
                    random_state=SEED,
                ),
            ),
        ]
    )

    results = {}

    for name, model in [
        ("ridge_linear_regression", ridge_model),
        ("random_forest_regression", rf_regression_model),
    ]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # MAE and RMSE are calculated in 'days' for clear interpretation of shadow error.
        results[name] = {
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        }

    return results


# =============================================================================
# Report Section 5.4: Association Rule Mining
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

    # Generate 1-item, 2-item, and 3-item antecedent combinations.
    for size in [1, 2, 3]:
        for antecedent in itertools.combinations(item_df.columns, size):
            mask = item_df[list(antecedent)].all(axis=1)
            support_x = mask.mean()

            # Filter out rare itemsets to avoid spurious rules.
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
# Report Section 6-7: Business Intelligence, Limitations, and Risk Controls
# =============================================================================

# Model outputs support typical Business Intelligence (BI) scenarios:
#   - Ward risk dashboard: Driven by the classification risk scores.
#   - Cluster census: Categorized via K-means cluster labels.
#   - LOS forecast: Enabled via the regression output module.
#   - Rule alerts: Implemented via association rule triggers.
#
# Key Limitations:
#   - This dataset consists entirely of simulated data and cannot be deployed directly in clinical practice.
#   - Real-world deployment demands rigorous external validation, calibration monitoring, 
#     fairness metrics assessments, privacy reviews, and human-in-the-loop oversight.


# =============================================================================
# Report Section 8 and Appendix: Main End-to-End Pipeline
# =============================================================================


def main() -> None:
    """Execute the complete end-to-end analytics pipeline in report sequence."""
    
    # -------------------------------------------------------------------------
    # Global Font Setting: Configure all plots to use "Times New Roman"
    # -------------------------------------------------------------------------
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Set seaborn theme style (font_scale applies scale factor to our set font)
    sns.set_theme(style="whitegrid", font_scale=0.95)
    
    # Re-enforce Times New Roman after seaborn reset
    plt.rcParams["font.family"] = "Times New Roman"

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

    # -------------------------------------------------------------------------
    # Step 2. Data preprocessing
    # -------------------------------------------------------------------------
    clean = raw.copy()

    # Impute missing values for numerical variables using the median.
    for col in numeric:
        clean[col] = clean[col].fillna(clean[col].median())

    # Apply 1%/99% winsorization to numerical variables.
    clean, winsor_bounds = winsorize_frame(clean, numeric)
    clean.to_csv(OUT / "simulated_patient_monitoring_preprocessed.csv", index=False)

    # Generate EDA charts.
    plot_eda(clean)

    # -------------------------------------------------------------------------
    # Step 3. Prepare feature matrix and targets
    # -------------------------------------------------------------------------
    X = clean[numeric + categorical]
    y_class = clean["deterioration_24h"]
    y_reg = clean["length_of_stay_days"]

    preprocessor = make_preprocessor(numeric, categorical)

    # -------------------------------------------------------------------------
    # Step 4. Clustering
    # -------------------------------------------------------------------------
    clustering, cluster_profile = run_clustering(clean, numeric)
    cluster_profile.to_csv(OUT / "cluster_profile.csv", index=False)
    plot_clustering(cluster_profile, clustering["silhouette"])

    # -------------------------------------------------------------------------
    # Step 5. Classification
    # -------------------------------------------------------------------------
    classification, logit_model, rf_model, y_test, logit_prob, rf_prob = run_classification(
        X,
        y_class,
        preprocessor,
    )
    plot_classification(y_test, logit_prob, rf_prob, classification)

    # Random Forest feature importance for model explanation.
    feature_importance = extract_feature_importance(rf_model)
    feature_importance.head(15).to_csv(OUT / "feature_importance_top15.csv", index=False)
    plot_feature_importance(feature_importance)

    # -------------------------------------------------------------------------
    # Step 6. Regression
    # -------------------------------------------------------------------------
    regression = run_regression(X, y_reg, preprocessor)

    # -------------------------------------------------------------------------
    # Step 7. Association rule mining
    # -------------------------------------------------------------------------
    association_rules = mine_association_rules(clean)
    association_rules.head(20).to_csv(OUT / "association_rules_top20.csv", index=False)
    plot_association_rules(association_rules)

    # -------------------------------------------------------------------------
    # Step 8. Save all results for report tables
    # -------------------------------------------------------------------------
    metrics = {
        "dataset": {
            "n": int(len(raw)),
            "deterioration_rate": float(clean["deterioration_24h"].mean()),
            "mean_los_days": float(clean["length_of_stay_days"].mean()),
            "missingness_percent": {
                col: float(raw[col].isna().mean() * 100)
                for col in raw.columns
                if raw[col].isna().any()
            },
            "winsor_bounds": winsor_bounds,
        },
        "clustering": clustering,
        "classification": classification,
        "regression": regression,
        "association_rules": association_rules.head(10).to_dict(orient="records"),
        "top_features": feature_importance.head(10).to_dict(orient="records"),
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Analysis completed. Outputs saved in:", OUT.resolve())


if __name__ == "__main__":
    main()
"""
ML Training Pipeline — Urgency & Intent Classifiers.

Features:
- Stratified K-Fold (k=5) cross validation
- Nested Cross Validation for unbiased evaluation
- TF-IDF + RandomForest baseline pipeline
- Focus on Recall (medical priority — minimize false negatives)
- Exports trained models to ml/models/
"""

import os
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    cross_validate,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("ml/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("ml/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Sample Training Data ──────────────────────────────────────────────────────
# In production: load from annotated dataset / RDS

URGENCY_DATA = {
    "text": [
        # CRÍTICA
        "dor no peito muito intensa, não consigo respirar",
        "estou desmaiando, falta de ar grave",
        "convulsão, perda de consciência",
        "sangramento intenso que não para",
        "dor no peito irradiando para o braço esquerdo",
        "dificuldade extrema para respirar, lábios roxos",
        # ALTA
        "febre de 40 graus há dois dias",
        "dor muito forte na cabeça, pior da minha vida",
        "vomitando sangue, fraqueza extrema",
        "pressão muito alta, tontura intensa",
        "fratura exposta, dor insuportável",
        "reação alérgica severa, inchaço no rosto",
        # MÉDIA
        "dor moderada no abdômen há 3 dias",
        "febre de 38 graus, mal-estar",
        "tosse persistente há uma semana",
        "dor nas costas que piora com movimento",
        "tontura ocasional, sem outros sintomas",
        "manchas vermelhas na pele, coçando muito",
        # BAIXA
        "quero agendar consulta de rotina",
        "preciso de resultado de exame",
        "dúvida sobre medicamento prescrito",
        "consulta preventiva, estou bem",
        "renovar receita de pressão",
        "checkup anual",
    ],
    "urgency": [
        "critica", "critica", "critica", "critica", "critica", "critica",
        "alta", "alta", "alta", "alta", "alta", "alta",
        "media", "media", "media", "media", "media", "media",
        "baixa", "baixa", "baixa", "baixa", "baixa", "baixa",
    ],
}

INTENT_DATA = {
    "text": [
        "quero marcar uma consulta",
        "preciso agendar horário com o médico",
        "gostaria de agendar para semana que vem",
        "cancelar minha consulta de amanhã",
        "preciso desmarcar o agendamento",
        "remarcar consulta do dia 15",
        "resultado do meu exame de sangue",
        "quando fica pronto o exame?",
        "dúvida sobre o resultado da tomografia",
        "estou com dor de cabeça forte",
        "febre e dor no corpo há dois dias",
        "sintomas de gripe, o que fazer?",
        "EMERGÊNCIA, dor no peito",
        "socorro, não consigo respirar",
        "situação urgente, preciso de ajuda",
        "bom dia",
        "olá, tudo bem?",
        "oi, preciso de informações",
    ],
    "intent": [
        "agendamento", "agendamento", "agendamento",
        "cancelamento", "cancelamento", "cancelamento",
        "duvida_exame", "duvida_exame", "duvida_exame",
        "sintomas", "sintomas", "sintomas",
        "emergencia", "emergencia", "emergencia",
        "saudacao", "saudacao", "saudacao",
    ],
}


# ── Pipeline Builder ──────────────────────────────────────────────────────────

def build_pipeline(classifier=None) -> Pipeline:
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            analyzer="word",
        )),
        ("clf", classifier),
    ])


# ── Stratified K-Fold Cross Validation ───────────────────────────────────────

def stratified_cross_validation(
    X: list, y: list, label: str, n_splits: int = 5
) -> dict:
    """
    Perform Stratified K-Fold cross validation.
    Emphasizes Recall to minimize false negatives (critical in medical context).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Stratified K-Fold CV ({n_splits} folds) — {label}")
    logger.info("="*60)

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = {
        "f1_macro":       "f1_macro",
        "recall_macro":   "recall_macro",
        "precision_macro":"precision_macro",
    }

    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)

    metrics = {}
    for metric, values in results.items():
        mean, std = values.mean(), values.std()
        metrics[metric] = {"mean": round(mean, 4), "std": round(std, 4)}
        logger.info(f"  {metric:<30}: {mean:.4f} ± {std:.4f}")

    return metrics


# ── Nested Cross Validation ───────────────────────────────────────────────────

def nested_cross_validation(X: list, y: list, label: str) -> dict:
    """
    Nested CV for unbiased model selection + hyperparameter tuning.
    Outer loop: generalization estimate | Inner loop: hyperparameter search.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Nested Cross Validation — {label}")
    logger.info("="*60)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth":    [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }

    pipeline = build_pipeline()
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1, refit=True
    )

    outer_scores = cross_val_score(
        grid_search, X, y, cv=outer_cv, scoring="f1_macro"
    )

    result = {
        "nested_f1_mean": round(outer_scores.mean(), 4),
        "nested_f1_std":  round(outer_scores.std(), 4),
        "outer_scores":   [round(s, 4) for s in outer_scores.tolist()],
    }

    logger.info(f"  Nested F1 (outer): {result['nested_f1_mean']:.4f} ± {result['nested_f1_std']:.4f}")
    logger.info(f"  Per-fold scores:   {result['outer_scores']}")
    return result


# ── Final Training & Export ───────────────────────────────────────────────────

def train_and_export(X: list, y: list, model_name: str) -> Pipeline:
    """Train on full dataset and save model + label encoder."""
    logger.info(f"\nTraining final model: {model_name}")

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info(f"  Model saved: {model_path}")
    return pipeline


def plot_confusion_matrix(pipeline: Pipeline, X: list, y: list, name: str):
    """Quick confusion matrix on full training set (for sanity check)."""
    y_pred = pipeline.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    logger.info(f"\nClassification Report — {name}:\n{classification_report(y, y_pred)}")

    report_path = REPORTS_DIR / f"{name}_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"  Report saved: {report_path}")


# ── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    logger.info("🏥 HealthAI — ML Training Pipeline")
    logger.info("="*60)

    all_metrics = {}

    # ── Train Urgency Classifier ──────────────────────────────────────────────
    logger.info("\n📊 URGENCY CLASSIFIER")
    X_urg = URGENCY_DATA["text"]
    y_urg = URGENCY_DATA["urgency"]

    urg_cv_metrics  = stratified_cross_validation(X_urg, y_urg, "Urgency Classifier")
    urg_nested      = nested_cross_validation(X_urg, y_urg, "Urgency Classifier")
    urg_model       = train_and_export(X_urg, y_urg, "urgency_classifier")
    plot_confusion_matrix(urg_model, X_urg, y_urg, "urgency_classifier")

    all_metrics["urgency"] = {**urg_cv_metrics, "nested_cv": urg_nested}

    # ── Train Intent Classifier ───────────────────────────────────────────────
    logger.info("\n📊 INTENT CLASSIFIER")
    X_int = INTENT_DATA["text"]
    y_int = INTENT_DATA["intent"]

    int_cv_metrics  = stratified_cross_validation(X_int, y_int, "Intent Classifier")
    int_nested      = nested_cross_validation(X_int, y_int, "Intent Classifier")
    int_model       = train_and_export(X_int, y_int, "intent_classifier")
    plot_confusion_matrix(int_model, X_int, y_int, "intent_classifier")

    all_metrics["intent"] = {**int_cv_metrics, "nested_cv": int_nested}

    # ── Save consolidated metrics ─────────────────────────────────────────────
    metrics_path = REPORTS_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\n✅ All metrics saved: {metrics_path}")
    logger.info("✅ Training complete!")


if __name__ == "__main__":
    main()

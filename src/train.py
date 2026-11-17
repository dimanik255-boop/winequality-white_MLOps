"""Обучение RandomForestRegressor для предсказания quality белого вина.

По результатам EDA: alcohol — главный признак; дубликаты лучше оставить;
лучшая модель — RandomForest.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
TARGET = "quality"


def train(params_path: str = "params.yaml") -> None:
    cfg = yaml.safe_load(Path(params_path).read_text())["model"]

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = RandomForestRegressor(**cfg)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }

    Path("models").mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "features": FEATURES, "target": TARGET},
        "models/model.joblib",
    )

    Path("metrics.json").write_text(json.dumps(metrics, indent=2))

    Path("reports").mkdir(exist_ok=True)
    report_lines = [
        "Wine Quality — RandomForestRegressor",
        f"Features ({len(FEATURES)}): {', '.join(FEATURES)}",
        f"Test size: {len(test_df)}",
        f"MAE:  {metrics['mae']:.4f}",
        f"RMSE: {metrics['rmse']:.4f}",
        f"R2:   {metrics['r2']:.4f}",
        "",
        "Feature importances (top):",
    ]
    importances = sorted(
        zip(FEATURES, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )
    for name, imp in importances:
        report_lines.append(f"  {name:25s} {imp:.4f}")

    Path("reports/metrics_report.txt").write_text("\n".join(report_lines))
    print(metrics)


if __name__ == "__main__":
    train()

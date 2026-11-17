"""Подготовка данных Wine Quality (white).

Загружает CSV с UCI, при необходимости убирает дубликаты (по params.yaml),
делает train/test split и сохраняет CSV в data/.
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-white.csv"
)

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


def prepare(params_path: str = "params.yaml") -> None:
    cfg = yaml.safe_load(Path(params_path).read_text())["data"]

    df = pd.read_csv(DATA_URL, sep=";")
    print(f"Загружено строк: {len(df)}")

    if cfg.get("drop_duplicates", False):
        before = len(df)
        df = df.drop_duplicates()
        print(f"Удалено дубликатов: {before - len(df)}")
    else:
        print(f"Дубликаты оставлены (duplicated rows ≈ {df.duplicated().sum()})")

    train, test = train_test_split(
        df,
        test_size=cfg["test_size"],
        random_state=cfg["seed"],
        stratify=df[TARGET],
    )

    Path("data").mkdir(exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)
    print(f"Prepared {len(train)} train / {len(test)} test examples")


if __name__ == "__main__":
    prepare()

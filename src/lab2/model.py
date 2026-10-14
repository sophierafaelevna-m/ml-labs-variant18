from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

N_FEATURES = 2**16


def build_vectorizer():
    return HashingVectorizer(
        n_features=N_FEATURES,
        alternate_sign=False,
        lowercase=True,
        stop_words="english",
        norm="l2",
    )


def build_classifier(c_value: float, random_state: int = 42):
    return SGDClassifier(
        loss="hinge",
        alpha=1.0 / (c_value * 1000.0),
        max_iter=1000,
        tol=1e-3,
        random_state=random_state,
        shuffle=False,
    )


def build_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", build_vectorizer()),
            ("classifier", build_classifier(c_value)),
        ]
    )


def tune_hyperparameter(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    text_column: str,
    target_column: str,
    c_values: list[float],
) -> dict[str, Any]:
    results = []
    best_score = -1.0
    best_c = None
    best_model = None

    for c_value in c_values:
        model = build_pipeline(c_value)

        model.fit(
            train_df[text_column],
            train_df[target_column],
        )

        predictions = model.predict(validation_df[text_column])

        score = float(
            f1_score(
                validation_df[target_column],
                predictions,
                average="macro",
            )
        )

        results.append(
            {
                "C": c_value,
                "macro_f1": score,
            }
        )

        if score > best_score:
            best_score = score
            best_c = c_value
            best_model = model

    return {
        "best_c": best_c,
        "best_score": best_score,
        "best_model": best_model,
        "all_results": results,
    }


def train_final_model(
    train_df: pd.DataFrame,
    text_column: str,
    target_column: str,
    c_value: float,
) -> Pipeline:
    model = build_pipeline(c_value)
    model.fit(
        train_df[text_column],
        train_df[target_column],
    )
    return model

from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(y_true, y_pred):
    rare_class = Counter(y_true).most_common()[-1][0]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "rare_class_recall": recall_score(
            y_true,
            y_pred,
            labels=[rare_class],
            average=None,
            zero_division=0,
        )[0],
    }


def naive_baseline(y_true, majority_label):
    most_common = majority_label

    predictions = np.full(
        len(y_true),
        most_common,
    )

    return calculate_metrics(
        y_true,
        predictions,
    )

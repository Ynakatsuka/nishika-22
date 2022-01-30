import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sklearn_accuracy(
    pred, target, threshold=0.5,
):
    pred = pred > threshold
    return accuracy_score(target, pred)


def sklearn_roc_auc_score(
    pred, target,
):
    return roc_auc_score(target, pred)


def sklearn_precision_score(pred, target, threshold=0.5):
    return precision_score(target, pred >= threshold)


def sklearn_recall_score(pred, target, threshold=0.5):
    return recall_score(target, pred >= threshold)


def sklearn_micro_f1(y_pred, y_true):
    return f1_score(np.argmax(y_true, 1), np.argmax(y_pred, 1), average="micro")


def sklearn_macro_f1(y_pred, y_true):
    return f1_score(np.argmax(y_true, 1), np.argmax(y_pred, 1), average="macro")

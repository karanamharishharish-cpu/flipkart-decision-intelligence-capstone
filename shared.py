"""Shared helper module for data loading, preprocessing, model definitions, and evaluation.
Used by part2.py and part4.py to ensure zero code duplication.
"""
import csv
import math
import numpy as np


def load_raw_data(filepath="data/records.csv"):
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def get_stratified_split(raw_rows):
    """Stratified 70/30 split preserving file order."""
    rows_0 = [r for r in raw_rows if int(r[8]) == 0]
    rows_1 = [r for r in raw_rows if int(r[8]) == 1]

    n0_train = int(math.floor(7 * len(rows_0) / 10))
    n1_train = int(math.floor(7 * len(rows_1) / 10))

    train_0 = rows_0[:n0_train]
    test_0 = rows_0[n0_train:]
    train_1 = rows_1[:n1_train]
    test_1 = rows_1[n1_train:]

    record_id_to_idx = {r[0]: i for i, r in enumerate(raw_rows)}
    train_all = sorted(train_0 + train_1, key=lambda r: record_id_to_idx[r[0]])
    test_all = sorted(test_0 + test_1, key=lambda r: record_id_to_idx[r[0]])

    return train_all, test_all


def standardise_train_test(train_records, test_records):
    """Standardises X1 to X7 with training mean and sample SD (ddof=1)."""
    X_train_raw = np.array([[float(x) for x in r[1:8]] for r in train_records])
    y_train = np.array([int(r[8]) for r in train_records])
    X_test_raw = np.array([[float(x) for x in r[1:8]] for r in test_records])
    y_test = np.array([int(r[8]) for r in test_records])

    mu_train = np.mean(X_train_raw, axis=0)
    sd_train = np.std(X_train_raw, axis=0, ddof=1)

    X_train = (X_train_raw - mu_train) / sd_train
    X_test = (X_test_raw - mu_train) / sd_train

    return X_train_raw, y_train, X_test_raw, y_test, X_train, X_test, mu_train, sd_train


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(y, p):
    eps = 1e-15
    p_c = np.clip(p, eps, 1.0 - eps)
    return -np.mean(y * np.log(p_c) + (1.0 - y) * np.log(1.0 - p_c))


def train_logistic_regression(X, y, eta=0.1, epochs=1000):
    """Trains binary logistic regression from scratch with full batch gradient descent."""
    n = len(y)
    p = X.shape[1]
    w = np.zeros(p)
    b = 0.0

    loss_history = {}
    p0 = sigmoid(X @ w + b)
    loss_history[0] = bce_loss(y, p0)

    for ep in range(1, epochs + 1):
        y_hat = sigmoid(X @ w + b)
        grad_w = (1.0 / n) * (X.T @ (y_hat - y))
        grad_b = (1.0 / n) * np.sum(y_hat - y)
        w -= eta * grad_w
        b -= eta * grad_b
        if ep in [1, 10, 100, 200, 500, 1000]:
            loss_history[ep] = bce_loss(y, sigmoid(X @ w + b))

    return w, b, loss_history


def compute_confusion_metrics(y_true, y_pred):
    """Computes confusion matrix cells and evaluation metrics."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n = len(y_true)
    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    spec = tn / (tn + fp) if tn + fp > 0 else 0.0
    tpr = rec
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "tpr": tpr,
        "fpr": fpr,
    }


def knn_predict(X_ref, y_ref, X_query, K):
    """Predicts class and positive probability for queries using K nearest reference points."""
    preds = []
    probs = []
    for q in X_query:
        # Squared Euclidean distance (monotonic with Euclidean distance)
        dists = np.sum((X_ref - q) ** 2, axis=1)
        idx_sorted = np.argsort(dists, kind="stable")[:K]
        pos_count = int(np.sum(y_ref[idx_sorted]))
        probs.append(pos_count / K)
        preds.append(1 if pos_count >= (K + 1) // 2 else 0)
    return np.array(preds), np.array(probs)

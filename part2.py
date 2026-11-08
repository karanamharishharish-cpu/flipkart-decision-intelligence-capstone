"""Part 2: Classification and evaluation.
Produces:
  outputs/part2_results.txt
  outputs/part2_thresholds.csv
  outputs/part2_knn_cv.csv
"""
import csv
import math
import os
import numpy as np
from shared import (
    load_raw_data,
    get_stratified_split,
    standardise_train_test,
    sigmoid,
    bce_loss,
    train_logistic_regression,
    compute_confusion_metrics,
    knn_predict,
)


def run_part2():
    os.makedirs("outputs", exist_ok=True)
    header, raw_rows = load_raw_data("data/records.csv")

    # Task 2.1: Split and standardise
    train_records, test_records = get_stratified_split(raw_rows)
    (
        X_train_raw,
        y_train,
        X_test_raw,
        y_test,
        X_train,
        X_test,
        mu_train,
        sd_train,
    ) = standardise_train_test(train_records, test_records)

    train_rows = len(train_records)
    train_positives = int(np.sum(y_train == 1))
    test_rows = len(test_records)
    test_positives = int(np.sum(y_test == 1))

    # Task 2.2: Logistic regression from scratch (eta=0.1, 1000 epochs)
    w, b, loss_hist = train_logistic_regression(X_train, y_train, eta=0.1, epochs=1000)
    loss_epoch_0 = loss_hist[0]
    loss_epoch_1 = loss_hist[1]
    loss_epoch_10 = loss_hist[10]
    loss_epoch_100 = loss_hist[100]
    loss_epoch_1000 = loss_hist[1000]

    prob_test = sigmoid(X_test @ w + b)
    test_bce = bce_loss(y_test, prob_test)

    # Task 2.3: Learning-rate effect (200 updates with eta in {0.01, 0.1, 1.0})
    loss_200 = {}
    for test_eta in [0.01, 0.1, 1.0]:
        _, _, hist = train_logistic_regression(X_train, y_train, eta=test_eta, epochs=200)
        loss_200[test_eta] = hist[200]

    # Task 2.4: Odds ratios = exp(w_j)
    odds_ratios = np.exp(w)

    # Task 2.5: Evaluate on test rows at tau = 0.5 and tau = 0.3
    m_05 = compute_confusion_metrics(y_test, (prob_test >= 0.5).astype(int))
    m_03 = compute_confusion_metrics(y_test, (prob_test >= 0.3).astype(int))
    blind_accuracy = float(np.sum(y_test == 0) / test_rows)

    # Task 2.6: Threshold sweep for tau in 0.1, 0.2, ..., 0.9
    threshold_rows = []
    for k in range(1, 10):
        tau = round(k * 0.1, 1)
        m_tau = compute_confusion_metrics(y_test, (prob_test >= tau).astype(int))
        threshold_rows.append(
            [
                f"{tau:.1f}",
                str(m_tau["tp"]),
                str(m_tau["fp"]),
                str(m_tau["tn"]),
                str(m_tau["fn"]),
                f"{m_tau['tpr']:.4f}",
                f"{m_tau['fpr']:.4f}",
            ]
        )

    with open("outputs/part2_thresholds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
        writer.writerows(threshold_rows)

    # Task 2.7: K-nearest neighbours with 5-fold cross-validation
    K_vals = list(range(1, 23, 2))
    fold_indices = np.array([i % 5 for i in range(train_rows)])
    cv_rows = []
    best_K = None
    best_cv_f1 = -1.0

    for K in K_vals:
        fold_f1s = []
        for f in range(5):
            val_mask = fold_indices == f
            ref_mask = ~val_mask
            X_ref = X_train[ref_mask]
            y_ref = y_train[ref_mask]
            X_val = X_train[val_mask]
            y_val = y_train[val_mask]

            preds, _ = knn_predict(X_ref, y_ref, X_val, K)
            m_f = compute_confusion_metrics(y_val, preds)
            fold_f1s.append(m_f["f1"])

        mean_cv_f1 = float(np.mean(fold_f1s))
        cv_rows.append([str(K), f"{mean_cv_f1:.4f}"])
        if mean_cv_f1 > best_cv_f1:
            best_cv_f1 = mean_cv_f1
            best_K = K

    with open("outputs/part2_knn_cv.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["K", "mean_cv_f1"])
        writer.writerows(cv_rows)

    knn_test_preds, _ = knn_predict(X_train, y_train, X_test, best_K)
    knn_m = compute_confusion_metrics(y_test, knn_test_preds)

    # Task 2.8: Choose alert threshold with money (cost = 50 * fp + 400 * fn)
    prob_train = sigmoid(X_train @ w + b)
    cost_candidates = []
    for k in range(1, 20):
        tau = k / 20.0
        m_t = compute_confusion_metrics(y_train, (prob_train >= tau).astype(int))
        cost = 50 * m_t["fp"] + 400 * m_t["fn"]
        cost_candidates.append((tau, cost))

    # Tie break: smaller tau
    cost_candidates.sort(key=lambda item: (item[1], item[0]))
    cost_tau, cost_train_at_tau = cost_candidates[0]

    # Evaluate cost on test rows
    m_test_tau = compute_confusion_metrics(y_test, (prob_test >= cost_tau).astype(int))
    cost_test_at_tau = 50 * m_test_tau["fp"] + 400 * m_test_tau["fn"]
    cost_test_at_05 = 50 * m_05["fp"] + 400 * m_05["fn"]
    cost_test_flag_nobody = 400 * test_positives

    results = [
        ("train_rows", str(train_rows)),
        ("train_positives", str(train_positives)),
        ("test_rows", str(test_rows)),
        ("test_positives", str(test_positives)),
        ("loss_epoch_0", f"{loss_epoch_0:.4f}"),
        ("loss_epoch_1", f"{loss_epoch_1:.4f}"),
        ("loss_epoch_10", f"{loss_epoch_10:.4f}"),
        ("loss_epoch_100", f"{loss_epoch_100:.4f}"),
        ("loss_epoch_1000", f"{loss_epoch_1000:.4f}"),
        ("weight_X1", f"{w[0]:.4f}"),
        ("weight_X2", f"{w[1]:.4f}"),
        ("weight_X3", f"{w[2]:.4f}"),
        ("weight_X4", f"{w[3]:.4f}"),
        ("weight_X5", f"{w[4]:.4f}"),
        ("weight_X6", f"{w[5]:.4f}"),
        ("weight_X7", f"{w[6]:.4f}"),
        ("bias", f"{b:.4f}"),
        ("test_bce", f"{test_bce:.4f}"),
        ("loss_epoch_200_eta_0.01", f"{loss_200[0.01]:.4f}"),
        ("loss_epoch_200_eta_0.1", f"{loss_200[0.1]:.4f}"),
        ("loss_epoch_200_eta_1.0", f"{loss_200[1.0]:.4f}"),
        ("odds_ratio_X1", f"{odds_ratios[0]:.4f}"),
        ("odds_ratio_X2", f"{odds_ratios[1]:.4f}"),
        ("odds_ratio_X3", f"{odds_ratios[2]:.4f}"),
        ("odds_ratio_X4", f"{odds_ratios[3]:.4f}"),
        ("odds_ratio_X5", f"{odds_ratios[4]:.4f}"),
        ("odds_ratio_X6", f"{odds_ratios[5]:.4f}"),
        ("odds_ratio_X7", f"{odds_ratios[6]:.4f}"),
        ("tau_0.5_tp", str(m_05["tp"])),
        ("tau_0.5_fp", str(m_05["fp"])),
        ("tau_0.5_tn", str(m_05["tn"])),
        ("tau_0.5_fn", str(m_05["fn"])),
        ("tau_0.5_accuracy", f"{m_05['accuracy']:.4f}"),
        ("tau_0.5_precision", f"{m_05['precision']:.4f}"),
        ("tau_0.5_recall", f"{m_05['recall']:.4f}"),
        ("tau_0.5_f1", f"{m_05['f1']:.4f}"),
        ("tau_0.5_specificity", f"{m_05['specificity']:.4f}"),
        ("tau_0.3_tp", str(m_03["tp"])),
        ("tau_0.3_fp", str(m_03["fp"])),
        ("tau_0.3_tn", str(m_03["tn"])),
        ("tau_0.3_fn", str(m_03["fn"])),
        ("tau_0.3_accuracy", f"{m_03['accuracy']:.4f}"),
        ("tau_0.3_precision", f"{m_03['precision']:.4f}"),
        ("tau_0.3_recall", f"{m_03['recall']:.4f}"),
        ("tau_0.3_f1", f"{m_03['f1']:.4f}"),
        ("tau_0.3_specificity", f"{m_03['specificity']:.4f}"),
        ("blind_accuracy", f"{blind_accuracy:.4f}"),
        ("knn_best_k", str(best_K)),
        ("knn_tp", str(knn_m["tp"])),
        ("knn_fp", str(knn_m["fp"])),
        ("knn_tn", str(knn_m["tn"])),
        ("knn_fn", str(knn_m["fn"])),
        ("knn_accuracy", f"{knn_m['accuracy']:.4f}"),
        ("knn_precision", f"{knn_m['precision']:.4f}"),
        ("knn_recall", f"{knn_m['recall']:.4f}"),
        ("knn_f1", f"{knn_m['f1']:.4f}"),
        ("cost_tau", f"{cost_tau:.4f}"),
        ("cost_train_at_tau", str(cost_train_at_tau)),
        ("cost_test_at_tau", str(cost_test_at_tau)),
        ("cost_test_at_0.5", str(cost_test_at_05)),
        ("cost_test_flag_nobody", str(cost_test_flag_nobody)),
    ]

    with open("outputs/part2_results.txt", "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 2 complete. Written {len(results)} keys to outputs/part2_results.txt")


if __name__ == "__main__":
    run_part2()

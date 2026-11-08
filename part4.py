"""Part 4: Model-choice and architecture memo.
Produces:
  outputs/part4_results.txt
  outputs/part4_splits.csv
  outputs/part4_voting.csv
  outputs/part4_adaboost.csv
  outputs/part4_architecture.csv
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
    train_logistic_regression,
    compute_confusion_metrics,
    knn_predict,
)


def run_part4():
    os.makedirs("outputs", exist_ok=True)
    header, raw_rows = load_raw_data("data/records.csv")

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

    n_train = len(y_train)
    pos_train = int(np.sum(y_train == 1))

    # Task 4.1: Which split is purest?
    def calc_entropy(pos, total):
        if total == 0 or pos == 0 or pos == total:
            return 0.0
        p1 = pos / total
        p0 = 1.0 - p1
        return -(p1 * math.log2(p1) + p0 * math.log2(p0))

    def calc_gini(pos, total):
        if total == 0:
            return 0.0
        p1 = pos / total
        p0 = 1.0 - p1
        return 1.0 - (p1 ** 2 + p0 ** 2)

    root_entropy = calc_entropy(pos_train, n_train)
    root_gini = calc_gini(pos_train, n_train)

    # Four candidate splits (original units)
    # S1: X6 == 1 (index 5)
    # S2: X7 == 1 (index 6)
    # S3: X3 > 0.5 (index 2)
    # S4: X1 <= 24 (index 0)
    splits = [
        ("S1", X_train_raw[:, 5] == 1),
        ("S2", X_train_raw[:, 6] == 1),
        ("S3", X_train_raw[:, 2] > 0.5),
        ("S4", X_train_raw[:, 0] <= 24),
    ]

    split_csv_rows = []
    best_ig = -1.0
    best_split_by_info_gain = None
    best_gg = -1.0
    best_split_by_gini_gain = None

    for s_name, cond in splits:
        n_yes = int(np.sum(cond))
        pos_yes = int(np.sum(y_train[cond]))
        n_no = n_train - n_yes
        pos_no = pos_train - pos_yes

        ent_yes = calc_entropy(pos_yes, n_yes)
        ent_no = calc_entropy(pos_no, n_no)
        w_ent = (n_yes / n_train) * ent_yes + (n_no / n_train) * ent_no
        info_gain = root_entropy - w_ent

        gi_yes = calc_gini(pos_yes, n_yes)
        gi_no = calc_gini(pos_no, n_no)
        w_gi = (n_yes / n_train) * gi_yes + (n_no / n_train) * gi_no
        gini_gain = root_gini - w_gi

        split_csv_rows.append(
            [
                s_name,
                str(n_yes),
                str(pos_yes),
                str(n_no),
                str(pos_no),
                f"{info_gain:.4f}",
                f"{gini_gain:.4f}",
            ]
        )

        if info_gain > best_ig:
            best_ig = info_gain
            best_split_by_info_gain = s_name

        if gini_gain > best_gg:
            best_gg = gini_gain
            best_split_by_gini_gain = s_name

    with open("outputs/part4_splits.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"])
        writer.writerows(split_csv_rows)

    # Task 4.2: A one-question model and two ways to combine models
    # The stump is the best split by info gain (S4: X1 <= 24)
    cond_stump_train = X_train_raw[:, 0] <= 24
    n_stump_yes = int(np.sum(cond_stump_train))
    pos_stump_yes = int(np.sum(y_train[cond_stump_train]))
    n_stump_no = n_train - n_stump_yes
    pos_stump_no = pos_train - pos_stump_yes

    stump_rate_yes = pos_stump_yes / n_stump_yes
    stump_rate_no = pos_stump_no / n_stump_no

    # Logistic model
    w, b, _ = train_logistic_regression(X_train, y_train, eta=0.1, epochs=1000)
    prob_logistic = sigmoid(X_test @ w + b)
    pred_logistic = (prob_logistic >= 0.3).astype(int)

    # KNN model (K = 5 from Task 2.7)
    K = 5
    pred_knn, prob_knn = knn_predict(X_train, y_train, X_test, K)

    # Stump model on test set
    cond_stump_test = X_test_raw[:, 0] <= 24
    prob_stump = np.where(cond_stump_test, stump_rate_yes, stump_rate_no)
    pred_stump = (prob_stump >= 0.3).astype(int)

    # Hard vote: positive when at least two of the three predict positive
    pred_hard_vote = ((pred_logistic + pred_knn + pred_stump) >= 2).astype(int)

    # Soft vote: positive when the mean of the three probabilities is >= 0.3
    prob_soft = (prob_logistic + prob_knn + prob_stump) / 3.0
    pred_soft_vote = (prob_soft >= 0.3).astype(int)

    voting_csv_rows = []
    models_to_eval = [
        ("logistic", pred_logistic),
        ("knn", pred_knn),
        ("stump", pred_stump),
        ("hard_vote", pred_hard_vote),
        ("soft_vote", pred_soft_vote),
    ]

    for m_name, preds in models_to_eval:
        metrics = compute_confusion_metrics(y_test, preds)
        voting_csv_rows.append(
            [
                m_name,
                str(metrics["tp"]),
                str(metrics["fp"]),
                str(metrics["tn"]),
                str(metrics["fn"]),
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1']:.4f}",
            ]
        )

    with open("outputs/part4_voting.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])
        writer.writerows(voting_csv_rows)

    # Task 4.3: Two rounds of AdaBoost by hand
    y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
    h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

    adaboost_csv_rows = []

    # Round 1: equal weights 1/8
    w1_before = np.ones(8) / 8.0
    mis1 = h1 != y_ada
    round1_error = float(np.sum(w1_before[mis1]))
    round1_alpha = 0.5 * math.log((1.0 - round1_error) / round1_error)
    w1_unnorm = w1_before * np.exp(-round1_alpha * y_ada * h1)
    w1_after = w1_unnorm / np.sum(w1_unnorm)

    for pt in range(8):
        adaboost_csv_rows.append(
            [
                "1",
                str(pt + 1),
                f"{w1_before[pt]:.4f}",
                f"{w1_unnorm[pt]:.4f}",
                f"{w1_after[pt]:.4f}",
            ]
        )

    # Round 2: weights from round 1
    w2_before = w1_after.copy()
    mis2 = h2 != y_ada
    round2_error = float(np.sum(w2_before[mis2]))
    round2_alpha = 0.5 * math.log((1.0 - round2_error) / round2_error)
    w2_unnorm = w2_before * np.exp(-round2_alpha * y_ada * h2)
    w2_after = w2_unnorm / np.sum(w2_unnorm)

    for pt in range(8):
        adaboost_csv_rows.append(
            [
                "2",
                str(pt + 1),
                f"{w2_before[pt]:.4f}",
                f"{w2_unnorm[pt]:.4f}",
                f"{w2_after[pt]:.4f}",
            ]
        )

    with open("outputs/part4_adaboost.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "point", "weight_before", "weight_unnormalised", "weight_after"])
        writer.writerows(adaboost_csv_rows)

    ensemble_preds = np.sign(round1_alpha * h1 + round2_alpha * h2)
    ensemble_accuracy = float(np.mean(ensemble_preds == y_ada))

    # Task 4.4: Out-of-bag share
    oob_fraction = float((1.0 - 1.0 / 560.0) ** 560)

    # Task 4.5: Size the neural networks
    # Model A Option 1: MLP
    mlp_hidden1 = 784 * 16 + 16
    mlp_hidden2 = 16 * 16 + 16
    mlp_output = 16 * 10 + 10
    mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

    # Model A Option 2: CNN
    cnn_conv1 = 5 * 5 * 1 * 6 + 6
    cnn_conv2 = 5 * 5 * 6 * 16 + 16
    cnn_dense1 = 400 * 32 + 32
    cnn_dense2 = 32 * 10 + 10
    cnn_total = cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2

    arch_rows = [
        ["mlp_hidden1", str(mlp_hidden1)],
        ["mlp_hidden2", str(mlp_hidden2)],
        ["mlp_output", str(mlp_output)],
        ["cnn_conv1", str(cnn_conv1)],
        ["cnn_conv2", str(cnn_conv2)],
        ["cnn_dense1", str(cnn_dense1)],
        ["cnn_dense2", str(cnn_dense2)],
    ]

    with open("outputs/part4_architecture.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "parameters"])
        writer.writerows(arch_rows)

    fc_layer_params = 128 * 128 * 3 * 1024 + 1024
    conv_layer_params = 3 * 3 * 3 * 64 + 64

    # Model B: RNN
    rnn_total = 128 * 64 + 128 * 128 + 128 + 16 * 128 + 16

    results = [
        ("root_entropy", f"{root_entropy:.4f}"),
        ("root_gini", f"{root_gini:.4f}"),
        ("best_split_by_info_gain", best_split_by_info_gain),
        ("best_split_by_gini_gain", best_split_by_gini_gain),
        ("stump_rate_yes", f"{stump_rate_yes:.4f}"),
        ("stump_rate_no", f"{stump_rate_no:.4f}"),
        ("round1_error", f"{round1_error:.4f}"),
        ("round1_alpha", f"{round1_alpha:.4f}"),
        ("round2_error", f"{round2_error:.4f}"),
        ("round2_alpha", f"{round2_alpha:.4f}"),
        ("ensemble_accuracy", f"{ensemble_accuracy:.4f}"),
        ("oob_fraction", f"{oob_fraction:.4f}"),
        ("mlp_total", str(mlp_total)),
        ("cnn_total", str(cnn_total)),
        ("fc_layer_params", str(fc_layer_params)),
        ("conv_layer_params", str(conv_layer_params)),
        ("rnn_total", str(rnn_total)),
    ]

    with open("outputs/part4_results.txt", "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 4 complete. Written {len(results)} keys to outputs/part4_results.txt")


if __name__ == "__main__":
    run_part4()

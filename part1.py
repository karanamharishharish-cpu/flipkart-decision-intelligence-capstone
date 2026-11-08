"""Part 1: Estimation, inference and a regression baseline.
Produces outputs/part1_results.txt.
"""
import csv
import math
import os
import numpy as np


def run_part1():
    os.makedirs("outputs", exist_ok=True)
    with open("data/records.csv", "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_rows = [row for row in reader]

    data = np.array([[float(x) for x in r[1:]] for r in raw_rows])
    X1 = data[:, 0]
    X2 = data[:, 1]
    X3 = data[:, 2]
    X4 = data[:, 3]
    X5 = data[:, 4]
    X6 = data[:, 5]
    X7 = data[:, 6]
    churned = data[:, 7]
    val = data[:, 8]

    # Task 1.1: Estimate with an interval
    n = len(val)
    mean_val = float(np.mean(val))
    sd_val = float(np.std(val, ddof=1))
    se_val = sd_val / math.sqrt(n)
    ci95_low = mean_val - 1.96 * se_val
    ci95_high = mean_val + 1.96 * se_val

    # Task 1.2: Test a planning claim
    z_stat = (mean_val - 8.0) / se_val
    reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

    # Task 1.3: Size a future sample
    E = 0.5
    n_required = math.ceil((1.96 * sd_val / E) ** 2)

    # Task 1.4: Simple regression by closed form
    cov_x4_val = float(np.sum((X4 - np.mean(X4)) * (val - mean_val)) / (n - 1))
    var_x4 = float(np.var(X4, ddof=1))
    reg_slope = cov_x4_val / var_x4
    reg_intercept = mean_val - reg_slope * float(np.mean(X4))
    fitted = reg_intercept + reg_slope * X4
    tss = float(np.sum((val - mean_val) ** 2))
    ess = float(np.sum((fitted - mean_val) ** 2))
    rss = float(np.sum((val - fitted) ** 2))
    tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
    r2 = 1.0 - rss / tss
    rmse = math.sqrt(rss / n)
    mae = float(np.mean(np.abs(val - fitted)))

    # Task 1.5: Fit the same line by gradient descent
    mean_x4 = float(np.mean(X4))
    sd_x4 = float(np.std(X4, ddof=1))
    z = (X4 - mean_x4) / sd_x4
    m = 0.0
    c = 0.0
    eta = 0.1

    e0 = val - (m * z + c)
    gd_J0 = float(np.mean(e0 ** 2))
    costs = {}
    for step in range(1, 501):
        e = val - (m * z + c)
        grad_m = -(2.0 / n) * float(np.sum(z * e))
        grad_c = -(2.0 / n) * float(np.sum(e))
        m -= eta * grad_m
        c -= eta * grad_c
        if step in [1, 10, 100, 500]:
            costs[step] = float(np.mean((val - (m * z + c)) ** 2))

    gd_J1 = costs[1]
    gd_J10 = costs[10]
    gd_J100 = costs[100]
    gd_J500 = costs[500]
    gd_slope = m / sd_x4
    gd_intercept = c - m * mean_x4 / sd_x4
    gd_slope_gap = abs(gd_slope - reg_slope)

    # Task 1.6: Check for a funnel, then apply log remedy
    order = np.argsort(fitted, kind="stable")
    low_idx = order[:400]
    up_idx = order[400:]
    resid = val - fitted
    resid_sd_lower_raw = float(np.std(resid[low_idx], ddof=1))
    resid_sd_upper_raw = float(np.std(resid[up_idx], ddof=1))
    resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

    # Log-log regression: ln(value) on ln(X4)
    y_log = np.log(val)
    x_log = np.log(X4)
    cov_log = float(np.sum((x_log - np.mean(x_log)) * (y_log - np.mean(y_log))) / (n - 1))
    var_x_log = float(np.var(x_log, ddof=1))
    loglog_slope = cov_log / var_x_log
    loglog_intercept = float(np.mean(y_log)) - loglog_slope * float(np.mean(x_log))
    fitted_log = loglog_intercept + loglog_slope * x_log
    tss_log = float(np.sum((y_log - np.mean(y_log)) ** 2))
    rss_log = float(np.sum((y_log - fitted_log) ** 2))
    loglog_r2 = 1.0 - rss_log / tss_log
    loglog_rmse = math.sqrt(rss_log / n)

    order_log = np.argsort(fitted_log, kind="stable")
    low_log_idx = order_log[:400]
    up_log_idx = order_log[400:]
    resid_log = y_log - fitted_log
    resid_sd_lower_log = float(np.std(resid_log[low_log_idx], ddof=1))
    resid_sd_upper_log = float(np.std(resid_log[up_log_idx], ddof=1))
    resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

    # Task 1.7: Multiple regression by normal equation
    # Order: constant, ln(X4), X1, X2, X3, X5, X6, X7
    X_mat = np.column_stack([np.ones(n), np.log(X4), X1, X2, X3, X5, X6, X7])
    beta_hat = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ y_log)
    mr_fitted = X_mat @ beta_hat
    mr_rss = float(np.sum((y_log - mr_fitted) ** 2))
    mr_tss = tss_log
    mr_r2 = 1.0 - mr_rss / mr_tss
    p = 7
    mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - p - 1)
    loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

    results = [
        ("n", str(n)),
        ("mean_value", f"{mean_val:.4f}"),
        ("sd_value", f"{sd_val:.4f}"),
        ("se_value", f"{se_val:.4f}"),
        ("ci95_low", f"{ci95_low:.4f}"),
        ("ci95_high", f"{ci95_high:.4f}"),
        ("z_stat", f"{z_stat:.4f}"),
        ("reject_h0", reject_h0),
        ("n_required", str(n_required)),
        ("reg_slope", f"{reg_slope:.4f}"),
        ("reg_intercept", f"{reg_intercept:.4f}"),
        ("tss", f"{tss:.4f}"),
        ("ess", f"{ess:.4f}"),
        ("rss", f"{rss:.4f}"),
        ("tss_equals_ess_plus_rss", tss_equals_ess_plus_rss),
        ("r2", f"{r2:.4f}"),
        ("rmse", f"{rmse:.4f}"),
        ("mae", f"{mae:.4f}"),
        ("gd_J0", f"{gd_J0:.4f}"),
        ("gd_J1", f"{gd_J1:.4f}"),
        ("gd_J10", f"{gd_J10:.4f}"),
        ("gd_J100", f"{gd_J100:.4f}"),
        ("gd_J500", f"{gd_J500:.4f}"),
        ("gd_slope", f"{gd_slope:.4f}"),
        ("gd_intercept", f"{gd_intercept:.4f}"),
        ("gd_slope_gap", f"{gd_slope_gap:.4f}"),
        ("resid_sd_lower_raw", f"{resid_sd_lower_raw:.4f}"),
        ("resid_sd_upper_raw", f"{resid_sd_upper_raw:.4f}"),
        ("resid_sd_ratio_raw", f"{resid_sd_ratio_raw:.4f}"),
        ("loglog_intercept", f"{loglog_intercept:.4f}"),
        ("loglog_slope", f"{loglog_slope:.4f}"),
        ("loglog_r2", f"{loglog_r2:.4f}"),
        ("loglog_rmse", f"{loglog_rmse:.4f}"),
        ("resid_sd_lower_log", f"{resid_sd_lower_log:.4f}"),
        ("resid_sd_upper_log", f"{resid_sd_upper_log:.4f}"),
        ("resid_sd_ratio_log", f"{resid_sd_ratio_log:.4f}"),
        ("mr_beta_0", f"{beta_hat[0]:.4f}"),
        ("mr_beta_1", f"{beta_hat[1]:.4f}"),
        ("mr_beta_2", f"{beta_hat[2]:.4f}"),
        ("mr_beta_3", f"{beta_hat[3]:.4f}"),
        ("mr_beta_4", f"{beta_hat[4]:.4f}"),
        ("mr_beta_5", f"{beta_hat[5]:.4f}"),
        ("mr_beta_6", f"{beta_hat[6]:.4f}"),
        ("mr_beta_7", f"{beta_hat[7]:.4f}"),
        ("mr_r2", f"{mr_r2:.4f}"),
        ("mr_adj_r2", f"{mr_adj_r2:.4f}"),
        ("loglog_adj_r2", f"{loglog_adj_r2:.4f}"),
    ]

    with open("outputs/part1_results.txt", "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 1 complete. Written {len(results)} keys to outputs/part1_results.txt")


if __name__ == "__main__":
    run_part1()

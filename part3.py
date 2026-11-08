"""Part 3: Segmentation and dimensionality reduction.
Produces:
  outputs/part3_results.txt
  outputs/part3_segments.csv
  outputs/part3_hierarchical.csv
"""
import csv
import os
import numpy as np


def run_part3():
    os.makedirs("outputs", exist_ok=True)
    with open("data/records.csv", "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_rows = [row for row in reader]

    raw_data = np.array([[float(x) for x in r[1:]] for r in raw_rows])
    X5_orig = raw_data[:, :5]
    outcome = raw_data[:, 7]
    value = raw_data[:, 8]
    n_records = len(raw_data)

    # Standardise X1 to X5 over all 800 rows (mean and sample SD, ddof=1)
    mean_5 = np.mean(X5_orig, axis=0)
    sd_5 = np.std(X5_orig, axis=0, ddof=1)
    X_std = (X5_orig - mean_5) / sd_5

    # Task 3.1: K-means from scratch for K = 2, 3, 4, 5, 6
    wcss = {}
    cluster_labels_k3 = None

    for K in [2, 3, 4, 5, 6]:
        # Pinned starting centroids
        init_indices = [k * (n_records // K) for k in range(K)]
        centroids = X_std[init_indices].copy()

        prev_labels = None
        while True:
            # Nearest centroid by squared Euclidean distance, ties to lower-numbered centroid
            dists = np.sum((X_std[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(dists, axis=1)

            if prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels.copy()

            # Update centroids with mean of members
            for k in range(K):
                members = X_std[labels == k]
                if len(members) > 0:
                    centroids[k] = np.mean(members, axis=0)

        # Final WCSS
        final_wcss = float(np.sum((X_std - centroids[labels]) ** 2))
        wcss[K] = final_wcss
        if K == 3:
            cluster_labels_k3 = labels

    # Task 3.2: Choose K with the elbow
    elbow_k = None
    for K in [3, 4, 5]:
        drop_curr = wcss[K] - wcss[K + 1]
        drop_prev = wcss[K - 1] - wcss[K]
        if drop_curr < 0.5 * drop_prev:
            elbow_k = K
            break

    # Task 3.3: Profile the K = 3 segments
    # Order the three clusters by ascending mean of X3 (in original units) -> A, B, C
    means_x3 = []
    for k in range(3):
        means_x3.append((k, float(np.mean(X5_orig[cluster_labels_k3 == k, 2]))))
    means_x3.sort(key=lambda item: item[1])
    seg_names = ["A", "B", "C"]
    cluster_to_seg = {old_k: seg_names[idx] for idx, (old_k, _) in enumerate(means_x3)}
    segment_assignment = np.array([cluster_to_seg[k] for k in cluster_labels_k3])

    segment_rows = []
    sizes = {}
    for s_name in ["A", "B", "C"]:
        mask = segment_assignment == s_name
        size_s = int(np.sum(mask))
        sizes[s_name] = size_s
        mean_X = np.mean(X5_orig[mask], axis=0)
        out_rate = float(np.mean(outcome[mask]))
        mean_val = float(np.mean(value[mask]))
        segment_rows.append(
            [
                s_name,
                str(size_s),
                f"{mean_X[0]:.4f}",
                f"{mean_X[1]:.4f}",
                f"{mean_X[2]:.4f}",
                f"{mean_X[3]:.4f}",
                f"{mean_X[4]:.4f}",
                f"{out_rate:.4f}",
                f"{mean_val:.4f}",
            ]
        )

    with open("outputs/part3_segments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "segment",
                "size",
                "mean_X1",
                "mean_X2",
                "mean_X3",
                "mean_X4",
                "mean_X5",
                "outcome_rate",
                "mean_value",
            ]
        )
        writer.writerows(segment_rows)

    # Task 3.4: Hierarchical clustering on supplied dissimilarity table
    elem_names = ["E1", "E2", "E3", "E4", "E5", "E6"]
    D = np.array(
        [
            [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
            [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
            [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
            [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
            [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
            [6.1, 2.0, 1.3, 4.5, 3.4, 0.0],
        ]
    )

    def run_agglomerative(linkage):
        clusters = [[i] for i in range(6)]
        merges = []

        for step in range(5):
            best_dist = float("inf")
            best_pair = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    c1, c2 = clusters[i], clusters[j]
                    pair_dists = [D[a, b] for a in c1 for b in c2]
                    if linkage == "single":
                        d = min(pair_dists)
                    elif linkage == "complete":
                        d = max(pair_dists)
                    elif linkage == "average":
                        d = sum(pair_dists) / len(pair_dists)

                    if d < best_dist:
                        best_dist = d
                        best_pair = (i, j)

            i, j = best_pair
            c1, c2 = clusters[i], clusters[j]
            # Name formatting: cluster_a has smaller minimum member number
            m1 = sorted(c1)
            m2 = sorted(c2)
            if m1[0] > m2[0]:
                m1, m2 = m2, m1
            name_a = "+".join([elem_names[x] for x in m1])
            name_b = "+".join([elem_names[x] for x in m2])
            merged = sorted(c1 + c2)
            merges.append((step + 1, name_a, name_b, best_dist))

            clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
            clusters.append(merged)

        heights = [m[3] for m in merges]
        gaps = [heights[k + 1] - heights[k] for k in range(4)]
        max_gap_idx = int(np.argmax(gaps))
        merges_below_cut = max_gap_idx + 1
        clusters_at_cut = 6 - merges_below_cut
        return merges, clusters_at_cut

    hierarchical_csv_rows = []
    clusters_at_cut_dict = {}
    for link in ["single", "complete", "average"]:
        merges, n_cut = run_agglomerative(link)
        clusters_at_cut_dict[link] = n_cut
        for m in merges:
            hierarchical_csv_rows.append([link, str(m[0]), m[1], m[2], f"{m[3]:.4f}"])

    with open("outputs/part3_hierarchical.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
        writer.writerows(hierarchical_csv_rows)

    # Task 3.5: PCA by the seven steps
    # (1) Mean vector is zero
    # (2) Data is centered
    # (3) Covariance matrix S = (1/(n-1)) X^T X
    S = (1.0 / (n_records - 1)) * (X_std.T @ X_std)
    eigvals, eigvecs = np.linalg.eigh(S)
    # Sort descending
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Sign convention: flip each eigenvector so entry of largest absolute value is positive
    for j in range(eigvecs.shape[1]):
        max_abs_idx = np.argmax(np.abs(eigvecs[:, j]))
        if eigvecs[max_abs_idx, j] < 0:
            eigvecs[:, j] = -eigvecs[:, j]

    eigenvalue_sum = float(np.sum(eigvals))
    explained_ratios = eigvals / eigenvalue_sum
    cumulative_ratios = np.cumsum(explained_ratios)
    components_for_80pct = int(np.where(cumulative_ratios >= 0.8)[0][0] + 1)

    # Top two components projection
    W2 = eigvecs[:, :2]
    Z = X_std @ W2

    mean_pc1_seg_A = float(np.mean(Z[segment_assignment == "A", 0]))
    mean_pc1_seg_B = float(np.mean(Z[segment_assignment == "B", 0]))
    mean_pc1_seg_C = float(np.mean(Z[segment_assignment == "C", 0]))

    results = [
        ("wcss_k2", f"{wcss[2]:.4f}"),
        ("wcss_k3", f"{wcss[3]:.4f}"),
        ("wcss_k4", f"{wcss[4]:.4f}"),
        ("wcss_k5", f"{wcss[5]:.4f}"),
        ("wcss_k6", f"{wcss[6]:.4f}"),
        ("elbow_k", str(elbow_k)),
        ("size_segment_A", str(sizes["A"])),
        ("size_segment_B", str(sizes["B"])),
        ("size_segment_C", str(sizes["C"])),
        ("single_clusters_at_cut", str(clusters_at_cut_dict["single"])),
        ("complete_clusters_at_cut", str(clusters_at_cut_dict["complete"])),
        ("average_clusters_at_cut", str(clusters_at_cut_dict["average"])),
        ("eigenvalue_sum", f"{eigenvalue_sum:.4f}"),
        ("eigenvalue_1", f"{eigvals[0]:.4f}"),
        ("eigenvalue_2", f"{eigvals[1]:.4f}"),
        ("eigenvalue_3", f"{eigvals[2]:.4f}"),
        ("eigenvalue_4", f"{eigvals[3]:.4f}"),
        ("eigenvalue_5", f"{eigvals[4]:.4f}"),
        ("explained_ratio_1", f"{explained_ratios[0]:.4f}"),
        ("explained_ratio_2", f"{explained_ratios[1]:.4f}"),
        ("explained_ratio_3", f"{explained_ratios[2]:.4f}"),
        ("explained_ratio_4", f"{explained_ratios[3]:.4f}"),
        ("explained_ratio_5", f"{explained_ratios[4]:.4f}"),
        ("cumulative_ratio_1", f"{cumulative_ratios[0]:.4f}"),
        ("cumulative_ratio_2", f"{cumulative_ratios[1]:.4f}"),
        ("cumulative_ratio_3", f"{cumulative_ratios[2]:.4f}"),
        ("cumulative_ratio_4", f"{cumulative_ratios[3]:.4f}"),
        ("cumulative_ratio_5", f"{cumulative_ratios[4]:.4f}"),
        ("components_for_80pct", str(components_for_80pct)),
        ("pc1_loading_X1", f"{eigvecs[0, 0]:.4f}"),
        ("pc1_loading_X2", f"{eigvecs[1, 0]:.4f}"),
        ("pc1_loading_X3", f"{eigvecs[2, 0]:.4f}"),
        ("pc1_loading_X4", f"{eigvecs[3, 0]:.4f}"),
        ("pc1_loading_X5", f"{eigvecs[4, 0]:.4f}"),
        ("pc2_loading_X1", f"{eigvecs[0, 1]:.4f}"),
        ("pc2_loading_X2", f"{eigvecs[1, 1]:.4f}"),
        ("pc2_loading_X3", f"{eigvecs[2, 1]:.4f}"),
        ("pc2_loading_X4", f"{eigvecs[3, 1]:.4f}"),
        ("pc2_loading_X5", f"{eigvecs[4, 1]:.4f}"),
        ("row1_pc1", f"{Z[0, 0]:.4f}"),
        ("row1_pc2", f"{Z[0, 1]:.4f}"),
        ("row2_pc1", f"{Z[1, 0]:.4f}"),
        ("row2_pc2", f"{Z[1, 1]:.4f}"),
        ("row3_pc1", f"{Z[2, 0]:.4f}"),
        ("row3_pc2", f"{Z[2, 1]:.4f}"),
        ("mean_pc1_segment_A", f"{mean_pc1_seg_A:.4f}"),
        ("mean_pc1_segment_B", f"{mean_pc1_seg_B:.4f}"),
        ("mean_pc1_segment_C", f"{mean_pc1_seg_C:.4f}"),
    ]

    with open("outputs/part3_results.txt", "w") as f:
        for k, v in results:
            f.write(f"{k} = {v}\n")

    print(f"Part 3 complete. Written {len(results)} keys to outputs/part3_results.txt")


if __name__ == "__main__":
    run_part3()

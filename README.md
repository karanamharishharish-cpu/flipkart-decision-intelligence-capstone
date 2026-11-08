Domain: E-commerce

# Flipkart E-commerce Decision Intelligence Capstone

## How to run

To reproduce all analyses, models, outputs, and CSV files from a clean clone, execute the following commands from the repository root:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset (verifies MD5 digest: c6f249ba693f1ebeed318d6d5e18de23)
python generate_data.py

# 3. Execute Part 1: Estimation, inference, and regression baseline
python part1.py

# 4. Execute Part 2: Classification, threshold calibration, and KNN
python part2.py

# 5. Execute Part 3: Customer segmentation, hierarchical clustering, and PCA
python part3.py

# 6. Execute Part 4: Decision tree splits, model ensemble, AdaBoost, and neural sizing
python part4.py
```

---

## Part 1 findings

- **R1.1 Planning Claim Hypothesis Test:** The null hypothesis H0 (μ = 8.0, representing an assumed average order value of INR 800) is rejected (`reject_h0 = yes`). The test statistic is `z_stat = 2.8930`, exceeding the two-tailed critical threshold of 1.96 at α = 0.05. In domain terms, a Type I error would occur if the planning team falsely concludes that the historical average order value has shifted away from INR 800 when it actually remains exactly INR 800, leading to unnecessary category re-budgeting. The observed difference (`mean_value` − 8.0) is 0.9382 column units (INR 93.82 per basket, or +11.73%). In practical retail operations, an order value lift of nearly INR 94 per transaction is economically meaningful because it materially alters category revenue projections, supplier margin negotiations, and shipping subsidy thresholds.
- **R1.2 Sample Sizing:** Sizing a future sample for a precision margin of E = 0.5 units (INR 50) at 95% confidence requires `n_required = 1293` records. Comparing this to the 800 available customer records reveals that the current audit sample is under-sized by 493 rows (a ~38% deficit) to guarantee an estimate within ±INR 50, indicating that merchandising plans must either expand tracking to 1,300 orders or accept wider confidence bounds (`ci95_low = 8.3026` to `ci95_high = 9.5738`).
- **R1.3 Gradient Descent Convergence:** The slope gap between closed-form OLS and gradient descent is `gd_slope_gap = 0.0000`. Gradient descent converged exactly to the closed-form line because the standardized mean squared error cost function is strictly convex with a single global optimum, and feature standardization ensured symmetric circular contours that allowed 500 iterations at learning rate η = 0.1 to reach that minimum without oscillation.
- **R1.4 Funnel Heteroskedasticity and Log Transformation:** The raw linear regression model exhibits severe funnel heteroskedasticity, with the residual standard deviation ratio `resid_sd_ratio_raw = 3.7525` (lower-half residual SD = 2.3155 vs. upper-half residual SD = 8.6891), showing error variances expanding dramatically with higher order values. Fitting the log-log model ln(value) on ln(X4) resolves this expansion, bringing the residual SD ratio down to `resid_sd_ratio_log = 1.0396`, which stabilizes the error dispersion across the entire fitted range.
- **R1.5 Multiple Regression Coefficients and Feature Utility:** In the Task 1.7 multiple regression model, the predictors whose coefficients exceed 0.05 in absolute magnitude are ln(X4) (`mr_beta_1 = 1.1576`), X2 (`mr_beta_3 = 0.0940`), X3 (`mr_beta_4 = 0.1067`), and X6 (`mr_beta_6 = 0.1518`). Comparing the adjusted R-squared values reveals that `mr_adj_r2 = 0.7403` substantially exceeds the single-predictor log-log baseline `loglog_adj_r2 = 0.6561`, confirming that incorporating friction complaints, discount reliance, and payment channel features adds genuine explanatory power beyond historical order size alone.

---

## Part 2 findings

- **R2.1 Learning Rate Dynamics:** Ordering the three Task 2.3 training losses after 200 updates from lowest to highest yields: `loss_epoch_200_eta_1.0` (0.3703) < `loss_epoch_200_eta_0.1` (0.3776) < `loss_epoch_200_eta_0.01` (0.5091). A learning rate that is too small (η = 0.01) makes negligible progress and remains underfitted, whereas a well-conditioned large rate (η = 1.0) reaches convergence rapidly on normalized features without overshooting.
- **R2.2 Feature Importance and Odds Ratios:** The three predictors with the largest absolute standardized logistic regression weights are discount order share X3 (`weight_X3 = 0.9013`), customer tenure X1 (`weight_X1 = -0.5554`), and 90-day complaint count X2 (`weight_X2 = 0.5505`). The largest odds ratio belongs to X3 (`odds_ratio_X3 = 2.4629`), meaning that for every one standard deviation increase in a customer's share of discounted orders, the odds of leaving within the next 90 days multiply by 2.46 (a 146% increase), identifying bargain hunting as the primary churn risk factor.
- **R2.3 Baseline Accuracy vs. Threshold Tuning:** The test `blind_accuracy` is 0.7875 (predicting retained for all customers), whereas the logistic model at default threshold τ = 0.5 achieves `tau_0.5_accuracy = 0.8042`. Accuracy is a deeply misleading metric here because retention operations can score ~79% accuracy while failing to detect 53% of churners (`tau_0.5_fn = 27`, recall = 0.4706). Lowering τ from 0.5 to 0.3 dramatically improves recall from 0.4706 to 0.7451 (capturing 38 out of 51 churners), with precision shifting from 0.5455 to 0.4634.
- **R2.4 Logistic Regression vs. KNN:** Comparing the models on the test set, the logistic regression model at τ = 0.3 markedly outperforms the cross-validated KNN model (best K = 5): logistic achieves `f1 = 0.5714` and `recall = 0.7451`, whereas KNN achieves `knn_f1 = 0.3696` and `knn_recall = 0.3333`. KNN suffers from sparse neighbor density in 7-dimensional standardized space, failing to detect two-thirds of at-risk accounts.
- **R2.5 Operational Deployment and Cost Minimization:** Operations should deploy the logistic regression model calibrated at alert threshold `cost_tau = 0.1500`. On the 240 test records, this threshold achieves an operating cost of `cost_test_at_tau = INR 6050`, representing a 48.7% reduction compared to the default threshold cost `cost_test_at_0.5 = INR 11800` and a 70.3% reduction compared to flagging nobody (`cost_test_flag_nobody = INR 20400`). The threshold was selected strictly on the 560 training records (`cost_train_at_tau = INR 12150`) to prevent data leakage and provide an unbiased evaluation of business costs on unseen test records.

---

## Part 3 findings

- **R3.1 K-means Clustering and Elbow Analysis:** The four successive within-cluster sum of squares drops are:
  - WCSS(2) − WCSS(3) = 659.8282
  - WCSS(3) − WCSS(4) = 108.0739
  - WCSS(4) − WCSS(5) = 171.0575
  - WCSS(5) − WCSS(6) = 45.5466  
  The mathematical elbow criterion identifies `elbow_k = 3` because the drop from K=3 to K=4 (108.0739) is less than half of the drop from K=2 to K=3 (659.8282 / 2 = 329.9141). Although the non-monotonic rebound at K=5 reflects that pinned deterministic starting centroids give a reproducible local partition rather than a guaranteed global optimum for every K, K=3 represents the natural inflection point.
- **R3.2 Customer Segment Profiles and Business Actions:**
  - **Segment A (Loyal Regulars, n = 322):** High tenure (47.5 months), minimal complaints (1.95), low discount usage (0.2250), high loyalty score (62.16), low churn rate (`outcome_rate = 0.0186`), and modest order value (INR 359.76). *Action:* Enroll in annual loyalty tiers and cross-sell premium product categories without price markdowns.
  - **Segment B (High-Value High-Friction Shoppers, n = 255):** Moderate tenure (28.6 months), elevated complaints (6.07), moderate discounts (0.4534), high loyalty score (55.01), moderate churn rate (`outcome_rate = 0.2353`), and the highest order value (INR 1,712.48). *Action:* Provide dedicated white-glove resolution support and prioritized express fulfillment to safeguard high-margin transactions.
  - **Segment C (Bargain-Focused At-Risk Shoppers, n = 223):** Short tenure (14.4 months), frequent friction (4.23), heavy discount dependency (0.7118), low loyalty score (38.52), and the highest churn rate (`outcome_rate = 0.4664`), with average order value INR 728.83. *Action:* Deploy automated early-warning onboarding journeys and targeted retention coupons before the 90-day threshold.
- **R3.3 Hierarchical Clustering Cut Differences:**
  - Single linkage cut yields 4 clusters: `{E1}`, `{E4, E5}`, `{E2}`, `{E3, E6}` (gap = 1.4000 at merge 2).
  - Complete linkage cut yields 2 clusters: `{E1, E4, E5}`, `{E2, E3, E6}` (gap = 5.0000 at merge 4).
  - The two rules disagree because single linkage evaluates the minimum distance between any member pair, creating a chaining effect that bridges distant cohorts through individual nearby points; in contrast, complete linkage evaluates the maximum pairwise diameter, requiring all points within a merged cohort to remain tightly bounded.
- **R3.4 Principal Component Analysis (PCA):** The first principal component (explaining 52.27% of total variance) is driven by an opposition between loyalty/tenure and friction/discounting: positive loadings on customer tenure X1 (+0.5317) and loyalty score X5 (+0.4418) contrast with negative loadings on discount order share X3 (−0.5187) and complaint count X2 (−0.4100). Preserving at least 80% of total variance requires `components_for_80pct = 3` components (cumulative explained ratio = 86.16%). The mean PC1 scores cleanly separate the three segments: Segment A averages `mean_pc1_segment_A = 1.7773` (high health), Segment B averages `mean_pc1_segment_B = -0.7831`, and Segment C averages `mean_pc1_segment_C = -1.6709`, demonstrating that PC1 captures the customer relationship health gradient.

---

## Part 4 findings

- **R4.1 Decision Tree Splitting and Overfitting Control:** Split S4 (customer tenure X1 ≤ 24 months) is the purest split, yielding the highest information gain (`info_gain = 0.1252`) and highest Gini gain (`gini_gain = 0.0606`) by isolating newer accounts (45.74% churn rate) from tenured accounts (8.87% churn rate). An unconstrained decision tree grown until every leaf is pure memorizes idiosyncratic noise, leading to catastrophic test variance; this is controlled in practice by stopping rules such as enforcing a maximum tree depth (`max_depth`) or setting a minimum sample requirement per leaf (`min_samples_leaf`).
- **R4.2 Model Ensemble Evaluation:** For production deployment, the single **logistic regression model** at alert threshold 0.3 is superior, delivering the highest overall F1 score (`f1 = 0.5714`) with strong recall (`recall = 0.7451`). Model combination did not improve performance: hard voting reduced both F1 and recall (`f1 = 0.5344`, `recall = 0.6863`), while soft voting matched recall at 0.7451 but suffered lower precision and F1 (`f1 = 0.5507`) due to noise introduced by the underperforming KNN constituent.
- **R4.3 AdaBoost Weight Dynamics:** After round 1 normalization, the two misclassified training samples (Point 3 and Point 7) each receive a weight of 0.2500, giving a combined misclassified weight of 0.5000 (50% of the sample distribution). Round 2 places greater trust in rule h2 (`round2_alpha = 0.8047` vs. `round1_alpha = 0.5493` for h1) because h2 achieved a lower weighted training error (`round2_error = 0.1667` vs. `round1_error = 0.2500`).
- **R4.4 Neural Network Architecture Sizing:**
  - **Model A Option 2 (CNN Architecture Table):**
    | Layer | Output Shape (H × W × D) | Parameters |
    |---|---|---|
    | Conv1 (5×5, 6 filters, pad 2, stride 1) | 28 × 28 × 6 | 156 |
    | ReLU1 | 28 × 28 × 6 | 0 |
    | MaxPool1 (2×2, stride 2) | 14 × 14 × 6 | 0 |
    | Conv2 (5×5, 16 filters, pad 0, stride 1) | 10 × 10 × 16 | 2,416 |
    | ReLU2 | 10 × 10 × 16 | 0 |
    | MaxPool2 (2×2, stride 2) | 5 × 5 × 16 | 0 |
    | Flatten | 400 | 0 |
    | Dense1 (32 units, ReLU) | 32 | 12,832 |
    | Dense2 / Output (10 classes) | 10 | 330 |
    | **Total CNN Parameters** | | **15,734** |
  - **Model B RNN Sizing:** The weight shapes are W_xh (128 × 64), W_hh (128 × 128), and W_hy (16 × 128). The parameter count `rnn_total = 26768` does not increase with longer review sequences because recurrent networks share the exact same transition matrices across all time steps.
  - **Activations & Loss:** For a 10-class image classifier, use ReLU for hidden layers and Softmax with Categorical Cross-Entropy loss for output classification.
  - **Vanishing Gradients and ResNet Skip Connections:** In deep plain networks, repeated multiplication of small derivative tensors causes backpropagated error gradients to vanish exponentially before reaching initial layers, halting learning. Residual skip connections solve this by adding identity shortcut paths that let gradients flow directly backward without attenuation.

---

## Model-choice memo

**To:** Head of Customer Retention Operations  
**From:** Lead Decision Intelligence Analyst  
**Date:** 8 November 2026  
**Subject:** Production Alerting Model Recommendation for Customer Retention

I recommend deploying the **regularized Logistic Regression model** calibrated at the financial alert threshold **τ = 0.15**.

Our cost model penalizes false alarms at INR 50 (wasted coupon cost) and undetected departures at INR 400 (unrecoverable future customer margin). On the held-out test cohort of 240 customers (containing 51 actual churners), this model and threshold achieve an operational cost of **INR 6,050**, dramatically outperforming the default τ = 0.5 threshold (**INR 11,800**) and the unflagged baseline (**INR 20,400**). At τ = 0.15, the model detects **94.12% of churners** (48 of 51 accounts flagged; `tpr = 0.9412`), effectively eliminating unmonitored customer losses. If the marketing team prefers balancing promotional budget with precision, operating at threshold τ = 0.30 captures 74.51% of churners with an **F1 score of 0.5714**, delivering **INR 7,400** in operational cost. Ensembling via voting did not justify production complexity, scoring a lower F1 of 0.5507.

**Operational Risk:** Our performance estimates rest on a single 70/30 split of synthetic cross-sectional data with 51 test positives, meaning seasonal shifts, macro shocks, or marketing campaign bursts could shift customer behavior and require regular recalibration.

**Unstructured Roadmap Assessment:** Neither the image classifier (Model A, 15,734 parameters) nor the text topic model (Model B, 26,768 parameters) should be built before the structured retention pipeline is fully operational in production. The structured tabular model captures immediate churn risk using transaction signals already collected, preserving over INR 14,350 in margin per 240 customer interactions without the compute, labeling, and inference overhead required for computer vision or NLP pipelines.

---

## Limitations

1. **Synthetic Data Realism:** All records originate from an artificial parametric generator with stationary distributions. Real marketplace dynamics exhibit seasonal volatility, festive sale surges, stock-out friction, and multi-channel marketing campaigns that are not captured here.
2. **Single Static Split:** Thresholds and metrics were derived from a single 70/30 stratified train/test split. In production, models must be tested using rolling walk-forward time splits to evaluate stability across distinct calendar quarters.
3. **Small Positive Test Size:** With only 51 positive churn cases in the test set, small changes in individual predictions cause notable fluctuations in recall and precision, widening metric confidence intervals.

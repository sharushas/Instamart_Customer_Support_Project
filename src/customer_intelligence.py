"""src/customer_intelligence.py
Part B: Customer Segments (RFM, KMeans, PCA) and Lapse Prediction (Pipeline, Tuning, Evaluation).
"""
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_part_b():
    clean_dir = os.path.join("data", "clean")
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    cust = pd.read_csv(os.path.join(clean_dir, "customers.csv"))
    orders = pd.read_csv(os.path.join(clean_dir, "orders.csv"))
    prod = pd.read_csv(os.path.join(clean_dir, "products.csv"))

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])

    completed = orders[orders["status"] == "completed"].copy()

    # ---------------- 1. Build RFM Table ----------------
    print("--- 1. RFM Table ---")
    snapshot = pd.to_datetime("2025-10-01 00:00:00")
    
    rfm = completed.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

    print(f"RFM Customers: {len(rfm)} (Reference: 1175)")
    print(f"Recency - min: {rfm['recency'].min()}, median: {rfm['recency'].median()}, max: {rfm['recency'].max()}")
    print(f"Frequency - mean: {rfm['frequency'].mean():.2f}, max: {rfm['frequency'].max()}")
    print(f"Monetary - mean: {rfm['monetary'].mean():.2f}, max: {rfm['monetary'].max():.2f}")

    # ---------------- 2. Cluster with K-Means ----------------
    print("\n--- 2. K-Means Clustering Evaluation (K=2 to 8) ---")
    scaler = StandardScaler()
    X_rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans_records = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_rfm_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_rfm_scaled, labels)
        kmeans_records.append({"K": k, "Inertia": round(inertia, 1), "Silhouette": round(sil, 3)})
        print(f"K={k}: Inertia={inertia:.1f}, Silhouette={sil:.3f}")

    # Final choice: K=4
    chosen_k = 4
    km_final = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    rfm["segment_id"] = km_final.fit_predict(X_rfm_scaled)

    # ---------------- 3. Profile, Name & Project Segments ----------------
    print("\n--- 3. Segment Profiling for K=4 ---")
    seg_profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).reset_index()
    print(seg_profile.to_string(index=False))

    # Identify premium segment (highest mean monetary value)
    premium_cluster = seg_profile.sort_values("mean_monetary", ascending=False).iloc[0]["segment_id"]
    print(f"Premium Cluster ID (Highest Monetary): {premium_cluster}")

    # Name segments based on profiles
    # Segment naming map
    def assign_segment_name(row):
        cid = row["segment_id"]
        if cid == premium_cluster:
            return "Premium Shoppers"
        elif row["mean_frequency"] > 15:
            return "Everyday Loyalists"
        elif row["mean_recency"] > 60:
            return "At-Risk / Lapsed"
        else:
            return "Occasional Value Seekers"

    # Merge names
    name_map = {}
    for _, row in seg_profile.iterrows():
        name_map[row["segment_id"]] = assign_segment_name(row)

    rfm["segment_name"] = rfm["segment_id"].map(name_map)
    rfm["is_premium"] = rfm["segment_id"] == premium_cluster

    # PCA 2 components
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_rfm_scaled)
    var_ratios = pca.explained_variance_ratio_
    print(f"PCA explained variance ratios: {var_ratios[0]:.3f}, {var_ratios[1]:.3f} (Total: {var_ratios.sum():.3f})")

    # Export outputs/customer_segments.csv
    rfm_export = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].copy()
    rfm_export.to_csv(os.path.join(outputs_dir, "customer_segments.csv"), index=False)
    print("Saved outputs/customer_segments.csv")

    # ---------------- 4. Define Lapse Problem ----------------
    print("\n--- 4. Lapse Dataset Construction ---")
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    window_end = pd.to_datetime("2025-09-30 23:59:59")

    # Cohort: customers with >= 1 completed order before cutoff
    pre_orders = orders[orders["order_ts"] < cutoff].copy()
    pre_completed = pre_orders[pre_orders["status"] == "completed"].copy()
    cohort_cids = sorted(pre_completed["customer_id"].unique())
    print(f"Cohort size (>=1 completed order before cutoff): {len(cohort_cids)} (Reference: 1156)")

    # Target label: lapsed = 1 if NO completed order in Sept 2025 [2025-09-01 .. 2025-09-30]
    sept_completed = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] <= window_end) & (orders["status"] == "completed")]
    sept_active_cids = set(sept_completed["customer_id"].unique())

    # Build features before cutoff
    # 1. Orders before cutoff merged with product info
    pre_orders_prod = pre_orders.merge(prod[["product_id", "category"]], on="product_id", how="left")
    pre_comp_prod = pre_completed.merge(prod[["product_id", "category"]], on="product_id", how="left")

    feat_records = []
    for cid in cohort_cids:
        c_comp = pre_comp_prod[pre_comp_prod["customer_id"] == cid]
        c_all = pre_orders_prod[pre_orders_prod["customer_id"] == cid]
        
        last_ts = c_comp["order_ts"].max()
        rec = (cutoff - last_ts).days
        freq = len(c_comp)
        mon = c_comp["order_amount"].sum()
        avg_disc = c_comp["discount_pct"].mean()
        
        # Cancellation rate: cancelled orders / all orders with final status
        all_final = c_all[c_all["status"].isin(["completed", "cancelled"])]
        canc_count = (all_final["status"] == "cancelled").sum()
        canc_rate = canc_count / len(all_final) if len(all_final) > 0 else 0.0

        avg_deliv = c_comp["delivery_minutes"].mean()
        
        # Favourite category (mode)
        fav_cat = c_comp["category"].mode()[0] if len(c_comp) > 0 else "Grocery"

        lapsed = 1 if cid not in sept_active_cids else 0

        feat_records.append({
            "customer_id": cid,
            "recency": rec,
            "frequency": freq,
            "monetary": mon,
            "avg_discount_pct": avg_disc,
            "cancellation_rate": canc_rate,
            "avg_delivery_minutes": avg_deliv,
            "favourite_category": fav_cat,
            "lapsed": lapsed
        })

    df_feats = pd.DataFrame(feat_records)
    # Merge demographic & tenure features from clean customers
    df_feats = df_feats.merge(cust[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
    df_feats["tenure_days"] = (cutoff - pd.to_datetime(df_feats["signup_date"])).dt.days
    df_feats = df_feats.drop(columns=["signup_date"])

    total_cohort = len(df_feats)
    total_lapsed = df_feats["lapsed"].sum()
    lapsed_pct = (total_lapsed / total_cohort) * 100.0
    print(f"Lapsed count: {total_lapsed} ({lapsed_pct:.2f}%) (Reference: 443, 38.32%)")

    # Features and target
    feature_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "tenure_days", "age",
        "favourite_category", "city", "gender"
    ]
    numeric_cols = [
        "recency", "frequency", "monetary", "avg_discount_pct",
        "cancellation_rate", "avg_delivery_minutes", "tenure_days", "age"
    ]
    categorical_cols = ["favourite_category", "city", "gender"]

    X = df_feats[feature_cols]
    y = df_feats["lapsed"]

    # Check for missing values in features
    assert X.isna().sum().sum() == 0, f"Missing values found in features: {X.isna().sum()}"

    # Split train/test
    X_train, X_test, y_train, y_test, cids_train, cids_test = train_test_split(
        X, y, df_feats["customer_id"], test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)} (Reference: 924 / 232)")

    # ---------------- 5. Preprocessing Pipeline ----------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols)
        ]
    )

    # Compare frequency mean
    scaler_num = StandardScaler()
    scaler_num.fit(X_train[numeric_cols])
    freq_idx = numeric_cols.index("frequency")
    learned_mean_freq = scaler_num.mean_[freq_idx]
    cohort_mean_freq = df_feats["frequency"].mean()
    print(f"Learned frequency mean (train only): {learned_mean_freq:.4f}")
    print(f"Cohort frequency mean (full cohort): {cohort_mean_freq:.4f}")

    # Baseline: Always predict 0 (Not Lapsed)
    y_pred_baseline = np.zeros_like(y_test)
    base_acc = accuracy_score(y_test, y_pred_baseline)
    base_f1 = f1_score(y_test, y_pred_baseline, zero_division=0)
    print(f"\nBaseline (predict all 0s): Accuracy = {base_acc*100:.1f}%, Lapsed F1 = {base_f1:.4f}")

    # ---------------- 6. Train, Tune & Evaluate ----------------
    print("\n--- 6. Grid Search Model Tuning ---")
    # (b) Logistic Regression
    pipe_lr = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid_lr = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    print(f"Logistic Regression Best Params: {grid_lr.best_params_}, Best CV F1: {grid_lr.best_score_:.4f}")

    # (c) Random Forest
    pipe_rf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    param_grid_rf = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    print(f"Random Forest Best Params: {grid_rf.best_params_}, Best CV F1: {grid_rf.best_score_:.4f}")

    # (d) Test Set Evaluation
    y_pred_lr = best_lr.predict(X_test)
    y_pred_rf = best_rf.predict(X_test)

    print("\n--- Test Set Evaluation: Logistic Regression ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_lr):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_lr):.4f}")

    print("\n--- Test Set Evaluation: Random Forest ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_rf):.4f}")

    # ---------------- 7. Class-Balance Experiment ----------------
    print("\n--- 7. Class-Balance Experiment (Over-sampling Lapsed in Train) ---")
    train_df = X_train.copy()
    train_df["lapsed"] = y_train.values
    majority = train_df[train_df["lapsed"] == 0]
    minority = train_df[train_df["lapsed"] == 1]
    
    n_maj = len(majority)
    n_min = len(minority)
    n_to_add = n_maj - n_min
    
    minority_oversampled = minority.sample(n=n_to_add, replace=True, random_state=42)
    train_balanced = pd.concat([majority, minority, minority_oversampled], ignore_index=True)
    
    X_train_bal = train_balanced[feature_cols]
    y_train_bal = train_balanced["lapsed"]
    print(f"Balanced Train Counts: 0={sum(y_train_bal==0)}, 1={sum(y_train_bal==1)} (Total: {len(y_train_bal)})")

    best_C = grid_lr.best_params_["clf__C"]
    pipe_lr_bal = Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(C=best_C, max_iter=1000, random_state=42))
    ])
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_lr_bal = pipe_lr_bal.predict(X_test)

    print("\n--- Balanced Logistic Regression on Unchanged Test Set ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_bal))
    print(f"Precision: {precision_score(y_test, y_pred_lr_bal):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_lr_bal):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_lr_bal):.4f}")

    # ---------------- 8. Feature Importance, Persist & Retention List ----------------
    print("\n--- 8. Interpretation & Persistence ---")
    # Feature names after one-hot encoding
    cat_encoder = best_rf.named_steps["pre"].named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(categorical_cols))
    all_feature_names = numeric_cols + cat_feature_names

    # Top 5 RF features
    rf_importances = best_rf.named_steps["clf"].feature_importances_
    feat_imp = sorted(zip(all_feature_names, rf_importances), key=lambda x: x[1], reverse=True)
    print("Top 5 Random Forest Features:")
    for f_name, imp in feat_imp[:5]:
        print(f"   {f_name}: {imp:.4f}")

    # Sign of 3 features in Logistic Regression
    lr_coefs = best_lr.named_steps["clf"].coef_[0]
    coef_dict = dict(zip(all_feature_names, lr_coefs))
    sample_feats = ["recency", "frequency", "monetary"]
    print("\nLogistic Regression Selected Coefficients:")
    for sf in sample_feats:
        print(f"   {sf}: coef={coef_dict[sf]:.4f} (Sign: {'+' if coef_dict[sf]>0 else '-'})")

    # Select best model: Random Forest
    selected_model = best_rf
    joblib.dump(selected_model, os.path.join(outputs_dir, "lapse_model.joblib"))
    print("\nSaved outputs/lapse_model.joblib")

    # Reload model and verify identical predictions
    reloaded_model = joblib.load(os.path.join(outputs_dir, "lapse_model.joblib"))
    reloaded_preds = reloaded_model.predict(X_test)
    assert np.array_equal(y_pred_rf, reloaded_preds), "Reloaded model predictions mismatch!"
    print("Verified: Reloaded model reproduces original predictions exactly.")

    # Retention List: Premium customers in test set ranked by predicted lapse probability
    # Segment labels from task 3
    rfm_segments = rfm_export.set_index("customer_id")
    test_cids = cids_test.values
    test_probs = selected_model.predict_proba(X_test)[:, 1]

    test_eval_df = pd.DataFrame({
        "customer_id": test_cids,
        "lapse_probability": test_probs,
        "actually_lapsed": y_test.values
    })
    test_eval_df["is_premium"] = test_eval_df["customer_id"].map(rfm_segments["is_premium"])

    prem_test = test_eval_df[test_eval_df["is_premium"] == True].copy()
    prem_test = prem_test.sort_values("lapse_probability", ascending=False).reset_index(drop=True)
    top10_retention = prem_test.head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
    top10_retention.to_csv(os.path.join(outputs_dir, "retention_list.csv"), index=False)
    print(f"\nPremium customers in test set: {len(prem_test)}")
    print(f"Top 10 Retention List written to outputs/retention_list.csv:")
    print(top10_retention.to_string(index=False))
    truly_lapsed = top10_retention["actually_lapsed"].sum()
    print(f"Truly lapsed among top 10: {truly_lapsed} / {len(top10_retention)}")

if __name__ == "__main__":
    run_part_b()

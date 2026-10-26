"""src/data_cleaner.py
Part A: Clean, Store, Validate, Query and Analyze Instamart data.
"""
import os
import sqlite3
import pandas as pd
import numpy as np

def run_part_a():
    raw_dir = os.path.join("data", "raw")
    clean_dir = os.path.join("data", "clean")
    os.makedirs(clean_dir, exist_ok=True)
    db_path = os.path.join("data", "instamart.db")

    # 1. Audit raw data
    print("--- 1. Auditing Raw Data ---")
    cust_raw = pd.read_csv(os.path.join(raw_dir, "customers.csv"))
    prod_raw = pd.read_csv(os.path.join(raw_dir, "products.csv"))
    ord_raw = pd.read_csv(os.path.join(raw_dir, "orders.csv"))
    tick_raw = pd.read_csv(os.path.join(raw_dir, "tickets.csv"), keep_default_na=False)

    print(f"customers.csv shape: {cust_raw.shape}")
    print(f"products.csv shape: {prod_raw.shape}")
    print(f"orders.csv shape: {ord_raw.shape}")
    print(f"tickets.csv shape: {tick_raw.shape}")

    # Specific audit figures
    raw_order_rows = len(ord_raw)
    exact_dup_orders = ord_raw.duplicated().sum()
    pending_orders = (ord_raw["status"] == "pending").sum()
    distinct_order_ids = ord_raw["order_id"].nunique()

    # Non-numeric amounts
    non_numeric_amt = pd.to_numeric(ord_raw["order_amount"], errors="coerce").isna().sum()

    # Negative delivery times & blank delivery times
    deliv_numeric = pd.to_numeric(ord_raw["delivery_minutes"], errors="coerce")
    neg_deliv = (deliv_numeric < 0).sum()
    blank_deliv = ord_raw["delivery_minutes"].isna().sum() + (ord_raw["delivery_minutes"] == "").sum()

    # Customer audit figures
    blank_age = cust_raw["age"].isna().sum() + (cust_raw["age"] == "").sum()
    age_numeric = pd.to_numeric(cust_raw["age"], errors="coerce")
    outside_age = ((age_numeric < 10) | (age_numeric > 100)).sum()
    distinct_raw_cities = cust_raw["city"].nunique()
    distinct_raw_gender = cust_raw["gender"].dropna().replace("", np.nan).dropna().nunique()
    blank_gender = cust_raw["gender"].isna().sum() + (cust_raw["gender"] == "").sum()

    audit_summary = {
        "raw_orders": raw_order_rows,
        "exact_duplicates": exact_dup_orders,
        "pending_rows": pending_orders,
        "distinct_order_ids": distinct_order_ids,
        "non_numeric_amounts": non_numeric_amt,
        "negative_delivery": neg_deliv,
        "blank_delivery": blank_deliv,
        "blank_age": blank_age,
        "age_outside_10_100": outside_age,
        "distinct_raw_cities": distinct_raw_cities,
        "distinct_raw_gender": distinct_raw_gender,
        "blank_gender": blank_gender,
    }
    for k, v in audit_summary.items():
        print(f"  {k}: {v}")

    # 2. Clean Customers
    print("\n--- 2. Cleaning Customers ---")
    cust_clean = cust_raw.copy()
    # (a) Standardise city
    cust_clean["city"] = cust_clean["city"].astype(str).str.strip().str.title()
    # (b) Map gender
    gender_map = {
        "male": "male", "Male": "male", "M": "male", "m": "male", "MALE": "male",
        "female": "female", "Female": "female", "F": "female", "f": "female", "FEMALE": "female"
    }
    cust_clean["gender"] = cust_clean["gender"].map(lambda g: gender_map.get(str(g).strip(), "unknown") if str(g).strip() != "" and pd.notna(g) else "unknown")
    # (c) Treat ages outside 10-100 as missing, then fill with median age of customer's city
    cust_clean["age"] = pd.to_numeric(cust_clean["age"], errors="coerce")
    cust_clean.loc[(cust_clean["age"] < 10) | (cust_clean["age"] > 100), "age"] = np.nan
    city_median_age = cust_clean.groupby("city")["age"].transform("median")
    cust_clean["age"] = cust_clean["age"].fillna(city_median_age).astype(int)
    # (d) Convert signup_date to datetime
    cust_clean["signup_date"] = pd.to_datetime(cust_clean["signup_date"])

    print("City distribution:\n", cust_clean["city"].value_counts())
    print("Gender distribution:\n", cust_clean["gender"].value_counts())
    print("Missing ages:", cust_clean["age"].isna().sum())
    print(f"Age range: {cust_clean['age'].min()} to {cust_clean['age'].max()}")

    # 3. Clean Products
    prod_clean = prod_raw.copy()

    # 4. Clean Orders
    print("\n--- 3. Cleaning Orders ---")
    ord_clean = ord_raw.copy()
    # (1) Convert order_ts to datetime
    ord_clean["order_ts"] = pd.to_datetime(ord_clean["order_ts"])
    # (2) Remove exact duplicates
    ord_clean = ord_clean.drop_duplicates()
    # (3) Keep only record with latest order_ts per order_id
    ord_clean = ord_clean.sort_values("order_ts").drop_duplicates(subset=["order_id"], keep="last")
    # (4) Convert order_amount to numeric, fill missing from price
    ord_clean["order_amount"] = pd.to_numeric(ord_clean["order_amount"], errors="coerce")
    ord_clean = ord_clean.merge(prod_clean[["product_id", "price_inr"]], on="product_id", how="left")
    calc_amt = (ord_clean["price_inr"] * ord_clean["quantity"] * (1 - ord_clean["discount_pct"] / 100.0)).round(2)
    ord_clean["order_amount"] = ord_clean["order_amount"].fillna(calc_amt)
    ord_clean = ord_clean.drop(columns=["price_inr"])

    # (5) Delivery minutes
    ord_clean["delivery_minutes"] = pd.to_numeric(ord_clean["delivery_minutes"], errors="coerce")
    ord_clean.loc[ord_clean["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    # Merge category from products to compute median delivery time by category among completed orders
    ord_clean = ord_clean.merge(prod_clean[["product_id", "category"]], on="product_id", how="left")
    
    # Calculate median delivery time per product category among completed orders
    completed_mask = ord_clean["status"] == "completed"
    cancelled_mask = ord_clean["status"] == "cancelled"
    
    cat_median_deliv = ord_clean[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    ord_clean.loc[completed_mask, "delivery_minutes"] = ord_clean.loc[completed_mask, "delivery_minutes"].fillna(cat_median_deliv)
    # Ensure cancelled orders have delivery_minutes as missing
    ord_clean.loc[cancelled_mask, "delivery_minutes"] = np.nan
    ord_clean = ord_clean.drop(columns=["category"])

    # Sort clean orders by order_id or order_ts
    ord_clean = ord_clean.sort_values("order_id").reset_index(drop=True)
    cust_clean = cust_clean.sort_values("customer_id").reset_index(drop=True)

    is_completed = ord_clean["status"] == "completed"
    is_cancelled = ord_clean["status"] == "cancelled"

    print(f"Clean orders rows: {len(ord_clean)}")
    print(f"Status counts:\n{ord_clean['status'].value_counts()}")
    print(f"Missing order_amount: {ord_clean['order_amount'].isna().sum()}")
    print(f"Missing deliv (completed): {ord_clean[is_completed]['delivery_minutes'].isna().sum()}")
    print(f"Missing deliv (cancelled): {ord_clean[is_cancelled]['delivery_minutes'].isna().sum()}")
    completed_revenue = ord_clean[is_completed]['order_amount'].sum()
    print(f"Completed revenue sum: INR {completed_revenue:.2f}")

    # 5. Store in data/clean/ and data/instamart.db
    print("\n--- 4. Storing clean data ---")
    cust_export = cust_clean.copy()
    cust_export["signup_date"] = cust_export["signup_date"].dt.strftime("%Y-%m-%d")
    cust_export.to_csv(os.path.join(clean_dir, "customers.csv"), index=False)

    prod_clean.to_csv(os.path.join(clean_dir, "products.csv"), index=False)

    ord_export = ord_clean.copy()
    ord_export["order_ts"] = ord_export["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ord_export.to_csv(os.path.join(clean_dir, "orders.csv"), index=False)

    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cust_export.to_sql("customers", conn, if_exists="replace", index=False)
    prod_clean.to_sql("products", conn, if_exists="replace", index=False)
    ord_export.to_sql("orders", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print("Clean data saved to CSV and SQLite successfully.")

if __name__ == "__main__":
    run_part_a()

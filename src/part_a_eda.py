"""src/part_a_eda.py
Part A Task 7: pandas cross-check, pivot table, summary stats, IQR fences, and correlation matrix.
"""
import os
import pandas as pd
import numpy as np

def run_eda():
    clean_dir = os.path.join("data", "clean")
    cust = pd.read_csv(os.path.join(clean_dir, "customers.csv"))
    prod = pd.read_csv(os.path.join(clean_dir, "products.csv"))
    orders = pd.read_csv(os.path.join(clean_dir, "orders.csv"))

    completed = orders[orders["status"] == "completed"].copy()
    merged = completed.merge(cust, on="customer_id").merge(prod, on="product_id")

    # (a) Pivot table with margins=True
    pt = pd.pivot_table(
        merged,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="All"
    )
    print("--- Pivot Table (Revenue by City and Category) ---")
    print(pt.round(2))
    grand_total = pt.loc["All", "All"]
    print(f"\nPivot Table Grand Total: INR {grand_total:.2f}")

    # Groupby per city: sum, mean, count
    city_stats = merged.groupby("city")["order_amount"].agg(
        sum="sum",
        mean="mean",
        count="count"
    ).round(2)
    print("\n--- Per-City Aggregation ---")
    print(city_stats)

    # (b) Distribution statistics for completed order_amount
    amt = completed["order_amount"]
    mean_val = amt.mean()
    median_val = amt.median()
    skew_val = amt.skew()
    q1 = amt.quantile(0.25)
    q2 = amt.quantile(0.50)
    q3 = amt.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_below = (amt < lower_fence).sum()
    outliers_above = (amt > upper_fence).sum()

    print("\n--- Completed Order Amount Distribution Statistics ---")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Skewness: {skew_val:.4f}")
    print(f"Q1: {q1:.2f}")
    print(f"Q2 (Median): {q2:.2f}")
    print(f"Q3: {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Lower Fence (Q1 - 1.5*IQR): {lower_fence:.2f} (Outliers below: {outliers_below})")
    print(f"Upper Fence (Q3 + 1.5*IQR): {upper_fence:.2f} (Outliers above: {outliers_above})")

    # Correlation matrix
    corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr_matrix = completed[corr_cols].corr()
    print("\n--- Correlation Matrix ---")
    print(corr_matrix.round(4))

if __name__ == "__main__":
    run_eda()

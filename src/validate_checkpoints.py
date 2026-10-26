"""src/validate_checkpoints.py
Validates reference checkpoints for Part A and asserts data integrity.
Prints exactly 'ALL CHECKPOINTS PASSED' when all hold.
"""
import os
import sqlite3
import pandas as pd

def validate():
    db_path = os.path.join("data", "instamart.db")
    assert os.path.exists(db_path), f"Database not found at {db_path}"

    conn = sqlite3.connect(db_path)
    cust = pd.read_sql_query("SELECT * FROM customers", conn)
    prod = pd.read_sql_query("SELECT * FROM products", conn)
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    conn.close()

    # Checkpoint 1: Total customer rows is 1200
    assert len(cust) == 1200, f"Expected 1200 customers, got {len(cust)}"

    # Checkpoint 2: Exactly 6 distinct standardized cities with reference counts
    city_counts = cust["city"].value_counts().to_dict()
    expected_cities = {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197,
        "Chennai": 191, "Hyderabad": 191, "Mumbai": 184
    }
    assert city_counts == expected_cities, f"City counts mismatch: {city_counts} != {expected_cities}"

    # Checkpoint 3: Gender counts match reference: female 610, male 560, unknown 30
    gender_counts = cust["gender"].value_counts().to_dict()
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    assert gender_counts == expected_genders, f"Gender counts mismatch: {gender_counts} != {expected_genders}"

    # Checkpoint 4: 0 missing ages and all ages between 18 and 60
    assert cust["age"].isna().sum() == 0, "Missing ages found"
    assert cust["age"].min() >= 18 and cust["age"].max() <= 60, f"Ages outside 18-60: [{cust['age'].min()}, {cust['age'].max()}]"

    # Checkpoint 5: Total clean orders is 12,309 (one per order_id)
    assert len(orders) == 12309, f"Expected 12309 orders, got {len(orders)}"
    assert orders["order_id"].nunique() == 12309, "Duplicate order_id found"

    # Checkpoint 6: Order status counts: 11,110 completed, 1,199 cancelled, 0 pending
    status_counts = orders["status"].value_counts().to_dict()
    assert status_counts.get("completed") == 11110, f"Completed orders mismatch: {status_counts.get('completed')}"
    assert status_counts.get("cancelled") == 1199, f"Cancelled orders mismatch: {status_counts.get('cancelled')}"
    assert "pending" not in status_counts, "Pending orders found in clean data"

    # Checkpoint 7: 0 missing order_amount
    assert orders["order_amount"].isna().sum() == 0, "Missing order_amount found"

    # Checkpoint 8: Delivery minutes: 0 missing for completed, exactly 1,199 missing for cancelled
    completed_deliv_missing = orders[orders["status"] == "completed"]["delivery_minutes"].isna().sum()
    cancelled_deliv_missing = orders[orders["status"] == "cancelled"]["delivery_minutes"].isna().sum()
    assert completed_deliv_missing == 0, f"Completed delivery missing: {completed_deliv_missing}"
    assert cancelled_deliv_missing == 1199, f"Cancelled delivery missing: {cancelled_deliv_missing}"

    # Checkpoint 9: Sum of order_amount over completed orders is INR 2,998,668.80
    completed_rev = round(orders[orders["status"] == "completed"]["order_amount"].sum(), 2)
    assert abs(completed_rev - 2998668.80) < 0.01, f"Revenue sum mismatch: {completed_rev} != 2998668.80"

    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate()

"""src/run_sql_queries.py
Executes the seven required SQL queries from sql/queries.sql against data/instamart.db
and verifies results against reference figures.
"""
import os
import sqlite3
import pandas as pd

def run_queries():
    db_path = os.path.join("data", "instamart.db")
    sql_file = os.path.join("sql", "queries.sql")
    conn = sqlite3.connect(db_path)

    # Read queries from file
    with open(sql_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip comment lines first, then split on semicolon
    lines = [line for line in content.splitlines() if not line.strip().startswith("--")]
    clean_sql = "\n".join(lines)
    queries = [q.strip() for q in clean_sql.split(";") if q.strip()]

    print(f"Loaded {len(queries)} executable queries from {sql_file}")

    # Q1
    df1 = pd.read_sql_query(queries[0], conn)
    print(f"Q1 result count: {len(df1)} (Reference: 30)")
    assert len(df1) == 30, f"Q1 mismatch: {len(df1)}"

    # Q2
    df2 = pd.read_sql_query(queries[1], conn)
    total_rev = df2["revenue"].sum()
    largest_cell = df2.sort_values("revenue", ascending=False).iloc[0]
    print(f"Q2 rows: {len(df2)} (Reference: 30), Total revenue: INR {total_rev:.2f} (Reference: 2998668.80)")
    print(f"   Largest cell: {largest_cell['city']} / {largest_cell['category']} at INR {largest_cell['revenue']:.2f}")
    assert len(df2) == 30 and abs(total_rev - 2998668.80) < 0.05
    assert largest_cell["city"] == "Bengaluru" and largest_cell["category"] == "Electronics" and abs(largest_cell["revenue"] - 233909.45) < 0.05

    # Q3
    df3 = pd.read_sql_query(queries[2], conn)
    print(f"Q3 products count: {len(df3)} (Reference: 11)")
    assert len(df3) == 11

    # Q4
    df4 = pd.read_sql_query(queries[3], conn)
    print(f"Q4 customers with no completed order: {len(df4)} (Reference: 25)")
    assert len(df4) == 25

    # Q5
    df5 = pd.read_sql_query(queries[4], conn)
    print(f"Q5 customers above average: {len(df5)} (Reference: 550)")
    assert len(df5) == 550

    # CTE average check
    avg_check = pd.read_sql_query("""
        WITH customer_spend AS (
            SELECT customer_id, SUM(order_amount) AS total_spend
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
        )
        SELECT AVG(total_spend) AS avg_total_spend FROM customer_spend
    """, conn).iloc[0]["avg_total_spend"]
    print(f"Q5 Average completed customer spend: INR {avg_check:.2f} (Reference: 2552.06)")
    assert abs(avg_check - 2552.06) < 0.05

    # Q6
    df6 = pd.read_sql_query(queries[5], conn)
    # Check top customer per city
    top_per_city = df6[df6["rnk"] == 1][["city", "customer_id", "total_spend"]].to_dict(orient="records")
    print("Q6 top customers per city:")
    for row in top_per_city:
        print(f"   {row['city']} {row['customer_id']} ({row['total_spend']:.2f})")
    
    diff_rank_row = (df6["rnk"] != df6["row_num"]).sum()
    print(f"Q6 rows with RANK != ROW_NUMBER: {diff_rank_row} (Reference: 8)")
    assert diff_rank_row == 8

    # Q7
    df7 = pd.read_sql_query(queries[6], conn)
    last_cum = df7.iloc[-1]["cumulative_revenue"]
    print(f"Q7 daily rows: {len(df7)} (Reference: 183), Last cumulative revenue: INR {last_cum:.2f} (Reference: 2998668.80)")
    assert len(df7) == 183
    assert abs(last_cum - 2998668.80) < 0.05

    conn.close()
    print("\nALL SQL QUERY CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_queries()

"""tests/test_chatbot_integration.py
Automated validation of Copilot components against Capstone benchmarks.
"""
import os
import sys
import re
import sqlite3
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

try:
    import pytest
except ImportError:
    pytest = None

from mock_llm import est_tokens
from src.part_c_pipeline import redact_pii
from src.part_d_pipeline import (
    load_policy_chunks,
    PolicyRetriever,
    answer_policy_question,
    make_tool_map,
    run_agent,
)

def test_part_a_database_presence():
    db_path = os.path.join("data", "instamart.db")
    assert os.path.exists(db_path), f"Database missing at {db_path}"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check table presence
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('customers', 'products', 'orders')")
    tables = {r[0] for r in cur.fetchall()}
    assert tables == {'customers', 'products', 'orders'}, f"Missing tables: {{'customers', 'products', 'orders'}} - {tables}"

    # Validate completed orders sum
    df_orders = pd.read_sql_query("SELECT SUM(order_amount) as total_rev, COUNT(*) as cnt FROM orders WHERE status = 'completed'", conn)
    assert df_orders["cnt"].iloc[0] == 11110, f"Expected 11,110 completed orders, got {df_orders['cnt'].iloc[0]}"
    assert abs(df_orders["total_rev"].iloc[0] - 2998668.80) < 0.05, f"Revenue checkpoint mismatch: {df_orders['total_rev'].iloc[0]}"
    conn.close()

def test_part_c_redaction_rules():
    sample = "Call 9876543210 or email user@test.com regarding order ORD-100048 delivery."
    redacted, em_c, ph_c, ord_c = redact_pii(sample)

    assert em_c == 1, "Failed email count"
    assert ph_c == 1, "Failed phone count"
    assert ord_c == 1, "Failed order count"
    assert redacted == "Call [PHONE] or email [EMAIL] regarding order [ORDER_ID] delivery."

    # Verify clean re-match
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", redacted)
    assert not re.search(r"\b[6-9]\d{9}\b", redacted)
    assert not re.search(r"ORD-\d{6}", redacted)

def test_part_d_policy_retrieval_and_answer():
    chunks = load_policy_chunks(overlap=10)
    assert len(chunks) == 15, f"Expected 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2), got {len(chunks)}"

    retriever = PolicyRetriever(chunks)
    query = "What delivery time makes a completed order eligible for a full refund?"
    ans, sources = answer_policy_question(retriever, query)

    assert "more than 45 minutes" in ans.lower()
    assert any("refund_policy.txt" in s for s in sources)

def test_part_d_agent_scenarios():
    chunks = load_policy_chunks(overlap=10)
    retriever = PolicyRetriever(chunks)
    tool_map = make_tool_map(retriever)

    # Test Scenario S1: Late delivery (< INR 1000) -> Full Refund
    res_s1 = run_agent(
        "My order ORD-100048 reached me very late. I would like a refund.",
        scenario_id="TEST_S1",
        tool_map=tool_map,
        max_iterations=6,
        token_limit=6000
    )
    assert res_s1["status"] == "REFUND"
    assert abs(float(re.search(r"[\d.]+", res_s1["final_answer"]).group()) - 128.25) < 0.05

    # Test Scenario S7: Endless loop fallback
    res_s7 = run_agent(
        "[LOOP] Refund order ORD-100048.",
        scenario_id="TEST_S7",
        tool_map=tool_map,
        max_iterations=6,
        token_limit=6000
    )
    assert res_s7["status"] == "FALLBACK_MAX_ITERATIONS"

def test_smart_copilot_intent_routing():
    from copilot import InstamartCopilot
    copilot = InstamartCopilot()

    assert copilot.detect_intent("Why was ORD-100048 delayed?") == "policy_qa"
    assert copilot.detect_intent("How do I request a refund?") == "policy_qa"
    assert copilot.detect_intent("Can wallet balance be refunded as cash?") == "policy_qa"
    assert copilot.detect_intent("What is the refund policy?") == "policy_qa"
    assert copilot.detect_intent("Refund ORD-100048") == "refund_agent"
    assert copilot.detect_intent("ORD-100145 was late") == "refund_agent"
    assert copilot.detect_intent("C0082") == "customer_lookup"
    assert copilot.detect_intent("The app crashed during checkout") == "triage"

def test_part_b_customer_intelligence():
    from copilot import InstamartCopilot
    copilot = InstamartCopilot()
    assert "C0082" in copilot.seg_map
    assert copilot.seg_map["C0082"]["is_premium"] == True
    assert copilot.model is not None

if __name__ == "__main__":
    print("Running integration tests...")
    test_part_a_database_presence()
    print("  [PASS] test_part_a_database_presence")
    test_part_c_redaction_rules()
    print("  [PASS] test_part_c_redaction_rules")
    test_part_d_policy_retrieval_and_answer()
    print("  [PASS] test_part_d_policy_retrieval_and_answer")
    test_part_d_agent_scenarios()
    print("  [PASS] test_part_d_agent_scenarios")
    test_smart_copilot_intent_routing()
    print("  [PASS] test_smart_copilot_intent_routing")
    test_part_b_customer_intelligence()
    print("  [PASS] test_part_b_customer_intelligence")
    print("\nALL INTEGRATION TESTS PASSED SUCCESSFULLY!")


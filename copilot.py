"""copilot.py
Instamart Customer Intelligence & Support Copilot -- Unified CLI & Interactive Copilot
Provides:
  1. Guided Sequential Project Runner (Parts A to D end-to-end tour).
  2. Live Policy RAG Q&A Assistant (Part D - TF-IDF policy grounding over data/policies/*.txt).
  3. Live Autonomous Refund ReAct Agent with Tool Execution (Part D - database query, policy check, refund ledger).
  4. Live Support Ticket Triage with Privacy Guardrails & Routing (Part C - PII redaction, token bounds, few-shot classification).
  5. Live Customer Intelligence & ML Churn Predictor (Part B - RFM segmentation & trained lapse ML model).
  6. Smart Conversational Copilot (Natural-language intent auto-routing).

Run: python copilot.py
"""
import os
import sys
import re
import json
import sqlite3
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath("."))

from src.data_cleaner import run_part_a
from src.validate_checkpoints import validate as validate_part_a
from src.run_sql_queries import run_queries as run_sql
from src.eda_summary import run_eda
from src.customer_intelligence import run_part_b
from mock_llm import TransientLLMError
from src.ticket_triage import (
    redact_pii,
    call_mock_with_retry,
    validate_response,
    FEW_SHOT_SYSTEM_PROMPT,
    est_tokens,
    MAX_INPUT_TOKENS
)
from src.policy_agent import (
    load_policy_chunks,
    PolicyRetriever,
    answer_policy_question,
    make_tool_map,
    run_agent
)

def get_input(prompt=""):
    """Reads input safely, stripping Unicode BOM markers and handling exit signals."""
    try:
        val = input(prompt)
        return val.lstrip("\ufeff\xef\xbb\xbf\u200b").strip()
    except (KeyboardInterrupt, EOFError):
        return None

class InstamartCopilot:
    def __init__(self):
        print("Initializing Instamart Copilot services...")
        self.chunks = load_policy_chunks(overlap=10)
        self.retriever = PolicyRetriever(self.chunks)
        self.tool_map = make_tool_map(self.retriever)

        # Cached segments and model (loaded if Part B has run)
        self.seg_map = {}
        seg_file = os.path.join("outputs", "customer_segments.csv")
        if os.path.exists(seg_file):
            df_seg = pd.read_csv(seg_file)
            self.seg_map = df_seg.set_index("customer_id").to_dict(orient="index")

        self.model = None
        model_file = os.path.join("outputs", "lapse_model.joblib")
        if os.path.exists(model_file):
            self.model = joblib.load(model_file)

        self.scenario_id_seq = 1

    def run_sequential_tour(self):
        """Runs the complete project sequentially from Step 1 to Step 8."""
        steps = [
            ("Step 1: Generate Synthetic Raw Data & Policies", "generate_data.py", None),
            ("Step 2: Clean Raw Data & Populate SQLite DB", "src/data_cleaner.py", run_part_a),
            ("Step 3: Validate 9 Reference Checkpoints", "src/validate_checkpoints.py", validate_part_a),
            ("Step 4: Analytical SQL Queries (Q1 to Q7)", "src/run_sql_queries.py", run_sql),
            ("Step 5: Exploratory Data Analysis & Statistics", "src/eda_summary.py", run_eda),
            ("Step 6: Customer Intelligence, RFM, & Lapse Model", "src/customer_intelligence.py", run_part_b),
            ("Step 7: Support Ticket Triage with Guardrails", "src/ticket_triage.py", None),
            ("Step 8: Policy RAG & ReAct Refund Agent Scenarios", "src/policy_agent.py", None),
        ]

        print("\n" + "="*70)
        print(" GUIDED SEQUENTIAL TOUR -- RUNNING ENTIRE PROJECT STEP-BY-STEP")
        print("="*70)
        print("Options at each step: Press [Enter] to run, 'all' to run remaining, 'skip', or 'q' to quit.\n")

        auto_mode = False
        for idx, (title, script_name, func) in enumerate(steps, 1):
            print(f"\n[{idx}/8] {title} ({script_name})")
            if not auto_mode:
                cmd = get_input("  -> Run this step? [Enter=Yes / all / skip / q]: ")
                if cmd is None or cmd.lower() == "q":
                    print("Sequential execution stopped.")
                    return
                elif cmd.lower() == "all":
                    auto_mode = True
                elif cmd.lower() == "skip":
                    print("  Skipped.")
                    continue

            try:
                if func is not None:
                    func()
                else:
                    import subprocess
                    subprocess.run([sys.executable, script_name], check=True)
                print(f"  [SUCCESS] {title} passed.")
            except Exception as err:
                print(f"  [ERROR] {title} failed: {err}")
                cont = get_input("  Continue to next step anyway? (y/N): ")
                if not cont or cont.lower() != "y":
                    return

        print("\n" + "="*70)
        print(" ALL 8 STEPS OF THE PROJECT COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")

    def ask_policy(self, question):
        """Part D Policy RAG Assistant: answers questions grounded in policy text."""
        print(f"\n[RAG] Searching policy documents for: '{question}'...")
        answer, sources = answer_policy_question(self.retriever, question)
        print("\n--- Policy Assistant Answer ---")
        print(f"{answer}\n")
        print(f"Grounded In: {', '.join(sorted(set(sources)))}\n")

    def run_refund_agent(self, user_msg):
        """Part D Autonomous ReAct Refund Agent: plans and executes tool calls."""
        sc_id = f"LIVE_{self.scenario_id_seq}"
        self.scenario_id_seq += 1

        print(f"\n[ReAct Agent] Processing request: '{user_msg}'...")
        res = run_agent(user_msg, scenario_id=sc_id, tool_map=self.tool_map, max_iterations=6, token_limit=6000)

        print("\n--- ReAct Agent Execution Results ---")
        print(f"Status:       {res.get('status', 'N/A')}")
        print(f"Final Answer: {res.get('final_answer', '')}")
        print(f"Model Calls:  {res.get('llm_calls', 0)}")
        print(f"Tokens Used:  {res.get('tokens_used', 0)}")
        print("Tools Executed:")
        steps = res.get("steps", [])
        if not steps:
            print("  (Direct response without tool invocations)")
        for s in steps:
            tool_name = s.get("tool", "unknown")
            tool_args = s.get("arguments", s.get("args", {}))
            tool_obs = s.get("observation", "")
            print(f"  -> Tool: {tool_name} | Args: {tool_args} => Result: {tool_obs}")
        print()

    def triage_ticket(self, raw_text, customer_id=None):
        """Part C Triage Pipeline: PII redaction, token bounds, classification & routing."""
        print(f"\n[Triage] Processing ticket for customer '{customer_id or 'Guest'}'...")

        # 1. Input Guardrails
        if not raw_text.strip():
            print("[Guardrail] Text is empty -> Status: skipped_empty -> Desk: human_review (Priority: NORMAL)\n")
            return

        tokens = est_tokens(raw_text)
        if tokens > MAX_INPUT_TOKENS:
            print(f"[Guardrail] Text length ({tokens} tokens) exceeds {MAX_INPUT_TOKENS} -> Status: flagged_long -> Desk: human_review (Priority: NORMAL)\n")
            return

        # 2. PII Redaction in strict order: Email -> Phone -> Order ID
        redacted, n_em, n_ph, n_ord = redact_pii(raw_text)
        print(f"[PII Guardrail] Redacted {n_em} email(s), {n_ph} phone(s), {n_ord} order ID(s).")
        print(f"[Sanitized Prompt]: \"{redacted}\"")

        # 3. Model Classification with Few-Shot Prompt (user content is strictly redacted text)
        messages = [
            {"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT},
            {"role": "user", "content": redacted}
        ]
        try:
            resp = call_mock_with_retry(messages, ticket_id="LIVE_TICKET")
            raw_reply = resp.choices[0].message.content
            val_status, cat = validate_response(raw_reply)
        except TransientLLMError as err:
            print(f"[TransientLLMError] Service temporarily unavailable after retries: {err}")
            val_status, cat = "budget_exhausted", None
            raw_reply = ""
        except Exception as err:
            print(f"[Model Error] Classification error: {err}")
            val_status, cat = "flagged_invalid_output", None
            raw_reply = ""

        print(f"[Output Validation]: Status={val_status}, Category={cat or 'N/A'}")

        # 4. Routing
        is_premium = False
        seg_name = "Standard"
        if customer_id and customer_id in self.seg_map:
            is_premium = bool(self.seg_map[customer_id].get("is_premium", False))
            seg_name = self.seg_map[customer_id].get("segment_name", "Standard")

        if val_status in ("flagged_invalid_output", "budget_exhausted"):
            route, prio = "human_review", "NORMAL"
        elif cat == "fraud":
            route, prio = "security_team", "HIGH"
        elif is_premium:
            route, prio = "premium_desk", "HIGH"
        else:
            route, prio = "support_queue", "NORMAL"

        print("\n--- Support Ticket Triage Routing ---")
        print(f"Customer Segment: {seg_name} (Premium: {is_premium})")
        print(f"Assigned Category: {cat or 'N/A'}")
        print(f"Priority:          {prio}")
        print(f"Destination Desk:  {route}\n")

    def lookup_customer(self, customer_id):
        """Part B Customer Intelligence: RFM metrics, segment, and ML churn prediction."""
        cid = customer_id.upper().strip()
        if cid not in self.seg_map:
            print(f"Customer '{cid}' not found in segmentation database.\n")
            return

        row = self.seg_map[cid]
        print(f"\n--- Customer Profile: {cid} ---")
        print(f"Segment:    {row.get('segment_name', 'N/A')} (Cluster {row.get('segment_id', 'N/A')})")
        print(f"Premium:    {'YES' if row.get('is_premium') else 'NO'}")
        print(f"Recency:    {row.get('recency', 'N/A')} days since last order")
        print(f"Frequency:  {row.get('frequency', 'N/A')} completed orders")
        print(f"Monetary:   INR {float(row.get('monetary', 0)):.2f}")

        # Calculate Churn Risk via ML Model if model and clean data exist
        if self.model is None:
            print("[ML Churn Model] Model file not loaded (outputs/lapse_model.joblib missing). Run Part B first.\n")
            return

        try:
            cust_df = pd.read_csv(os.path.join("data", "clean", "customers.csv"))
            orders_df = pd.read_csv(os.path.join("data", "clean", "orders.csv"))
            prod_df = pd.read_csv(os.path.join("data", "clean", "products.csv"))

            orders_df["order_ts"] = pd.to_datetime(orders_df["order_ts"])
            cust_df["signup_date"] = pd.to_datetime(cust_df["signup_date"])
            cutoff = pd.to_datetime("2025-09-01 00:00:00")

            cust_meta = cust_df[cust_df["customer_id"] == cid]
            if cust_meta.empty:
                print(f"[ML Churn Model] Customer metadata not found for '{cid}'.\n")
                return

            # Strictly compute features before 2025-09-01 cutoff
            pre_orders = orders_df[(orders_df["order_ts"] < cutoff) & (orders_df["customer_id"] == cid)]
            pre_completed = pre_orders[pre_orders["status"] == "completed"]

            if pre_completed.empty:
                print("[ML Churn Model] Customer had no completed orders prior to 2025-09-01 cutoff; lapse probability is not defined for this cohort.\n")
                return

            merged = pre_completed.merge(prod_df[["product_id", "category"]], on="product_id", how="left")
            fav_cat = merged["category"].mode().iloc[0] if not merged.empty else "Grocery"

            # Match Part B pipeline cancellation rate: cancelled / all orders with final status ("completed", "cancelled")
            all_final = pre_orders[pre_orders["status"].isin(["completed", "cancelled"])]
            canc_rate = (all_final["status"] == "cancelled").sum() / len(all_final) if len(all_final) > 0 else 0.0

            signup_dt = pd.to_datetime(cust_meta.iloc[0]["signup_date"])
            tenure_days = (cutoff - signup_dt).days

            # Exactly match training feature schema and column order from Part B
            feature_cols = [
                "recency", "frequency", "monetary", "avg_discount_pct",
                "cancellation_rate", "avg_delivery_minutes", "tenure_days", "age",
                "favourite_category", "city", "gender"
            ]

            features = {
                "recency": int((cutoff - pre_completed["order_ts"].max()).days),
                "frequency": int(len(pre_completed)),
                "monetary": float(pre_completed["order_amount"].sum()),
                "avg_discount_pct": float(pre_completed["discount_pct"].mean()),
                "cancellation_rate": float(canc_rate),
                "avg_delivery_minutes": float(pre_completed["delivery_minutes"].mean()),
                "tenure_days": int(tenure_days),
                "age": int(cust_meta.iloc[0]["age"]),
                "favourite_category": str(fav_cat),
                "city": str(cust_meta.iloc[0]["city"]),
                "gender": str(cust_meta.iloc[0]["gender"]),
            }
            X_df = pd.DataFrame([features])[feature_cols]

            churn_prob = float(self.model.predict_proba(X_df)[0][1])
            churn_pred = int(self.model.predict(X_df)[0])
            risk_label = "HIGH RISK (LAPSED / CHURN)" if churn_pred == 1 else "LOW RISK (ACTIVE)"
            print(f"ML Churn Probability: {churn_prob*100:.1f}% -> Status: {risk_label}\n")

        except Exception as ex:
            print(f"[ML Churn Model] Could not compute churn estimate: {ex}\n")

    def detect_intent(self, text):
        """Disambiguates user intent for smart routing:
        1. Customer ID (e.g. C0082) -> customer_lookup
        2. Questions starting with question words or ending with '?' -> policy_qa (precedence over refund)
        3. Refund keywords or Order ID -> refund_agent
        4. Default -> triage
        """
        raw = text.strip()
        if re.match(r"^C\d{4}$", raw.upper()):
            return "customer_lookup"

        question_starters = ("what", "why", "how", "can", "could", "should", "is", "does", "do", "who", "when", "where", "which")
        words = raw.lower().split()
        first_word = words[0] if words else ""
        is_question = raw.endswith("?") or (first_word in question_starters)

        if is_question:
            return "policy_qa"

        has_order_id = bool(re.search(r"ORD-\d{6}", raw.upper()))
        has_refund_intent = any(kw in raw.lower() for kw in ["refund", "money back", "late delivery", "delayed"])

        if has_order_id or has_refund_intent:
            return "refund_agent"

        return "triage"

    def smart_copilot_chat(self):
        """Conversational mode: automatically detects intent and routes accordingly."""
        print("\n--- Smart Conversational Copilot ---")
        print("Type anything in natural language. The Copilot will automatically detect intent:")
        print("  - Customer ID (e.g. C0082) -> Profile & Churn Risk Lookup")
        print("  - Order / Refund (e.g. 'Refund ORD-100048', 'ORD-100145 was late') -> Autonomous ReAct Agent")
        print("  - Policy Questions (e.g. 'Can wallet balance be refunded as cash?', 'Why was ORD-100048 delayed?') -> Policy QA Assistant")
        print("  - Support Tickets (e.g. 'App crashed, email user@test.com') -> PII Redaction & Triage")
        print("Type 'back' to return to main menu.\n")

        while True:
            text = get_input("[You] > ")
            if not text:
                continue
            if text.lower() in ("back", "exit", "quit"):
                break

            intent = self.detect_intent(text)
            if intent == "customer_lookup":
                self.lookup_customer(text.upper())
            elif intent == "policy_qa":
                self.ask_policy(text)
            elif intent == "refund_agent":
                self.run_refund_agent(text)
            else:
                self.triage_ticket(text)

def main():
    copilot = InstamartCopilot()

    while True:
        print("="*60)
        print("       INSTAMART COPILOT - INTERACTIVE TERMINAL MENU")
        print("="*60)
        print(" 1. Run Complete Project Step-by-Step (Sequential Tour)")
        print(" 2. Live Policy RAG Q&A Assistant (Ask Policy Questions)")
        print(" 3. Live Autonomous Refund Agent (ReAct Tool Execution)")
        print(" 4. Live Support Ticket Triage (PII Redaction & Routing)")
        print(" 5. Live Customer Intelligence & Churn Risk Lookup")
        print(" 6. Smart Conversational Copilot (Auto-detects Intent)")
        print(" 7. Exit")
        print("="*60)

        choice = get_input("\nEnter choice (1-7): ")
        if choice is None or choice.lower() in ("7", "q", "exit", "quit"):
            print("\nThank you for using the Instamart Copilot! Goodbye.\n")
            break

        if choice == "1":
            copilot.run_sequential_tour()
        elif choice == "2":
            print("\n--- Policy Q&A Assistant (type 'back' to return) ---")
            while True:
                q = get_input("[Question] > ")
                if not q:
                    continue
                if q.lower() in ("back", "exit", "quit"):
                    break
                copilot.ask_policy(q)
        elif choice == "3":
            print("\n--- Autonomous Refund ReAct Agent (type 'back' to return) ---")
            print("Examples: 'Refund ORD-100048', 'ORD-100145 was late', 'ORD-100003 was on time'")
            while True:
                msg = get_input("[Refund Request] > ")
                if not msg:
                    continue
                if msg.lower() in ("back", "exit", "quit"):
                    break
                copilot.run_refund_agent(msg)
        elif choice == "4":
            print("\n--- Support Ticket Triage (type 'back' to return) ---")
            while True:
                cid = get_input("Customer ID (e.g. C0001, or Enter for guest): ")
                if cid is not None and cid.lower() in ("back", "exit", "quit"):
                    break
                ticket = get_input("Raw Ticket Text: ")
                if ticket is not None and ticket.lower() in ("back", "exit", "quit"):
                    break
                if ticket:
                    copilot.triage_ticket(ticket, customer_id=cid)
        elif choice == "5":
            print("\n--- Customer Intelligence & Churn Lookup (type 'back' to return) ---")
            while True:
                cid = get_input("Enter Customer ID (e.g. C0082, C0693, C0426): ")
                if not cid:
                    continue
                if cid.lower() in ("back", "exit", "quit"):
                    break
                copilot.lookup_customer(cid)
        elif choice == "6":
            copilot.smart_copilot_chat()
        else:
            print("Invalid choice. Please enter a number from 1 to 7.")

if __name__ == "__main__":
    main()

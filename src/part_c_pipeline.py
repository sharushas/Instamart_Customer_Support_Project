import os
import sys
import re
import json
import random
import time
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
import mock_llm
from mock_llm import TransientLLMError, est_tokens

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

ZERO_SHOT_SYSTEM_PROMPT = (
    "You are an AI support ticket classifier for Instamart. CLASSIFY the following customer ticket "
    "into exactly one of these four categories: billing, delivery, technical, fraud. "
    "Respond ONLY with a valid JSON object of the form: {\"category\": \"<category>\", \"reason\": \"<brief reason>\"}."
)

FEW_SHOT_SYSTEM_PROMPT = (
    "You are an AI support ticket classifier for Instamart. CLASSIFY the following customer ticket "
    "into exactly one of these four categories: billing, delivery, technical, fraud. "
    "Respond ONLY with a valid JSON object of the form: {\"category\": \"<category>\", \"reason\": \"<brief reason>\"}.\n"
    "EXAMPLES:\n"
    "Ticket: I was charged twice on my card for order [ORDER_ID]. Please refund.\n"
    "Response: {\"category\": \"billing\", \"reason\": \"double charge refund request\"}\n"
    "Ticket: Unauthorized transaction on my account for order [ORDER_ID], I did not place this!\n"
    "Response: {\"category\": \"fraud\", \"reason\": \"unauthorized card transaction\"}\n"
    "Ticket: The rider marked [ORDER_ID] as delivered but I never received anything.\n"
    "Response: {\"category\": \"delivery\", \"reason\": \"missing delivery package\"}\n"
    "Ticket: The app crashes every time I try to open cart or checkout.\n"
    "Response: {\"category\": \"technical\", \"reason\": \"application crash bug\"}"
)

def redact_pii(text):
    r"""
    Apply in exact order:
    1. email: [\w.+-]+@[\w-]+\.[\w.]+ -> [EMAIL]
    2. phone: \b[6-9]\d{9}\b -> [PHONE]
    3. order id: ORD-\d{6} -> [ORDER_ID]
    """
    email_pat = r"[\w.+-]+@[\w-]+\.[\w.]+"
    phone_pat = r"\b[6-9]\d{9}\b"
    order_pat = r"ORD-\d{6}"

    n_email = len(re.findall(email_pat, text))
    t1 = re.sub(email_pat, "[EMAIL]", text)

    n_phone = len(re.findall(phone_pat, t1))
    t2 = re.sub(phone_pat, "[PHONE]", t1)

    n_order = len(re.findall(order_pat, t2))
    t3 = re.sub(order_pat, "[ORDER_ID]", t2)

    return t3, n_email, n_phone, n_order

def call_mock_with_retry(messages, ticket_id):
    """
    Wrap model call in retry function: max 3 retries, exponential backoff with jitter.
    Wait before retry n (0, 1, 2) is BASE_DELAY * (2**n) * random.uniform(0.9, 1.15).
    Logs each retry.
    """
    for n in range(3):
        try:
            resp = mock_llm.chat(messages)
            return resp
        except TransientLLMError as e:
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"[RETRY] Ticket {ticket_id}: attempt {n+1}/3 failed with '{e}'. Backoff wait {wait:.4f}s")
            time.sleep(wait)
    # Final try
    return mock_llm.chat(messages)

def validate_response(content):
    """
    Parse as JSON:
      if category in ALLOWED_CATEGORIES -> classified_json
    If not valid JSON:
      extract lowercase words with re.findall(r"[a-z]+", text.lower())
      if exactly one allowed category word is present -> classified_keyword
    Else -> flagged_invalid_output
    """
    if not content:
        return "flagged_invalid_output", None

    try:
        data = json.loads(content)
        if isinstance(data, dict):
            cat = data.get("category")
            if cat in ALLOWED_CATEGORIES:
                return "classified_json", cat
    except Exception:
        pass

    # Keyword fallback
    words = set(re.findall(r"[a-z]+", content.lower()))
    matching_cats = [c for c in ALLOWED_CATEGORIES if c in words]
    if len(matching_cats) == 1:
        return "classified_keyword", matching_cats[0]

    return "flagged_invalid_output", None

def run_triage_pipeline(prompt_type="few-shot", token_budget=100000):
    raw_tickets = pd.read_csv(os.path.join("data", "raw", "tickets.csv"), keep_default_na=False)
    sys_prompt = FEW_SHOT_SYSTEM_PROMPT if prompt_type == "few-shot" else ZERO_SHOT_SYSTEM_PROMPT

    mock_llm.reset()
    random.seed(42)

    running_tokens = 0
    results = []

    for _, row in raw_tickets.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        oid = row["order_id"]
        raw_text = row["text"]
        true_cat = row["true_category"]

        # 1. Input guardrails
        if not raw_text or raw_text.strip() == "":
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "skipped_empty",
                "category": "", "true_category": true_cat, "tokens_used": 0, "response_raw": ""
            })
            continue

        est_tok = est_tokens(raw_text)
        if est_tok > MAX_INPUT_TOKENS:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "flagged_long",
                "category": "", "true_category": true_cat, "tokens_used": 0, "response_raw": ""
            })
            continue

        # 2. Redact PII
        redacted_text, _, _, _ = redact_pii(raw_text)

        # 3. Budget Check
        if running_tokens >= token_budget:
            results.append({
                "ticket_id": tid, "customer_id": cid, "status": "budget_exhausted",
                "category": "", "true_category": true_cat, "tokens_used": 0, "response_raw": ""
            })
            continue

        # 4. LLM Call
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": redacted_text}
        ]

        resp = call_mock_with_retry(messages, tid)
        call_tokens = resp.usage.total_tokens
        running_tokens += call_tokens
        content = resp.choices[0].message.content

        # 5. Output Validation
        status, pred_cat = validate_response(content)

        results.append({
            "ticket_id": tid, "customer_id": cid, "status": status,
            "category": pred_cat if pred_cat else "", "true_category": true_cat,
            "tokens_used": call_tokens, "response_raw": content
        })

    df_res = pd.DataFrame(results)
    return df_res, running_tokens

def run_part_c():
    raw_tickets = pd.read_csv(os.path.join("data", "raw", "tickets.csv"), keep_default_na=False)
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. PII Audit across all 164 tickets
    print("--- 1. PII Redaction Audit across all 164 tickets ---")
    tot_email, tot_phone, tot_order = 0, 0, 0
    all_redacted = []
    for _, r in raw_tickets.iterrows():
        red, ne, np, no = redact_pii(r["text"])
        tot_email += ne
        tot_phone += np
        tot_order += no
        all_redacted.append(red)

    print(f"Emails replaced: {tot_email} (Reference: 55)")
    print(f"Phones replaced: {tot_phone} (Reference: 91)")
    print(f"Order IDs replaced: {tot_order} (Reference: 132)")
    print(f"Total substitutions: {tot_email + tot_phone + tot_order} (Reference: 278)")

    # Assert 0 PII remains after redaction
    for red in all_redacted:
        assert re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", red) is None
        assert re.search(r"\b[6-9]\d{9}\b", red) is None
        assert re.search(r"ORD-\d{6}", red) is None
    print("Assertion passed: 0 PII patterns found in redacted texts.")

    # 2. Compare Zero-Shot vs Few-Shot (no budget restriction)
    print("\n--- 2. Comparing Zero-Shot vs Few-Shot ---")
    df_zero, tokens_zero = run_triage_pipeline(prompt_type="zero-shot", token_budget=1000000)
    df_few, tokens_few = run_triage_pipeline(prompt_type="few-shot", token_budget=1000000)

    # Check status counts on eligible tickets (156 tickets)
    status_counts_zero = df_zero["status"].value_counts().to_dict()
    status_counts_few = df_few["status"].value_counts().to_dict()
    print("Zero-shot status counts:\n", status_counts_zero)
    print("Few-shot status counts:\n", status_counts_few)

    # Accuracy over classified tickets (status in classified_json, classified_keyword)
    classified_zero = df_zero[df_zero["status"].isin(["classified_json", "classified_keyword"])].copy()
    classified_few = df_few[df_few["status"].isin(["classified_json", "classified_keyword"])].copy()

    acc_zero = (classified_zero["category"] == classified_zero["true_category"]).mean() * 100.0
    acc_few = (classified_few["category"] == classified_few["true_category"]).mean() * 100.0

    print(f"\nZero-shot: {len(classified_zero)} classified, Accuracy: {acc_zero:.1f}%, Total tokens: {tokens_zero}, Tokens/classified: {tokens_zero/len(classified_zero):.1f}")
    print(f"Few-shot:  {len(classified_few)} classified, Accuracy: {acc_few:.1f}%, Total tokens: {tokens_few}, Tokens/classified: {tokens_few/len(classified_few):.1f}")

    # Zero-shot Confusion Matrix
    print("\nZero-shot Confusion Matrix (classified tickets):")
    ct_zero = pd.crosstab(classified_zero["true_category"], classified_zero["category"], rownames=["True"], colnames=["Predicted"])
    print(ct_zero)

    # 3. Budget Demo (Zero-shot with TOKEN_BUDGET = 6000)
    print("\n--- 3. Budget Demo (TOKEN_BUDGET = 6000) ---")
    df_budget, tokens_budget = run_triage_pipeline(prompt_type="zero-shot", token_budget=6000)
    exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
    print(f"Tickets ending budget_exhausted: {exhausted_count} (out of {len(df_budget)})")
    
    # Save outputs/triage_budget_demo.csv
    budget_export = df_budget[["ticket_id", "customer_id", "status", "category", "tokens_used"]].copy()
    budget_export.to_csv(os.path.join(outputs_dir, "triage_budget_demo.csv"), index=False)
    print("Saved outputs/triage_budget_demo.csv")

    # 4. Final Triage Results with Routing (Few-shot, TOKEN_BUDGET = 100000)
    print("\n--- 4. Final Triage Results with Customer Segment Routing ---")
    segments_path = os.path.join(outputs_dir, "customer_segments.csv")
    seg_df = pd.read_csv(segments_path).set_index("customer_id") if os.path.exists(segments_path) else pd.DataFrame()

    final_rows = []
    for _, row in df_few.iterrows():
        cid = row["customer_id"]
        status = row["status"]
        cat = row["category"]
        tokens = row["tokens_used"]

        is_prem = False
        seg_name = ""
        if cid in seg_df.index:
            is_prem = bool(seg_df.loc[cid, "is_premium"])
            seg_name = str(seg_df.loc[cid, "segment_name"])

        # Routing rules:
        # skipped_empty, flagged_long, flagged_invalid_output, budget_exhausted -> human_review, NORMAL
        # else if fraud -> security_team, HIGH
        # else if is_premium -> premium_desk, HIGH
        # else -> support_queue, NORMAL
        if status in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            route = "human_review"
            priority = "NORMAL"
        elif cat == "fraud":
            route = "security_team"
            priority = "HIGH"
        elif is_prem:
            route = "premium_desk"
            priority = "HIGH"
        else:
            route = "support_queue"
            priority = "NORMAL"

        final_rows.append({
            "ticket_id": row["ticket_id"],
            "customer_id": cid,
            "status": status,
            "category": cat if status in ["classified_json", "classified_keyword"] else "",
            "segment_name": seg_name if cid in seg_df.index else "",
            "priority": priority,
            "route": route,
            "tokens_used": tokens
        })

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(os.path.join(outputs_dir, "triage_results.csv"), index=False)
    print("Saved outputs/triage_results.csv (164 rows)")
    print("\nRouting Summary:")
    print(final_df["route"].value_counts())
    print("\nPriority Summary:")
    print(final_df["priority"].value_counts())

if __name__ == "__main__":
    run_part_c()

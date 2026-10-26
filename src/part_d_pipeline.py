"""src/part_d_pipeline.py
Part D: Policy RAG Assistant and ReAct Refund Agent with State Checkpointing,
Safe Fallbacks, Ledger Logging, and Resume Capability.
"""
import os
import sys
import re
import json
import time
import sqlite3
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath("."))
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- 1. Chunking ----------------
def chunk_text(text, doc_name, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    step = chunk_size - overlap if (chunk_size - overlap) > 0 else chunk_size
    idx = 0
    while idx < len(words):
        chunk_words = words[idx:idx + chunk_size]
        chunks.append({
            "source": doc_name,
            "text": " ".join(chunk_words)
        })
        if idx + chunk_size >= len(words):
            break
        idx += step
    return chunks

def load_policy_chunks(overlap=10):
    pol_dir = os.path.join("data", "policies")
    all_chunks = []
    files = sorted(os.listdir(pol_dir))
    for fn in files:
        if fn.endswith(".txt"):
            with open(os.path.join(pol_dir, fn), "r", encoding="utf-8") as f:
                content = f.read().strip()
            chunks = chunk_text(content, fn, chunk_size=40, overlap=overlap)
            all_chunks.extend(chunks)
    return all_chunks

# ---------------- 2. Retrieval ----------------
class PolicyRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query, k=3):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        ranked_indices = sims.argsort()[::-1][:k]
        results = []
        for idx in ranked_indices:
            results.append({
                "source": self.chunks[idx]["source"],
                "score": float(sims[idx]),
                "text": self.chunks[idx]["text"]
            })
        return results

# ---------------- 3. Policy Assistant QA ----------------
def answer_policy_question(retriever, question):
    top_chunks = retriever.retrieve(question, k=3)
    
    # User message exact format:
    # CONTEXT:\n[<source file name>]\n<chunk text>\n... \nQUESTION:\n<question>
    context_blocks = []
    for c in top_chunks:
        context_blocks.append(f"[{c['source']}]\n{c['text']}")
    context_str = "\n".join(context_blocks)
    
    user_msg = f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
    sys_msg = "You are a POLICY ASSISTANT. Answer only from the supplied context. If not found, say you could not find the information in the policy documents."
    
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg}
    ]
    resp = mock_llm.chat(messages)
    ans_text = resp.choices[0].message.content
    sources = [c["source"] for c in top_chunks]
    return ans_text, sources

# ---------------- 4. Tools & Tool Map ----------------
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order by order_id in data/instamart.db.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Retrieve policy guidelines for late deliveries and refunds.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund amount based on order status and delivery minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Issue refund for orders up to INR 1000 and log to ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate refund above INR 1000 to human support lead and log to ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

def make_tool_map(retriever, db_path="data/instamart.db", ledger_path="outputs/refund_ledger.jsonl"):
    def get_order(order_id):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "order_id": row[0],
                "status": row[1],
                "order_amount": float(row[2]),
                "delivery_minutes": float(row[3]) if row[3] is not None else ""
            }
        return {"error": "order not found"}

    def search_policy(query):
        res = retriever.retrieve(query, k=3)
        return {"results": res}

    def calculate_refund(order_amount, delivery_minutes):
        try:
            amt = float(order_amount)
            deliv = float(delivery_minutes)
            if deliv > 45:
                return {"refund_inr": amt}
            return {"refund_inr": 0.0}
        except (ValueError, TypeError):
            return {"refund_inr": 0.0}
    def _append_ledger(tool_name, order_id, refund_inr):
        os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tool": tool_name,
            "order_id": order_id,
            "refund_inr": float(refund_inr)
        }
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def issue_refund(order_id, refund_inr):
        _append_ledger("issue_refund", order_id, refund_inr)
        return {"refund_inr": float(refund_inr), "status": "issued"}

    def escalate_to_human(order_id, refund_inr):
        _append_ledger("escalate_to_human", order_id, refund_inr)
        return {"refund_inr": float(refund_inr), "status": "escalated"}

    return {
        "get_order": get_order,
        "search_policy": search_policy,
        "calculate_refund": calculate_refund,
        "issue_refund": issue_refund,
        "escalate_to_human": escalate_to_human
    }

# ---------------- 5. ReAct Agent Loop ----------------
def save_checkpoint(run_id, scenario_id, messages, tokens_used, step_num, tools_executed):
    state_dir = "state"
    os.makedirs(state_dir, exist_ok=True)
    ckpt_path = os.path.join(state_dir, f"{run_id}.json")
    data = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_num,
        "tools_executed": tools_executed
    }
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def run_agent(user_message, scenario_id, tool_map, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, simulate_crash_after_tool=None):
    if not run_id:
        run_id = f"run_{scenario_id}_{int(time.time()*1000)}"

    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "You are a customer support refund AGENT for Instamart."},
        {"role": "user", "content": user_message}
    ]
    tokens_used = 0
    llm_calls = 0
    steps = []
    tools_executed = []

    for it in range(max_iterations):
        # Time limit check at top of iteration
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            summary = f"Handing over to human: elapsed time {elapsed:.2f}s exceeded limit of {time_limit_seconds}s."
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT",
                "final_answer": summary, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "tools_executed": tools_executed
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        # Token limit check
        if tokens_used > token_limit:
            summary = f"Handing over to human: tokens {tokens_used} exceeded limit of {token_limit}."
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": summary, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "tools_executed": tools_executed
            }

        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)

        if not tool_calls:
            final_content = choice.message.content or ""
            # Normal status is word before colon
            m = re.match(r"^([A-Z_]+):", final_content)
            status = m.group(1) if m else "COMPLETED"
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": status,
                "final_answer": final_content, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "tools_executed": tools_executed
            }

        # Handle tool calls
        tc = tool_calls[0]
        t_id = tc.id
        t_name = tc.name
        t_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": t_id, "name": t_name, "arguments": t_args}]
        })

        # Execute tool
        fn = tool_map.get(t_name)
        obs = fn(**t_args) if fn else {"error": f"Tool {t_name} not found"}
        obs_str = json.dumps(obs)

        messages.append({
            "role": "tool",
            "tool_call_id": t_id,
            "name": t_name,
            "content": obs_str
        })

        tools_executed.append(t_name)
        steps.append({
            "step": len(tools_executed),
            "tool": t_name,
            "arguments": t_args,
            "observation": obs
        })

        save_checkpoint(run_id, scenario_id, messages, tokens_used, len(tools_executed), tools_executed)

        # Simulated crash trigger for testing resumability
        if simulate_crash_after_tool and len(tools_executed) == simulate_crash_after_tool:
            raise RuntimeError(f"Simulated crash after tool {len(tools_executed)} ({t_name})")

    # If reached max_iterations without stop
    summary = f"Handing over to human agent: automated resolution exceeded {max_iterations} steps."
    return {
        "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": summary, "steps": steps, "llm_calls": llm_calls,
        "tokens_used": tokens_used, "tools_executed": tools_executed
    }

def resume_agent(run_id, tool_map, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    ckpt_path = os.path.join("state", f"{run_id}.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_time = time.monotonic()
    scenario_id = data["scenario_id"]
    messages = data["messages"]
    tokens_used = data["tokens_used"]
    tools_executed = list(data["tools_executed"])
    steps = []
    llm_calls = 0

    for it in range(max_iterations):
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            summary = f"Handing over to human: elapsed time {elapsed:.2f}s exceeded limit of {time_limit_seconds}s."
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT",
                "final_answer": summary, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "tools_executed": tools_executed
            }

        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        if tokens_used > token_limit:
            summary = f"Handing over to human: tokens {tokens_used} exceeded limit of {token_limit}."
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": summary, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "tools_executed": tools_executed
            }

        choice = resp.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None)

        if not tool_calls:
            final_content = choice.message.content or ""
            m = re.match(r"^([A-Z_]+):", final_content)
            status = m.group(1) if m else "COMPLETED"
            return {
                "run_id": run_id, "scenario_id": scenario_id, "status": status,
                "final_answer": final_content, "steps": steps, "llm_calls": llm_calls,
                "tokens_used": tokens_used, "tools_executed": tools_executed
            }

        tc = tool_calls[0]
        t_id = tc.id
        t_name = tc.name
        t_args = tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments)

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": t_id, "name": t_name, "arguments": t_args}]
        })

        fn = tool_map.get(t_name)
        obs = fn(**t_args) if fn else {"error": f"Tool {t_name} not found"}
        obs_str = json.dumps(obs)

        messages.append({
            "role": "tool",
            "tool_call_id": t_id,
            "name": t_name,
            "content": obs_str
        })

        tools_executed.append(t_name)
        steps.append({
            "step": len(tools_executed),
            "tool": t_name,
            "arguments": t_args,
            "observation": obs
        })

        save_checkpoint(run_id, scenario_id, messages, tokens_used, len(tools_executed), tools_executed)

    summary = f"Handing over to human agent: automated resolution exceeded {max_iterations} steps."
    return {
        "run_id": run_id, "scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": summary, "steps": steps, "llm_calls": llm_calls,
        "tokens_used": tokens_used, "tools_executed": tools_executed
    }

# ---------------- Main Part D Execution ----------------
def run_part_d():
    print("--- 1. Document Chunking & Verification ---")
    chunks_10 = load_policy_chunks(overlap=10)
    print(f"Total chunks with overlap=10: {len(chunks_10)} (Reference: 15)")
    by_doc = {}
    for c in chunks_10:
        by_doc[c["source"]] = by_doc.get(c["source"], 0) + 1
    for d, cnt in sorted(by_doc.items()):
        print(f"   {d}: {cnt} chunks")

    # ---------------- 2. Vectorisation & Retrieval ----------------
    print("\n--- 2. Vectorisation & Top-3 Retrieval Check ---")
    retriever_10 = PolicyRetriever(chunks_10)
    q_test = "refund policy late delivery"
    top_test = retriever_10.retrieve(q_test, k=3)
    print(f"Query: '{q_test}' -> Top 3 sources: {[c['source'] for c in top_test]}")
    assert all(c["source"] == "refund_policy.txt" for c in top_test), "Top 3 chunks not all refund_policy.txt"

    # ---------------- 3 & 4. Evaluate Policy Assistant on policy_eval.csv ----------------
    print("\n--- 3 & 4. Evaluating Policy QA Assistant on policy_eval.csv ---")
    eval_csv = os.path.join("data", "raw", "policy_eval.csv")
    eval_df = pd.read_csv(eval_csv, keep_default_na=False)

    def evaluate_retriever(retriever, label="overlap=10"):
        doc_matches = 0
        fact_matches = 0
        fallback_matches = 0
        records = []
        for _, row in eval_df.iterrows():
            q = row["question"]
            exp_doc = row["expected_doc"]
            exp_fact = row["expected_fact"]

            ans, sources = answer_policy_question(retriever, q)
            has_doc = (exp_doc in sources) if exp_doc != "NONE" else False
            has_fact = (exp_fact in ans) if exp_fact != "" else False
            is_fallback = ans.startswith("I could not find")

            if exp_doc != "NONE":
                if has_doc: doc_matches += 1
                if has_fact: fact_matches += 1
            else:
                if is_fallback: fallback_matches += 1

            records.append({
                "question": q, "expected_doc": exp_doc, "sources": sources,
                "expected_fact": exp_fact, "answer": ans,
                "doc_match": has_doc, "fact_match": has_fact, "fallback": is_fallback
            })

        print(f"\nResults for {label}:")
        print(f"   Answerable Doc in Top-3: {doc_matches}/12")
        print(f"   Answerable Fact contained: {fact_matches}/12")
        print(f"   Unanswerable Fallbacks:  {fallback_matches}/2")
        return pd.DataFrame(records), doc_matches, fact_matches, fallback_matches

    res_10, d10, f10, fb10 = evaluate_retriever(retriever_10, "overlap=10 (standard)")

    # Repeat with zero overlap
    chunks_0 = load_policy_chunks(overlap=0)
    retriever_0 = PolicyRetriever(chunks_0)
    res_0, d0, f0, fb0 = evaluate_retriever(retriever_0, "overlap=0 (zero overlap)")

    # Identify broken question(s) with zero overlap
    broken = res_0[(res_0["expected_doc"] != "NONE") & ((~res_0["doc_match"]) | (~res_0["fact_match"]))]
    print("\nQuestions broken by zero overlap:")
    for _, b in broken.iterrows():
        print(f"   Question: '{b['question']}' | Fact: '{b['expected_fact']}' | Answer: '{b['answer']}'")

    # ---------------- 5 & 6. Run ReAct Refund Agent ----------------
    print("\n--- 5 & 6. Running ReAct Refund Agent on 7 Scenarios ---")
    tool_map = make_tool_map(retriever_10)
    scenarios_df = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"))

    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    runs_file = os.path.join(outputs_dir, "agent_runs.jsonl")
    ledger_file = os.path.join(outputs_dir, "refund_ledger.jsonl")
    
    # Reset files
    if os.path.exists(runs_file): os.remove(runs_file)
    if os.path.exists(ledger_file): os.remove(ledger_file)

    scenario_results = []
    with open(runs_file, "w", encoding="utf-8") as f_runs:
        for _, row in scenarios_df.iterrows():
            sid = row["scenario_id"]
            cid = row["customer_id"]
            msg = row["message"]
            exp_status = row["expected_outcome"]
            exp_refund = float(row["expected_refund_inr"])

            run_res = run_agent(msg, sid, tool_map, max_iterations=6, token_limit=6000, time_limit_seconds=10)
            
            # Extract refund amount if issued or escalated
            amt = 0.0
            for step in run_res["steps"]:
                if step["tool"] in ["issue_refund", "escalate_to_human"]:
                    amt = step["arguments"].get("refund_inr", 0.0)

            # S7 expected outcome is FALLBACK_MAX_ITERATIONS
            status = run_res["status"]
            scenario_results.append({
                "scenario_id": sid,
                "status": status,
                "refund_inr": amt,
                "expected_outcome": exp_status,
                "expected_refund": exp_refund,
                "llm_calls": run_res["llm_calls"],
                "tokens_used": run_res["tokens_used"]
            })
            
            # Write agent_runs.jsonl line
            f_runs.write(json.dumps({
                "scenario_id": sid,
                "status": status,
                "final_answer": run_res["final_answer"],
                "llm_calls": run_res["llm_calls"],
                "tokens_used": run_res["tokens_used"],
                "steps": run_res["steps"]
            }) + "\n")

    df_sc_res = pd.DataFrame(scenario_results)
    print("\n--- Seven Scenario Results ---")
    print(df_sc_res.to_string(index=False))

    # ---------------- 7. Stop Safely Demonstrations ----------------
    print("\n--- 7. Stop Safely Demonstrations ---")
    s1_row = scenarios_df.iloc[0]
    
    # (a) Token limit demo: S1 with token_limit=200
    res_tok_demo = run_agent(s1_row["message"], "S1_TOKEN_DEMO", tool_map, token_limit=200)
    print(f"Token limit demo status: {res_tok_demo['status']} (Reference: FALLBACK_TOKEN_LIMIT)")
    assert res_tok_demo["status"] == "FALLBACK_TOKEN_LIMIT"

    # (b) Timeout demo: S1 with time_limit_seconds=0
    res_time_demo = run_agent(s1_row["message"], "S1_TIME_DEMO", tool_map, time_limit_seconds=0)
    print(f"Timeout demo status:     {res_time_demo['status']} (Reference: FALLBACK_TIMEOUT)")
    assert res_time_demo["status"] == "FALLBACK_TIMEOUT"

    # ---------------- 8. Crash and Resume Demonstration ----------------
    print("\n--- 8. Crash & Resume Demonstration with S1 ---")
    demo_run_id = f"s1_crash_resume_demo_{int(time.time())}"
    
    # 1. Start run and raise simulated crash after tool 2 (search_policy)
    crashed = False
    try:
        run_agent(s1_row["message"], "S1", tool_map, run_id=demo_run_id, simulate_crash_after_tool=2)
    except RuntimeError as e:
        crashed = True
        print(f"Simulated crash successfully triggered: '{e}'")

    assert crashed, "Simulated crash failed to raise"

    # Inspect checkpoint saved in state/
    ckpt_path = os.path.join("state", f"{demo_run_id}.json")
    assert os.path.exists(ckpt_path), f"Checkpoint file missing at {ckpt_path}"
    with open(ckpt_path) as f:
        ckpt_data = json.load(f)
    print(f"Checkpoint before resume: tools_executed={ckpt_data['tools_executed']}")

    # 2. Resume agent from checkpoint
    resumed_res = resume_agent(demo_run_id, tool_map)
    print(f"Resume completed. Final status: {resumed_res['status']}, Final Answer: {resumed_res['final_answer']}")
    print(f"Tools executed across both parts: {resumed_res['tools_executed']}")
    assert resumed_res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"], \
        f"Tools mismatch: {resumed_res['tools_executed']}"
    assert resumed_res["status"] == "REFUND", f"Status mismatch: {resumed_res['status']}"

    print("\nALL PART D TESTS AND DEMONSTRATIONS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_part_d()

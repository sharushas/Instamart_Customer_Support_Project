"""app.py
Instamart Customer Intelligence & Support Copilot Web Application Server.
Built using Python Standard Library http.server and integrates Parts A, B, C, D.
Run: python app.py
"""
import os
import sys
import json
import sqlite3
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

sys.path.insert(0, os.path.abspath("."))
import joblib
import pandas as pd
import numpy as np

import mock_llm
from src.part_c_pipeline import redact_pii, validate_response, ZERO_SHOT_SYSTEM_PROMPT, FEW_SHOT_SYSTEM_PROMPT
from src.part_d_pipeline import (
    load_policy_chunks, PolicyRetriever, answer_policy_question,
    make_tool_map, run_agent
)

# Global caches
retriever = None
tool_map = None
lapse_model = None

def init_services():
    global retriever, tool_map, lapse_model
    chunks = load_policy_chunks(overlap=10)
    retriever = PolicyRetriever(chunks)
    tool_map = make_tool_map(retriever)
    model_path = os.path.join("outputs", "lapse_model.joblib")
    if os.path.exists(model_path):
        lapse_model = joblib.load(model_path)

class CopilotRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(os.path.join("web", "index.html"), "rb") as f:
                self.wfile.write(f.read())
            return

        if path == "/api/ledger":
            ledger_file = os.path.join("outputs", "refund_ledger.jsonl")
            records = []
            if os.path.exists(ledger_file):
                with open(ledger_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try: records.append(json.loads(line))
                            except: pass
            self._send_json(records)
            return

        if path == "/api/triage_results":
            res_file = os.path.join("outputs", "triage_results.csv")
            if os.path.exists(res_file):
                df = pd.read_csv(res_file, keep_default_na=False)
                self._send_json(df.to_dict(orient="records"))
            else:
                self._send_json([])
            return

        if path == "/api/retention_list":
            ret_file = os.path.join("outputs", "retention_list.csv")
            if os.path.exists(ret_file):
                df = pd.read_csv(ret_file)
                self._send_json(df.to_dict(orient="records"))
            else:
                self._send_json([])
            return

        if path == "/api/customer":
            cid = query.get("id", [""])[0].strip()
            seg_file = os.path.join("outputs", "customer_segments.csv")
            db_file = os.path.join("data", "instamart.db")
            if not cid or not os.path.exists(seg_file) or not os.path.exists(db_file):
                self._send_json({"error": "Customer not found or data missing"})
                return

            seg_df = pd.read_csv(seg_file)
            cust_row = seg_df[seg_df["customer_id"] == cid]
            if cust_row.empty:
                self._send_json({"error": f"Customer ID {cid} not found in active database"})
                return

            cust_dict = cust_row.iloc[0].to_dict()
            # Fetch demographic from sqlite
            conn = sqlite3.connect(db_file)
            cur = conn.cursor()
            cur.execute("SELECT city, age, gender FROM customers WHERE customer_id = ?", (cid,))
            db_row = cur.fetchone()
            conn.close()
            if db_row:
                cust_dict["city"] = db_row[0]
                cust_dict["age"] = db_row[1]
                cust_dict["gender"] = db_row[2]

            self._send_json(cust_dict)
            return

        if path == "/api/search_policy":
            q = query.get("q", ["refund policy"])[0]
            chunks = retriever.retrieve(q, k=4)
            self._send_json(chunks)
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            data = json.loads(body)
        except:
            data = {}

        if path == "/api/chat":
            msg = data.get("message", "")
            mode = data.get("mode", "agent")

            if mode == "policy":
                ans, sources = answer_policy_question(retriever, msg)
                self._send_json({
                    "mode": "policy",
                    "answer": ans,
                    "sources": sources
                })
            else:
                # ReAct Refund Agent
                sid = f"WEB_{int(time.time())}"
                run_res = run_agent(msg, sid, tool_map)
                self._send_json(run_res)
            return

        if path == "/api/triage_test":
            raw_text = data.get("text", "")
            redacted, ne, np, no = redact_pii(raw_text)

            # Classify using few-shot prompt
            messages = [
                {"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT},
                {"role": "user", "content": redacted}
            ]
            resp = mock_llm.chat(messages)
            content = resp.choices[0].message.content
            status, cat = validate_response(content)

            # Routing
            if cat == "fraud":
                route, priority = "security_team", "HIGH"
            elif status in ["skipped_empty", "flagged_long", "flagged_invalid_output"]:
                route, priority = "human_review", "NORMAL"
            else:
                route, priority = "support_queue", "NORMAL"

            self._send_json({
                "original": raw_text,
                "redacted": redacted,
                "category": cat,
                "status": status,
                "route": route,
                "priority": priority
            })
            return

        if path == "/api/predict_lapse":
            # Feature payload
            rec = float(data.get("recency", 20))
            freq = float(data.get("frequency", 5))
            mon = float(data.get("monetary", 1200))
            disc = float(data.get("avg_discount_pct", 10))
            canc = float(data.get("cancellation_rate", 0.05))
            deliv = float(data.get("avg_delivery_minutes", 30))
            tenure = float(data.get("tenure_days", 300))
            age = float(data.get("age", 35))
            fav_cat = data.get("favourite_category", "Grocery")
            city = data.get("city", "Bengaluru")
            gender = data.get("gender", "female")

            row_df = pd.DataFrame([{
                "recency": rec,
                "frequency": freq,
                "monetary": mon,
                "avg_discount_pct": disc,
                "cancellation_rate": canc,
                "avg_delivery_minutes": deliv,
                "tenure_days": tenure,
                "age": age,
                "favourite_category": fav_cat,
                "city": city,
                "gender": gender
            }])

            prob = float(lapse_model.predict_proba(row_df)[0, 1]) if lapse_model else 0.5
            pred_class = int(prob >= 0.5)

            self._send_json({
                "lapse_probability": round(prob, 4),
                "predicted_lapsed": pred_class
            })
            return

        self.send_error(404, "Endpoint not found")

    def _send_json(self, obj):
        content = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

def run_server(port=8000):
    init_services()
    server_address = ("", port)
    httpd = HTTPServer(server_address, CopilotRequestHandler)
    print(f"Instamart Copilot Web App running at http://localhost:{port}")
    httpd.serve_forever()

if __name__ == "__main__":
    import time
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)

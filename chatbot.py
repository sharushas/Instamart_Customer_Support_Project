"""chatbot.py
Instamart Customer Intelligence & Support Copilot -- CLI & Interactive Chatbot

Allows users to:
  1. Run guided sequential pipelines (Parts A, B, C, D).
  2. Ask policy questions grounded in official documentation via TF-IDF RAG.
  3. Simulate autonomous refund decisions via ReAct Agent.
  4. Perform support ticket triage and classification.
  5. Predict customer lapse probability and inspect RFM clusters.
  6. Engage in natural-language conversational queries.

Run: python chatbot.py
"""
import sys
import os

from copilot import InstamartCopilot, main, get_input

if __name__ == "__main__":
    main()

"""chatbot.py
Backward-compatibility alias for copilot.py (Instamart Customer Intelligence & Support Copilot).
"""
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from copilot import InstamartCopilot, main, get_input

if __name__ == "__main__":
    main()

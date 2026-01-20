# -*- coding: utf-8 -*-
"""post_data.py

Puzzle Mode submissions -> SharePoint-hosted online Excel.

This module is intentionally **webhook-based** (Power Automate / Logic Apps / custom API).
Reason: Microsoft Graph's Excel TableRow operations are often used with **delegated**
permissions in many tenants; a public Streamlit app usually shouldn't implement interactive
Microsoft sign-in just to write rows.

Backend contract
----------------
Your webhook should accept JSON like:

{
  "email": "name.surname@uzh.ch",
  "cost_eur": 1234.56,
  "co2_total_ton": 78.9,
  "details": "... free text ...",
  "payload": { "any": "json" }
}

And it should enforce the rule:
  - If multiple submissions exist for the same email, keep only the **lowest-cost** row
    in the Excel table and delete the rest.

Secrets / environment variables
-------------------------------
PUZZLE_SUBMISSION_WEBHOOK_URL : required
PUZZLE_SUBMISSION_API_KEY     : optional (sent as x-api-key)
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests


def _get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st  # local import to avoid hard dependency

        if hasattr(st, "secrets") and name in st.secrets:
            val = st.secrets[name]
            return default if val is None else str(val)
    except Exception:
        pass

    return os.getenv(name, default)


def push_puzzle_submission(
    *,
    email: str,
    cost_eur: float,
    co2_total_ton: float,
    details: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Send a Puzzle Mode submission to the configured webhook.

    Returns a parsed JSON dict if the webhook responds with JSON; otherwise returns
    a minimal status dict.
    """

    url = _get_secret("PUZZLE_SUBMISSION_WEBHOOK_URL")
    if not url:
        raise RuntimeError(
            "Missing PUZZLE_SUBMISSION_WEBHOOK_URL. Set it in Streamlit secrets or as an env var."
        )

    api_key = _get_secret("PUZZLE_SUBMISSION_API_KEY")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    body = {
        "email": email,
        "cost_eur": float(cost_eur),
        "co2_total_ton": float(co2_total_ton),
        "details": details,
        "payload": payload,
    }

    # Basic retry for transient errors
    retries = 2
    for attempt in range(retries + 1):
        resp = requests.post(url, headers=headers, json=body, timeout=20)
        if resp.status_code in (200, 201, 202, 204):
            if not resp.text:
                return {"status": "ok"}
            try:
                return resp.json()
            except Exception:
                return {"status": "ok", "raw": resp.text}

        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(1.5 * (attempt + 1))
            continue

        raise RuntimeError(f"Webhook returned {resp.status_code}: {resp.text[:500]}")

SYSTEM_PROMPT = """You are TransitPulse Lite, a public transit service-status assistant.
You MUST base your response ONLY on the provided anomaly summary.
Be calm, practical, and non-alarmist. If data is insufficient, say so.

Return STRICT JSON with keys:
en_short, en_detail, bm_short, bm_detail, actions (array), disclaimer
"""

USER_TEMPLATE = """Context:
- Area: {area}
- Timestamp (local): {ts}
- Route: {route_name}
- Active vehicles: {active}
- ML indicators:
  - Bunched vehicles: {bunched}
  - Gap vehicles: {gap}

Task:
1) Explain what riders may experience (uneven waiting time, bunching, service gaps).
2) Provide 3-6 practical actions riders can take.
3) Provide a short disclaimer about realtime GPS & refresh interval.

Output STRICT JSON only.

Do NOT use markdown. Do NOT wrap JSON in backticks. Output a single JSON object only.
"""

import os
import requests

def _get_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)

def groq_chat(messages, model="llama-3.1-8b-instant", temperature=0.0, max_tokens=450):
    api_key = _get_secret("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY (env var or Streamlit secrets)")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": _get_secret("GROQ_MODEL", model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,

        # ✅ Force valid JSON
        "response_format": {"type": "json_object"},
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# app.py
import os
import json
import requests
import streamlit as st
from typing import List

# ---------- Gemini Config ----------
MODEL_NAME = "gemini-2.0-flash-lite"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def get_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", st.secrets.get("GEMINI_API_KEY", ""))

# ---------- Tiny RAG (Knowledge Base) ----------
class TinyRAG:
    def __init__(self):
        self.docs: List[str] = [
            "High latency may indicate overloaded switches or links.",
            "Packet drops can occur due to congestion or faulty hardware.",
            "Load balancing across multiple paths can prevent bottlenecks.",
            "Prioritize critical traffic flows to ensure SLA compliance."
        ]

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        query_words = query.lower().split()
        scored = []
        for doc in self.docs:
            score = sum(1 for w in query_words if w in doc.lower())
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: -x[0])
        return [doc for _, doc in scored[:k]]

# ---------- Gemini Client ----------
def gemini_generate(api_key: str, prompt: str, max_tokens: int = 300) -> str:
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens}
    }
    try:
        resp = requests.post(API_URL, headers=headers, params=params, json=payload, timeout=30)
        data = resp.json()
    except Exception as e:
        return f"⚠️ Network/Request error: {e}"

    # Extract text safely
    if "candidates" in data and data["candidates"]:
        cand = data["candidates"][0]
        parts = cand.get("content", {}).get("parts", [])
        for p in parts:
            if "text" in p and p["text"]:
                return p["text"].strip()
    return "⚠️ Gemini API returned no usable text."

# ---------- Agentic AI Pipeline ----------
def agentic_network_optimizer(api_key: str, traffic_summary: str, rag: TinyRAG) -> str:
    # Retrieve relevant KB
    context = " ".join(rag.retrieve(traffic_summary))

    # Build prompt
    prompt = f"""
You are an expert network AI assistant. Analyze the following network traffic data and summary:

Traffic Summary:
{traffic_summary}

Knowledge Base:
{context}

Tasks:
1. Predict future traffic patterns
2. Identify potential bottlenecks
3. Suggest dynamic routing adjustments and load balancing
4. Provide actionable recommendations for optimizing network performance

Return a concise, structured report.
"""
    return gemini_generate(api_key, prompt, max_tokens=300)

# ---------- Optional LangGraph ----------
try:
    import langgraph as lg
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Autonomous Network Optimizer", layout="wide")
st.title("📡 MANISH - Autonomous Network Optimizer (Lightweight Demo)")

# Input
traffic_input = st.text_area(
    "Paste recent network traffic summary or logs:",
    height=200,
    placeholder="Example: Link1 avg 80% utilization, Link2 avg 60%, packet loss ~2%"
)

if st.button("Run Optimization"):
    api_key = get_api_key()
    if not api_key:
        st.error("GEMINI_API_KEY not found. Set environment variable or Streamlit secrets.")
    elif not traffic_input.strip():
        st.warning("Please paste a network traffic summary.")
    else:
        rag = TinyRAG()
        if LANGGRAPH_AVAILABLE:
            g = lg.Graph()
            g.add_node("network_input", {"length": len(traffic_input)})
        with st.spinner("Analyzing network traffic..."):
            report = agentic_network_optimizer(api_key, traffic_input.strip(), rag)
        st.subheader("📝 Optimization Report")
        st.markdown(report)
        if LANGGRAPH_AVAILABLE:
            g.add_node("optimizer_output", {"length": len(report)})
            g.add_edge("network_input", "optimizer_output")
            st.info("Pipeline recorded in LangGraph (local).")

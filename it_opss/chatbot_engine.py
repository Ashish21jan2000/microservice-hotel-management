"""IT Ops Chatbot Engine — powered by Azure OpenAI GPT-4o.

Uses RAG: retrieves relevant data from local artifacts (incidents, events,
predictions, model metrics) and feeds it as context to GPT-4o for
intelligent, conversational answers.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SEPT1011 = BASE / "structured" / "September1011"
PREDICTION_OUT = BASE / "outputs" / "anomaly_v1"

_DEMO_RCA = SEPT1011 / "demo_incidents.csv"
_DEMO_EV = SEPT1011 / "demo_events.csv"
_USE_DEMO = _DEMO_RCA.exists() and _DEMO_EV.exists()
RCA_PATH = _DEMO_RCA if _USE_DEMO else SEPT1011 / "rca_incident_report_mcs.csv"
EV_PATH = _DEMO_EV if _USE_DEMO else SEPT1011 / "events_with_incidents.csv"


def _log(msg: str) -> None:
    print(f"[CHATBOT] {msg}", file=sys.stderr, flush=True)


# ── Load .env ───────────────────────────────────────────────────────────────────

def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_manual()
        return
    for p in [BASE / ".env", BASE / "scripts" / ".env"]:
        if p.exists():
            load_dotenv(dotenv_path=str(p), override=False)
            return


def _load_env_manual() -> None:
    p = BASE / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


_load_env()


# ── Data Loading ────────────────────────────────────────────────────────────────

def _read_flex(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


class KnowledgeBase:
    """Loads all data artifacts once and provides retrieval methods for RAG."""

    def __init__(self) -> None:
        self.rca = self._load_rca()
        self.events = self._load_events()
        self.metrics = self._load_json("model_metrics.json")
        self.latest_snapshot = self._load_json("latest_risk_snapshot.json")
        self.scores = self._load_scores()
        self.alerts = self._load_alerts()
        self.backtest = self._load_text("backtest_report.md")

    def _load_rca(self) -> pd.DataFrame:
        df = _read_flex(RCA_PATH)
        df["time_start"] = pd.to_datetime(df.get("time_start"), utc=True, errors="coerce")
        df["time_end"] = pd.to_datetime(df.get("time_end"), utc=True, errors="coerce")
        if "incident_id" in df.columns:
            df["incident_id"] = df["incident_id"].astype(str).str.strip()
        if "confidence_score" in df.columns:
            df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="coerce")
        return df

    def _load_events(self) -> pd.DataFrame:
        df = _read_flex(EV_PATH)
        df["time"] = pd.to_datetime(df.get("time"), utc=True, errors="coerce")
        if "incident_id" in df.columns:
            df["incident_id"] = df["incident_id"].astype(str).str.strip()
        return df

    def _load_json(self, name: str) -> dict:
        p = PREDICTION_OUT / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _load_scores(self) -> pd.DataFrame:
        p = PREDICTION_OUT / "scores_all_windows.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
        return df

    def _load_alerts(self) -> pd.DataFrame:
        p = PREDICTION_OUT / "alerts_triggered.csv"
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
        return df

    def _load_text(self, name: str) -> str:
        p = PREDICTION_OUT / name
        return p.read_text(encoding="utf-8") if p.exists() else ""

    # ── Retrieval helpers for RAG ───────────────────────────────────────────

    def get_overview_stats(self) -> str:
        rca = self.rca
        ev = self.events
        sev_dist = rca["severity"].astype(str).str.upper().value_counts().to_dict()
        conf = pd.to_numeric(rca.get("confidence_score", 0), errors="coerce")
        dur = pd.to_numeric(rca.get("duration_sec", 0), errors="coerce")

        unique_svcs = set()
        for s in rca["services"].astype(str):
            unique_svcs.update([x.strip() for x in s.replace(",", ";").split(";") if x.strip()])
        unique_hosts = set()
        for h in rca["hosts"].astype(str):
            unique_hosts.update([x.strip() for x in h.replace(",", ";").split(";") if x.strip()])

        return (
            f"DATASET OVERVIEW:\n"
            f"- Total incidents: {len(rca)}\n"
            f"- Total events: {len(ev)}\n"
            f"- Date range: {rca['time_start'].min()} to {rca['time_start'].max()}\n"
            f"- Unique services: {len(unique_svcs)} ({', '.join(list(unique_svcs)[:8])}...)\n"
            f"- Unique hosts: {len(unique_hosts)}\n"
            f"- Severity distribution: {sev_dist}\n"
            f"- Avg confidence score: {conf.mean():.2f}\n"
            f"- Avg duration: {dur.mean():.1f}s, Median: {dur.median():.1f}s\n"
        )

    def get_incident_detail(self, iid: str) -> str:
        match = self.rca[self.rca["incident_id"] == iid]
        if match.empty:
            return f"No incident found with ID {iid}."
        row = match.iloc[0]
        cols_text = []
        for col in row.index:
            v = row[col]
            if pd.notna(v):
                cols_text.append(f"  {col}: {v}")
        return f"INCIDENT {iid} FULL RECORD:\n" + "\n".join(cols_text)

    def get_incident_events_summary(self, iid: str) -> str:
        ev = self.events[self.events["incident_id"] == iid]
        if ev.empty:
            return f"No events found for incident {iid}."
        n = len(ev)
        sev_dist = ev["severity"].value_counts().to_dict() if "severity" in ev.columns else {}
        svcs = ev["service.name"].unique().tolist() if "service.name" in ev.columns else []
        hosts = ev["host.name"].unique().tolist() if "host.name" in ev.columns else []
        signals = ev["signal_name"].unique().tolist()[:5] if "signal_name" in ev.columns else []
        sample_msgs = ev["message"].dropna().head(3).tolist() if "message" in ev.columns else []
        return (
            f"EVENTS FOR INCIDENT {iid}:\n"
            f"  Total events: {n}\n"
            f"  Severity breakdown: {sev_dist}\n"
            f"  Services: {svcs}\n"
            f"  Hosts: {hosts}\n"
            f"  Top signals: {signals}\n"
            f"  Sample messages: {sample_msgs}\n"
        )

    def get_prediction_for_incident(self, iid: str) -> str:
        inc = self.rca[self.rca["incident_id"] == iid]
        if inc.empty:
            return f"Incident {iid} not found."
        row = inc.iloc[0]
        t_start = row.get("time_start")
        if pd.isna(t_start):
            return f"Incident {iid} has no valid start time for prediction analysis."

        svc = str(row.get("services", "")).split(";")[0].strip()
        host = str(row.get("hosts", "")).split(";")[0].strip()
        az = str(row.get("azs", "")).split(";")[0].strip()
        threshold = float(self.metrics.get("threshold", 0.5))

        for etype, evalue in [("host", host), ("service", svc), ("az", az), ("global", "GLOBAL")]:
            if not evalue:
                continue
            sc = self._get_entity_scores(etype, evalue)
            if sc.empty:
                continue
            pre = sc[(sc["timestamp"] >= t_start - pd.Timedelta(minutes=35)) &
                     (sc["timestamp"] < t_start)]
            if pre.empty:
                continue
            peak = float(pre["risk_score"].max())
            alerted = pre[pre["risk_score"] >= threshold]
            was_predicted = not alerted.empty
            lead_min = 0.0
            if was_predicted:
                lead_min = (t_start - alerted["timestamp"].min()).total_seconds() / 60.0

            post = sc[(sc["timestamp"] >= t_start) &
                      (sc["timestamp"] <= t_start + pd.Timedelta(hours=2))]
            post_peak = float(post["risk_score"].max()) if not post.empty else 0.0

            return (
                f"PREDICTION ANALYSIS FOR INCIDENT {iid}:\n"
                f"  Entity scope: {etype}={evalue}\n"
                f"  Was predicted: {'YES' if was_predicted else 'NO'}\n"
                f"  Peak pre-incident risk score: {peak:.4f}\n"
                f"  Threshold: {threshold:.4f}\n"
                f"  Lead time: {lead_min:.1f} minutes (before incident start)\n"
                f"  Post-incident peak risk: {post_peak:.4f}\n"
                f"  Pre-incident scores: {[round(x, 3) for x in pre['risk_score'].tolist()[-10:]]}\n"
            )
        return f"No prediction scores available for incident {iid}."

    def get_recurrence_info(self, iid: str) -> str:
        inc = self.rca[self.rca["incident_id"] == iid]
        if inc.empty:
            return f"Incident {iid} not found."
        row = inc.iloc[0]
        t_end = row.get("time_end")
        if pd.isna(t_end):
            return f"Incident {iid} has no end time for recurrence analysis."

        host = str(row.get("hosts", "")).split(";")[0].strip()
        svc = str(row.get("services", "")).split(";")[0].strip()
        rca = self.rca.dropna(subset=["time_start"])

        parts = [f"RECURRENCE ANALYSIS FOR INCIDENT {iid}:"]
        if host:
            future_h = rca[(rca["hosts"].astype(str).str.contains(host, case=False, na=False)) &
                           (rca["time_start"] > t_end)]
            parts.append(f"  Subsequent incidents on host {host}: {len(future_h)}")
            if not future_h.empty:
                nxt = future_h.sort_values("time_start").iloc[0]
                gap = (nxt["time_start"] - t_end).total_seconds() / 60
                parts.append(f"  Next one: {nxt['incident_id']} ({nxt.get('incident_title', '')[:60]}) in {gap:.0f} min")
        if svc:
            future_s = rca[(rca["services"].astype(str).str.contains(svc, case=False, na=False)) &
                           (rca["time_start"] > t_end)]
            parts.append(f"  Subsequent incidents on service {svc}: {len(future_s)}")
        return "\n".join(parts)

    def get_model_metrics(self) -> str:
        m = self.metrics
        if not m:
            return "No model metrics available."
        test = m.get("test_metrics", {})
        val = m.get("val_metrics", {})
        lt = m.get("test_lead_time_min", {})
        return (
            f"MODEL PERFORMANCE METRICS:\n"
            f"  Threshold: {m.get('threshold', 'N/A')}\n"
            f"  Validation — PR-AUC: {val.get('pr_auc', 0):.4f}, ROC-AUC: {val.get('roc_auc', 0):.4f}, "
            f"Precision: {val.get('precision', 0):.4f}, Recall: {val.get('recall', 0):.4f}, F1: {val.get('f1', 0):.4f}\n"
            f"  Test — PR-AUC: {test.get('pr_auc', 0):.4f}, ROC-AUC: {test.get('roc_auc', 0):.4f}, "
            f"Precision: {test.get('precision', 0):.4f}, Recall: {test.get('recall', 0):.4f}, F1: {test.get('f1', 0):.4f}\n"
            f"  Lead time — Median: {lt.get('median', 0):.1f} min, P90: {lt.get('p90', 0):.1f} min\n"
        )

    def get_feature_importance(self) -> str:
        contribs = self.latest_snapshot.get("top_feature_contributors", [])
        if not contribs:
            return "No feature importance data available."
        lines = ["TOP FEATURE CONTRIBUTORS:"]
        for i, c in enumerate(contribs, 1):
            lines.append(f"  {i}. {c['feature']}: {c['score']:.4f}")
        return "\n".join(lines)

    def get_top_services(self, n: int = 10) -> str:
        svc_col = self.rca["services"].astype(str).str.strip()
        all_svcs: list[str] = []
        for s in svc_col:
            all_svcs.extend([x.strip() for x in s.replace(",", ";").split(";") if x.strip()])
        counts = pd.Series(all_svcs).value_counts().head(n)
        lines = ["TOP SERVICES BY INCIDENT COUNT:"]
        for svc, cnt in counts.items():
            lines.append(f"  {svc}: {cnt}")
        return "\n".join(lines)

    def get_top_hosts(self, n: int = 10) -> str:
        host_col = self.rca["hosts"].astype(str).str.strip()
        all_hosts: list[str] = []
        for h in host_col:
            all_hosts.extend([x.strip() for x in h.replace(",", ";").split(";") if x.strip()])
        counts = pd.Series(all_hosts).value_counts().head(n)
        lines = ["TOP HOSTS BY INCIDENT COUNT:"]
        for host, cnt in counts.items():
            lines.append(f"  {host}: {cnt}")
        return "\n".join(lines)

    def get_severity_distribution(self) -> str:
        dist = self.rca["severity"].astype(str).str.upper().value_counts()
        lines = ["SEVERITY DISTRIBUTION:"]
        for sev, cnt in dist.items():
            pct = cnt / len(self.rca) * 100
            lines.append(f"  {sev}: {cnt} ({pct:.1f}%)")
        lines.append(f"  Total: {len(self.rca)}")
        return "\n".join(lines)

    def get_az_distribution(self) -> str:
        az_col = self.rca["azs"].astype(str).str.strip()
        all_azs: list[str] = []
        for a in az_col:
            all_azs.extend([x.strip() for x in a.replace(",", ";").split(";") if x.strip()])
        counts = pd.Series(all_azs).value_counts()
        lines = ["AZ DISTRIBUTION:"]
        total = sum(counts.values)
        for az, cnt in counts.items():
            lines.append(f"  {az}: {cnt} ({cnt / total * 100:.1f}%)")
        return "\n".join(lines)

    def search_incidents(self, **filters) -> str:
        df = self.rca.copy()
        if "severity" in filters and filters["severity"]:
            df = df[df["severity"].astype(str).str.upper() == filters["severity"].upper()]
        if "service" in filters and filters["service"]:
            df = df[df["services"].astype(str).str.contains(filters["service"], case=False, na=False)]
        if "host" in filters and filters["host"]:
            df = df[df["hosts"].astype(str).str.contains(filters["host"], case=False, na=False)]
        if df.empty:
            return "No incidents matching those filters."
        sample = df.head(20)
        lines = [f"FOUND {len(df)} INCIDENTS (showing first {len(sample)}):"]
        for _, r in sample.iterrows():
            lines.append(f"  {r['incident_id']}: {str(r.get('incident_title', ''))[:60]} | "
                         f"Sev: {r.get('severity', '')} | Conf: {r.get('confidence_score', '')}")
        return "\n".join(lines)

    def get_backtest_report(self) -> str:
        if self.backtest:
            return f"BACKTEST REPORT:\n{self.backtest}"
        return "No backtest report available."

    def _get_entity_scores(self, entity_type: str, entity_value: str) -> pd.DataFrame:
        if self.scores.empty:
            return pd.DataFrame()
        if {"entity_type", "entity_value"}.issubset(self.scores.columns):
            return self.scores[(self.scores["entity_type"] == entity_type) &
                               (self.scores["entity_value"] == entity_value)]
        return self.scores


# ── Entity Extraction ───────────────────────────────────────────────────────────

_INC_RE = re.compile(r"(INC-?\d+)", re.I)


def _extract_incident_ids(text: str) -> list[str]:
    found = _INC_RE.findall(text)
    normalized = []
    for m in found:
        m_up = m.upper().replace("INC", "INC-").replace("INC--", "INC-")
        if not m_up.startswith("INC-"):
            m_up = "INC-" + m_up.replace("INC", "")
        normalized.append(m_up)
    return normalized


# ── Context Builder (RAG) ──────────────────────────────────────────────────────

def _build_context(query: str, history: list[dict], kb: KnowledgeBase) -> str:
    """Retrieve relevant data based on the query and inject it as context."""
    q_lower = query.lower()
    context_parts: list[str] = []

    inc_ids = _extract_incident_ids(query)
    if not inc_ids:
        for msg in reversed(history[-6:]):
            if msg.get("role") == "user":
                inc_ids = _extract_incident_ids(msg.get("content", ""))
                if inc_ids:
                    break

    if inc_ids:
        for iid in inc_ids[:2]:
            context_parts.append(kb.get_incident_detail(iid))
            context_parts.append(kb.get_incident_events_summary(iid))
            if any(w in q_lower for w in ["predict", "risk", "alert", "warning", "detect", "advance"]):
                context_parts.append(kb.get_prediction_for_incident(iid))
            if any(w in q_lower for w in ["recur", "repeat", "again", "future", "breakdown", "next time"]):
                context_parts.append(kb.get_recurrence_info(iid))

    if any(w in q_lower for w in ["overview", "summary", "total", "how many", "stats", "dashboard",
                                   "all incident", "general", "big picture"]):
        context_parts.append(kb.get_overview_stats())

    if any(w in q_lower for w in ["severity", "high", "medium", "low", "critical"]):
        context_parts.append(kb.get_severity_distribution())
        for sev in ["HIGH", "MEDIUM", "LOW", "CRITICAL"]:
            if sev.lower() in q_lower:
                context_parts.append(kb.search_incidents(severity=sev))
                break

    if any(w in q_lower for w in ["list", "show me", "give me", "all of"]):
        for msg in reversed(history[-4:]):
            if msg.get("role") != "user":
                continue
            prev = msg.get("content", "").lower()
            for sev in ["HIGH", "MEDIUM", "LOW", "CRITICAL"]:
                if sev.lower() in prev:
                    context_parts.append(kb.search_incidents(severity=sev))
                    break
            break

    if any(w in q_lower for w in ["service", "top service", "which service"]):
        context_parts.append(kb.get_top_services())

    if any(w in q_lower for w in ["host", "server", "machine", "ip", "200."]):
        context_parts.append(kb.get_top_hosts())

    if any(w in q_lower for w in ["az", "availability zone", "zone"]):
        context_parts.append(kb.get_az_distribution())

    if any(w in q_lower for w in ["accuracy", "precision", "recall", "f1", "auc", "metric",
                                   "performance", "model", "how good", "how well"]):
        context_parts.append(kb.get_model_metrics())

    if any(w in q_lower for w in ["feature", "contributor", "important", "what drove", "signal"]):
        context_parts.append(kb.get_feature_importance())

    if any(w in q_lower for w in ["method", "approach", "pipeline", "how does", "how do",
                                   "architecture", "technique", "algorithm", "explain"]):
        context_parts.append(kb.get_model_metrics())
        context_parts.append(kb.get_feature_importance())

    if any(w in q_lower for w in ["predict", "risk", "alert", "detect"]) and not inc_ids:
        context_parts.append(kb.get_model_metrics())
        context_parts.append(kb.get_backtest_report())

    if any(w in q_lower for w in ["compare", "vs", "versus", "difference"]) and len(inc_ids) >= 2:
        for iid in inc_ids[:2]:
            context_parts.append(kb.get_prediction_for_incident(iid))
            context_parts.append(kb.get_recurrence_info(iid))

    if not context_parts:
        context_parts.append(kb.get_overview_stats())
        context_parts.append(kb.get_model_metrics())

    return "\n\n".join(context_parts)


# ── System Prompt ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert IT Operations AI Assistant for a cloud infrastructure monitoring platform.
You answer questions about incidents, root cause analysis (RCA), anomaly prediction, and operational insights.

YOUR DATA SOURCES:
- 965 real incidents from Sep 10-11, 2025 in a Cloud Foundry environment
- 159,000+ events across those incidents
- An anomaly prediction model (LightGBM + isotonic calibration) that scores risk every 5 minutes
- Model uses 100+ features across 3 time windows (5m, 15m, 30m): event counts, unique entities, severity ratios, entropy, burstiness, growth rates
- Alerting policy: 2 consecutive windows above threshold triggers an alert

HOW THE PREDICTION PIPELINE WORKS:
1. Feature Engineering: Every 5 minutes, rolling features are computed from the event stream (event volume, unique services/hosts, severity ratios, signal repetition, host concentration, AZ entropy)
2. Model: LightGBM classifier with isotonic calibration producing risk probabilities (0 to 1)
3. Labeling: Each window is labeled 1 if an incident starts within next 30 minutes
4. Alerting: Alert fires when 2 consecutive windows exceed the threshold
5. Scoring: Per entity (service, host, AZ) and global scoring

RULES:
- Answer based ONLY on the retrieved context data provided below. Do not invent data.
- Use markdown formatting: headers, tables, bold for emphasis.
- When discussing incidents, reference their IDs (e.g., INC-00042).
- For predictions, mention risk scores, threshold, lead time, and entity scope.
- If the context doesn't contain enough information, say so honestly.
- Be conversational but precise — this is for a production operations team.
- Remember the chat history for follow-up questions.
- If the user says "hi" or greets you, respond warmly and list your capabilities.
"""


# ── Azure OpenAI Call ───────────────────────────────────────────────────────────

_llm_client = None
_llm_deployment = "gpt-4o-4"


def _get_llm_client():
    global _llm_client, _llm_deployment
    if _llm_client is not None:
        return _llm_client

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    _llm_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-4").strip()
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview").strip()

    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env"
        )

    import httpx
    from openai import AzureOpenAI

    http_client = httpx.Client(
        timeout=httpx.Timeout(60.0, connect=15.0),
        verify=True,
    )
    _llm_client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
        http_client=http_client,
    )
    return _llm_client


def _call_llm(messages: list[dict]) -> str:
    client = _get_llm_client()
    resp = client.chat.completions.create(
        model=_llm_deployment,
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )
    return resp.choices[0].message.content if resp.choices else "Sorry, I couldn't generate a response."


# ── Main chat function ──────────────────────────────────────────────────────────

_kb: KnowledgeBase | None = None


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


def chat(query: str, history: list[dict]) -> str:
    kb = _get_kb()

    _log(f"QUERY: {query!r}")

    context = _build_context(query, history, kb)
    _log(f"CONTEXT LENGTH: {len(context)} chars")

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    recent_history = history[-12:]
    for msg in recent_history:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = f"""RETRIEVED CONTEXT (from local data):
---
{context}
---

USER QUESTION: {query}"""

    messages.append({"role": "user", "content": user_message})

    _log(f"SENDING {len(messages)} messages to GPT-4o...")
    response = _call_llm(messages)
    _log(f"RESPONSE: {response[:120].replace(chr(10), ' ')}...")
    _log("-" * 60)

    return response

#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
SEPT1011 = BASE / "structured" / "September1011"
_DEMO_RCA = SEPT1011 / "demo_incidents.csv"
_DEMO_EV  = SEPT1011 / "demo_events.csv"
_USE_DEMO = _DEMO_RCA.exists() and _DEMO_EV.exists()

RCA_MCS = _DEMO_RCA if _USE_DEMO else SEPT1011 / "rca_incident_report_mcs.csv"
EV_MCS  = _DEMO_EV  if _USE_DEMO else SEPT1011 / "events_with_incidents.csv"
# PAGE_SIZE_OPTIONS = [50, 100, 200, 500]
DETAIL_MIN_CONFIDENCE = 0.80

st.set_page_config(page_title="Incident RCA Dashboard", page_icon="??", layout="wide", initial_sidebar_state="expanded")


def _read_csv_flexible(path: Path, **kwargs: Any) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        fallback_kwargs = {k: v for k, v in kwargs.items() if k != "low_memory"}
        return pd.read_csv(path, engine="python", on_bad_lines="skip", **fallback_kwargs)


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


def _rca_field_to_markdown(text: str) -> str:
    """Turn LLM/CSV RCA text (often one long line with 1. 2. steps) into readable Markdown lists."""
    t = _safe_text(text)
    if not t:
        return ""
    t = unescape(t)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"</p>\s*<p[^>]*>", "\n", t, flags=re.I)
    t = re.sub(r"<li[^>]*>", "\n", t, flags=re.I)
    t = re.sub(r"</li>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse duplicate step markers from prompts/UI: "1. 1. Do X" -> "1. Do X"
    t = re.sub(r"\b(\d+)\.\s+\1\.\s*", r"\1. ", t)
    segments: list[str] = []
    for block in re.split(r"\n+", t):
        block = block.strip()
        if not block:
            continue
        for piece in re.split(r"\s+(?=\d+\.\s)", block):
            p = piece.strip()
            if p:
                segments.append(p)
    if not segments:
        return t.strip()
    bullets: list[str] = []
    for seg in segments:
        body = re.sub(r"^(?:\d+\.\s*)+", "", seg).strip()
        body = re.sub(r"^[-*]\s+", "", body).strip()
        if body:
            bullets.append(f"- {body}")
    return "\n\n".join(bullets) if bullets else t.strip()


def _normalize_rca_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "time_start" in out.columns and "start_time" not in out.columns:
        out["start_time"] = pd.to_datetime(out["time_start"], utc=True, errors="coerce")
    if "time_end" in out.columns and "end_time" not in out.columns:
        out["end_time"] = pd.to_datetime(out["time_end"], utc=True, errors="coerce")
    if "confidence_score" in out.columns and "model_confidence" not in out.columns:
        out["model_confidence"] = pd.to_numeric(out["confidence_score"], errors="coerce")
    if "severity_counts" in out.columns and "severity_distribution" not in out.columns:
        out["severity_distribution"] = out["severity_counts"]
    if "incident_id" in out.columns:
        out["incident_id"] = out["incident_id"].astype(str).str.strip()
    return out


@st.cache_data(ttl=300)
def load_rca() -> pd.DataFrame:
    return _normalize_rca_columns(_read_csv_flexible(RCA_MCS, low_memory=False))


@st.cache_data(ttl=300)
def load_events() -> pd.DataFrame:
    df = _read_csv_flexible(EV_MCS, low_memory=False)
    if "incident_id" in df.columns:
        df["incident_id"] = df["incident_id"].astype(str).str.strip()
    return df


def _rca_filtered_by_confidence(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    if "model_confidence" not in df.columns:
        return df
    c = pd.to_numeric(df["model_confidence"], errors="coerce").fillna(0.0)
    return df[(c >= lo) & (c <= hi)]


def events_display(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in ("root_cause", "impact", "fix", "incident_title") if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def summarize_incident_events(ev: pd.DataFrame) -> str:
    if ev.empty:
        return "No events are available for this incident."

    parts: list[str] = [f"This incident includes {len(ev):,} events."]

    if "time" in ev.columns:
        ts = pd.to_datetime(ev["time"], errors="coerce", utc=True).dropna()
        if not ts.empty:
            parts.append(
                f"Time window spans from {ts.min().strftime('%Y-%m-%d %H:%M:%S UTC')} "
                f"to {ts.max().strftime('%Y-%m-%d %H:%M:%S UTC')}."
            )

    if "service.name" in ev.columns:
        svc_counts = ev["service.name"].astype(str).str.strip().replace("", pd.NA).dropna().value_counts()
        if not svc_counts.empty:
            parts.append(f"It spans {len(svc_counts):,} distinct services.")
            top_services = ", ".join(f"{name} ({count})" for name, count in svc_counts.head(3).items())
            parts.append(f"Top services: {top_services}.")

    if "host.name" in ev.columns:
        host_counts = ev["host.name"].astype(str).str.strip().replace("", pd.NA).dropna().value_counts()
        if not host_counts.empty:
            parts.append(f"Affected hosts: {len(host_counts):,}.")
            top_hosts = ", ".join(f"{name} ({count})" for name, count in host_counts.head(3).items())
            parts.append(f"Most active hosts: {top_hosts}.")

    if "attr.az" in ev.columns:
        az_counts = ev["attr.az"].astype(str).str.strip().replace("", pd.NA).dropna().value_counts()
        if not az_counts.empty:
            az_text = ", ".join(f"{name} ({count})" for name, count in az_counts.items())
            parts.append(f"AZ distribution: {az_text}.")

    if "attr.deployment" in ev.columns:
        dep_counts = ev["attr.deployment"].astype(str).str.strip().replace("", pd.NA).dropna().value_counts()
        if not dep_counts.empty:
            top_dep = ", ".join(f"{name} ({count})" for name, count in dep_counts.head(2).items())
            parts.append(f"Top deployment groups: {top_dep}.")

    if "severity" in ev.columns:
        sev_counts = ev["severity"].astype(str).str.upper().str.strip().replace("", pd.NA).dropna().value_counts()
        if not sev_counts.empty:
            sev_text = ", ".join(f"{name}: {count}" for name, count in sev_counts.items())
            parts.append(f"Severity mix: {sev_text}.")

    if "signal_name" in ev.columns:
        sig_counts = ev["signal_name"].astype(str).str.strip().replace("", pd.NA).dropna().value_counts()
        if not sig_counts.empty:
            top_signals = ", ".join(f"{name} ({count})" for name, count in sig_counts.head(3).items())
            parts.append(f"Frequent signals: {top_signals}.")

    if "message" in ev.columns:
        msg_counts = ev["message"].astype(str).str.strip().replace("", pd.NA).dropna().value_counts()
        if not msg_counts.empty:
            top_freq = int(msg_counts.iloc[0])
            repetition = (top_freq / len(ev)) * 100.0
            parts.append(f"Most frequent message appears {top_freq:,} times ({repetition:.1f}% of incident events).")
            top_msg = str(msg_counts.index[0])[:180]
            parts.append(f"Most repeated signal/message: {top_msg}.")

    return " ".join(parts)


PREDICTION_OUT = BASE / "outputs" / "anomaly_v1"


@st.cache_data(ttl=300)
def load_prediction_artifacts() -> dict | None:
    if not PREDICTION_OUT.exists():
        return None
    try:
        scores = pd.read_csv(PREDICTION_OUT / "scores_all_windows.csv")
        scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True, errors="coerce")
        metrics = json.loads((PREDICTION_OUT / "model_metrics.json").read_text(encoding="utf-8"))
        return {"scores": scores, "metrics": metrics}
    except Exception:
        return None


def _feature_to_plain(feat: str) -> str:
    mapping = {
        "event_count": "event volume",
        "unique_services": "number of distinct services",
        "unique_hosts": "number of distinct hosts",
        "unique_signals": "signal diversity",
        "unique_messages": "message diversity",
        "sev_ERROR": "ERROR-severity event count",
        "sev_WARN": "WARN-severity event count",
        "sev_INFO": "INFO-severity event count",
        "sev_DEBUG": "DEBUG-severity event count",
        "repeated_message_ratio": "message repetition rate",
        "repeated_signal_ratio": "signal repetition rate",
        "host_concentration": "host concentration (single-host dominance)",
        "az_entropy": "availability zone spread",
        "deployment_entropy": "deployment group diversity",
        "event_rate_per_min": "event arrival rate (per minute)",
        "event_growth": "event volume growth rate",
    }
    base = feat
    window = ""
    for suffix in ("_sum_30m", "_mean_30m", "_sum_15m", "_mean_15m", "_sum_5m", "_mean_5m", "_ratio"):
        if feat.endswith(suffix):
            base = feat[: -len(suffix)]
            window = suffix.replace("_sum_", " (").replace("_mean_", " (avg ").replace("_ratio", " ratio")
            if window.startswith(" (") and not window.endswith(")"):
                window += " window)"
            elif window.startswith(" (avg"):
                window += ")"
            break
    plain = mapping.get(base, base.replace("_", " "))
    return f"{plain}{window}"


def _select_entity_scores(scores_df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Pick the most specific scored entity slice for this incident."""
    if not {"entity_type", "entity_value"}.issubset(scores_df.columns):
        return scores_df

    inc_service = _safe_text(row.get("services", "")).split(";")[0].strip()
    inc_host = _safe_text(row.get("hosts", "")).split(";")[0].strip()
    inc_az = _safe_text(row.get("azs", "")).split(";")[0].strip()

    for etype, evalue in [
        ("host", inc_host),
        ("service", inc_service),
        ("az", inc_az),
    ]:
        if evalue:
            subset = scores_df[(scores_df["entity_type"] == etype) &
                               (scores_df["entity_value"] == evalue)]
            if not subset.empty:
                return subset

    g = scores_df[(scores_df["entity_type"] == "global") & (scores_df["entity_value"] == "GLOBAL")]
    return g if not g.empty else scores_df


def build_prediction_narrative(row: pd.Series, ev: pd.DataFrame,
                               artifacts: dict) -> str:
    scores_df = artifacts["scores"]
    metrics = artifacts["metrics"]
    threshold = float(metrics.get("threshold", 0.5))

    t_start = pd.to_datetime(row.get("time_start"), utc=True, errors="coerce")
    t_end = pd.to_datetime(row.get("time_end"), utc=True, errors="coerce")
    if pd.isna(t_start):
        return "Prediction data unavailable — incident timestamp is missing."

    entity_scores = _select_entity_scores(scores_df, row)
    entity_label = ""
    if {"entity_type", "entity_value"}.issubset(entity_scores.columns) and not entity_scores.empty:
        et = entity_scores.iloc[0]["entity_type"]
        ev_val = entity_scores.iloc[0]["entity_value"]
        if str(et) != "global":
            entity_label = f" (scoped to **{et}: {ev_val}**)"

    pre_window = entity_scores[
        (entity_scores["timestamp"] >= t_start - pd.Timedelta(minutes=35)) &
        (entity_scores["timestamp"] < t_start)
    ].sort_values("timestamp")

    during_window = entity_scores[
        (entity_scores["timestamp"] >= t_start) &
        (entity_scores["timestamp"] <= (t_end if pd.notna(t_end) else t_start + pd.Timedelta(minutes=10)))
    ].sort_values("timestamp")

    parts: list[str] = []

    inc_service = _safe_text(row.get("services", ""))
    inc_host = _safe_text(row.get("hosts", ""))
    inc_az = _safe_text(row.get("azs", ""))
    context_parts = []
    if inc_service:
        context_parts.append(f"service **{inc_service.split(';')[0].strip()}**")
    if inc_host:
        context_parts.append(f"host **{inc_host.split(';')[0].strip()}**")
    if inc_az:
        context_parts.append(f"AZ **{inc_az.split(';')[0].strip()}**")
    context_str = ", ".join(context_parts) if context_parts else "this entity"

    if pre_window.empty and during_window.empty:
        return (f"No prediction data is available for {context_str} in the time window "
                f"surrounding this incident. This may occur if the incident falls outside "
                f"the scored timeline.")

    peak_pre = float(pre_window["risk_score"].max()) if not pre_window.empty else 0.0
    peak_during = float(during_window["risk_score"].max()) if not during_window.empty else 0.0

    alerted_pre = pre_window[pre_window["risk_score"] >= threshold] if not pre_window.empty else pd.DataFrame()
    was_predicted = not alerted_pre.empty

    if was_predicted:
        first_alert_ts = alerted_pre["timestamp"].min()
        lead_min = (t_start - first_alert_ts).total_seconds() / 60.0
        parts.append(
            f"**Predicted: Yes** — The anomaly detection model flagged elevated risk for "
            f"{context_str}{entity_label} **{lead_min:.1f} minutes before** this incident started. "
            f"The risk score first breached the alert threshold ({threshold:.3f}) at "
            f"{first_alert_ts.strftime('%H:%M:%S UTC')}."
        )
    else:
        if peak_pre > threshold * 0.6:
            parts.append(
                f"**Predicted: Partial** — The model detected rising risk for {context_str}{entity_label} "
                f"(peak score: {peak_pre:.3f}) in the 30 minutes before this incident, but it did not "
                f"reach the alert threshold ({threshold:.3f}). With threshold tuning or additional data, "
                f"this incident could become fully predictable."
            )
        else:
            parts.append(
                f"**Predicted: No** — The model did not detect significant pre-incident anomaly signals "
                f"for {context_str}{entity_label}. The pre-incident risk score peaked at {peak_pre:.3f}, "
                f"below the alert threshold ({threshold:.3f}). This incident may have had a sudden onset "
                f"without gradual escalation."
            )

    if not pre_window.empty and len(pre_window) >= 2:
        scores_list = pre_window["risk_score"].tolist()
        trend = "increasing" if scores_list[-1] > scores_list[0] + 0.05 else (
            "decreasing" if scores_list[-1] < scores_list[0] - 0.05 else "stable"
        )
        parts.append(
            f"In the 30 minutes before the incident, the risk score for {context_str} showed a "
            f"**{trend}** trajectory, moving from {scores_list[0]:.3f} to {scores_list[-1]:.3f} "
            f"(peak: {peak_pre:.3f})."
        )

    sev_text = _safe_text(row.get("severity", ""))
    conf_text = _safe_text(row.get("confidence_score", ""))

    if sev_text or conf_text:
        parts.append(
            f"The RCA analysis rated this incident as **{sev_text or 'N/A'}** severity "
            f"with a confidence score of **{conf_text or 'N/A'}**."
        )

    if was_predicted:
        parts.append(
            "**Operational recommendation:** With this level of advance warning, the operations team "
            "could have initiated proactive mitigation — such as scaling affected services, rerouting "
            "traffic, or alerting on-call engineers — before the incident fully materialized."
        )
    elif peak_pre > threshold * 0.4:
        parts.append(
            "**Improvement path:** This incident showed partial pre-indicators. Enriching the feature set "
            "with application-level metrics (latency, error rates, queue depths) and training on more "
            "historical data would likely improve early detection for incidents of this type."
        )
    else:
        parts.append(
            "**Improvement path:** This incident had a sudden onset pattern. Future iterations of the model "
            "could incorporate change-event detection (deployments, config changes, scaling events) to "
            "catch this class of incidents earlier."
        )

    return "\n\n".join(parts)


def build_recurrence_narrative(row: pd.Series, rca_df: pd.DataFrame,
                               artifacts: dict) -> str:
    scores_df = artifacts["scores"]
    metrics = artifacts["metrics"]
    threshold = float(metrics.get("threshold", 0.5))

    t_start = pd.to_datetime(row.get("time_start"), utc=True, errors="coerce")
    t_end = pd.to_datetime(row.get("time_end"), utc=True, errors="coerce")
    if pd.isna(t_start) or pd.isna(t_end):
        return "Recurrence analysis unavailable — incident timestamps are missing."

    entity_scores = _select_entity_scores(scores_df, row)

    post_window = entity_scores[
        (entity_scores["timestamp"] > t_end) &
        (entity_scores["timestamp"] <= t_end + pd.Timedelta(hours=2))
    ].sort_values("timestamp")

    inc_service = _safe_text(row.get("services", "")).split(";")[0].strip()
    inc_host = _safe_text(row.get("hosts", "")).split(";")[0].strip()
    inc_az = _safe_text(row.get("azs", "")).split(";")[0].strip()

    parts: list[str] = []

    # --- 1. Post-incident risk trajectory ---
    if not post_window.empty:
        post_scores = post_window["risk_score"].tolist()
        post_peak = max(post_scores)
        post_latest = post_scores[-1]
        post_above = post_window[post_window["risk_score"] >= threshold]
        time_span_min = (post_window["timestamp"].max() - post_window["timestamp"].min()).total_seconds() / 60

        if post_peak >= threshold:
            pct_above = len(post_above) / len(post_window) * 100
            parts.append(
                f"**Post-incident risk: ELEVATED** — In the {time_span_min:.0f} minutes after this incident "
                f"resolved, the risk score remained above the alert threshold for "
                f"**{pct_above:.0f}% of scored windows** (peak: {post_peak:.3f}). "
                f"This indicates the underlying conditions that caused this incident have **not fully stabilized** "
                f"and a recurrence is probable."
            )
        elif post_peak >= threshold * 0.6:
            parts.append(
                f"**Post-incident risk: MODERATE** — After resolution, the risk score declined but remained "
                f"moderately elevated (peak: {post_peak:.3f}, latest: {post_latest:.3f}). "
                f"The system is recovering but has not returned to a healthy baseline. "
                f"A milder recurrence or related follow-on incident is possible within the next 30-60 minutes."
            )
        else:
            parts.append(
                f"**Post-incident risk: LOW** — The risk score dropped to {post_latest:.3f} after resolution, "
                f"well below the alert threshold. The system appears to have **stabilized** and "
                f"immediate recurrence is unlikely."
            )

        if len(post_scores) >= 3:
            tail = post_scores[-3:]
            if tail[-1] > tail[0] + 0.05:
                parts.append(
                    "The risk trajectory in the latest scored windows is **rising again**, "
                    "which may indicate a fresh escalation cycle beginning."
                )
            elif tail[-1] < tail[0] - 0.05:
                parts.append(
                    "The risk trajectory is **trending downward**, suggesting continued recovery."
                )
    else:
        parts.append(
            "**Post-incident risk: UNKNOWN** — No scored windows are available after this incident's "
            "end time. This may be because the incident is near the end of the scored timeline."
        )

    # --- 2. Historical recurrence pattern ---
    rca_ts = rca_df.copy()
    rca_ts["_ts"] = pd.to_datetime(rca_ts["time_start"], utc=True, errors="coerce")
    rca_ts = rca_ts.dropna(subset=["_ts"])

    same_service = rca_ts[rca_ts["services"].astype(str).str.contains(inc_service, case=False, na=False)] if inc_service else pd.DataFrame()
    same_host = rca_ts[rca_ts["hosts"].astype(str).str.contains(inc_host, case=False, na=False)] if inc_host else pd.DataFrame()

    future_same_svc = same_service[same_service["_ts"] > t_end] if not same_service.empty else pd.DataFrame()
    future_same_host = same_host[same_host["_ts"] > t_end] if not same_host.empty else pd.DataFrame()

    recurrence_signals = []
    if not future_same_host.empty:
        next_inc = future_same_host.sort_values("_ts").iloc[0]
        gap_min = (next_inc["_ts"] - t_end).total_seconds() / 60
        recurrence_signals.append(
            f"Host **{inc_host}** experienced **{len(future_same_host)} subsequent incident(s)** "
            f"in the data. The next one occurred **{gap_min:.0f} minutes later** "
            f"({next_inc.get('incident_title', 'N/A')[:80]})."
        )
    if not future_same_svc.empty and inc_service:
        next_inc = future_same_svc.sort_values("_ts").iloc[0]
        gap_min = (next_inc["_ts"] - t_end).total_seconds() / 60
        recurrence_signals.append(
            f"Service **{inc_service}** had **{len(future_same_svc)} subsequent incident(s)** in the data. "
            f"Next occurrence was **{gap_min:.0f} minutes later**."
        )

    if recurrence_signals:
        parts.append("**Historical recurrence pattern:**\n- " + "\n- ".join(recurrence_signals))
    else:
        parts.append(
            f"**Historical recurrence pattern:** No further incidents were recorded for this "
            f"host/service combination in the available data window, suggesting this was either "
            f"a one-time event or was fully remediated."
        )

    # --- 3. Recurrence risk verdict ---
    post_elevated = (not post_window.empty and post_window["risk_score"].max() >= threshold * 0.6)
    has_historical_recurrence = not future_same_svc.empty or not future_same_host.empty

    if post_elevated and has_historical_recurrence:
        risk_level = "HIGH"
        risk_color = "red"
        verdict = (
            "Both the post-incident risk trajectory and historical pattern indicate a **high likelihood "
            "of recurrence**. Immediate action is recommended: verify that the root-cause fix has been "
            "fully applied, monitor the affected service/host closely, and consider pre-emptive scaling "
            "or traffic rerouting."
        )
    elif post_elevated or has_historical_recurrence:
        risk_level = "MEDIUM"
        risk_color = "orange"
        if post_elevated:
            verdict = (
                "The post-incident risk score remains elevated, indicating incomplete recovery. "
                "While historical data does not show immediate repetition for this entity, the "
                "current telemetry warrants **continued monitoring** and readiness for rapid response."
            )
        else:
            verdict = (
                "While the post-incident risk score has stabilized, historical data shows this "
                "host/service has experienced repeat incidents. **Proactive investigation** of the "
                "underlying root cause is recommended to prevent future occurrences."
            )
    else:
        risk_level = "LOW"
        risk_color = "green"
        verdict = (
            "Post-incident risk has returned to baseline and no historical recurrence pattern was "
            "detected. The incident appears to have been **fully resolved**. Standard monitoring "
            "is sufficient."
        )

    parts.append(f"**Recurrence Risk: {risk_level}** — {verdict}")

    return "\n\n".join(parts)


def render_home(rca_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
    conf = pd.to_numeric(rca_df.get("model_confidence", 0.0), errors="coerce").fillna(0.0)
    lo, hi = st.sidebar.slider("Model confidence range", 0.0, 1.0, (float(conf.min()), float(conf.max())), 0.01)
    view = _rca_filtered_by_confidence(rca_df, lo, hi)
    st.sidebar.caption(f"Showing **{len(view):,}** out of **{len(rca_df):,}** incidents")
    st.sidebar.metric("Total incidents", len(rca_df))
    st.sidebar.metric("Total event rows", f"{len(events_df):,}")
    st.sidebar.markdown("---")
    st.title("Home - All incidents")
    st.dataframe(view, width="stretch", height=500)


def render_detail(rca_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
    detail = _rca_filtered_by_confidence(rca_df, DETAIL_MIN_CONFIDENCE, 1.0)
    ids = [str(x).strip() for x in detail["incident_id"].tolist()]
    if not ids:
        st.warning("No incidents available")
        return
    sid = st.selectbox("Choose incident", ids)
    row = detail[detail["incident_id"].astype(str).str.strip() == str(sid).strip()].iloc[0]
    st.title("Incident RCA")
    st.metric("Incident ID", str(row["incident_id"]))
    st.subheader("Selected Incident Row (All Columns)")
    st.dataframe(row.to_frame().T, width="stretch", hide_index=True)
    ev = events_df[events_df["incident_id"].astype(str).str.strip() == str(sid).strip()].copy()
    st.subheader("Event Summary")
    st.write(summarize_incident_events(ev))
    for col in ("root_cause", "impact", "fix"):
        st.subheader(col.replace("_", " ").title())
        raw = _safe_text(row.get(col, ""))
        if not raw:
            st.write("-")
        else:
            st.markdown(_rca_field_to_markdown(raw))
    st.subheader("Severity Distribution")
    st.write(_safe_text(row.get("severity_distribution", "")) or _safe_text(row.get("severity_counts", "")) or "-")

    # TEMP: Hide these two sections from Incident Detail (keep code for later).
    # - "Anomaly Prediction Insight"
    # - "Recurrence / Future Breakdown Risk"
    if False:
        st.markdown("---")
        st.subheader("Anomaly Prediction Insight")
        artifacts = load_prediction_artifacts()
        if artifacts is not None:
            narrative = build_prediction_narrative(row, ev, artifacts)
            st.markdown(narrative)

            t_start = pd.to_datetime(row.get("time_start"), utc=True, errors="coerce")
            t_end = pd.to_datetime(row.get("time_end"), utc=True, errors="coerce")
            if pd.notna(t_start):
                scores_df = artifacts["scores"]
                threshold = float(artifacts["metrics"].get("threshold", 0.5))
                entity_sc = _select_entity_scores(scores_df, row)
                window = entity_sc[
                    (entity_sc["timestamp"] >= t_start - pd.Timedelta(minutes=35)) &
                    (entity_sc["timestamp"] <= (t_end if pd.notna(t_end) else t_start + pd.Timedelta(minutes=10)))
                ].sort_values("timestamp")

                if not window.empty:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates

                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.fill_between(window["timestamp"], window["risk_score"], alpha=0.3, color="#4fc3f7")
                    ax.plot(window["timestamp"], window["risk_score"], "o-", color="#4fc3f7",
                            markersize=4, linewidth=1.5)
                    ax.axhline(threshold, color="#ef5350", ls="--", lw=1.5, label=f"Threshold ({threshold:.3f})")
                    if pd.notna(t_start) and pd.notna(t_end):
                        ax.axvspan(t_start, t_end, alpha=0.2, color="#ff9800", label="Incident Active")
                    ax.set_ylabel("Risk Score")
                    ax.set_ylim(-0.05, 1.05)
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    ax.legend(loc="upper left", fontsize=8)
                    ax.set_title("Risk Score Around This Incident", fontsize=11, fontweight="bold")
                    ax.grid(alpha=0.15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            st.markdown("---")
            st.subheader("Recurrence / Future Breakdown Risk")
            recurrence_text = build_recurrence_narrative(row, rca_df, artifacts)
            st.markdown(recurrence_text)

            if pd.notna(t_start) and pd.notna(t_end):
                entity_sc_post = _select_entity_scores(scores_df, row)
                extended = entity_sc_post[
                    (entity_sc_post["timestamp"] >= t_start - pd.Timedelta(minutes=15)) &
                    (entity_sc_post["timestamp"] <= t_end + pd.Timedelta(hours=2))
                ].sort_values("timestamp")

                if not extended.empty:
                    import matplotlib.pyplot as plt
                    import matplotlib.dates as mdates

                    fig2, ax2 = plt.subplots(figsize=(10, 3))
                    pre_ext = extended[extended["timestamp"] <= t_end]
                    post_ext = extended[extended["timestamp"] > t_end]

                    if not pre_ext.empty:
                        ax2.plot(pre_ext["timestamp"], pre_ext["risk_score"], "o-",
                                 color="#4fc3f7", markersize=3, linewidth=1.2, label="Pre/During Incident")
                    if not post_ext.empty:
                        ax2.fill_between(post_ext["timestamp"], post_ext["risk_score"],
                                         alpha=0.25, color="#ab47bc")
                        ax2.plot(post_ext["timestamp"], post_ext["risk_score"], "o-",
                                 color="#ab47bc", markersize=3, linewidth=1.5, label="Post-Incident")

                    ax2.axhline(threshold, color="#ef5350", ls="--", lw=1.5)
                    ax2.axvspan(t_start, t_end, alpha=0.15, color="#ff9800")
                    if pd.notna(t_end):
                        ax2.axvline(t_end, color="#66bb6a", ls=":", lw=1.5, label="Incident Resolved")
                    ax2.set_ylabel("Risk Score")
                    ax2.set_ylim(-0.05, 1.05)
                    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    ax2.legend(loc="upper right", fontsize=8)
                    ax2.set_title("Post-Incident Risk Trajectory (Next 2 Hours)", fontsize=11, fontweight="bold")
                    ax2.grid(alpha=0.15)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
        else:
            st.info("Run `py scripts/anomaly_prediction_v1.py` to generate prediction artifacts.")

    st.subheader("Events for this incident")
    st.dataframe(events_display(ev), width="stretch", height=420)


def render_prediction() -> None:
    import json
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    OUT = BASE / "outputs" / "anomaly_v1"
    if not OUT.exists():
        st.warning("Run `py scripts/anomaly_prediction_v1.py` first to generate prediction artifacts.")
        return

    scores = pd.read_csv(OUT / "scores_all_windows.csv")
    alerts = pd.read_csv(OUT / "alerts_triggered.csv")
    metrics = json.loads((OUT / "model_metrics.json").read_text(encoding="utf-8"))
    latest = json.loads((OUT / "latest_risk_snapshot.json").read_text(encoding="utf-8"))
    inc = pd.read_csv(RCA_MCS)
    for df_ in [scores, alerts, inc]:
        for c in ["timestamp", "time_start", "time_end"]:
            if c in df_.columns:
                df_[c] = pd.to_datetime(df_[c], utc=True, errors="coerce")
    threshold = float(metrics.get("threshold", 0.5))

    st.title("Anomaly Prediction Report")

    c1, c2, c3, c4 = st.columns(4)
    test = metrics.get("test_metrics", {})
    c1.metric("PR-AUC", f"{test.get('pr_auc', 0):.4f}")
    c2.metric("F1 Score", f"{test.get('f1', 0):.4f}")
    c3.metric("Precision", f"{test.get('precision', 0):.4f}")
    c4.metric("Recall", f"{test.get('recall', 0):.4f}")

    st.metric("Threshold", f"{threshold:.3f}")

    st.subheader("Risk Score Over 24 Hours")
    plot_df = scores.copy()
    if {"entity_type", "entity_value"}.issubset(plot_df.columns):
        g = plot_df[(plot_df["entity_type"] == "global") & (plot_df["entity_value"] == "GLOBAL")]
        if not g.empty:
            plot_df = g
    plot_df = plot_df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(plot_df["timestamp"], plot_df["risk_score"], alpha=0.25, color="#4fc3f7")
    ax.plot(plot_df["timestamp"], plot_df["risk_score"], linewidth=1.2, color="#4fc3f7")
    ax.axhline(threshold, color="#ef5350", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.3f})")
    for _, ir in inc.dropna(subset=["time_start", "time_end"]).iterrows():
        ax.axvspan(ir["time_start"], ir["time_end"], alpha=0.18, color="#ff9800")
    ax.set_ylabel("Risk Score")
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    legend_elements = [
        plt.Line2D([0], [0], color="#4fc3f7", lw=2, label="Risk Score"),
        plt.Line2D([0], [0], color="#ef5350", lw=2, ls="--", label=f"Threshold"),
        Patch(facecolor="#ff9800", alpha=0.3, label="Incident Window"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Incident Detection Coverage")
    cov_rows = []
    for _, ir in inc.dropna(subset=["time_start", "time_end"]).iterrows():
        window = plot_df[(plot_df["timestamp"] >= ir["time_start"] - pd.Timedelta(minutes=35)) &
                         (plot_df["timestamp"] <= ir["time_start"])]
        pre_alerts = window[window["risk_score"] >= threshold]
        if not pre_alerts.empty:
            first = pre_alerts["timestamp"].min()
            lead = (ir["time_start"] - first).total_seconds() / 60
            detected = True
        else:
            lead = 0
            detected = False
        cov_rows.append({
            "Incident": ir.get("incident_id", ""),
            "Start": ir["time_start"].strftime("%H:%M") if pd.notna(ir["time_start"]) else "-",
            "Severity": ir.get("severity", "-"),
            "Detected": "Yes" if detected else "MISSED",
            "Lead Time (min)": f"{lead:.0f}" if detected else "-",
        })
    cov_df = pd.DataFrame(cov_rows)
    detected_count = (cov_df["Detected"] == "Yes").sum()
    st.metric("Detection Rate", f"{detected_count}/{len(cov_df)} ({detected_count/max(len(cov_df),1)*100:.0f}%)")
    st.dataframe(cov_df, width="stretch", hide_index=True)

    st.subheader("Top Feature Contributors (Latest Window)")
    contribs = latest.get("top_feature_contributors", [])
    if contribs:
        feat_df = pd.DataFrame(contribs)
        st.dataframe(feat_df, width="stretch", hide_index=True)
    else:
        st.write("No contributors available.")

    if st.button("Back to Home"):
        st.session_state["view"] = "home"
        st.rerun()


def render_predictive_maintenance() -> None:
    """Interactive Predictive Maintenance — select window, run prediction, see results."""

    WINDOW_OPTIONS = {
        "5 minutes": 5,
        "10 minutes": 10,
        "15 minutes": 15,
        "30 minutes": 30,
    }

    OUT = BASE / "outputs" / "anomaly_v1"
    if not OUT.exists():
        st.warning("Run `py scripts/anomaly_prediction_v1.py` first to generate prediction artifacts.")
        return

    scores = pd.read_csv(OUT / "scores_all_windows.csv")
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True, errors="coerce")
    metrics = json.loads((OUT / "model_metrics.json").read_text(encoding="utf-8"))
    threshold = float(metrics.get("threshold", 0.5))

    svc_scores = scores[scores["entity_type"] == "service"].copy()
    base_timestamps = sorted(svc_scores["timestamp"].dropna().unique())

    st.title("Predictive Maintenance")

    col_win, col_ts = st.columns([1, 2])
    with col_win:
        window_label = st.selectbox("Select Time Window", list(WINDOW_OPTIONS.keys()))
    window_min = WINDOW_OPTIONS[window_label]

    aligned_timestamps = _align_timestamps(base_timestamps, window_min)
    ts_labels = [pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S+00:00") for t in aligned_timestamps]

    with col_ts:
        selected_label = st.selectbox(
            f"Select {window_label} Window",
            ts_labels,
            index=len(ts_labels) - 1 if ts_labels else 0,
        )
    selected_ts = pd.Timestamp(selected_label, tz="UTC") if selected_label else None

    run_clicked = st.button("Run Prediction", type="primary")

    if run_clicked and selected_ts is not None:
        st.session_state["pm_selected_ts"] = selected_ts
        st.session_state["pm_window_min"] = window_min
        st.session_state["pm_ran"] = True
        st.session_state["pm_email_sent"] = False

    if not st.session_state.get("pm_ran") or st.session_state.get("pm_selected_ts") is None:
        return

    sel_ts = st.session_state["pm_selected_ts"]
    win_min = st.session_state.get("pm_window_min", 5)
    win_end = sel_ts + pd.Timedelta(minutes=win_min)
    window_scores = _aggregate_service_scores(svc_scores, sel_ts, win_min)

    if window_scores.empty:
        st.warning("No prediction data available for this window.")
        return

    window_scores = window_scores.sort_values("risk_score", ascending=False)

    total_services = len(window_scores)
    failing_services: list[dict] = []

    for _, row in window_scores.iterrows():
        if row["risk_score"] >= threshold:
            failing_services.append({"service": row["entity_value"], "risk_score": row["risk_score"]})

    failure_count = len(failing_services)

    global_scores = scores[scores["entity_type"] == "global"].copy()
    global_window = global_scores[
        (global_scores["timestamp"] >= sel_ts) & (global_scores["timestamp"] < win_end)
    ]
    global_risk = float(global_window["risk_score"].max()) if not global_window.empty else 0.0

    # --- Auto-send email & show alert box at top ---
    if failure_count > 0 and not st.session_state.get("pm_email_sent"):
        from alert_service import send_alert_email
        result = send_alert_email(
            window_start=sel_ts.strftime("%Y-%m-%d %H:%M:%S"),
            window_end=win_end.strftime("%Y-%m-%d %H:%M:%S"),
            window_min=win_min,
            failures=failing_services,
            total_services=total_services,
            global_risk=global_risk,
        )
        st.session_state["pm_email_sent"] = True
        st.session_state["pm_email_result"] = result

    if failure_count > 0 and st.session_state.get("pm_email_result"):
        result = st.session_state["pm_email_result"]
        if result["success"]:
            st.markdown(
                f"""<div style="background:#e8f5e9;border:1px solid #a5d6a7;border-left:4px solid #43a047;
                padding:14px;border-radius:6px;margin-bottom:16px">
                <strong>📧 Failure Alert Email Sent</strong><br>
                <span style="color:#555">To: {', '.join(result['sent_to'])}</span><br>
                <span style="color:#555">Subject: {result['subject']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.warning(f"Email alert failed: {result['error']}")

    # --- Prediction Results ---
    st.subheader("Prediction Results")

    if failure_count == 0:
        st.markdown(
            '<div style="background:#e8f5e9;border-left:4px solid #43a047;padding:20px;'
            'border-radius:6px;margin:12px 0">'
            '<h3 style="margin:0;color:#2e7d32">✅ All Services Healthy</h3>'
            '<p style="margin:8px 0 0;color:#333;font-size:15px">'
            'No failures predicted in the next 30 minutes. All services are operating normally.'
            '</p></div>',
            unsafe_allow_html=True,
        )
    else:
        summary_lines = []
        for svc in failing_services:
            summary_lines.append(
                f"<strong>{svc['service']}</strong> — High risk of failure. "
                f"Check service health, review recent changes, and prepare for possible restart."
            )

        svc_bullets = "".join(f'<li style="margin-bottom:6px">{line}</li>' for line in summary_lines)

        st.markdown(
            f'<div style="background:#ffebee;border-left:4px solid #d32f2f;padding:20px;'
            f'border-radius:6px;margin:12px 0">'
            f'<h3 style="margin:0;color:#c62828">⚠ {failure_count} out of {total_services} Service(s) Expected to Fail</h3>'
            f'<p style="margin:8px 0;color:#333;font-size:15px">'
            f'Based on log pattern analysis, the following services are expected to fail '
            f'in the next <strong>30 minutes</strong>:</p>'
            f'<ul style="margin:8px 0;color:#333;font-size:14px">{svc_bullets}</ul>'
            f'<p style="margin:10px 0 0;color:#c62828;font-weight:bold;font-size:14px">'
            f'Immediate action recommended.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _align_timestamps(base_timestamps: list, window_min: int) -> list:
    """Produce timestamp boundaries aligned to *window_min* minute intervals."""
    if not base_timestamps or window_min <= 5:
        return list(base_timestamps)
    all_ts = pd.DatetimeIndex(base_timestamps)
    freq = f"{window_min}min"
    aligned = all_ts.floor(freq).unique().sort_values()
    return list(aligned)


def _aggregate_service_scores(
    svc_scores: pd.DataFrame, start: pd.Timestamp, window_min: int
) -> pd.DataFrame:
    """Aggregate per-service 5-min scores within a larger window using MAX."""
    end = start + pd.Timedelta(minutes=window_min)
    window = svc_scores[
        (svc_scores["timestamp"] >= start) & (svc_scores["timestamp"] < end)
    ].copy()
    if window.empty:
        return window
    agg = (
        window.groupby("entity_value", as_index=False)
        .agg(risk_score=("risk_score", "max"), bins_in_window=("risk_score", "size"))
    )
    agg["entity_type"] = "service"
    return agg


STARTER_QUESTIONS = [
    "Give me an overview of all incidents",
    "What happened in INC-00000?",
    "Was INC-00042 predicted by the model?",
    "Will INC-00010 happen again?",
    "How many HIGH severity incidents?",
    "Which service had the most incidents?",
    "What is the model's accuracy?",
    "How does the prediction pipeline work?",
    "Compare INC-00001 and INC-00003",
]


def render_chatbot() -> None:
    from chatbot_engine import chat as chatbot_reply

    st.title("AI Assistant")
    st.caption(
        "Powered by GPT-4o — ask me anything about incidents, predictions, root causes, or the anomaly detection model. "
        "If the screen looks washed out or stuck on “CONNECTING”, wait a few seconds or refresh (F5); VPN or long AI replies can interrupt the live browser session."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if not st.session_state["chat_history"]:
        st.markdown("**Try one of these:**")
        cols = st.columns(3)
        for i, q in enumerate(STARTER_QUESTIONS):
            col = cols[i % 3]
            if col.button(q, key=f"starter_{i}", width="stretch"):
                st.session_state["chat_history"].append({"role": "user", "content": q})
                try:
                    with st.spinner("Thinking with GPT-4o..."):
                        response = chatbot_reply(q, st.session_state["chat_history"])
                except Exception as e:
                    response = f"Error: {e}"
                st.session_state["chat_history"].append({"role": "assistant", "content": response})
                st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about incidents, predictions, root causes..."):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with GPT-4o..."):
                try:
                    response = chatbot_reply(prompt, st.session_state["chat_history"])
                except Exception as e:
                    response = f"Error communicating with Azure OpenAI: {e}"
            st.markdown(response)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

    if st.session_state["chat_history"]:
        st.sidebar.markdown("---")
        if st.sidebar.button("Clear Chat History", type="secondary"):
            st.session_state["chat_history"] = []
            st.rerun()


def main() -> None:
    st.session_state.setdefault("view", "home")
    if not RCA_MCS.exists() or not EV_MCS.exists():
        st.error("Missing MCS files under structured/September1011")
        return
    rca_df = load_rca()
    events_df = load_events()

    nav_options = ["Home", "Incident Detail", "Predictive Maintenance", "AI Assistant"]
    view_keys = ["home", "detail", "pred_maint", "chat"]
    current = st.session_state.get("view", "home")
    current_idx = view_keys.index(current) if current in view_keys else 0
    nav = st.sidebar.radio("Navigation", nav_options, index=current_idx)
    st.session_state["view"] = view_keys[nav_options.index(nav)]

    if st.session_state["view"] == "detail":
        render_detail(rca_df, events_df)
    elif st.session_state["view"] == "pred_maint":
        render_predictive_maintenance()
    elif st.session_state["view"] == "chat":
        render_chatbot()
    else:
        render_home(rca_df, events_df)


if __name__ == "__main__":
    main()

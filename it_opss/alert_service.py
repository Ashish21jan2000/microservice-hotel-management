"""Email alerting service for Predictive Maintenance.

Uses SendGrid HTTP API (HTTPS, port 443) to bypass corporate SMTP blocks.
Credentials are loaded from .env:
    SMTP_PASSWORD      – SendGrid API key (starts with SG.)
    SMTP_SENDER        – Verified sender email
    ALERT_RECIPIENTS   – comma-separated recipient emails
"""
from __future__ import annotations

import os
import urllib3
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

SENDGRID_API_KEY = os.getenv("SMTP_PASSWORD", "")
SENDER_EMAIL = os.getenv("SMTP_SENDER", "")
ALERT_RECIPIENTS = [
    r.strip() for r in os.getenv("ALERT_RECIPIENTS", "").split(",") if r.strip()
]
SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"


def _build_html_body(
    window_start: str,
    window_end: str,
    window_min: int,
    failures: list[dict[str, Any]],
    total_services: int,
    global_risk: float,
) -> str:
    svc_items = ""
    for f in failures:
        svc_items += (
            f'<li style="margin-bottom:10px"><strong>{f["service"]}</strong> — '
            f'High risk of failure. Check service health, review recent changes, '
            f'and prepare for possible restart.</li>'
        )

    return f"""
    <html>
    <body style="font-family:Segoe UI,Arial,sans-serif;color:#333;max-width:600px;margin:auto">
        <div style="background:#d32f2f;color:#fff;padding:20px;border-radius:8px 8px 0 0">
            <h2 style="margin:0">⚠️ PREDICTIVE MAINTENANCE ALERT</h2>
        </div>
        <div style="background:#fff;border:1px solid #ddd;border-top:none;padding:20px;border-radius:0 0 8px 8px">

            <p style="font-size:16px;color:#333">
                Based on log pattern analysis,
                <strong style="color:#d32f2f">{len(failures)} out of {total_services} service(s)</strong>
                are expected to fail in the next <strong>30 minutes</strong>.
            </p>

            <p style="font-size:13px;color:#666">
                Window analyzed: {window_start} → {window_end} UTC ({window_min} min)
            </p>

            <h3 style="border-bottom:2px solid #d32f2f;padding-bottom:6px">
                Services Expected to Fail
            </h3>
            <ul style="font-size:14px;line-height:1.7">{svc_items}</ul>

            <div style="background:#fff3e0;border-left:4px solid #ff9800;padding:12px;margin-top:20px;border-radius:4px">
                <strong>Immediate action recommended.</strong><br>
                Verify service health, check recent deployments, review logs, and
                prepare rollback or scaling plans for the affected services.
            </div>

            <p style="color:#999;font-size:12px;margin-top:20px">
                Sent by IT Ops Predictive Maintenance System at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
            </p>
        </div>
    </body>
    </html>"""


def send_alert_email(
    window_start: str,
    window_end: str,
    window_min: int,
    failures: list[dict[str, Any]],
    total_services: int,
    global_risk: float,
    recipients: list[str] | None = None,
) -> dict[str, Any]:
    """Send a prediction alert email via SendGrid HTTP API."""
    to_list = recipients or ALERT_RECIPIENTS
    if not to_list:
        return {"success": False, "error": "No recipients configured. Set ALERT_RECIPIENTS in .env"}
    if not SENDGRID_API_KEY:
        return {"success": False, "error": "SMTP_PASSWORD (SendGrid API key) not set in .env"}
    if not SENDER_EMAIL:
        return {"success": False, "error": "SMTP_SENDER not set in .env"}

    subject = (
        f"⚠️ Failure Predicted — {len(failures)} out of {total_services} service(s) "
        f"expected to fail in next 30 minutes [{window_start}]"
    )

    html = _build_html_body(
        window_start, window_end, window_min, failures, total_services, global_risk,
    )

    plain = (
        f"PREDICTIVE MAINTENANCE ALERT\n"
        f"{'=' * 40}\n\n"
        f"{len(failures)} out of {total_services} service(s) are expected to fail in the next 30 minutes.\n"
        f"Window analyzed: {window_start} → {window_end} UTC ({window_min} min)\n\n"
        f"Services expected to fail:\n\n"
    )
    for f in failures:
        plain += f"  - {f['service']} — High risk of failure. Check service health and prepare for restart.\n"
    plain += f"\n{'=' * 40}\nImmediate action recommended.\n"

    payload = {
        "personalizations": [{"to": [{"email": r} for r in to_list]}],
        "from": {"email": SENDER_EMAIL, "name": "IT Ops Predictive Maintenance"},
        "subject": subject,
        "content": [
            {"type": "text/plain", "value": plain},
            {"type": "text/html", "value": html},
        ],
    }

    try:
        resp = requests.post(
            SENDGRID_URL,
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
            verify=False,
        )
        if resp.status_code in (200, 201, 202):
            return {"success": True, "sent_to": to_list, "subject": subject}
        else:
            return {"success": False, "error": f"SendGrid {resp.status_code}: {resp.text}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}

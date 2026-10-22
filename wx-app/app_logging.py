"""Weather Bricks — Flask app.

A thin web layer over the `weather` data module. Three JSON endpoints plus the
single-page UI:

    GET  /api/weather/conditions?city=<name>    current observation
    GET  /api/weather/forecast?city=<name>      multi-period forecast
    POST /api/weather/report   {"city": ...}    LLM-written plain-language summary

The report endpoint calls a Databricks model serving endpoint (set ENDPOINT_NAME
in app.yaml). The app authenticates as its own service principal, which must have
CAN_QUERY on that endpoint (attached as an app resource).
"""
import json
import logging
import os
import sys

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from flask import Flask, jsonify, render_template, request

import weather

APP_NAME = os.environ.get("DATABRICKS_APP_NAME", "weather-app")

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME")
if not ENDPOINT_NAME:
    raise RuntimeError("ENDPOINT_NAME is not set. Add this to app.yaml")

app = Flask(__name__)
w = WorkspaceClient()

# --- Logging: emit to stdout so lines land in the Databricks app logs. ------
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    logger.addHandler(handler)


REPORT_SYSTEM_PROMPT = """\
You are a friendly local meteorologist writing a short weather brief for a general audience.
Write two short paragraphs of plain language — no markdown, no headings, no bullet lists.
First paragraph: the current conditions and how it feels right now.
Second paragraph: the notable trend over the coming periods and one piece of practical advice
(umbrella, layers, etc.). Be warm and concise. Use only the data provided; do not invent specifics."""


def query_llm(prompt: str, system_prompt: str = REPORT_SYSTEM_PROMPT, max_tokens: int = 400):
    """Send a single-shot chat request to the serving endpoint and return the text."""
    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        messages=[
            ChatMessage(role=ChatMessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatMessageRole.USER, content=prompt),
        ],
        max_tokens=max_tokens,
    )
    if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content
    return None


def _weather_error_response(exc: Exception, city: str):
    """Map a data-layer exception to (json, status), or None if it's not a known
    weather error (caller should re-raise so it surfaces as a 500)."""
    if isinstance(exc, LookupError):
        return jsonify({"error": str(exc)}), 404
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else "unknown"
        return jsonify({"error": f"NWS returned HTTP {status} for '{city}'."}), 502
    if isinstance(exc, requests.RequestException):
        return jsonify({"error": "Weather service is unreachable. Please try again."}), 503
    if isinstance(exc, ValueError):
        return jsonify({"error": str(exc)}), 502
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/weather/conditions")
def conditions():
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "Query parameter 'city' is required."}), 400
    logger.info("conditions requested city=%r", city)
    try:
        return jsonify(weather.get_conditions(city))
    except Exception as exc:
        mapped = _weather_error_response(exc, city)
        if mapped is None:
            raise
        return mapped


@app.get("/api/weather/forecast")
def forecast():
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "Query parameter 'city' is required."}), 400
    logger.info("forecast requested city=%r", city)
    try:
        return jsonify(weather.get_forecast(city))
    except Exception as exc:
        mapped = _weather_error_response(exc, city)
        if mapped is None:
            raise
        return mapped


@app.post("/api/weather/report")
def report():
    body = request.get_json(silent=True) or {}
    city = (body.get("city") or "").strip()
    if not city:
        return jsonify({"error": "JSON body with a 'city' field is required."}), 400
    if not ENDPOINT_NAME:
        logger.warning("report rejected: ENDPOINT_NAME not configured")
        return jsonify({"error": "Model serving endpoint is not configured."}), 500

    logger.info("report requested city=%r endpoint=%s", city, ENDPOINT_NAME)

    # Fetch both sources; tolerate one failing so a partial report still works.
    conditions_data = None
    forecast_data = None
    try:
        conditions_data = weather.get_conditions(city)
    except Exception as exc:
        logger.warning("report: conditions fetch failed city=%r error=%r", city, exc)
    try:
        forecast_data = weather.get_forecast(city)
    except Exception as exc:
        logger.warning("report: forecast fetch failed city=%r error=%r", city, exc)

    if conditions_data is None and forecast_data is None:
        return jsonify({"error": f"Could not retrieve any weather data for '{city}'."}), 502

    location = (conditions_data or forecast_data).get("location", city)
    prompt = f"""Write the weather brief for {location}.

Current conditions (JSON, may be null if unavailable):
{json.dumps(conditions_data, indent=2)}

Forecast (JSON, may be null if unavailable):
{json.dumps(forecast_data, indent=2)}
"""
    try:
        report_text = query_llm(prompt)
    except Exception as exc:
        logger.error("report generation failed city=%r error=%r", city, exc)
        return jsonify({"error": "Could not generate the weather summary."}), 502

    if not report_text:
        return jsonify({"error": "The model returned an empty summary."}), 502

    return jsonify({"report": report_text, "location": location})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")))

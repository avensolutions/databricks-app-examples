"""Weather Bricks — National Weather Service data layer.

A framework-free module that talks to api.weather.gov. Three public entry points:

    geocode_us_city(query) -> {name, state, lat, lon}   (offline, GeoNames)
    get_conditions(query)  -> current-observation summary dict
    get_forecast(query)    -> multi-period forecast summary dict

NWS requires no API key — it only asks for a User-Agent identifying the app
(set NWS_USER_AGENT in app.yaml so each deployment identifies itself). Geocoding
is offline via the `geonamescache` package; NWS only covers US locations.

Errors are raised, not swallowed, so the Flask layer can map them to HTTP codes:
  - LookupError                 -> city not found
  - ValueError                  -> NWS returned nothing usable
  - requests.RequestException   -> NWS unreachable / HTTP error
"""
import os
from typing import Any, Optional

import geonamescache
import requests
from cachetools import TTLCache

APP_NAME = os.environ.get("DATABRICKS_APP_NAME", "weather-app")

NWS_API_BASE = "https://api.weather.gov"
# NWS asks for a User-Agent identifying the app ("AppName (contact)") so they can
# reach you about API issues. Override via env var in app.yaml.
NWS_USER_AGENT = os.environ.get("NWS_USER_AGENT", f"{APP_NAME} (https://www.databricks.com)")
REQUEST_TIMEOUT = 5

# Short TTLs keep data fresh (observations/forecasts refresh ~hourly) while
# sparing the NWS API on repeated lookups of the same city. A grid's station
# mapping is effectively static, so it gets a long TTL.
OBSERVATION_TTL_SECONDS = 300
FORECAST_TTL_SECONDS = 600
STATION_TTL_SECONDS = 24 * 60 * 60
CACHE_MAXSIZE = 256

MAX_STATIONS_TO_TRY = 5    # skip silent specialty stations (RAWS/fire-weather) reporting nulls
MAX_FORECAST_PERIODS = 8   # ~4 days of NWS day/night periods

# --- Offline US-city geocoding index --------------------------------------
# geonamescache.get_cities() returns ~28k world cities (population >= 15k). We
# collapse to US and index by primary name AND GeoNames' alternate names (which
# include forms like "NYC"/"Big Apple"). Primary-name matches win over altname
# matches so "Manhattan" -> Manhattan, KS, not NYC (which lists it as an altname).
_gc = geonamescache.GeonamesCache()
_US_CITIES_BY_NAME: dict[str, list[dict[str, Any]]] = {}
_US_CITIES_BY_ALTNAME: dict[str, list[dict[str, Any]]] = {}
for _city in _gc.get_cities().values():
    if _city["countrycode"] != "US":
        continue
    _US_CITIES_BY_NAME.setdefault(_city["name"].lower(), []).append(_city)
    for _alt in _city.get("alternatenames", []):
        _US_CITIES_BY_ALTNAME.setdefault(_alt.lower(), []).append(_city)

stations_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=STATION_TTL_SECONDS)
observation_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=OBSERVATION_TTL_SECONDS)
forecast_cache: TTLCache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=FORECAST_TTL_SECONDS)

# Single shared session so the User-Agent header rides every request.
session = requests.Session()
session.headers.update({"User-Agent": NWS_USER_AGENT, "Accept": "application/geo+json"})


# --- Unit + format helpers -------------------------------------------------
def c_to_f(value: Optional[float]) -> Optional[int]:
    return round(value * 9 / 5 + 32) if value is not None else None


def kmh_to_mph(value: Optional[float]) -> Optional[int]:
    return round(value * 0.621371) if value is not None else None


_COMPASS_POINTS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


def degrees_to_compass(degrees: Optional[float]) -> Optional[str]:
    if degrees is None:
        return None
    return _COMPASS_POINTS[int((degrees + 11.25) % 360 / 22.5)]


# --- Geocoding (offline) ---------------------------------------------------
def geocode_us_city(query: str) -> dict[str, Any]:
    """Resolve free-text to a US city with lat/lon. Highest population wins ties
    (Springfield, Portland, ...). Raises LookupError if nothing matches."""
    key = query.strip().lower()
    matches = _US_CITIES_BY_NAME.get(key) or _US_CITIES_BY_ALTNAME.get(key) or []
    if not matches:
        raise LookupError(f"No US city found matching '{query}'. NWS only covers US locations.")
    best = max(matches, key=lambda c: c["population"])
    return {
        "name": best["name"],
        "state": best.get("admin1code", ""),
        "lat": best["latitude"],
        "lon": best["longitude"],
    }


# --- NWS: current conditions ----------------------------------------------
def get_stations(lat: float, lon: float) -> list[str]:
    """lat/lon -> ordered list of nearby NWS station IDs (closest first).
    Two NWS hops: /points -> observationStations URL -> station list."""
    key = (round(lat, 4), round(lon, 4))
    if key in stations_cache:
        return stations_cache[key]

    resp = session.get(f"{NWS_API_BASE}/points/{lat},{lon}", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    stations_url = resp.json()["properties"]["observationStations"]

    resp = session.get(stations_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise ValueError("No observation stations found near this location.")
    station_ids = [f["properties"]["stationIdentifier"] for f in features]

    stations_cache[key] = station_ids
    return station_ids


def _shape_observation(city: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    # Each measurement is {"value": X, "unitCode": ...}; `or {}` keeps it null-safe.
    temp_c = (obs.get("temperature") or {}).get("value")
    heat_index_c = (obs.get("heatIndex") or {}).get("value")
    wind_chill_c = (obs.get("windChill") or {}).get("value")
    wind_kmh = (obs.get("windSpeed") or {}).get("value")
    wind_deg = (obs.get("windDirection") or {}).get("value")
    humidity = (obs.get("relativeHumidity") or {}).get("value")

    # NWS sets heatIndex when hot, windChill when cold (mutually exclusive); when
    # neither applies, "feels like" is just the actual temperature.
    feels_like_c = heat_index_c if heat_index_c is not None else wind_chill_c
    if feels_like_c is None:
        feels_like_c = temp_c

    state = city.get("state", "")
    return {
        "location": f"{city['name']}, {state}" if state else city["name"],
        "city": city["name"],
        "state": state,
        "condition": obs.get("textDescription"),
        "temperature": c_to_f(temp_c),
        "feels_like": c_to_f(feels_like_c),
        "humidity": round(humidity) if humidity is not None else None,
        "wind_speed": kmh_to_mph(wind_kmh),
        "wind_direction": degrees_to_compass(wind_deg),
        "units": "imperial",
        "temperature_unit": "°F",
        "wind_unit": "mph",
        "icon_url": obs.get("icon"),
        "observed_at": obs.get("timestamp"),
        "source": "National Weather Service",
    }


def get_conditions(query: str) -> dict[str, Any]:
    """Current observation for a US city. Walks nearby stations until one is
    reporting real data (the closest is sometimes a silent specialty station)."""
    cache_key = query.strip().lower()
    if cache_key in observation_cache:
        return observation_cache[cache_key]

    city = geocode_us_city(query)
    station_ids = get_stations(city["lat"], city["lon"])

    for station_id in station_ids[:MAX_STATIONS_TO_TRY]:
        try:
            resp = session.get(
                f"{NWS_API_BASE}/stations/{station_id}/observations/latest",
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.HTTPError:
            continue

        obs = resp.json().get("properties", {})
        if (obs.get("temperature") or {}).get("value") is None:
            continue

        shaped = _shape_observation(city, obs)
        observation_cache[cache_key] = shaped
        return shaped

    raise ValueError(f"No NWS station near '{query}' is currently reporting observations.")


# --- NWS: forecast ---------------------------------------------------------
def _format_wind(period: dict[str, Any]) -> Optional[str]:
    speed = period.get("windSpeed")
    direction = period.get("windDirection")
    if speed and direction:
        return f"{speed} {direction}"
    return speed or direction or None


def fetch_forecast(lat: float, lon: float) -> list[dict[str, Any]]:
    """Two NWS hops: /points -> forecast URL -> list of forecast periods."""
    resp = session.get(f"{NWS_API_BASE}/points/{lat},{lon}", timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    forecast_url = resp.json()["properties"]["forecast"]

    resp = session.get(forecast_url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("properties", {}).get("periods", [])


def _summarize_forecast(periods: list[dict[str, Any]], resolved: dict[str, Any]) -> dict[str, Any]:
    """Trim NWS forecast periods to a compact, display- and LLM-friendly summary.
    NWS gives natural period names ("Today", "Tonight", "Tuesday Night"), so we
    pass them through rather than computing our own labels.

    `short_forecast` ("Partly Sunny") drives the GUI strip; `detailed_forecast`
    is NWS's richer human prose ("Sunny, with a high near 72. NW wind 5 to 10
    mph...") — too verbose for the cards, but great context for the AI summary."""
    trimmed = periods[:MAX_FORECAST_PERIODS]
    state = resolved.get("state", "")
    return {
        "location": f"{resolved['name']}, {state}" if state else resolved["name"],
        "source": "National Weather Service",
        "forecast_window": f"Next {len(trimmed)} periods (NWS day/night)",
        "periods": [
            {
                "name": p.get("name"),
                "is_daytime": p.get("isDaytime"),
                "temperature": p.get("temperature"),
                "temperature_unit": f"°{p.get('temperatureUnit', 'F')}",
                "wind": _format_wind(p),
                "precipitation_chance": (p.get("probabilityOfPrecipitation") or {}).get("value"),
                "short_forecast": p.get("shortForecast"),
                "detailed_forecast": p.get("detailedForecast"),
                "icon_url": p.get("icon"),
            }
            for p in trimmed
        ],
    }


def get_forecast(query: str) -> dict[str, Any]:
    """Multi-period forecast for a US city."""
    cache_key = query.strip().lower()
    if cache_key in forecast_cache:
        return forecast_cache[cache_key]

    resolved = geocode_us_city(query)
    periods = fetch_forecast(resolved["lat"], resolved["lon"])
    if not periods:
        raise ValueError(f"NWS returned no forecast periods for '{query}'.")

    summary = _summarize_forecast(periods, resolved)
    forecast_cache[cache_key] = summary
    return summary

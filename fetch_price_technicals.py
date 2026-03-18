"""
fetch_price_technicals.py
=========================
Layer 2 — Data Ingestion

Reads watchlist.txt and api_config.txt, fetches historical EOD price data
from the FMP API for every ticker, and saves the raw JSON to cache_.

Usage:
    python fetch_price_technicals.py

Dependencies: requests (stdlib otherwise)
"""

import json
import logging
import os
import sys
import requests
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Config / Watchlist Parsers
# ---------------------------------------------------------------------------

def parse_config(path: Path) -> dict:
    """Parse key = value config file; skip # comments and blank lines."""
    cfg = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                cfg[key.strip().upper()] = val.strip()
    return cfg


def parse_watchlist(path: Path) -> list:
    """Return a deduplicated, uppercased list of tickers from Tickers_List.csv."""
    seen, tickers = set(), []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            t = t.split(",")[0].strip().upper()
            if t == "TICKER":   # skip header row
                continue
            if t not in seen:
                seen.add(t)
                tickers.append(t)
    return tickers


# ---------------------------------------------------------------------------
# Date Resolution
# ---------------------------------------------------------------------------

def resolve_dates(cfg: dict):
    """
    Return (start_date_str, end_date_str) based on DATE_MODE in config.

    DATE_MODE = range     → use START_DATE / END_DATE (END_DATE = today allowed)
    DATE_MODE = days_back → compute start as today minus DAYS_BACK calendar days
    """
    today = datetime.today()
    mode  = cfg.get("DATE_MODE", "days_back").lower()

    if mode == "range":
        start   = cfg.get("START_DATE", "")
        end_raw = cfg.get("END_DATE", "today")
        end     = today.strftime("%Y-%m-%d") if end_raw.lower() == "today" else end_raw
        return start, end

    # days_back (default)
    days_back = int(cfg.get("DAYS_BACK", "760"))
    start     = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end       = today.strftime("%Y-%m-%d")
    return start, end


# ---------------------------------------------------------------------------
# FMP API Call
# ---------------------------------------------------------------------------

def fetch_ticker(
    ticker: str,
    api_key: str,
    start_date: str,
    end_date: str,
    frequency: str,
    session: requests.Session,
) -> dict:
    """
    Fetch historical price data for one ticker from FMP.

    Uses the FMP Stable API (post-August 2025):
        EOD  → GET /stable/historical-price-eod/full
               ?symbol={ticker}&from={start}&to={end}&apikey={key}
               Response: flat list of OHLCV records (no adjClose on this plan)

    Supported DATA_FREQUENCY values in api_config.txt:
        eod               → /stable/historical-price-eod/full
        1hour/4hour/etc.  → /stable/historical-chart/{interval}  (if available)

    Returns a normalised dict with keys:
        symbol      : str
        historical  : list[dict]   (date-ascending records)
        _meta       : dict         (run metadata)
    """
    base = "https://financialmodelingprep.com/stable"
    freq = frequency.lower()

    if freq == "eod":
        url    = f"{base}/historical-price-eod/full"
        params = {
            "symbol": ticker,
            "from":   start_date,
            "to":     end_date,
            "apikey": api_key,
        }
    else:
        # Intraday intervals via stable API
        interval_map = {
            "1hour": "1hour", "4hour": "4hour",
            "30min": "30min", "15min": "15min",
            "5min":  "5min",  "1min":  "1min",
        }
        interval = interval_map.get(freq, "1hour")
        url      = f"{base}/historical-chart/{interval}"
        params   = {
            "symbol": ticker,
            "from":   start_date,
            "to":     end_date,
            "apikey": api_key,
        }

    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Stable API returns a bare list — wrap into a consistent envelope
    if isinstance(data, list):
        data = {"symbol": ticker, "historical": data}

    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    workspace = Path(__file__).parent.resolve()

    config_path    = workspace / "api_config.txt"
    watchlist_path = workspace / "Tickers_List.csv"
    api_key_path   = workspace / "fmp.env"

    # ---- Config ------------------------------------------------------------
    cfg       = parse_config(config_path)
    cache_dir = workspace / cfg.get("CACHE_DIR", "cache_")
    log_dir   = workspace / cfg.get("LOG_DIR",   "logs")
    cache_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # ---- Logging -----------------------------------------------------------
    log_file = log_dir / "fetch_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("fetch")
    log.info("=" * 60)
    log.info("fetch_price_technicals.py  START")
    log.info(f"Workspace : {workspace}")

    # ---- API Key -----------------------------------------------------------
    if not api_key_path.exists():
        log.error(f"fmp.env not found at: {api_key_path}")
        log.error("ERROR")
        sys.exit(1)

    api_key = ""
    for line in api_key_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("FMP_API_KEY="):
            api_key = line.split("=", 1)[1].strip()
            break
    if not api_key:
        log.error("FMP_API_KEY not found in fmp.env.")
        log.error("ERROR")
        sys.exit(1)

    # ---- Watchlist ---------------------------------------------------------
    tickers = parse_watchlist(watchlist_path)
    if not tickers:
        log.error("Tickers_List.csv contains no tickers.")
        log.error("ERROR")
        sys.exit(1)
    log.info(f"Tickers   : {len(tickers)}  →  {', '.join(tickers)}")

    # ---- Date range --------------------------------------------------------
    start_date, end_date = resolve_dates(cfg)
    frequency  = cfg.get("DATA_FREQUENCY", "eod").lower()
    price_type = cfg.get("PRICE_TYPE", "adjusted").lower()
    log.info(f"Date range: {start_date}  →  {end_date}")
    log.info(f"Frequency : {frequency}   |   Price type: {price_type}")
    log.info("-" * 60)

    # ---- Fetch loop --------------------------------------------------------
    session          = requests.Session()
    success, failed  = [], []

    for ticker in tickers:
        try:
            log.info(f"  Fetching {ticker} ...")
            data = fetch_ticker(ticker, api_key, start_date, end_date,
                                frequency, session)

            historical = data.get("historical", [])
            if not historical:
                log.warning(f"  {ticker}: API returned no historical rows.")
                failed.append(ticker)
                continue

            # Stamp metadata so the report generator knows what was requested
            data["_meta"] = {
                "ticker":      ticker,
                "fetched_at":  datetime.utcnow().isoformat() + "Z",
                "start_date":  start_date,
                "end_date":    end_date,
                "frequency":   frequency,
                "price_type":  price_type,
                "row_count":   len(historical),
            }

            out_path = cache_dir / f"{ticker}_price_data.json"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)

            # Log first and last dates for quick sanity check
            dates = [r.get("date", "") for r in historical if r.get("date")]
            date_range_str = f"{min(dates)} → {max(dates)}" if dates else "unknown range"
            log.info(f"  {ticker}: {len(historical)} rows  |  {date_range_str}  →  saved")
            success.append(ticker)

        except requests.HTTPError as exc:
            log.error(f"  {ticker}: HTTP {exc.response.status_code} — {exc}")
            failed.append(ticker)
        except Exception as exc:
            log.error(f"  {ticker}: FAILED — {exc}")
            failed.append(ticker)

    # ---- Summary -----------------------------------------------------------
    log.info("=" * 60)
    log.info(f"Succeeded : {len(success)}")
    log.info(f"Failed    : {len(failed)}")
    if failed:
        log.warning(f"Failed tickers: {', '.join(failed)}")

    if failed and not success:
        log.error("All tickers failed. ERROR")
        sys.exit(1)

    log.info("fetch_price_technicals.py  COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

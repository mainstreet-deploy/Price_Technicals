"""
generate_technicals_report.py
==============================
Layer 3 + 4 — Computation & Reporting

Reads cached JSON files from cache_, computes moving averages, return
periods, and Golden/Death Cross signals, then generates PDF reports
using matplotlib (charts + tables via PdfPages).

Output:
    REPORT_MODE = combined   → output_/Price_Technicals_Combined_YYYYMMDD.pdf
    REPORT_MODE = per_ticker → output_/{TICKER}_Technicals_YYYYMMDD.pdf
    REPORT_MODE = both       → both of the above

Usage:
    python generate_technicals_report.py

Dependencies: pandas, numpy, matplotlib
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display required
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Shared Helpers (same pattern as fetch_price_technicals.py)
# ---------------------------------------------------------------------------

def parse_config(path: Path) -> dict:
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
# Layer 3 — Computation
# ---------------------------------------------------------------------------

def load_price_df(cache_path: Path, price_type: str) -> pd.DataFrame:
    """
    Load a JSON cache file into a clean, date-sorted DataFrame.

    Selects 'adjClose' (adjusted) or 'close' (unadjusted) based on price_type.
    The selected column is aliased to 'price' for all downstream calculations.
    """
    with open(cache_path, encoding="utf-8") as fh:
        data = json.load(fh)

    historical = data.get("historical", [])
    if not historical:
        raise ValueError(f"No historical data in {cache_path.name}")

    df = pd.DataFrame(historical)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Select price column
    if price_type == "adjusted" and "adjClose" in df.columns:
        df["price"] = pd.to_numeric(df["adjClose"], errors="coerce")
    elif "close" in df.columns:
        df["price"] = pd.to_numeric(df["close"], errors="coerce")
    else:
        # Fall back to the first numeric-looking column
        for col in ["adjClose", "close", "price"]:
            if col in df.columns:
                df["price"] = pd.to_numeric(df[col], errors="coerce")
                break

    df = df.dropna(subset=["price"]).reset_index(drop=True)
    return df


def compute_moving_averages(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Add MA{n} columns for every window in the list."""
    for w in windows:
        df[f"MA{w}"] = df["price"].rolling(window=w, min_periods=w).mean()
    return df


def detect_crosses(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    lookback_days: int,
) -> dict:
    """
    Detect the most recent Golden Cross or Death Cross within lookback_days.

    Golden Cross : fast MA crosses ABOVE slow MA  (bullish)
    Death Cross  : fast MA crosses BELOW slow MA  (bearish)

    Returns a dict:
        type          : 'golden' | 'death' | 'above' | 'below'
        date          : pd.Timestamp or None
        days_ago      : int or None
        recent_event  : bool   (True only if a cross occurred within lookback)
    """
    fast_col = f"MA{fast_window}"
    slow_col = f"MA{slow_window}"

    if fast_col not in df.columns or slow_col not in df.columns:
        return {"type": None, "date": None, "days_ago": None, "recent_event": False}

    valid = df[[fast_col, slow_col, "date"]].dropna().copy()
    if len(valid) < 2:
        return {"type": None, "date": None, "days_ago": None, "recent_event": False}

    valid["diff"]      = valid[fast_col] - valid[slow_col]
    valid["prev_diff"] = valid["diff"].shift(1)

    golden_mask = (valid["prev_diff"] <= 0) & (valid["diff"] > 0)
    death_mask  = (valid["prev_diff"] >= 0) & (valid["diff"] < 0)

    today   = df["date"].max()
    cutoff  = today - pd.Timedelta(days=lookback_days)

    # Gather all cross events inside the lookback window
    recent = []
    for _, row in valid[golden_mask & (valid["date"] >= cutoff)].iterrows():
        recent.append(("golden", row["date"]))
    for _, row in valid[death_mask  & (valid["date"] >= cutoff)].iterrows():
        recent.append(("death",  row["date"]))

    if recent:
        # Return the most recent cross event
        recent.sort(key=lambda x: x[1], reverse=True)
        cross_type, cross_date = recent[0]
        days_ago = (today - cross_date).days
        return {
            "type":         cross_type,
            "date":         cross_date,
            "days_ago":     days_ago,
            "recent_event": True,
        }

    # No recent cross — return current positional state
    current_diff = valid["diff"].iloc[-1]
    return {
        "type":         "above" if current_diff > 0 else "below",
        "date":         None,
        "days_ago":     None,
        "recent_event": False,
    }


def compute_returns(df: pd.DataFrame, periods: list) -> dict:
    """
    Compute percentage returns for a list of period keywords.

    Supported: WTD, MTD, QTD, YTD, 1M, 3M, 6M, 1Y, 3Y, 5Y
    Formula: (price_today / price_start - 1) * 100

    For each period the function finds the closest available trading day
    on or before the target anchor date.
    """
    today         = df["date"].max()
    current_price = df.loc[df["date"] == today, "price"].values[0]
    returns       = {}

    for period in periods:
        p = period.strip().upper()
        try:
            # Determine target anchor date
            if   p == "WTD":
                target = today - pd.Timedelta(days=today.weekday())   # Monday
            elif p == "MTD":
                target = today.replace(day=1)
            elif p == "QTD":
                q_month = ((today.month - 1) // 3) * 3 + 1
                target  = today.replace(month=q_month, day=1)
            elif p == "YTD":
                target = today.replace(month=1, day=1)
            elif p == "1M":
                target = today - pd.DateOffset(months=1)
            elif p == "3M":
                target = today - pd.DateOffset(months=3)
            elif p == "6M":
                target = today - pd.DateOffset(months=6)
            elif p == "1Y":
                target = today - pd.DateOffset(years=1)
            elif p == "3Y":
                target = today - pd.DateOffset(years=3)
            elif p == "5Y":
                target = today - pd.DateOffset(years=5)
            else:
                continue

            available = df[df["date"] <= pd.Timestamp(target)]
            if available.empty:
                returns[p] = None
                continue

            start_price = available.iloc[-1]["price"]
            ret         = (current_price / start_price - 1) * 100
            returns[p]  = round(ret, 2)

        except Exception:
            returns[p] = None

    return returns


# ---------------------------------------------------------------------------
# Layer 4 — Charting helpers
# ---------------------------------------------------------------------------

# Shared colour palette — consistent across all figures
COLORS = {
    "price":     "#1a1a2e",    # near-black navy
    "MA_fast":   "#e07b2a",    # orange  (50-day)
    "MA_slow":   "#c0392b",    # crimson (200-day)
    "bullish":   "#27ae60",    # green
    "bearish":   "#e74c3c",    # red
    "neutral":   "#7f8c8d",    # grey
    "header_bg": "#1a1a2e",
    "row_alt":   "#f0f4f8",
    "bg_fig":    "#f8f9fa",
    "bg_ax":     "#ffffff",
}

PERIOD_ORDER = ["WTD", "MTD", "QTD", "YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y"]


def _cross_label(cross_info: dict) -> str:
    """Human-readable current signal label."""
    if cross_info.get("recent_event"):
        label = "Golden Cross" if cross_info["type"] == "golden" else "Death Cross"
        return f"{label}  ({cross_info['days_ago']}d ago)"
    ct = cross_info.get("type", "")
    if ct == "above":
        return "50MA > 200MA  (Bullish)"
    if ct == "below":
        return "50MA < 200MA  (Bearish)"
    return "N/A"


def _cross_color(cross_info: dict) -> str:
    if cross_info.get("recent_event"):
        return COLORS["bullish"] if cross_info["type"] == "golden" else COLORS["bearish"]
    ct = cross_info.get("type", "")
    return COLORS["bullish"] if ct == "above" else COLORS["bearish"] if ct == "below" else COLORS["neutral"]


# ---------------------------------------------------------------------------
# Per-Ticker Figure
# ---------------------------------------------------------------------------

def create_ticker_figure(
    ticker: str,
    df: pd.DataFrame,
    ma_windows: list,
    returns: dict,
    cross_info: dict,
    cfg: dict,
) -> plt.Figure:
    """
    Build a single-page figure for one ticker:

    ┌─────────────────────────────────────────────┐
    │  Price chart + 50MA/200MA  (full width top) │
    ├─────────────────────┬───────────────────────┤
    │  Returns bar chart  │   Statistics table    │
    └─────────────────────┴───────────────────────┘
    """
    chart_lookback  = int(cfg.get("CHART_LOOKBACK_DAYS", "365"))
    flag_crosses    = cfg.get("FLAG_CROSS_EVENTS", "true").lower() == "true"
    ma_windows_s    = sorted(ma_windows)

    today       = df["date"].max()
    chart_start = today - pd.Timedelta(days=chart_lookback)
    chart_df    = df[df["date"] >= chart_start].copy()

    # ---- Figure layout -----------------------------------------------------
    fig = plt.figure(figsize=(15, 10), facecolor=COLORS["bg_fig"])
    fig.suptitle(
        f"{ticker}   ·   Price Technicals",
        fontsize=15, fontweight="bold", color=COLORS["header_bg"],
        y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        height_ratios=[2.3, 1.0],
        width_ratios=[1.65, 1.0],
        hspace=0.44, wspace=0.30,
        left=0.06, right=0.97, top=0.92, bottom=0.07,
    )
    ax_price   = fig.add_subplot(gs[0, :])
    ax_returns = fig.add_subplot(gs[1, 0])
    ax_stats   = fig.add_subplot(gs[1, 1])

    # ========================================================================
    # Price Chart
    # ========================================================================
    fast_col = f"MA{ma_windows_s[0]}"   if ma_windows_s else None
    slow_col = f"MA{ma_windows_s[-1]}"  if len(ma_windows_s) > 1 else None

    # Shaded zone between the two MAs
    if fast_col and slow_col and fast_col in chart_df.columns and slow_col in chart_df.columns:
        ax_price.fill_between(
            chart_df["date"], chart_df[fast_col], chart_df[slow_col],
            where=chart_df[fast_col] >= chart_df[slow_col],
            alpha=0.07, color=COLORS["bullish"], label="_",
        )
        ax_price.fill_between(
            chart_df["date"], chart_df[fast_col], chart_df[slow_col],
            where=chart_df[fast_col] <  chart_df[slow_col],
            alpha=0.07, color=COLORS["bearish"], label="_",
        )

    # Price line
    ax_price.plot(
        chart_df["date"], chart_df["price"],
        color=COLORS["price"], linewidth=1.3, label="Price", zorder=3,
    )

    # MA lines — colour by position in sorted window list
    ma_line_colors = [COLORS["MA_fast"], COLORS["MA_slow"]] + ["#8e44ad", "#2980b9"]
    for idx, w in enumerate(ma_windows_s):
        col = f"MA{w}"
        if col in chart_df.columns:
            ax_price.plot(
                chart_df["date"], chart_df[col],
                color=ma_line_colors[min(idx, len(ma_line_colors) - 1)],
                linewidth=1.6, linestyle="--",
                label=f"{w}-Day MA", zorder=4,
            )

    # Cross annotation
    if flag_crosses and cross_info.get("recent_event"):
        cross_date = cross_info["date"]
        if isinstance(cross_date, pd.Timestamp) and cross_date >= chart_start:
            c_color     = COLORS["bullish"] if cross_info["type"] == "golden" else COLORS["bearish"]
            c_label     = "Golden Cross" if cross_info["type"] == "golden" else "Death Cross"
            price_at    = chart_df.loc[chart_df["date"] >= cross_date, "price"]
            y_anchor    = price_at.iloc[0] if not price_at.empty else chart_df["price"].mean()
            ax_price.axvline(cross_date, color=c_color, linewidth=1.5,
                             linestyle=":", alpha=0.85, zorder=5)
            ax_price.annotate(
                c_label,
                xy=(cross_date, y_anchor),
                xytext=(6, 6), textcoords="offset points",
                fontsize=8, color=c_color, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=c_color, lw=0.8),
            )

    # Axes formatting
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    ax_price.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.2f}")
    )
    ax_price.set_ylabel("Adjusted Close", fontsize=9)
    ax_price.set_title(
        f"Price History  ({chart_lookback}-day window)   |   "
        f"As of {today.strftime('%B %d, %Y')}",
        fontsize=9, loc="left", pad=6,
    )
    ax_price.legend(loc="upper left", fontsize=8, framealpha=0.75)
    ax_price.grid(True, alpha=0.25, linewidth=0.5)
    ax_price.set_facecolor(COLORS["bg_ax"])

    # ========================================================================
    # Returns Bar Chart
    # ========================================================================

    # Respect the canonical period ordering from PERIOD_ORDER
    ordered_returns = {
        p: returns[p]
        for p in PERIOD_ORDER
        if p in returns and returns[p] is not None
    }

    if ordered_returns:
        labels  = list(ordered_returns.keys())
        values  = list(ordered_returns.values())
        bcolors = [COLORS["bullish"] if v >= 0 else COLORS["bearish"] for v in values]

        bars = ax_returns.barh(
            labels, values,
            color=bcolors, alpha=0.85,
            edgecolor="white", linewidth=0.5,
        )

        # Inline value labels
        x_range = max(abs(v) for v in values) if values else 1
        for bar, val in zip(bars, values):
            offset   = x_range * 0.03
            ha       = "left" if val >= 0 else "right"
            x_label  = bar.get_width() + offset if val >= 0 else bar.get_width() - offset
            ax_returns.text(
                x_label, bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}%",
                va="center", ha=ha, fontsize=7.5, color="#333333",
            )

        ax_returns.axvline(0, color="#888888", linewidth=0.8)
        ax_returns.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.0f}%")
        )
        ax_returns.set_xlabel("Return (%)", fontsize=9)
        ax_returns.set_title("Return Periods", fontsize=9, loc="left", pad=4)
        ax_returns.grid(True, axis="x", alpha=0.25, linewidth=0.5)
        ax_returns.invert_yaxis()
    else:
        ax_returns.text(0.5, 0.5, "Insufficient data", ha="center",
                        va="center", fontsize=9, color=COLORS["neutral"])
        ax_returns.axis("off")

    ax_returns.set_facecolor(COLORS["bg_ax"])

    # ========================================================================
    # Statistics Table
    # ========================================================================
    ax_stats.axis("off")

    current_price = df["price"].iloc[-1]
    n_rows_year   = min(252, len(df))
    high_52w      = df["price"].tail(n_rows_year).max()
    low_52w       = df["price"].tail(n_rows_year).min()

    stat_rows = [
        ["Current Price",  f"${current_price:,.2f}"],
        ["52-Week High",   f"${high_52w:,.2f}"],
        ["52-Week Low",    f"${low_52w:,.2f}"],
    ]

    for w in ma_windows_s:
        col = f"MA{w}"
        if col in df.columns:
            last_ma = df[col].dropna().iloc[-1] if not df[col].dropna().empty else None
            if last_ma is not None and not np.isnan(last_ma):
                pct = (current_price / last_ma - 1) * 100
                stat_rows.append([f"{w}-Day MA",  f"${last_ma:,.2f}"])
                stat_rows.append([f"vs {w}MA",    f"{pct:+.1f}%"])

    stat_rows.append(["Signal", _cross_label(cross_info)])

    col_labels  = ["Value"]
    row_labels  = [r[0] for r in stat_rows]
    cell_values = [[r[1]] for r in stat_rows]

    tbl = ax_stats.table(
        cellText=cell_values,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="right",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.45)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(COLORS["header_bg"])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(COLORS["row_alt"])
        else:
            cell.set_facecolor(COLORS["bg_ax"])

    # Colour-code the Signal row
    signal_r = len(stat_rows)   # 1-indexed (header = 0)
    sig_color = _cross_color(cross_info)
    for c in range(-1, 1):
        cell = tbl.get_celld().get((signal_r, c))
        if cell:
            cell.set_text_props(color=sig_color, fontweight="bold")

    ax_stats.set_title("Statistics", fontsize=9, loc="left", pad=4)

    return fig


# ---------------------------------------------------------------------------
# Summary Figure (one page covering all tickers)
# ---------------------------------------------------------------------------

def create_summary_figure(
    tickers: list,
    all_data: dict,
    cfg: dict,
    return_periods: list,
) -> plt.Figure:
    """
    Summary table — one row per ticker, columns for each return period,
    MA signal, and cross status.  Return cells are colour-coded green/red.
    """
    headers = (
        ["Ticker", "Price"]
        + return_periods
        + ["vs 50MA", "vs 200MA", "Signal"]
    )

    rows       = []
    row_colors = []

    for ticker in tickers:
        d = all_data.get(ticker)
        if d is None:
            continue

        df_t       = d["df"]
        returns    = d["returns"]
        cross_info = d["cross_info"]
        ma_windows = d["ma_windows"]

        current_price = df_t["price"].iloc[-1]
        row           = [ticker, f"${current_price:,.2f}"]

        cell_colors = [
            COLORS["bg_ax"],  # Ticker (white)
            COLORS["bg_ax"],  # Price  (white)
        ]

        for p in return_periods:
            val = returns.get(p)
            if val is None:
                row.append("—")
                cell_colors.append(COLORS["bg_ax"])
            else:
                row.append(f"{val:+.1f}%")
                cell_colors.append(
                    "#d5f5e3" if val > 0 else "#fde8e8" if val < 0 else COLORS["bg_ax"]
                )

        # vs fast MA
        fast_w = min(ma_windows) if ma_windows else None
        slow_w = max(ma_windows) if len(ma_windows) > 1 else None
        for w in [fast_w, slow_w]:
            if w is None:
                row.append("—")
                cell_colors.append(COLORS["bg_ax"])
                continue
            col     = f"MA{w}"
            last_ma = df_t[col].dropna().iloc[-1] if (col in df_t.columns and not df_t[col].dropna().empty) else None
            if last_ma:
                pct = (current_price / last_ma - 1) * 100
                row.append(f"{pct:+.1f}%")
                cell_colors.append(
                    "#d5f5e3" if pct > 0 else "#fde8e8"
                )
            else:
                row.append("—")
                cell_colors.append(COLORS["bg_ax"])

        # Signal
        row.append(_cross_label(cross_info))
        cell_colors.append(COLORS["bg_ax"])

        rows.append(row)
        row_colors.append(cell_colors)

    if not rows:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.text(0.5, 0.5, "No ticker data available for summary.",
                ha="center", va="center", fontsize=12, color=COLORS["neutral"])
        ax.axis("off")
        return fig

    fig_height = max(5, len(rows) * 0.55 + 2.5)
    fig, ax    = plt.subplots(figsize=(16, fig_height), facecolor=COLORS["bg_fig"])
    ax.axis("off")

    run_date = datetime.today().strftime("%B %d, %Y")
    fig.suptitle(
        f"Price Technicals  —  Summary\n{run_date}",
        fontsize=14, fontweight="bold", color=COLORS["header_bg"],
        y=0.97,
    )

    tbl = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.75)

    n_cols    = len(headers)
    n_rows    = len(rows)
    period_cs = range(2, 2 + len(return_periods))

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.4)

        if r == 0:
            # Header row
            cell.set_facecolor(COLORS["header_bg"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            # Data rows — apply pre-computed cell colour
            data_r = r - 1
            if 0 <= data_r < len(row_colors) and 0 <= c < len(row_colors[data_r]):
                bg = row_colors[data_r][c]
                cell.set_facecolor(bg)
            else:
                cell.set_facecolor(
                    COLORS["row_alt"] if r % 2 == 0 else COLORS["bg_ax"]
                )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    workspace = Path(__file__).parent.resolve()

    config_path    = workspace / "api_config.txt"
    watchlist_path = workspace / "Tickers_List.csv"

    cfg        = parse_config(config_path)
    cache_dir  = workspace / cfg.get("CACHE_DIR",  "cache_")
    output_dir = workspace / cfg.get("OUTPUT_DIR", "output_")
    log_dir    = workspace / cfg.get("LOG_DIR",    "logs")

    output_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    # ---- Logging -----------------------------------------------------------
    log_file = log_dir / "report_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("report")
    log.info("=" * 60)
    log.info("generate_technicals_report.py  START")
    log.info(f"Workspace : {workspace}")

    # ---- Config values -----------------------------------------------------
    tickers        = parse_watchlist(watchlist_path)
    price_type     = cfg.get("PRICE_TYPE", "adjusted").lower()
    ma_windows     = sorted(
        int(w.strip()) for w in cfg.get("MA_WINDOWS", "50, 200").split(",")
    )
    cross_lookback = int(cfg.get("MA_CROSS_LOOKBACK", "30"))
    return_periods = [
        p.strip().upper()
        for p in cfg.get("RETURN_PERIODS", "WTD,MTD,YTD,1Y").split(",")
    ]
    report_mode      = cfg.get("REPORT_MODE", "combined").lower()
    include_summary  = cfg.get("INCLUDE_SUMMARY_PAGE",  "true").lower() == "true"
    flag_crosses     = cfg.get("FLAG_CROSS_EVENTS", "true").lower() == "true"

    log.info(f"Tickers   : {len(tickers)}")
    log.info(f"MA windows: {ma_windows}")
    log.info(f"Periods   : {return_periods}")
    log.info(f"Mode      : {report_mode}")
    log.info("-" * 60)

    # ---- Process each ticker -----------------------------------------------
    all_data = {}

    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}_price_data.json"
        if not cache_path.exists():
            log.warning(f"  {ticker}: cache file not found — skipping.")
            continue
        try:
            log.info(f"  Processing {ticker} ...")
            df         = load_price_df(cache_path, price_type)
            df         = compute_moving_averages(df, ma_windows)
            returns    = compute_returns(df, return_periods)
            fast       = ma_windows[0]
            slow       = ma_windows[-1] if len(ma_windows) > 1 else ma_windows[0]
            cross_info = detect_crosses(df, fast, slow, cross_lookback)

            all_data[ticker] = {
                "df":         df,
                "returns":    returns,
                "cross_info": cross_info,
                "ma_windows": ma_windows,
            }
            ytd = returns.get("YTD")
            log.info(
                f"  {ticker}: {len(df)} rows  |  "
                f"YTD={ytd:+.1f}%" if ytd is not None
                else f"  {ticker}: {len(df)} rows  |  YTD=N/A"
            )
        except Exception as exc:
            log.error(f"  {ticker}: FAILED — {exc}")

    if not all_data:
        log.error("No usable data for any ticker. ERROR")
        sys.exit(1)

    run_date_str = datetime.today().strftime("%Y%m%d")

    # ---- Combined PDF ------------------------------------------------------
    if report_mode in ("combined", "both"):
        combined_path = output_dir / f"Price_Technicals_Combined_{run_date_str}.pdf"
        log.info(f"Writing combined PDF → {combined_path.name}")

        with PdfPages(combined_path) as pdf:
            # Page 1 — Summary table
            if include_summary:
                log.info("  Building summary page ...")
                fig = create_summary_figure(
                    list(all_data.keys()), all_data, cfg, return_periods
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # One page per ticker
            for ticker in all_data:
                log.info(f"  Building page for {ticker} ...")
                d   = all_data[ticker]
                fig = create_ticker_figure(
                    ticker, d["df"], d["ma_windows"],
                    d["returns"], d["cross_info"], cfg,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        log.info(f"Combined PDF complete: {combined_path}")

    # ---- Per-Ticker PDFs ---------------------------------------------------
    if report_mode in ("per_ticker", "both"):
        log.info("Writing per-ticker PDFs ...")
        for ticker in all_data:
            d        = all_data[ticker]
            per_path = output_dir / f"{ticker}_Technicals_{run_date_str}.pdf"
            log.info(f"  {ticker} → {per_path.name}")

            with PdfPages(per_path) as pdf:
                fig = create_ticker_figure(
                    ticker, d["df"], d["ma_windows"],
                    d["returns"], d["cross_info"], cfg,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    # ---- Final summary ------------------------------------------------------
    log.info("=" * 60)
    log.info(f"Processed : {len(all_data)} / {len(tickers)} tickers")
    log.info("generate_technicals_report.py  COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

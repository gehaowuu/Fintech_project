import os
import logging
import time
import json
import datetime as dt
from datetime import datetime, timedelta, date
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fredapi import Fred


load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Finance Middleware", version="0.1.0")

FRED_API_KEY = os.getenv("FRED_API_KEY")
TE_CLIENT_KEY = os.getenv("TE_CLIENT_KEY")
TE_CLIENT_SECRET = os.getenv("TE_CLIENT_SECRET")

fred_client: Optional[Fred] = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

if not FRED_API_KEY:
    logging.warning("未检测到 FRED_API_KEY，FRED 数据将被跳过。")
if not TE_CLIENT_KEY or not TE_CLIENT_SECRET:
    logging.warning("未检测到 TradingEconomics 凭证，经济日历将返回空列表。")

NEWS_QUERY = "Federal Reserve OR CPI OR inflation"
NEWS_RSS = (
    f"https://news.google.com/rss/search?q={NEWS_QUERY.replace(' ', '+')}+when:3d"
    "&hl=en-US&gl=US&ceid=US:en"
)

PCR_URLS = [
    "https://cdn.cboe.com/data/us/options/market_statistics/daily_put_call_ratios.csv",
    "https://cdn.cboe.com/data/PUT/pc.csv",
    "https://cdn.cboe.com/resources/options/volume_and_put_call_ratios/totalpc.csv",
]

DEFAULT_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
TE_CACHE_PATH = os.getenv("TE_CACHE_PATH", "cache_te_calendar.json")
TE_CIRCUIT_PATH = os.getenv("TE_CIRCUIT_PATH", "cache_te_circuit.json")

# -------- TradingEconomics helpers --------
def te_credential_string() -> str:
    key = (TE_CLIENT_KEY or "").strip()
    secret = (TE_CLIENT_SECRET or "").strip()
    if key and secret:
        return f"{key}:{secret}"
    return key


def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: str, obj: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _circuit_open(now_ts: float) -> bool:
    data = _load_json(TE_CIRCUIT_PATH, {})
    until = float(data.get("blocked_until_ts", 0))
    return now_ts < until


def _set_circuit(hours: int, reason: str) -> None:
    now_ts = time.time()
    _save_json(
        TE_CIRCUIT_PATH,
        {
            "blocked_until_ts": now_ts + hours * 3600,
            "reason": reason,
            "set_at": dt.datetime.utcnow().isoformat() + "Z",
        },
    )


def get_te_calendar_events(
    countries: tuple = ("united states",),
    lookback_days: int = 7,
    lookforward_days: int = 7,
    importance_min: Optional[int] = 1,
    timeout: int = 15,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    使用 TradingEconomics REST 接口拉取经济日历，带缓存和断路器，避免 403/限频导致反复失败。
    - 窗口：回看+前看（避免当天无数据时为空）
    - 1 req/s throttle，409 退避重试
    - 403 时开启断路器（6h），优先返回缓存
    """
    cred = te_credential_string()
    if not cred:
        return [], "TradingEconomics credentials missing (TE_CLIENT_KEY/TE_CLIENT_SECRET)."

    now_ts = time.time()
    if _circuit_open(now_ts):
        cached = _load_json(TE_CACHE_PATH, {})
        if cached.get("events"):
            return cached["events"], "TradingEconomics calendar skipped (circuit open). Using cached events."
        return [], "TradingEconomics calendar skipped (circuit open). No cache available."

    today = dt.date.today()
    start = (today - dt.timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = (today + dt.timedelta(days=lookforward_days)).strftime("%Y-%m-%d")
    country_path = ",".join([c.replace(" ", "%20") for c in countries])
    url = f"https://api.tradingeconomics.com/calendar/country/{country_path}/{start}/{end}"

    params = {"c": cred, "f": "json"}
    if importance_min is not None:
        params["importance"] = str(importance_min)

    headers = {"User-Agent": "finance-middleware/1.0"}

    # 1 req/s throttle
    time.sleep(1.05)

    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 409:
            time.sleep(1.5)
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code == 403:
            _set_circuit(hours=6, reason="TE 403 Forbidden (quota/blocked)")
            cached = _load_json(TE_CACHE_PATH, {})
            if cached.get("events"):
                return cached["events"], "TradingEconomics 403; using cached events."
            return [], "TradingEconomics unauthorized/forbidden (status=403). Check TE quota/plan or unblock; no cache."
        if r.status_code == 401:
            return [], "TradingEconomics 401 Unauthorized. Check TE key/secret."
        r.raise_for_status()
        raw = r.json() or []
    except Exception as e:
        cached = _load_json(TE_CACHE_PATH, {})
        if cached.get("events"):
            return cached["events"], f"TradingEconomics calendar failed ({e}); using cached events."
        return [], f"TradingEconomics calendar request failed: {e}"

    events: List[Dict[str, Any]] = []
    for x in raw:
        events.append(
            {
                "datetime": x.get("Date"),
                "country": x.get("Country"),
                "category": x.get("Category"),
                "event": x.get("Event"),
                "actual": x.get("Actual"),
                "previous": x.get("Previous"),
                "forecast": x.get("Forecast") or x.get("TEForecast"),
                "importance": x.get("Importance"),
                "unit": x.get("Unit"),
                "currency": x.get("Currency"),
                "source": x.get("Source"),
                "source_url": x.get("SourceURL"),
                "ticker": x.get("Ticker") or x.get("Symbol"),
                "te_url": x.get("URL"),
                "last_update": x.get("LastUpdate"),
            }
        )

    _save_json(
        TE_CACHE_PATH,
        {
            "saved_at": dt.datetime.utcnow().isoformat() + "Z",
            "window": {"start": start, "end": end, "importance_min": importance_min, "countries": list(countries)},
            "events": events,
        },
    )

    note = f"TradingEconomics calendar window: {start} to {end}, importance>={importance_min}."
    return events, note


# -------- FRED helpers --------
def fred_latest_observation(series_id: str, timeout: int = 15, max_rows: int = 10) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    直接调用 FRED observations，按时间倒序取最近有效值。
    返回 {"value": float, "date": "YYYY-MM-DD"} 或 (None, note)
    """
    if not FRED_API_KEY:
        return None, "FRED_API_KEY missing."

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "series_id": series_id,
        "sort_order": "desc",
        "limit": max_rows,
    }
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        obs = (r.json() or {}).get("observations", []) or []
        for o in obs:
            v = o.get("value", None)
            if v is None:
                continue
            v = str(v).strip()
            if v in ("", ".", "nan", "NaN"):
                continue
            return {"value": float(v), "date": o.get("date")}, None
        return None, f"No valid observation found for {series_id}."
    except Exception as e:
        return None, f"FRED observations failed for {series_id}: {e}"


# -------- CBOE PCR helpers --------
def _read_csv_url(url: str, timeout: int = 15) -> pd.DataFrame:
    headers = {"User-Agent": "finance-middleware/1.0", "Accept": "text/csv,*/*"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

def _download_text(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "finance-middleware/1.0", "Accept": "text/csv,*/*"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text


def _find_header_line_index(lines: List[str], header_token: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if header_token in line:
            return i
    return None


def _parse_pc_csv(text: str) -> Tuple[float, str]:
    lines = text.splitlines()
    # 1) Date-based tables (totalpc/equitypc)
    idx = _find_header_line_index(lines, "Date,")
    if idx is not None:
        csv_text = "\n".join(lines[idx:])
        df = pd.read_csv(StringIO(csv_text))
        df.columns = [c.strip() for c in df.columns]
        date_col = "Date"
        ratio_col = None
        for c in df.columns:
            if "P/C" in c or "Put/Call" in c:
                ratio_col = c
                break
        if ratio_col is None:
            put_col = next((c for c in df.columns if "Put" in c and "Volume" in c), None)
            call_col = next((c for c in df.columns if "Call" in c and "Volume" in c), None)
            if put_col and call_col:
                df["__ratio__"] = pd.to_numeric(df[put_col], errors="coerce") / pd.to_numeric(
                    df[call_col], errors="coerce"
                )
                ratio_col = "__ratio__"
            else:
                raise ValueError("Cannot find ratio column in Date-based CSV.")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        df = df.dropna(subset=[date_col, ratio_col]).sort_values(date_col)
        last = df.iloc[-1]
        return float(last[ratio_col]), last[date_col].date().isoformat()

    # 2) indexpcarchive: header "Trade_date"
    idx = _find_header_line_index(lines, "Trade_date")
    if idx is not None:
        csv_text = "\n".join(lines[idx:])
        df = pd.read_csv(StringIO(csv_text))
        df.columns = [c.strip() for c in df.columns]
        date_col = "Trade_date"
        ratio_col = "P/C Ratio" if "P/C Ratio" in df.columns else None
        if ratio_col is None:
            put_col = "Put" if "Put" in df.columns else None
            call_col = "Call" if "Call" in df.columns else None
            if put_col and call_col:
                df["__ratio__"] = pd.to_numeric(df[put_col], errors="coerce") / pd.to_numeric(
                    df[call_col], errors="coerce"
                )
                ratio_col = "__ratio__"
            else:
                raise ValueError("Cannot find ratio column in Trade_date-based CSV.")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df[ratio_col] = pd.to_numeric(df[ratio_col], errors="coerce")
        df = df.dropna(subset=[date_col, ratio_col]).sort_values(date_col)
        last = df.iloc[-1]
        return float(last[ratio_col]), last[date_col].date().isoformat()

    raise ValueError("Unknown CBOE CSV format (no Date,/Trade_date header found).")


def get_cboe_put_call_ratio(timeout: int = 15) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    返回 total/equity/index put/call ratio 最新值，按顺序尝试三个官方 CSV，兼容说明行/不同表头。
    """
    url_candidates = [
        ("cboe-totalpc", "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv"),
        ("cboe-equitypc", "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"),
        ("cboe-indexpcarchive", "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv"),
    ]
    last_error = None
    for source, url in url_candidates:
        try:
            text = _download_text(url, timeout=timeout)
            value, date_str = _parse_pc_csv(text)
            return {"value": value, "date": date_str, "source": source, "url": url}, None
        except Exception as e:
            last_error = f"{url} failed: {e}"
            continue
    return None, last_error or "CBOE put/call fetch failed."


def _parse_date(date_str: Optional[str]) -> date:
    """Parse YYYY-MM-DD, default今天；若为周末自动回退到上一个工作日。"""
    if not date_str:
        parsed = datetime.utcnow().date()
    else:
        try:
            parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date, expected YYYY-MM-DD") from exc
    while parsed.weekday() >= 5:  # 周末退回最近交易日
        parsed -= timedelta(days=1)
    return parsed


def _safe_asof(series: pd.Series, target: pd.Timestamp) -> Optional[float]:
    if series is None or series.empty:
        return None
    try:
        value = series.asof(target)
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_pct(value: Optional[float], signed: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.2%}"


def _fmt_num(value: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}{suffix}"


def _to_billions(value: Optional[float]) -> Optional[float]:
    """FRED 货币量单位通常是百万，转为十亿美元以便阅读。"""
    if value is None:
        return None
    return value / 1000.0


def _download_prices(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, threads=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data
    except Exception:
        return pd.DataFrame()


def _first_available_price(
    tickers: List[str], target: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp
) -> Tuple[Optional[float], Optional[str]]:
    """依次尝试多个 ticker，返回第一个成功的收盘价与 ticker 名。"""
    for ticker in tickers:
        prices = _download_prices([ticker], start, end)
        if prices.empty or ticker not in prices:
            continue
        value = _safe_asof(prices[ticker], target)
        if value is not None:
            return value, ticker
    return None, None


def _extract_put_call_ratio(df: pd.DataFrame) -> Optional[float]:
    """更健壮地从 CBOE CSV 中提取 P/C，比对多种列名并兼容缺失。"""
    if df.empty:
        return None
    candidates = [col for col in df.columns if "P/C" in col.upper() or "PUT" in col.upper()]
    for col in candidates:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not series.empty:
            return float(series.iloc[-1])
    # 兜底：取最后一个数值列的最新值
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    for col in reversed(numeric_cols):
        series = df[col].dropna()
        if not series.empty:
            return float(series.iloc[-1])
    return None


def _fetch_fear_greed() -> Tuple[Optional[int], Optional[str], Optional[str]]:
    """优先使用 fear-greed-index 包，失败则调用 CNN API。"""
    try:
        from fear_greed_index.CNNFearAndGreedIndex import CNNFearAndGreedIndex

        fgi = CNNFearAndGreedIndex()
        data = fgi.get()
        if data:
            return int(data.get("score")), data.get("rating"), "fear-greed-index"
    except Exception as exc:
        logging.warning("fear-greed-index 包不可用，尝试直接请求 CNN: %s", exc)
    try:
        resp = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            timeout=10,
            headers=DEFAULT_HEADERS,
        )
        resp.raise_for_status()
        payload = resp.json()
        return int(payload["fear_and_greed"]["score"]), payload["fear_and_greed"].get("rating"), "cnn-api"
    except Exception as exc:
        logging.warning("CNN Fear & Greed 接口不可用: %s", exc)
    # 尝试历史 CSV 兜底
    try:
        hist_df = pd.read_csv(
            "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/master/fear-greed-historical.csv"
        )
        if not hist_df.empty:
            last_row = hist_df.iloc[-1]
            return int(last_row.get("fear_greed_index")), str(last_row.get("label")), "csv-fallback"
    except Exception as exc:
        logging.warning("FGI 历史 CSV 兜底失败: %s", exc)
    return None, None, None


def fetch_fundamentals(date_str: str) -> Dict[str, Any]:
    """获取利率、DXY、经济日历与新闻。"""
    target = pd.to_datetime(date_str)
    start = target - timedelta(days=10)

    rates = {
        "dgs2": None,
        "dgs10": None,
        "fedfunds": None,
        "term_spread": None,
        "ffr_minus_2y": None,
        "fedfunds_date": None,
        "ffr_minus_2y_note": None,
    }
    if fred_client:
        try:
            dgs2 = fred_client.get_series("DGS2", start, target)
            dgs10 = fred_client.get_series("DGS10", start, target)
            rates["dgs2"] = _safe_asof(dgs2, target)
            rates["dgs10"] = _safe_asof(dgs10, target)
            if rates["dgs10"] is not None and rates["dgs2"] is not None:
                rates["term_spread"] = rates["dgs10"] - rates["dgs2"]
        except Exception as exc:
            logging.warning("Failed to fetch FRED rates: %s", exc)

    # Fed Funds：优先 EFFR，再退 FEDFUNDS，取最近一次有效值
    effr, _ = fred_latest_observation("EFFR")
    ff_obs, ff_note = (effr, None) if effr else fred_latest_observation("FEDFUNDS")
    if ff_obs:
        rates["fedfunds"] = ff_obs["value"]
        rates["fedfunds_date"] = ff_obs["date"]
    else:
        rates["fedfunds"] = None
        rates["fedfunds_date"] = None
    # ffr_minus_2y：用最新 FedFunds/EFFR 减去最新 DGS2
    dgs2_latest, _ = fred_latest_observation("DGS2")
    if ff_obs and dgs2_latest:
        rates["ffr_minus_2y"] = ff_obs["value"] - dgs2_latest["value"]
        rates["ffr_minus_2y_note"] = f"FF({ff_obs['date']}) - DGS2({dgs2_latest['date']})"
    # asof 日期（使用最新 observation）
    dgs2_obs, _ = fred_latest_observation("DGS2")
    dgs10_obs, _ = fred_latest_observation("DGS10")
    rates["dgs2_date"] = dgs2_obs["date"] if dgs2_obs else None
    rates["dgs10_date"] = dgs10_obs["date"] if dgs10_obs else None

    dxy_value, dxy_ticker = _first_available_price(
        ["DX-Y.NYB", "DXY", "USDOLLAR"], target, start, target + timedelta(days=1)
    )

    events, events_note = get_te_calendar_events(
        countries=("united states",), lookback_days=7, lookforward_days=7, importance_min=2
    )

    news: List[Dict[str, Any]] = []
    try:
        feed = feedparser.parse(NEWS_RSS)
        news = [
            {
                "title": entry.title,
                "link": entry.link,
                "published": getattr(entry, "published", None),
                "source": getattr(entry, "source", {}).get("title") if hasattr(entry, "source") else "Google News",
            }
            for entry in feed.entries[:5]
        ]
    except Exception as exc:
        logging.warning("Failed to parse news feed: %s", exc)

    return {
        **rates,
        "dxy": dxy_value,
        "dxy_ticker": dxy_ticker,
        "events": events,
        "events_note": events_note,
        "news": news,
        "news_note": "Google News RSS 仅覆盖最近几天，如为历史日期请注意时效性。",
    }


def fetch_liquidity(date_str: str) -> Dict[str, Any]:
    """净流动性（十亿美元）及信用利差。"""
    target = pd.to_datetime(date_str)
    start = target - timedelta(days=40)

    walcl = rrp = tga = None
    net_liquidity = net_change_4w = None
    if fred_client:
        try:
            walcl_series = fred_client.get_series("WALCL", start, target)
            rrp_series = fred_client.get_series("RRPONTSYD", start, target)
            tga_series = fred_client.get_series("WTREGEN", start, target)
            walcl = _to_billions(_safe_asof(walcl_series, target))
            rrp = _to_billions(_safe_asof(rrp_series, target))
            tga = _to_billions(_safe_asof(tga_series, target))
            if walcl is not None and rrp is not None and tga is not None:
                net_liquidity = walcl - rrp - tga
                prior_date = target - timedelta(days=28)
                prior_val = _to_billions(_safe_asof(walcl_series, prior_date))
                prior_rrp = _to_billions(_safe_asof(rrp_series, prior_date))
                prior_tga = _to_billions(_safe_asof(tga_series, prior_date))
                if prior_val is not None and prior_rrp is not None and prior_tga is not None:
                    net_change_4w = net_liquidity - (prior_val - prior_rrp - prior_tga)
        except Exception as exc:
            logging.warning("Failed to fetch liquidity series: %s", exc)

    prices = _download_prices(["HYG", "IEI"], start, target + timedelta(days=1))
    credit_ratio = credit_change_20d = None
    if not prices.empty and {"HYG", "IEI"} <= set(prices.columns):
        ratio_series = prices["HYG"] / prices["IEI"]
        credit_ratio = _safe_asof(ratio_series, target)
        past_date = target - timedelta(days=20)
        past_ratio = _safe_asof(ratio_series, past_date)
        if credit_ratio is not None and past_ratio is not None:
            credit_change_20d = credit_ratio - past_ratio

    return {
        "walcl": walcl,
        "rrp": rrp,
        "tga": tga,
        "net_liquidity": net_liquidity,
        "net_change_4w": net_change_4w,
        "credit_ratio": credit_ratio,
        "credit_change_20d": credit_change_20d,
    }


def fetch_sentiment(date_str: str) -> Dict[str, Any]:
    """情绪相关指标：Fear & Greed、VIX 期限结构、PCR、风险偏好价差。"""
    target = pd.to_datetime(date_str)

    fgi_score, fgi_rating, fgi_source = _fetch_fear_greed()

    vix = vix3m = term_structure = vix_source = None
    vix_data = _download_prices(["^VIX", "^VIX3M"], target - timedelta(days=30), target + timedelta(days=1))
    if not vix_data.empty:
        vix = _safe_asof(vix_data.get("^VIX"), target)
        vix3m = _safe_asof(vix_data.get("^VIX3M"), target)
        vix_source = "^VIX3M"
    if vix is not None and vix3m is None:
        alt_data = _download_prices(["VIXY"], target - timedelta(days=30), target + timedelta(days=1))
        if not alt_data.empty and "VIXY" in alt_data:
            vix3m = _safe_asof(alt_data["VIXY"], target)
            vix_source = "VIXY"
    if vix is not None and vix3m is not None:
        if vix3m > vix:
            term_structure = "contango"
        elif vix3m < vix:
            term_structure = "backwardation"
        else:
            term_structure = "flat"

    put_call_ratio = None
    put_call_source = None
    put_call_date = None
    put_call_note = None
    pcr, pcr_note = get_cboe_put_call_ratio()
    if pcr:
        put_call_ratio = pcr["value"]
        put_call_source = pcr["source"]
        put_call_date = pcr["date"]
    else:
        put_call_note = pcr_note

    spreads = {}
    spread_pairs = [("SPY", "XLU", "spy_xlu"), ("HYG", "IEF", "hyg_ief"), ("BTC-USD", "GC=F", "btc_gold")]
    tickers = list({a for pair in spread_pairs for a in pair[:2]})
    prices = _download_prices(tickers, target - timedelta(days=15), target + timedelta(days=1))
    returns = prices.pct_change() if not prices.empty else pd.DataFrame()
    for lhs, rhs, name in spread_pairs:
        if not returns.empty and lhs in returns and rhs in returns:
            spreads[name] = _safe_asof(returns[lhs] - returns[rhs], target)

    return {
        "fgi_score": fgi_score,
        "fgi_rating": fgi_rating,
        "fgi_source": fgi_source,
        "vix": vix,
        "vix3m": vix3m,
        "vix_term_source": vix_source,
        "term_structure": term_structure,
        "put_call_ratio": put_call_ratio,
        "put_call_source": put_call_source,
        "put_call_date": put_call_date,
        "put_call_note": put_call_note,
        "put_call_url": pcr["url"] if pcr else None,
        **spreads,
    }


def _calc_indicators(series: pd.Series, target: pd.Timestamp) -> Dict[str, Any]:
    """计算单资产的价格结构指标。"""
    s = series.dropna()
    if s.empty:
        return {}
    ma20 = s.rolling(20).mean().asof(target)
    ma50 = s.rolling(50).mean().asof(target)
    ma200 = s.rolling(200).mean().asof(target)
    atr14 = s.diff().abs().rolling(14).mean().asof(target)
    close = s.asof(target)
    if pd.isna(close):
        return {}
    trend_label = None
    if ma50 and ma200:
        if close > ma50 > ma200:
            trend_label = "uptrend"
        elif close < ma50 < ma200:
            trend_label = "downtrend"
        else:
            trend_label = "range"
    return {
        "close": float(close),
        "distance_ma20_pct": float((close - ma20) / ma20) if ma20 else None,
        "distance_ma50_pct": float((close - ma50) / ma50) if ma50 else None,
        "distance_ma200_pct": float((close - ma200) / ma200) if ma200 else None,
        "atr_pct": float(atr14 / close) if atr14 else None,
        "boll_width": float(s.rolling(20).std().asof(target) * 2 / ma20) if ma20 else None,
        "trend_label": trend_label,
    }


def fetch_technicals(date_str: str) -> Dict[str, Any]:
    """技术面指标：主流指数/商品/加密的均线距离、ATR、布林带宽度等。"""
    target = pd.to_datetime(date_str)
    assets = {
        "SPX": "^GSPC",
        "NDX": "^NDX",
        "RSP": "RSP",
        "IWM": "IWM",
        "XLK": "XLK",
        "XLP": "XLP",
        "XLU": "XLU",
        "DXY": "DX-Y.NYB",
        "GOLD": "GC=F",
        "CRUDE": "CL=F",
        "BTC": "BTC-USD",
    }
    hist = _download_prices(list(assets.values()), target - timedelta(days=750), target + timedelta(days=1))
    technicals: Dict[str, Any] = {}
    for name, ticker in assets.items():
        if hist.empty or ticker not in hist:
            continue
        technicals[name] = _calc_indicators(hist[ticker], target)

    breadth_diff = style_ratio = None
    if not hist.empty and {"RSP", "^GSPC"} <= set(hist.columns):
        returns = hist.pct_change()
        breadth_diff = _safe_asof(returns["RSP"] - returns["^GSPC"], target)
    if not hist.empty and {"XLK", "XLP"} <= set(hist.columns):
        style_ratio = _safe_asof(hist["XLK"] / hist["XLP"], target)

    return {"assets": technicals, "breadth_diff": breadth_diff, "style_ratio": style_ratio}


def assign_labels(modules: Dict[str, Any]) -> Dict[str, str]:
    """根据阈值打标签，供 LLM 生成综合观点。"""
    labels: Dict[str, str] = {}
    fundamentals = modules.get("fundamentals", {})
    liquidity = modules.get("liquidity", {})
    sentiment = modules.get("sentiment", {})
    technicals = modules.get("technicals", {}).get("assets", {})

    term_spread = fundamentals.get("term_spread")
    if term_spread is None:
        labels["macro_regime"] = "Unknown"
    elif term_spread < -0.5:
        labels["macro_regime"] = "Hawkish"
    elif term_spread > 0.0:
        labels["macro_regime"] = "Dovish"
    else:
        labels["macro_regime"] = "Neutral"

    net_change = liquidity.get("net_change_4w")
    if net_change is None:
        labels["liquidity_regime"] = "Unknown"
    elif net_change > 500:
        labels["liquidity_regime"] = "Easing"
    elif net_change < -500:
        labels["liquidity_regime"] = "Tightening"
    else:
        labels["liquidity_regime"] = "Neutral"

    fgi_score = sentiment.get("fgi_score")
    vix = sentiment.get("vix") or 0
    if fgi_score is None:
        labels["sentiment_regime"] = "Unknown"
    elif fgi_score >= 60 and vix < 20:
        labels["sentiment_regime"] = "Greed"
    elif fgi_score <= 40 or vix > 25:
        labels["sentiment_regime"] = "Fear"
    else:
        labels["sentiment_regime"] = "Neutral"

    spx = technicals.get("SPX", {})
    trend = spx.get("trend_label")
    boll = spx.get("boll_width")
    if trend is None:
        labels["technical_regime"] = "Unknown"
    elif trend in {"uptrend", "downtrend"} and boll and boll > 0.04:
        labels["technical_regime"] = "Trending"
    else:
        labels["technical_regime"] = "Range"

    return labels


def build_prompt_context(modules: Dict[str, Any], labels: Dict[str, str]) -> str:
    """根据四大模块构建面向 LLM 的提示文本。"""
    fundamentals = modules["fundamentals"]
    liquidity = modules["liquidity"]
    sentiment = modules["sentiment"]
    technicals = modules["technicals"]

    news_lines = [f"- {n['title']} (source: {n.get('source')})" for n in fundamentals.get("news", [])]
    event_lines = [
        f"- {e.get('category')}: {e.get('actual')} (fcst {e.get('forecast')}, prev {e.get('previous')})"
        for e in fundamentals.get("events", [])
    ]
    assets = technicals.get("assets", {})
    asset_lines = []
    for name, data in assets.items():
        if not data:
            continue
        close_val = data.get("close")
        atr_text = _fmt_pct(data.get("atr_pct"), signed=False)
        asset_lines.append(
            f"- {name}: {_fmt_num(close_val, 2)} "
            f"(MA20 {_fmt_pct(data.get('distance_ma20_pct'))} / "
            f"MA50 {_fmt_pct(data.get('distance_ma50_pct'))} / "
            f"MA200 {_fmt_pct(data.get('distance_ma200_pct'))}, "
            f"ATR% {atr_text}, "
            f"trend {data.get('trend_label')})"
        )

    prompt = [
        "=== FUNDAMENTALS ===",
        f"- 2Y: {_fmt_num(fundamentals.get('dgs2'), 2, '%')} | 10Y: {_fmt_num(fundamentals.get('dgs10'), 2, '%')} "
        f"| Term spread: {_fmt_num(fundamentals.get('term_spread'), 2, '%')}",
        f"- Fed Funds: {_fmt_num(fundamentals.get('fedfunds'), 2, '%')} | FFR-2Y: {_fmt_num(fundamentals.get('ffr_minus_2y'), 2, '%')}",
        f"- DXY: {_fmt_num(fundamentals.get('dxy'))} (ticker {fundamentals.get('dxy_ticker') or 'DX-Y.NYB/DXY'})",
        f"- Key events:\n{chr(10).join(event_lines) if event_lines else '  (none, maybe credentials missing)'}",
        f"- Top news:\n{chr(10).join(news_lines) if news_lines else '  (none)'}",
        "",
        "=== LIQUIDITY ===",
        f"- Net liquidity: {_fmt_num(liquidity.get('net_liquidity'), 2, 'B')} (Δ4w {_fmt_num(liquidity.get('net_change_4w'), 2, 'B')})",
        f"- Components: WALCL {_fmt_num(liquidity.get('walcl'), 2, 'B')} | RRP {_fmt_num(liquidity.get('rrp'), 2, 'B')} | "
        f"TGA {_fmt_num(liquidity.get('tga'), 2, 'B')}",
        f"- Credit ratio (HYG/IEI): {_fmt_num(liquidity.get('credit_ratio'))} (Δ20d {_fmt_num(liquidity.get('credit_change_20d'))})",
        "",
        "=== SENTIMENT ===",
        f"- Fear & Greed: {sentiment.get('fgi_score')} ({sentiment.get('fgi_rating')}, source={sentiment.get('fgi_source')})",
        f"- VIX: {_fmt_num(sentiment.get('vix'))} | VIX3M: {_fmt_num(sentiment.get('vix3m'))} (source={sentiment.get('vix_term_source')}) "
        f"| Term structure: {sentiment.get('term_structure')}",
        f"- Put/Call: {_fmt_num(sentiment.get('put_call_ratio'))}",
        f"- Risk spreads: SPY-XLU {_fmt_pct(sentiment.get('spy_xlu'))} | HYG-IEF {_fmt_pct(sentiment.get('hyg_ief'))} | "
        f"BTC-Gold {_fmt_pct(sentiment.get('btc_gold'))}",
        "",
        "=== TECHNICALS ===",
        *asset_lines,
        f"- Breadth diff (RSP-SPX): {_fmt_pct(technicals.get('breadth_diff'))}",
        f"- Style ratio (XLK/XLP): {_fmt_num(technicals.get('style_ratio'))}",
        "",
        "=== LABELS ===",
        f"- Macro: {labels.get('macro_regime')} | Liquidity: {labels.get('liquidity_regime')} | "
        f"Sentiment: {labels.get('sentiment_regime')} | Technical: {labels.get('technical_regime')}",
    ]
    return "\n".join(prompt)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/context")
async def get_context(date: Optional[str] = Query(None, description="YYYY-MM-DD, default today")) -> Dict[str, Any]:
    target_date = _parse_date(date)
    date_str = target_date.isoformat()
    try:
        fundamentals = fetch_fundamentals(date_str)
        liquidity = fetch_liquidity(date_str)
        sentiment = fetch_sentiment(date_str)
        technicals = fetch_technicals(date_str)
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Failed to build context")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    modules = {
        "fundamentals": fundamentals,
        "liquidity": liquidity,
        "sentiment": sentiment,
        "technicals": technicals,
    }
    labels = assign_labels(modules)
    prompt_context = build_prompt_context(modules, labels)
    return {**modules, "date": date_str, "labels": labels, "prompt_context": prompt_context}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

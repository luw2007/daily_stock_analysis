# -*- coding: utf-8 -*-
"""Tencent direct daily K-line fetcher for A-share and HK-share fallback routing."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import requests

try:
    import exchange_calendars as xcals
except ImportError:  # pragma: no cover - dependency is present in supported installs
    xcals = None

from .base import (
    BaseFetcher,
    DataFetchError,
    STANDARD_COLUMNS,
    _is_hk_market,
    is_bse_code,
    normalize_stock_code,
)
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote, safe_float, safe_int

logger = logging.getLogger(__name__)

_MAX_KLINE_BARS = 800


class TencentFetcher(BaseFetcher):
    """Fetch qfq daily K-line data from Tencent's direct quote endpoint.

    A-shares use ``fqkline`` (qfq). Hong Kong shares use ``hkfqkline``: the
    A-share ``fqkline`` endpoint ignores the adjustment flag for ``hkXXXXX`` and
    returns *unadjusted* ``day`` bars, so it must not be used to serve HK qfq.
    """

    name = "TencentFetcher"
    # This direct endpoint is the last-resort A-share daily fallback. Keeping
    # it at priority 0 made a single Efinance failure skip the richer built-in
    # fallback chain and try Tencent before AkShare/PyTDX/Baostock/YFinance.
    priority = 5
    allow_empty_daily_data = True

    _KLINE_ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    _HK_KLINE_ENDPOINT = "https://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get"
    _QUOTE_ENDPOINT = "https://qt.gtimg.cn/q"
    _HTTP_TIMEOUT_SECONDS = 8

    def __init__(self) -> None:
        self.priority = _read_tencent_priority()

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol = _to_tencent_symbol(stock_code)
        if not symbol:
            raise DataFetchError(f"TencentFetcher unsupported stock code: {stock_code}")

        lookback = _estimate_lookback_days(start_date=start_date, end_date=end_date)
        explicit_start = _format_tencent_date(start_date)
        explicit_end = _format_tencent_date(end_date)
        explicit_window = (
            f"{explicit_start},{explicit_end}"
            if explicit_start and explicit_end
            else ","
        )
        is_hk = symbol.startswith("hk")
        response = requests.get(
            self._HK_KLINE_ENDPOINT if is_hk else self._KLINE_ENDPOINT,
            params={"param": f"{symbol},day,{explicit_window},{lookback},qfq"},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"},
            timeout=self._HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        rows = _extract_kline_rows(payload, symbol=symbol)
        if is_hk and rows is None:
            raise DataFetchError(
                f"TencentFetcher HK response for {stock_code} has no adjusted qfqday series"
            )
        rows = rows or []
        if not rows:
            logger.info("TencentFetcher empty daily history for %s", stock_code)
            return _empty_daily_frame()

        df = pd.DataFrame(rows)
        first_returned_date = _first_returned_date(df)
        if first_returned_date and _is_capped_history_incomplete(
            first_returned_date=first_returned_date,
            start_date=start_date,
            lookback=lookback,
            returned_rows=len(rows),
        ):
            logger.info(
                "TencentFetcher incomplete capped daily history for %s: first_date=%s requested_start=%s",
                stock_code,
                first_returned_date,
                start_date,
            )
            return _empty_daily_frame()

        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        if df.empty:
            logger.info(
                "TencentFetcher daily history outside requested range for %s: %s~%s",
                stock_code,
                start_date,
                end_date,
            )
            return _empty_daily_frame()
        return df

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        normalized = df.copy()
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if "pct_chg" not in normalized.columns:
            normalized["pct_chg"] = normalized["close"].pct_change().fillna(0.0) * 100
        normalized = normalized[["date", "open", "high", "low", "close", "volume", "amount", "pct_chg"]]
        return normalized

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        symbol = _to_tencent_symbol(stock_code)
        if not symbol:
            return None
        try:
            response = requests.get(
                f"{self._QUOTE_ENDPOINT}={symbol}",
                headers={"Referer": "https://finance.qq.com", "User-Agent": "Mozilla/5.0"},
                timeout=self._HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            response.encoding = "gbk"
            content = response.text.strip()
            data_start = content.find('"')
            data_end = content.rfind('"')
            if data_start == -1 or data_end <= data_start:
                return None
            fields = content[data_start + 1:data_end].split("~")
            if len(fields) <= 2 or fields[2].strip() != symbol[2:]:
                return None
            name = fields[1].strip()
            return name or None
        except Exception:
            logger.debug(
                "TencentFetcher stock name lookup failed for %s",
                stock_code,
                exc_info=True,
            )
            return None

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """Hong Kong realtime snapshot from ``qt.gtimg.cn`` (``r_hkXXXXX``).

        Only HK codes are served; A-share realtime keeps its own chain. Fields
        that the payload does not carry as a positive number stay ``None``
        instead of being synthesised from the last price.
        """
        symbol = _to_tencent_hk_symbol(stock_code)
        if not symbol:
            return None
        response = requests.get(
            f"{self._QUOTE_ENDPOINT}=r_{symbol}",
            headers={"Referer": "https://finance.qq.com", "User-Agent": "Mozilla/5.0"},
            timeout=self._HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response.encoding = "gbk"
        return _parse_hk_realtime_quote(response.text, symbol=symbol, stock_code=stock_code)


# qt.gtimg.cn ``r_hkXXXXX`` field layout (verified against live 00700/03800/06862
# on 2026-10-16): 1 name, 2 code, 3 last, 4 prev close, 5 open, 6 volume (shares),
# 30 "YYYY/MM/DD HH:MM:SS" (HK time), 31 change, 32 change %, 33 high, 34 low,
# 36 volume (shares), 37 turnover (HKD), 43 amplitude %, 44/45 market cap
# (100M HKD), 48/49 52-week high/low, 75 currency.
_HK_QUOTE_TZ = timezone(timedelta(hours=8))


def _positive_float(value: Any) -> Optional[float]:
    number = safe_float(value)
    return number if number is not None and number > 0 else None


def _hk_quote_timestamp(value: str) -> Optional[str]:
    try:
        parsed = datetime.strptime(value.strip(), "%Y/%m/%d %H:%M:%S")
    except (AttributeError, ValueError):
        return None
    return parsed.replace(tzinfo=_HK_QUOTE_TZ).isoformat()


def _parse_hk_realtime_quote(
    content: str, *, symbol: str, stock_code: str
) -> Optional[UnifiedRealtimeQuote]:
    content = (content or "").strip()
    data_start = content.find('"')
    data_end = content.rfind('"')
    if data_start == -1 or data_end <= data_start:
        return None
    fields = content[data_start + 1:data_end].split("~")
    if len(fields) < 38 or fields[2].strip() != symbol[2:]:
        return None
    price = _positive_float(fields[3])
    if price is None:
        return None
    pre_close = _positive_float(fields[4])
    volume = safe_int(fields[36])
    total_mv = _positive_float(fields[44]) if len(fields) > 44 else None
    circ_mv = _positive_float(fields[45]) if len(fields) > 45 else None
    return UnifiedRealtimeQuote(
        code=normalize_stock_code(stock_code),
        name=fields[1].strip(),
        source=RealtimeSource.TENCENT,
        provider_timestamp=_hk_quote_timestamp(fields[30]),
        market="hk",
        currency=(fields[75].strip().upper() if len(fields) > 75 and fields[75].strip() else "HKD"),
        price=price,
        change_pct=safe_float(fields[32]),
        change_amount=safe_float(fields[31]),
        volume=volume if volume and volume > 0 else None,
        amount=_positive_float(fields[37]),
        amplitude=_positive_float(fields[43]) if len(fields) > 43 else None,
        open_price=_positive_float(fields[5]),
        high=_positive_float(fields[33]),
        low=_positive_float(fields[34]),
        pre_close=pre_close,
        total_mv=total_mv * 1e8 if total_mv else None,
        circ_mv=circ_mv * 1e8 if circ_mv else None,
        high_52w=_positive_float(fields[48]) if len(fields) > 48 else None,
        low_52w=_positive_float(fields[49]) if len(fields) > 49 else None,
    )


def _to_tencent_hk_symbol(stock_code: str) -> str:
    """Return ``hk00700`` for HK00700 / 00700.HK / 700.HK / bare 4-5 digit codes."""
    if not _is_hk_market(stock_code):
        return ""
    code = normalize_stock_code(stock_code)
    if code.isdigit():  # bare 4-5 digit code accepted by _is_hk_market
        return f"hk{code.zfill(5)}"
    digits = code[2:]
    if code.startswith("HK") and digits.isdigit() and len(digits) == 5:
        return f"hk{digits}"
    return ""


def _to_tencent_symbol(stock_code: str) -> str:
    hk_symbol = _to_tencent_hk_symbol(stock_code)
    if hk_symbol:
        return hk_symbol
    raw_code = (stock_code or "").strip().upper()
    code = normalize_stock_code(stock_code)
    if not code or not code.isdigit() or len(code) != 6:
        return ""
    if raw_code.startswith(("SH", "SS")) or raw_code.endswith((".SH", ".SS")):
        return f"sh{code}"
    if raw_code.startswith("SZ") or raw_code.endswith(".SZ"):
        return f"sz{code}"
    if raw_code.startswith("BJ") or raw_code.endswith(".BJ"):
        return f"bj{code}"
    if is_bse_code(code):
        return f"bj{code}"
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


def _read_tencent_priority() -> int:
    raw_value = os.getenv("TENCENT_PRIORITY", "5")
    try:
        return int(str(raw_value).strip())
    except (TypeError, ValueError):
        logger.warning(
            "TENCENT_PRIORITY=%r is not a valid integer; falling back to 5",
            raw_value,
        )
        return 5


def _estimate_lookback_days(*, start_date: str, end_date: str) -> int:
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        calendar_days = max(1, (end - start).days + 1)
    except ValueError:
        calendar_days = 90
    # Trading days are sparse over calendar days; add margin for holidays/suspensions.
    return max(30, min(_MAX_KLINE_BARS, int(calendar_days * 1.8) + 20))


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=STANDARD_COLUMNS)


def _first_returned_date(df: pd.DataFrame) -> Optional[str]:
    if "date" not in df.columns or df.empty:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().strftime("%Y-%m-%d")


def _is_capped_history_incomplete(
    *,
    first_returned_date: str,
    start_date: str,
    lookback: int,
    returned_rows: int,
) -> bool:
    hit_cap = lookback >= _MAX_KLINE_BARS and returned_rows >= _MAX_KLINE_BARS
    if not hit_cap:
        return False
    try:
        first = datetime.strptime(first_returned_date, "%Y-%m-%d")
        requested_start = datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError:
        return False
    return first > _first_trading_date_on_or_after(requested_start)


def _first_trading_date_on_or_after(start_date: datetime) -> datetime:
    if xcals is not None:
        try:
            cal = xcals.get_calendar("XSHG")
            session = cal.date_to_session(start_date.date(), direction="next")
            return datetime.combine(session.date(), datetime.min.time())
        except Exception:
            pass

    current = start_date
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _format_tencent_date(date_text: str) -> Optional[str]:
    try:
        return datetime.strptime(date_text, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _is_star_market(symbol: str) -> bool:
    """Return True for STAR Market codes (688xxx, and 689xxx CDRs).

    Accepts Tencent symbols (``sh688111``), bare codes (``688111``) and
    suffixed codes (``688111.SH``).
    """
    code = (symbol or "").strip().lower()
    if code[:2] in ("sh", "sz", "bj"):
        code = code[2:]
    return code.startswith(("688", "689"))


def _lots_to_shares(volume: Any, symbol: str = "") -> Any:
    """Convert Tencent daily K-line volume to shares.

    Tencent reports volume in lots (手, 100 shares) for most A-shares, but in
    shares (股) for STAR Market stocks (688/689). Multiplying STAR Market volume
    by 100 inflates it 100x (Refs #2350).
    """
    try:
        value = float(volume)
    except (TypeError, ValueError):
        return volume
    return value if _is_star_market(symbol) else value * 100


def _extract_kline_rows(payload: dict[str, Any], *, symbol: str) -> Optional[list[dict[str, Any]]]:
    """Parse Tencent K-line rows.

    Returns ``[]`` for a well-formed empty result. For HK symbols returns
    ``None`` when rows exist only under the unadjusted ``day`` key, so a qfq
    request is never silently answered with unadjusted data.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    item = data.get(symbol) if isinstance(data, dict) else None
    if not isinstance(item, dict):
        return []
    is_hk = symbol.startswith("hk")
    if is_hk:
        rows = item.get("qfqday") or []
        if not rows and item.get("day"):
            return None
    else:
        rows = item.get("qfqday") or item.get("day") or []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        if is_hk:
            # [date, open, close, high, low, volume(shares), {events}, turnover%, amount(万 HKD)]
            amount = _wan_to_units(row[8]) if len(row) > 8 else None
            volume: Any = row[5]
        else:
            amount = row[6] if len(row) > 6 else None
            volume = _lots_to_shares(row[5], symbol)
        result.append(
            {
                "date": str(row[0]),
                "open": row[1],
                "close": row[2],
                "high": row[3],
                "low": row[4],
                "volume": volume,
                "amount": amount,
            }
        )
    return result


def _wan_to_units(value: Any) -> Any:
    """Convert Tencent HK turnover (万 HKD) to HKD."""
    try:
        return float(value) * 10000
    except (TypeError, ValueError):
        return None

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import os

import requests


logger = logging.getLogger(__name__)

FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"
TAIPEI_TZ = timezone(timedelta(hours=8), name="Asia/Taipei")

DEFAULT_FOCUS_SECURITIES = [
    "0050",
    "0056",
    "00878",
    "00919",
    "00981A",
    "00631L",
    "00632R",
    "2330",
    "2317",
    "2454",
]

SECURITY_NAMES = {
    "0050": "元大台灣50",
    "0056": "元大高股息",
    "00631L": "元大台灣50正2",
    "00632R": "元大台灣50反1",
    "00878": "國泰永續高股息",
    "00919": "群益台灣精選高息",
    "00981A": "主動統一台股增長",
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
}

INSTITUTION_NAMES = {
    "Foreign_Investor": "外資",
    "Foreign_Dealer_Self": "外資自營商",
    "Investment_Trust": "投信",
    "Dealer_self": "自營商自行買賣",
    "Dealer_Hedging": "自營商避險",
}


class FinMindApiError(RuntimeError):
    pass


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %r", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %r", name, raw, default)
        return default


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def _to_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    text = f"{value:,.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _fmt_signed(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    prefix = "+" if value > 0 else ""
    return f"{prefix}{_fmt_num(value, digits)}"


def _fmt_billion_ntd(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{_fmt_signed(value / 100_000_000, digits)} 億元"


def _latest_date(rows: list[dict], key: str = "date") -> str | None:
    dates = [str(row.get(key, "")).split()[0] for row in rows if row.get(key)]
    return max(dates) if dates else None


def _rows_on_date(rows: list[dict], day: str, key: str = "date") -> list[dict]:
    return [row for row in rows if str(row.get(key, "")).split()[0] == day]


def _series(rows: list[dict], value_key: str) -> list[tuple[str, float, dict]]:
    items = []
    for row in rows:
        day = str(row.get("date", "")).split()[0]
        value = _to_float(row.get(value_key))
        if day and value is not None:
            items.append((day, value, row))
    items.sort(key=lambda item: item[0])
    return items


def _sma(values: list[float], days: int) -> float | None:
    if len(values) < days:
        return None
    return sum(values[-days:]) / days


def _sma_trend(values: list[float], days: int) -> str:
    if len(values) < days + 1:
        return "NA"
    current = sum(values[-days:]) / days
    previous = sum(values[-days - 1:-1]) / days
    if current > previous:
        return "上彎"
    if current < previous:
        return "下彎"
    return "持平"


def _price_position(close: float, values: list[float]) -> str:
    parts = []
    for days in (5, 10, 20):
        ma = _sma(values, days)
        if ma is None:
            continue
        relation = "站上" if close >= ma else "跌破"
        parts.append(f"{relation}SMA{days}({_fmt_num(ma)})")
    return "、".join(parts) if parts else "均線資料不足"


class FinMindClient:
    def __init__(self):
        self.token = os.getenv("FINMIND_API_TOKEN") or None
        self.timeout = _env_float("FINMIND_TIMEOUT_SECONDS", 20.0)
        self.session = requests.Session()
        self.unavailable_reason: str | None = None

    def fetch(
        self,
        dataset: str,
        *,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict]:
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        if self.unavailable_reason:
            raise FinMindApiError(
                f"FinMind API connection unavailable: {self.unavailable_reason}"
            )

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            resp = self.session.get(
                FINMIND_API_URL,
                params=params,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ProxyError,
            requests.exceptions.Timeout,
        ) as err:
            self.unavailable_reason = f"{type(err).__name__}: {err}"
            raise
        payload = resp.json()
        data = payload.get("data") or []
        status = payload.get("status")
        if status not in (None, 200, "200") and not data:
            msg = payload.get("msg") or payload.get("message") or payload
            raise FinMindApiError(f"{dataset}: status={status}, msg={msg}")
        if not isinstance(data, list):
            raise FinMindApiError(f"{dataset}: unexpected data shape")
        return data


def _build_taiex_intraday_line(client: FinMindClient, end_day) -> str | None:
    for offset in range(0, 10):
        day = end_day - timedelta(days=offset)
        if day.weekday() >= 5:
            continue
        try:
            rows = client.fetch(
                "TaiwanVariousIndicators5Seconds",
                start_date=day.isoformat(),
            )
        except Exception as err:
            logger.debug("FinMind TAIEX intraday fetch failed for %s: %s", day, err)
            continue
        if not rows:
            continue

        rows.sort(key=lambda row: str(row.get("date", "")))
        latest = rows[-1]
        taiex = _to_float(latest.get("TAIEX"))
        return (
            "- 加權指數取樣（TaiwanVariousIndicators5Seconds）："
            f"{latest.get('date')} TAIEX {_fmt_num(taiex)}。[FinMind]"
        )
    return None


def _build_total_return_index_line(
    client: FinMindClient,
    data_id: str,
    label: str,
    start_date: str,
    end_date: str,
) -> str | None:
    rows = client.fetch(
        "TaiwanStockTotalReturnIndex",
        data_id=data_id,
        start_date=start_date,
        end_date=end_date,
    )
    values = _series(rows, "price")
    if not values:
        return None

    day, close, _ = values[-1]
    previous = values[-2][1] if len(values) >= 2 else None
    change = close - previous if previous is not None else None
    pct = (change / previous * 100) if previous else None
    closes = [item[1] for item in values]
    return (
        f"- {label}：{day} 收 {_fmt_num(close)}，"
        f"日變動 {_fmt_signed(change)} ({_fmt_signed(pct)}%)，"
        f"{_price_position(close, closes)}，"
        f"SMA5趨勢 {_sma_trend(closes, 5)}、SMA10趨勢 {_sma_trend(closes, 10)}、"
        f"SMA20趨勢 {_sma_trend(closes, 20)}。[FinMind]"
    )


def _build_index_section(
    client: FinMindClient,
    start_date: str,
    end_date: str,
    end_day,
    warnings: list[str],
) -> list[str]:
    lines = ["### 台灣指數與市場狀態"]
    try:
        intraday = _build_taiex_intraday_line(client, end_day)
        if intraday:
            lines.append(intraday)
        else:
            warnings.append("未取得 TaiwanVariousIndicators5Seconds 加權指數取樣。")
    except Exception as err:
        warnings.append(f"加權指數取樣擷取失敗：{type(err).__name__}: {err}")

    for data_id, label in (
        ("TAIEX", "加權報酬指數 TAIEX TR"),
        ("TPEx", "櫃買報酬指數 TPEx TR"),
    ):
        try:
            line = _build_total_return_index_line(
                client,
                data_id,
                label,
                start_date,
                end_date,
            )
            if line:
                lines.append(line)
            else:
                warnings.append(f"未取得 {label}。")
        except Exception as err:
            warnings.append(f"{label} 擷取失敗：{type(err).__name__}: {err}")

    return lines if len(lines) > 1 else []


def _build_total_institution_section(
    client: FinMindClient,
    start_date: str,
    end_date: str,
    warnings: list[str],
) -> list[str]:
    try:
        rows = client.fetch(
            "TaiwanStockTotalInstitutionalInvestors",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as err:
        warnings.append(f"整體三大法人擷取失敗：{type(err).__name__}: {err}")
        return []

    day = _latest_date(rows)
    if not day:
        warnings.append("未取得整體三大法人資料。")
        return []

    latest_rows = _rows_on_date(rows, day)
    by_name = {row.get("name"): row for row in latest_rows}
    lines = [f"### 整體三大法人買賣超（{day}）"]

    foreign_net = 0.0
    foreign_parts = 0
    for key in ("Foreign_Investor", "Foreign_Dealer_Self"):
        row = by_name.get(key)
        if not row:
            continue
        buy = _to_float(row.get("buy"))
        sell = _to_float(row.get("sell"))
        if buy is not None and sell is not None:
            foreign_net += buy - sell
            foreign_parts += 1
    if foreign_parts:
        lines.append(f"- 外資合計估算買賣超：{_fmt_billion_ntd(foreign_net)}。[FinMind]")

    for key in (
        "Foreign_Investor",
        "Investment_Trust",
        "Dealer_self",
        "Dealer_Hedging",
        "Foreign_Dealer_Self",
    ):
        row = by_name.get(key)
        if not row:
            continue
        buy = _to_float(row.get("buy"))
        sell = _to_float(row.get("sell"))
        net = None if buy is None or sell is None else buy - sell
        label = INSTITUTION_NAMES.get(key, key)
        lines.append(
            f"- {label}：買進 {_fmt_billion_ntd(buy)}，"
            f"賣出 {_fmt_billion_ntd(sell)}，買賣超 {_fmt_billion_ntd(net)}。[FinMind]"
        )

    return lines if len(lines) > 1 else []


def _institution_net_shares(rows: list[dict], names: tuple[str, ...]) -> float | None:
    day = _latest_date(rows)
    if not day:
        return None
    total = 0.0
    found = False
    for row in _rows_on_date(rows, day):
        if row.get("name") not in names:
            continue
        buy = _to_float(row.get("buy"))
        sell = _to_float(row.get("sell"))
        if buy is None or sell is None:
            continue
        total += buy - sell
        found = True
    return total if found else None


def _build_focus_security_line(
    client: FinMindClient,
    stock_id: str,
    start_date: str,
    end_date: str,
) -> str | None:
    price_rows = client.fetch(
        "TaiwanStockPrice",
        data_id=stock_id,
        start_date=start_date,
        end_date=end_date,
    )
    prices = _series(price_rows, "close")
    if not prices:
        return None

    day, close, latest = prices[-1]
    closes = [item[1] for item in prices]
    trading_money = _to_float(latest.get("Trading_money"))
    spread = _to_float(latest.get("spread"))
    name = SECURITY_NAMES.get(stock_id, stock_id)

    foreign_net_shares = None
    trust_net_shares = None
    try:
        institution_rows = client.fetch(
            "TaiwanStockInstitutionalInvestorsBuySell",
            data_id=stock_id,
            start_date=start_date,
            end_date=end_date,
        )
        foreign_net_shares = _institution_net_shares(
            institution_rows,
            ("Foreign_Investor", "Foreign_Dealer_Self"),
        )
        trust_net_shares = _institution_net_shares(
            institution_rows,
            ("Investment_Trust",),
        )
    except Exception as err:
        logger.debug("FinMind institution fetch failed for %s: %s", stock_id, err)

    foreign_est_amount = (
        None if foreign_net_shares is None else foreign_net_shares * close
    )
    trust_est_amount = None if trust_net_shares is None else trust_net_shares * close
    return (
        f"- {stock_id} {name}：{day} 收 {_fmt_num(close)}，"
        f"漲跌價差 {_fmt_signed(spread)}，成交金額 {_fmt_billion_ntd(trading_money)}，"
        f"{_price_position(close, closes)}；"
        f"外資買賣超 { _fmt_num(foreign_net_shares, 0) } 股"
        f"（估 {_fmt_billion_ntd(foreign_est_amount)}），"
        f"投信買賣超 { _fmt_num(trust_net_shares, 0) } 股"
        f"（估 {_fmt_billion_ntd(trust_est_amount)}）。[FinMind]"
    )


def _build_focus_security_section(
    client: FinMindClient,
    start_date: str,
    end_date: str,
    warnings: list[str],
) -> list[str]:
    stock_ids = _env_csv("FINMIND_FOCUS_SECURITIES", DEFAULT_FOCUS_SECURITIES)
    max_items = max(1, _env_int("FINMIND_FOCUS_LIMIT", len(stock_ids)))
    lines = ["### 關注 ETF 與權值股"]
    for stock_id in stock_ids[:max_items]:
        try:
            line = _build_focus_security_line(
                client,
                stock_id,
                start_date,
                end_date,
            )
            if line:
                lines.append(line)
            else:
                warnings.append(f"{stock_id} 未取得股價資料。")
        except Exception as err:
            warnings.append(f"{stock_id} 擷取失敗：{type(err).__name__}: {err}")
    return lines if len(lines) > 1 else []


def build_market_context_text(mode: str = "morning") -> str:
    """Return FinMind market context to append to Gemini-extracted facts."""
    if not _env_bool("MARKET_CONTEXT_ENABLED", True):
        return ""

    now = datetime.now(TAIPEI_TZ)
    end_day = now.date()
    lookback_days = max(25, _env_int("FINMIND_LOOKBACK_DAYS", 45))
    start_day = end_day - timedelta(days=lookback_days)
    start_date = start_day.isoformat()
    end_date = end_day.isoformat()
    client = FinMindClient()
    warnings: list[str] = []

    lines = [
        "## 程式補充資料（FinMind API）",
        f"- 擷取時間：{now.strftime('%Y-%m-%d %H:%M')} Asia/Taipei；"
        f"查詢區間：{start_date} 至 {end_date}。",
        "- 使用規則：以下為程式直接查詢的結構化資料；若與圖片抽取資料衝突，請標示分歧，不要自行改寫成不存在的事實。",
    ]
    if not client.token:
        lines.append("- API 狀態：未設定 FINMIND_API_TOKEN，使用匿名額度。[FinMind]")

    sections = [
        _build_index_section(client, start_date, end_date, end_day, warnings),
    ]
    if client.unavailable_reason:
        warnings.append(f"FinMind API 連線不可用：{client.unavailable_reason}")
    else:
        sections.extend([
            _build_total_institution_section(client, start_date, end_date, warnings),
            _build_focus_security_section(client, start_date, end_date, warnings),
        ])
    for section in sections:
        if section:
            lines.extend(["", *section])

    if warnings:
        lines.extend(["", "### 資料狀態"])
        for warning in warnings[:12]:
            lines.append(f"- {warning}")

    if mode == "after_hours":
        lines.append("- 報告模式：盤後；優先用最新台股收盤與法人資料。[FinMind]")
    else:
        lines.append("- 報告模式：晨報；若今日台股尚未開盤，請以最新可取得交易日解讀。[FinMind]")

    return "\n".join(lines).strip()

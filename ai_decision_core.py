from __future__ import annotations

import json
import os
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yfinance as yf
import yfinance.cache as yf_cache
from yfinance.exceptions import YFRateLimitError

from siliconflow_client import chat_deepseek


BTC_SYMBOL = "BTC-USD"
MAX_RAW_NEWS = 20
MAX_SELECTED_NEWS = 5
YFINANCE_CACHE_DIR = ".yf_cache"
NEWSAPI_URL = "https://newsapi.org/v2/everything"
SECRETS_PATH = Path(__file__).parent / ".streamlit" / "secrets.toml"

yf_cache.set_cache_location(YFINANCE_CACHE_DIR)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)


@dataclass
class NewsItem:
    title: str
    summary: str
    publisher: str = ""
    published_at: str = ""
    link: str = ""


def get_newsapi_key() -> str:
    api_key = os.getenv("NEWSAPI_API_KEY")
    if api_key:
        return api_key

    if SECRETS_PATH.exists():
        with SECRETS_PATH.open("rb") as secrets_file:
            api_key = tomllib.load(secrets_file).get("NEWSAPI_API_KEY")
        if api_key:
            return api_key

    raise RuntimeError(
        "Missing NEWSAPI_API_KEY. Set it in .streamlit/secrets.toml "
        "or as an environment variable."
    )


def get_btc_price() -> float:
    ticker = yf.Ticker(BTC_SYMBOL)
    try:
        try:
            price = ticker.fast_info.get("last_price")
            if price:
                return float(price)
        except Exception:
            pass

        history = ticker.history(period="1d", interval="1m")
        if history.empty:
            raise RuntimeError("Unable to fetch BTC price from yfinance.")
        return float(history["Close"].dropna().iloc[-1])
    except YFRateLimitError as exc:
        raise RuntimeError(
            "Yahoo Finance rate limited this request. Try again later."
        ) from exc


def get_bitcoin_news(limit: int = MAX_RAW_NEWS) -> list[NewsItem]:
    params = {
        "q": "Bitcoin OR BTC",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(limit, 100),
    }
    headers = {"X-Api-Key": get_newsapi_key()}

    try:
        response = requests.get(NEWSAPI_URL, params=params, headers=headers, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError("NewsAPI request failed before returning data.") from exc

    payload = response.json()

    if payload.get("status") != "ok":
        message = payload.get("message", "Unknown NewsAPI error.")
        raise RuntimeError(f"NewsAPI request failed: {message}")

    news_items: list[NewsItem] = []
    for item in payload.get("articles", [])[:limit]:
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        title = item.get("title") or ""
        summary = item.get("description") or item.get("content") or ""
        link = item.get("url") or ""
        publisher = source.get("name") or ""
        published_at = item.get("publishedAt") or ""

        if title.strip():
            news_items.append(
                NewsItem(
                    title=title.strip(),
                    summary=summary.strip(),
                    publisher=publisher.strip(),
                    published_at=published_at,
                    link=link.strip(),
                )
            )

    return news_items


def fetch_btc_news(limit: int = MAX_RAW_NEWS) -> list[NewsItem]:
    return get_bitcoin_news(limit=limit)


def _build_analysis_prompt(btc_price: float, news_items: list[NewsItem]) -> str:
    raw_news = [asdict(item) for item in news_items]
    return f"""
You are an AI market decision agent for Bitcoin trading.

Current BTC price from yfinance: {btc_price}

Raw BTC news:
{json.dumps(raw_news, ensure_ascii=False, indent=2)}

Tasks:
A. Filter out irrelevant ads, duplicated items, and old/non-actionable news. Keep the 5 most important items at most.
B. For each retained item, assign sentiment from -1 to 1 and weight from 0 to 1.
C. Write one short AI market view in Chinese.

Return only valid JSON:
{{
  "market_view": "简短中文观点",
  "items": [
    {{
      "title": "news title",
      "summary": "brief summary",
      "sentiment": 0.0,
      "weight": 0.0,
      "reason": "short Chinese reason"
    }}
  ]
}}
""".strip()


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _clamp(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(low, min(high, number))


def analyze_news_with_ai(btc_price: float, news_items: list[NewsItem]) -> dict[str, Any]:
    if not news_items:
        return {"market_view": "暂无可用新闻，AI 暂不形成方向性判断。", "items": []}

    content = chat_deepseek(
        messages=[
            {"role": "system", "content": "You filter and score Bitcoin news. Return strict JSON only."},
            {"role": "user", "content": _build_analysis_prompt(btc_price, news_items)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    result = _extract_json(content)
    result["items"] = result.get("items", [])[:MAX_SELECTED_NEWS]

    for item in result["items"]:
        item["sentiment"] = _clamp(item.get("sentiment"), -1.0, 1.0)
        item["weight"] = _clamp(item.get("weight"), 0.0, 1.0)

    return result


def calculate_final_score(items: list[dict[str, Any]]) -> float:
    total_weight = sum(_clamp(item.get("weight"), 0.0, 1.0) for item in items)
    if total_weight <= 0:
        return 0.0
    weighted_score = sum(
        _clamp(item.get("sentiment"), -1.0, 1.0) * _clamp(item.get("weight"), 0.0, 1.0)
        for item in items
    )
    return round(weighted_score / total_weight, 4)


def run_ai_decision() -> dict[str, Any]:
    btc_price = get_btc_price()
    raw_news = fetch_btc_news()
    ai_result = analyze_news_with_ai(btc_price, raw_news)
    return {
        "symbol": BTC_SYMBOL,
        "btc_price": btc_price,
        "raw_news_count": len(raw_news),
        "market_view": ai_result.get("market_view", ""),
        "items": ai_result["items"],
        "final_score": calculate_final_score(ai_result["items"]),
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


if __name__ == "__main__":
    print(json.dumps(run_ai_decision(), ensure_ascii=False, indent=2))

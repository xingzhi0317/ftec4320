from __future__ import annotations

import csv
import os
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ai_decision_core import (
    NewsItem,
    analyze_news_with_ai,
    calculate_final_score,
    get_bitcoin_news,
)
from lstm_predictor import LSTMPrediction, train_and_predict_lstm
from strategy_executor import (
    Account,
    add_rsi,
    fetch_btc_hourly_klines,
    should_stop_loss,
    should_trailing_stop,
)


INITIAL_BALANCE = 10_000.0
CACHE_TTL = timedelta(minutes=5)
LSTM_CACHE_TTL = timedelta(minutes=30)
AI_BUY_THRESHOLD = 0.056
TRADING_LOG_PATH = Path("trading_log.csv")
TRADE_HISTORY_PATH = Path("trade_history.csv")


st.set_page_config(
    page_title="BTC AI Trading Console",
    page_icon="BTC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0E1117 !important; color: #E6EDF7 !important; }
    section[data-testid="stSidebar"] {
        background: #111827 !important;
        border-right: 1px solid #263244;
    }
    div[data-testid="stMetric"] {
        background: #151B24;
        border: 1px solid #273244;
        border-radius: 8px;
        padding: 16px;
    }
    .rule-box {
        background: #151B24;
        border: 1px solid #273244;
        border-left: 4px solid #38BDF8;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 16px;
    }
    .demo-pill {
        display: inline-block;
        background: #7C2D12;
        color: #FED7AA;
        border: 1px solid #FB923C;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 700;
    }
    .safe { color: #22C55E; font-weight: 700; }
    .warn { color: #F59E0B; font-weight: 700; }
    .danger { color: #EF4444; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# State, secrets, and CSV helpers
# -----------------------------

def init_state() -> None:
    defaults = {
        "balance": INITIAL_BALANCE,
        "position": 0.0,
        "entry_price": 0.0,
        "highest_price": 0.0,
        "realized_pnl": 0.0,
        "market_cache": None,
        "ai_cache": None,
        "lstm_cache": None,
        "latest_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_secrets_to_environment() -> list[str]:
    missing = []
    for key in ("SILICONFLOW_API_KEY", "NEWSAPI_API_KEY"):
        value = st.secrets.get(key, "")
        if value:
            os.environ[key] = str(value)
        else:
            missing.append(key)
    return missing


def account_from_state() -> Account:
    return Account(
        balance=float(st.session_state.balance),
        position=float(st.session_state.position),
        entry_price=float(st.session_state.entry_price),
        highest_price=float(st.session_state.highest_price),
    )


def save_account(account: Account) -> None:
    st.session_state.balance = account.balance
    st.session_state.position = account.position
    st.session_state.entry_price = account.entry_price
    st.session_state.highest_price = account.highest_price


def cache_is_fresh(cache: dict[str, Any] | None) -> bool:
    if not cache or "time" not in cache:
        return False
    return datetime.now(tz=UTC) - cache["time"] < CACHE_TTL


def lstm_cache_is_fresh(cache: dict[str, Any] | None) -> bool:
    if not cache or "time" not in cache:
        return False
    return datetime.now(tz=UTC) - cache["time"] < LSTM_CACHE_TTL


def ensure_csv(path: Path, columns: list[str]) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(columns)


def append_csv(path: Path, columns: list[str], row: dict[str, Any]) -> None:
    ensure_csv(path, columns)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writerow(row)


def read_recent_csv(path: Path, limit: int = 10) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).tail(limit).iloc[::-1]


def append_trading_log(result: dict[str, Any]) -> None:
    account = account_from_state()
    append_csv(
        TRADING_LOG_PATH,
        [
            "time",
            "mode",
            "price",
            "change_24h_pct",
            "rsi",
            "ai_score",
            "decision",
            "market_view",
            "net_worth",
            "balance",
            "position",
        ],
        {
            "time": datetime.now(tz=UTC).isoformat(),
            "mode": "DEMO" if result.get("demo_mode") else "LIVE",
            "price": round(result.get("price", 0.0), 2),
            "change_24h_pct": round(result.get("change_24h", 0.0), 4),
            "rsi": round(result.get("rsi", 0.0), 2),
            "ai_score": round(result.get("final_score", 0.0), 4),
            "decision": result.get("signal", "HOLD"),
            "market_view": result.get("market_view", ""),
            "net_worth": round(result.get("net_worth", account.balance), 2),
            "balance": round(account.balance, 2),
            "position": account.position,
        },
    )


def append_trade(row: dict[str, Any]) -> None:
    append_csv(
        TRADE_HISTORY_PATH,
        [
            "time",
            "side",
            "buy_price",
            "sell_price",
            "quantity",
            "ai_score",
            "rsi",
            "pnl",
            "net_worth",
        ],
        row,
    )


# -----------------------------
# Fast market data and demo fallbacks
# -----------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data_cached() -> pd.DataFrame:
    return fetch_btc_hourly_klines()


def make_mock_price_data(hours: int = 100) -> pd.DataFrame:
    now = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
    times = [now - timedelta(hours=hours - 1 - i) for i in range(hours)]
    price = 78_000.0
    rows = []
    for _ in times:
        close = max(1_000.0, price + random.uniform(-350, 420))
        rows.append(
            (
                price,
                max(price, close) + random.uniform(20, 180),
                min(price, close) - random.uniform(20, 180),
                close,
            )
        )
        price = close
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times), columns=["open", "high", "low", "close"])


def load_market_data() -> dict[str, Any]:
    """Fast path for first screen: only price, 24h change, RSI, and chart data."""
    if cache_is_fresh(st.session_state.market_cache):
        return st.session_state.market_cache

    demo_mode = False
    try:
        data = fetch_market_data_cached()
    except Exception as exc:
        demo_mode = True
        data = make_mock_price_data()
        st.warning(f"Demo Mode: market data API failed, using mock candles. {exc}")

    data = add_rsi(data)
    price = float(data["close"].iloc[-1])
    rsi = float(data["RSI_14"].dropna().iloc[-1])
    change_24h = calculate_24h_change(data)
    market = {
        "time": datetime.now(tz=UTC),
        "demo_mode": demo_mode,
        "price_data": data,
        "price": price,
        "rsi": rsi,
        "change_24h": change_24h,
    }
    st.session_state.market_cache = market
    return market


def calculate_24h_change(data: pd.DataFrame) -> float:
    if len(data) < 25:
        return 0.0
    current = float(data["close"].iloc[-1])
    previous = float(data["close"].iloc[-25])
    return 0.0 if previous == 0 else ((current - previous) / previous) * 100


def make_mock_news() -> list[NewsItem]:
    return [
        NewsItem("Bitcoin ETF inflows strengthen", "ETF inflows suggest institutional demand.", "Demo News"),
        NewsItem("Macro uncertainty pressures crypto", "Dollar strength weighs on risk assets.", "Demo Macro"),
        NewsItem("On-chain accumulation improves", "Long-term holders keep accumulating BTC.", "Demo Chain"),
    ]


def make_mock_ai_result() -> dict[str, Any]:
    items = [
        {"title": "Bitcoin ETF inflows strengthen", "summary": "ETF inflows suggest institutional demand.", "sentiment": 0.65, "weight": 0.75, "reason": "机构资金流入偏利好。"},
        {"title": "Macro uncertainty pressures crypto", "summary": "Dollar strength weighs on risk assets.", "sentiment": -0.25, "weight": 0.45, "reason": "宏观压力降低风险偏好。"},
        {"title": "On-chain accumulation improves", "summary": "Long-term holders keep accumulating BTC.", "sentiment": 0.45, "weight": 0.55, "reason": "链上累积支撑中期情绪。"},
    ]
    return {"market_view": "Demo Mode：模拟舆情略偏多，但需等待真实新闻确认。", "items": items}


@st.cache_data(ttl=1800, show_spinner=False)
def train_lstm_cached() -> LSTMPrediction:
    return train_and_predict_lstm()


def make_mock_lstm_prediction(market: dict[str, Any]) -> LSTMPrediction:
    data = market["price_data"]
    last_price = float(market["price"])
    future_index = [data.index[-1] + timedelta(hours=step) for step in range(1, 25)]
    forecast = [last_price * (1 + 0.0008 * step) for step in range(1, 25)]
    prediction_frame = pd.DataFrame(
        {
            "actual_close": pd.concat(
                [data["close"].tail(48), pd.Series([None] * 24, index=future_index)]
            ),
            "predicted_close": pd.concat(
                [
                    pd.Series([None] * 48, index=data.index[-48:]),
                    pd.Series(forecast, index=future_index),
                ]
            ),
        }
    )
    return LSTMPrediction(
        trend_signal=1,
        last_price=last_price,
        forecast_end_price=float(forecast[-1]),
        forecast_return=float(forecast[-1] / last_price - 1),
        prediction_frame=prediction_frame,
        train_loss=0.0,
    )


def load_lstm_prediction(market: dict[str, Any]) -> tuple[LSTMPrediction, bool]:
    if lstm_cache_is_fresh(st.session_state.lstm_cache):
        cached = st.session_state.lstm_cache
        return cached["prediction"], bool(cached.get("demo_mode", False))

    try:
        prediction = train_lstm_cached()
        demo_mode = False
    except Exception as exc:
        st.warning(f"Demo Mode: LSTM forecast failed, using mock forecast. {exc}")
        prediction = make_mock_lstm_prediction(market)
        demo_mode = True

    st.session_state.lstm_cache = {
        "time": datetime.now(tz=UTC),
        "prediction": prediction,
        "demo_mode": demo_mode,
    }
    return prediction, demo_mode


# -----------------------------
# AI analysis and execution
# -----------------------------

def execute_paper_trade(account: Account, signal: str, price: float, rsi: float, ai_score: float) -> None:
    before_position = account.position
    before_entry = account.entry_price
    if signal == "BUY":
        account.buy_all(price)
        append_trade(
            {
                "time": datetime.now(tz=UTC).isoformat(),
                "side": "BUY",
                "buy_price": round(price, 2),
                "sell_price": "",
                "quantity": account.position,
                "ai_score": round(ai_score, 4),
                "rsi": round(rsi, 2),
                "pnl": 0.0,
                "net_worth": round(account.total_assets(price), 2),
            }
        )
    elif signal == "SELL" and before_position > 0:
        sell_value_after_fee = before_position * price * 0.999
        cost_basis = before_position * before_entry
        pnl = sell_value_after_fee - cost_basis
        account.sell_all(price)
        st.session_state.realized_pnl += pnl
        append_trade(
            {
                "time": datetime.now(tz=UTC).isoformat(),
                "side": "SELL",
                "buy_price": round(before_entry, 2),
                "sell_price": round(price, 2),
                "quantity": before_position,
                "ai_score": round(ai_score, 4),
                "rsi": round(rsi, 2),
                "pnl": round(pnl, 2),
                "net_worth": round(account.total_assets(price), 2),
            }
        )


def decide_signal_with_lstm(
    final_score: float,
    rsi: float,
    lstm_trend: int,
    account: Account,
    price: float,
) -> str:
    account.update_highest_price(price)
    stop_loss = should_stop_loss(account, price)
    trailing_stop = should_trailing_stop(account, price)

    if account.in_position:
        if final_score < -0.3 or rsi > 80 or stop_loss or trailing_stop:
            return "SELL"
        return "HOLD"

    if final_score > AI_BUY_THRESHOLD and rsi < 70 and lstm_trend == 1:
        return "BUY"
    return "HOLD"


def run_ai_analysis(news_limit: int, market: dict[str, Any], show_status: bool) -> dict[str, Any]:
    """Slow path: NewsAPI + DeepSeek + strategy execution. Triggered by user action."""
    if news_limit == 3 and cache_is_fresh(st.session_state.ai_cache):
        cached = st.session_state.ai_cache
        cached["from_cache"] = True
        return cached

    status = st.status("AI 正在读取全球舆情，请稍等...", expanded=True) if show_status else None

    def write(message: str) -> None:
        if status:
            status.write(message)

    demo_mode = bool(market.get("demo_mode"))
    try:
        news_items = get_bitcoin_news(limit=news_limit)
        write(f"[新闻抓取] NewsAPI 成功抓取 {len(news_items)} 条新闻。")
    except Exception as exc:
        demo_mode = True
        news_items = make_mock_news()
        write(f"[新闻抓取] Demo Mode: NewsAPI 调用失败，使用模拟新闻。{exc}")

    if status:
        with st.expander("前3条原始新闻摘要", expanded=False):
            for index, item in enumerate(news_items[:3], 1):
                st.markdown(f"**{index}. {item.title}**")
                st.caption(item.publisher)
                st.write(item.summary or "No summary.")

    try:
        write("[AI 决策过程] DeepSeek-V3 正在逐条分析：")
        for item in news_items[:5]:
            write(f"- {item.title}")
        ai_result = analyze_news_with_ai(float(market["price"]), news_items)
    except Exception as exc:
        demo_mode = True
        ai_result = make_mock_ai_result()
        write(f"[AI 决策过程] Demo Mode: DeepSeek 调用失败，使用模拟评分。{exc}")

    final_score = calculate_final_score(ai_result.get("items", []))
    write(f"**[AI 决策过程] Final Score: {final_score:.4f}**")

    write("[LSTM 预测] Training cached 2-layer LSTM and forecasting next 24 hours...")
    lstm_prediction, lstm_demo = load_lstm_prediction(market)
    demo_mode = demo_mode or lstm_demo
    write(
        "[LSTM 预测] "
        f"trend={lstm_prediction.trend_signal}; "
        f"24h forecast return={lstm_prediction.forecast_return * 100:.4f}%."
    )

    account = account_from_state()
    price = float(market["price"])
    rsi = float(market["rsi"])
    account.update_highest_price(price)
    stop_loss = should_stop_loss(account, price)
    trailing_stop = should_trailing_stop(account, price)
    buy_ready = (
        final_score > AI_BUY_THRESHOLD
        and rsi < 70
        and lstm_prediction.trend_signal == 1
        and not account.in_position
    )
    sell_ready = account.in_position and (final_score < -0.3 or rsi > 80 or stop_loss or trailing_stop)
    write(
        f"[量化过滤] RSI={rsi:.2f}; buy_ready={buy_ready}; "
        f"sell_ready={sell_ready}; stop_loss={stop_loss}; trailing_stop={trailing_stop}; "
        f"LSTM Trend={lstm_prediction.trend_signal}."
    )

    signal = decide_signal_with_lstm(
        final_score=final_score,
        rsi=rsi,
        lstm_trend=lstm_prediction.trend_signal,
        account=account,
        price=price,
    )
    execute_paper_trade(account, signal, price, rsi, final_score)
    save_account(account)

    result = {
        **market,
        "demo_mode": demo_mode,
        "final_score": final_score,
        "signal": signal,
        "market_view": ai_result.get("market_view", ""),
        "news": ai_result.get("items", []),
        "lstm_prediction": lstm_prediction,
        "lstm_trend": lstm_prediction.trend_signal,
        "lstm_forecast_return": lstm_prediction.forecast_return,
        "net_worth": account.total_assets(price),
        "from_cache": False,
    }
    st.session_state.latest_result = result
    if news_limit == 3:
        st.session_state.ai_cache = {"time": datetime.now(tz=UTC), **result}
    append_trading_log(result)

    if status:
        status.update(label="AI analysis completed.", state="complete", expanded=True)
    return result


# -----------------------------
# Charts and display
# -----------------------------

def rsi_label(rsi: float) -> tuple[str, str]:
    if rsi < 70:
        return "Safe", "safe"
    if rsi <= 80:
        return "Hot", "warn"
    return "Overbought", "danger"


def render_chart(data: pd.DataFrame) -> None:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("BTC Close Price", "RSI(14)"),
    )
    fig.add_trace(go.Scatter(x=data.index, y=data["close"], mode="lines", name="BTC close", line=dict(color="#38BDF8", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI_14"], mode="lines", name="RSI(14)", line=dict(color="#F59E0B", width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#F59E0B", row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="#EF4444", row=2, col=1)
    fig.update_yaxes(fixedrange=False, rangemode="normal", row=1, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111827",
        font=dict(color="#E6EDF7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")


def render_lstm_chart(prediction: LSTMPrediction) -> None:
    data = prediction.prediction_frame
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["actual_close"],
            mode="lines",
            name="Actual close",
            line=dict(color="#38BDF8", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["predicted_close"],
            mode="lines",
            name="LSTM forecast",
            line=dict(color="#A78BFA", width=2, dash="dash"),
        )
    )
    fig.update_yaxes(fixedrange=False, rangemode="normal")
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#111827",
        font=dict(color="#E6EDF7"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")


# -----------------------------
# Page skeleton renders first
# -----------------------------

init_state()
ensure_csv(TRADING_LOG_PATH, ["time", "mode", "price", "change_24h_pct", "rsi", "ai_score", "decision", "market_view", "net_worth", "balance", "position"])
ensure_csv(TRADE_HISTORY_PATH, ["time", "side", "buy_price", "sell_price", "quantity", "ai_score", "rsi", "pnl", "net_worth"])

st.sidebar.header("Decision Rules")
st.sidebar.markdown("**BUY**: `AI Score > 0.056 and RSI(14) < 70 and LSTM Trend == 1`")
st.sidebar.markdown("**SELL**: `AI Score < -0.3 or RSI(14) > 80 or risk stop`")
st.sidebar.markdown("Risk: 2% fixed stop loss; trailing stop after profit > 3% and 1% pullback.")
st.sidebar.header("Runtime Info")
st.sidebar.write("SiliconFlow DeepSeek-V3 + NewsAPI")
st.sidebar.write("Market data: CryptoCompare")
st.sidebar.write("Mode: paper trading")
missing_secrets = sync_secrets_to_environment()
if missing_secrets:
    st.sidebar.warning("Missing secrets: " + ", ".join(missing_secrets))
else:
    st.sidebar.success("Secrets loaded")

st.title("BTC AI Trading Console")
st.caption("Progressive rendering: market data appears first; AI news analysis runs on demand.")

price_slot = st.empty()
chart_slot = st.empty()
ai_slot = st.empty()
log_slot = st.empty()

with price_slot.container():
    c1, c2, c3 = st.columns(3)
    c1.metric("BTC Price", "正在获取最新报价...")
    c2.metric("Net Worth", f"{st.session_state.balance:,.2f} USDT")
    c3.metric("Cumulative Return", "0.00%")

market = load_market_data()
account = account_from_state()
price = float(market["price"])
net_worth = account.total_assets(price)
return_pct = ((net_worth / INITIAL_BALANCE) - 1) * 100

with price_slot.container():
    c1, c2, c3 = st.columns(3)
    c1.metric("BTC Price", f"${price:,.2f}", f"{market['change_24h']:.2f}% 24h")
    c2.metric("Net Worth", f"{net_worth:,.2f} USDT")
    c3.metric("Cumulative Return", f"{return_pct:.2f}%")

with chart_slot.container():
    st.success("技术指标已更新。")
    render_chart(market["price_data"])
    status, css = rsi_label(float(market["rsi"]))
    st.markdown(f'RSI(14): <span class="{css}">{market["rsi"]:.2f} / {status}</span>', unsafe_allow_html=True)
    cached_lstm = st.session_state.lstm_cache
    if cached_lstm and cached_lstm.get("prediction"):
        prediction = cached_lstm["prediction"]
        trend_text = {1: "Bullish", 0: "Sideways", -1: "Bearish"}.get(prediction.trend_signal, "Unknown")
        st.subheader("LSTM 24h Price Forecast")
        st.write(
            f"Trend Signal: `{prediction.trend_signal}` ({trend_text}) | "
            f"Forecast Return: `{prediction.forecast_return * 100:.4f}%` | "
            f"Train Loss: `{prediction.train_loss:.6f}`"
        )
        render_lstm_chart(prediction)
    else:
        st.info("LSTM forecast has not been generated yet. Click Update LSTM Forecast or run AI analysis.")

latest = st.session_state.latest_result
with ai_slot.container():
    st.subheader("AI Deep Insight")
    if latest and latest.get("market_view"):
        if latest.get("demo_mode"):
            st.markdown('<span class="demo-pill">Demo Mode</span>', unsafe_allow_html=True)
        if latest.get("from_cache"):
            st.info("Using cached AI analysis from the last 5 minutes.")
        st.chat_message("assistant").write(latest["market_view"])
        st.metric("AI Final Score", f"{float(latest.get('final_score', 0.0)):.4f}")
        st.metric("Decision", latest.get("signal", "HOLD"))
        with st.expander("AI selected news, scores, and reasons", expanded=True):
            for index, item in enumerate(latest.get("news", []), 1):
                st.markdown(f"**{index}. {item.get('title', '')}**")
                st.write(item.get("summary", ""))
                st.write(f"sentiment: `{item.get('sentiment', 0)}` | weight: `{item.get('weight', 0)}`")
                st.caption(item.get("reason", ""))
    else:
        st.info("AI 正在实时分析中，预计还需 10 秒。点击下方按钮开始舆情分析；你可以先查看价格图表。")

button_col_1, button_col_2, button_col_3 = st.columns(3)
run_quick = button_col_1.button("Analyze Top 3 News", type="primary", width="stretch")
run_full = button_col_2.button("Run Now: Full 20 News Analysis", width="stretch")
run_lstm = button_col_3.button("Update LSTM Forecast", width="stretch")

if run_lstm:
    with st.spinner("Training 2-layer LSTM and forecasting next 24 hours..."):
        load_lstm_prediction(market)
    st.rerun()

if run_quick or run_full:
    news_limit = 20 if run_full else 3
    result = run_ai_analysis(news_limit=news_limit, market=market, show_status=True)
    st.rerun()

st.subheader("Account State")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Cash", f"{st.session_state.balance:,.2f} USDT")
a2.metric("BTC Position", f"{st.session_state.position:.8f}")
a3.metric("Entry Price", f"${st.session_state.entry_price:,.2f}")
a4.metric("Realized PnL", f"{st.session_state.realized_pnl:,.2f} USDT")

tab_trades, tab_runs = st.tabs(["Recent Trades", "Run Log"])
with tab_trades:
    st.dataframe(read_recent_csv(TRADE_HISTORY_PATH), width="stretch", hide_index=True)
with tab_runs:
    st.dataframe(read_recent_csv(TRADING_LOG_PATH), width="stretch", hide_index=True)

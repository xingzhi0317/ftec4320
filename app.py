from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ai_decision_core import (
    analyze_news_with_ai,
    calculate_final_score,
    get_bitcoin_news,
)
from strategy_executor import (
    Account,
    add_rsi,
    decide_signal,
    fetch_btc_hourly_klines,
)


INITIAL_BALANCE = 10_000.0
TRADE_HISTORY_PATH = Path("trade_history.csv")
AUTO_RUN_INTERVAL = timedelta(minutes=5)


st.set_page_config(
    page_title="BTC AI Trading Console",
    page_icon="BTC",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #080b12;
        color: #e6edf7;
    }
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 8px;
        padding: 16px;
    }
    .logic-box {
        background: #101827;
        border: 1px solid #2b3950;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 16px;
        margin: 10px 0 18px 0;
    }
    .panel {
        background: #0f1724;
        border: 1px solid #243044;
        border-radius: 8px;
        padding: 16px;
    }
    .safe { color: #22c55e; font-weight: 700; }
    .warn { color: #f59e0b; font-weight: 700; }
    .danger { color: #ef4444; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "balance": INITIAL_BALANCE,
        "position": 0.0,
        "entry_price": 0.0,
        "highest_price": 0.0,
        "trade_history": [],
        "last_result": None,
        "last_run_at": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def require_secrets() -> bool:
    missing = []
    for key in ("SILICONFLOW_API_KEY", "NEWSAPI_API_KEY"):
        if key not in st.secrets or not st.secrets[key]:
            missing.append(key)

    if missing:
        st.error(
            "Missing required secrets: "
            + ", ".join(missing)
            + ". Please add them to .streamlit/secrets.toml."
        )
        return False
    return True


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


def ensure_trade_history_file() -> None:
    if TRADE_HISTORY_PATH.exists():
        return

    with TRADE_HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
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
            ]
        )


def append_trade(row: dict) -> None:
    ensure_trade_history_file()
    with TRADE_HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
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
        )
        writer.writerow(row)

    st.session_state.trade_history.append(row)


def load_recent_trades(limit: int = 10) -> pd.DataFrame:
    if not TRADE_HISTORY_PATH.exists():
        return pd.DataFrame(
            columns=[
                "time",
                "side",
                "buy_price",
                "sell_price",
                "quantity",
                "ai_score",
                "rsi",
                "pnl",
                "net_worth",
            ]
        )
    return pd.read_csv(TRADE_HISTORY_PATH).tail(limit).iloc[::-1]


def rsi_status(rsi: float) -> tuple[str, str]:
    if rsi < 70:
        return "SAFE", "safe"
    if rsi <= 80:
        return "HOT", "warn"
    return "OVERBOUGHT", "danger"


def run_trading_cycle() -> dict:
    data = fetch_btc_hourly_klines()
    data_with_rsi = add_rsi(data)
    latest = data_with_rsi.dropna(subset=["close", "RSI_14"]).iloc[-1]
    current_price = float(latest["close"])
    rsi = float(latest["RSI_14"])

    news_items = get_bitcoin_news(limit=20)
    ai_result = analyze_news_with_ai(current_price, news_items)
    final_score = calculate_final_score(ai_result["items"])

    account = account_from_state()
    before_position = account.position
    before_entry = account.entry_price
    signal = decide_signal(final_score, rsi, account, current_price)

    if signal == "BUY":
        account.buy_all(current_price)
        append_trade(
            {
                "time": datetime.now(tz=UTC).isoformat(),
                "side": "BUY",
                "buy_price": round(current_price, 2),
                "sell_price": "",
                "quantity": account.position,
                "ai_score": final_score,
                "rsi": round(rsi, 2),
                "pnl": 0.0,
                "net_worth": round(account.total_assets(current_price), 2),
            }
        )
    elif signal == "SELL":
        quantity = before_position
        cost_basis = before_entry * quantity
        sell_value_after_fee = quantity * current_price * 0.999
        pnl = sell_value_after_fee - cost_basis
        account.sell_all(current_price)
        append_trade(
            {
                "time": datetime.now(tz=UTC).isoformat(),
                "side": "SELL",
                "buy_price": round(before_entry, 2),
                "sell_price": round(current_price, 2),
                "quantity": quantity,
                "ai_score": final_score,
                "rsi": round(rsi, 2),
                "pnl": round(pnl, 2),
                "net_worth": round(account.total_assets(current_price), 2),
            }
        )

    save_account(account)
    result = {
        "price_data": data_with_rsi,
        "current_price": current_price,
        "rsi": rsi,
        "signal": signal,
        "final_score": final_score,
        "market_view": ai_result.get("market_view", ""),
        "news": ai_result.get("items", []),
        "net_worth": account.total_assets(current_price),
        "generated_at": datetime.now(tz=UTC),
    }
    st.session_state.last_result = result
    st.session_state.last_run_at = result["generated_at"]
    return result


def maybe_auto_run(enabled: bool) -> None:
    if not enabled:
        return

    components.html(
        """
        <script>
        setTimeout(function() {
            window.parent.location.reload();
        }, 300000);
        </script>
        """,
        height=0,
    )

    last_run_at = st.session_state.last_run_at
    should_run = (
        last_run_at is None
        or datetime.now(tz=UTC) - last_run_at >= AUTO_RUN_INTERVAL
    )
    if should_run and require_secrets():
        with st.spinner("Auto monitor is running the AI trading cycle..."):
            try:
                run_trading_cycle()
                st.success("Auto monitor completed one cycle.")
            except Exception as exc:
                st.error(f"Auto monitor failed: {exc}")


init_state()
ensure_trade_history_file()

st.sidebar.header("Runtime")
st.sidebar.write("SiliconFlow DeepSeek-V3 + NewsAPI")
st.sidebar.write("Market data: CryptoCompare histohour")
st.sidebar.write("Mode: paper trading")

auto_enabled = st.sidebar.toggle("开启自动交易监控", value=False)
maybe_auto_run(auto_enabled)

st.title("BTC AI Trading Console")

st.markdown(
    """
    <div class="logic-box">
    <b>Current decision rules</b><br>
    BUY: AI final_score &gt; 0.4 and RSI(14) &lt; 70.<br>
    SELL: AI final_score &lt; -0.3, or loss &gt; 2%, or RSI(14) &gt; 80.<br>
    Risk: 2% fixed stop loss; trailing stop after profit exceeds 3% and price pulls back 1%.
    </div>
    """,
    unsafe_allow_html=True,
)

top_left, top_mid, top_right = st.columns(3)

if st.button("Run Now", type="primary", use_container_width=True):
    if require_secrets():
        with st.spinner("Fetching data, scoring news, checking RSI, and executing paper trade..."):
            try:
                run_trading_cycle()
                st.success("Trading cycle completed.")
            except Exception as exc:
                st.error(f"Trading cycle failed: {exc}")

result = st.session_state.last_result
account = account_from_state()
current_price = result["current_price"] if result else 0.0
net_worth = account.total_assets(current_price) if result else account.balance
return_pct = ((net_worth / INITIAL_BALANCE) - 1) * 100

top_left.metric("BTC Price", f"${current_price:,.2f}" if result else "Waiting")
top_mid.metric("Net Worth", f"{net_worth:,.2f} USDT")
top_right.metric("Total Return", f"{return_pct:.2f}%")

left_col, right_col = st.columns([6, 4])

with left_col:
    st.subheader("Market & Indicators")
    if result:
        chart_data = result["price_data"][["close"]].rename(columns={"close": "BTC close"})
        st.line_chart(chart_data)
        status, css_class = rsi_status(result["rsi"])
        st.markdown(
            f'RSI(14): <span class="{css_class}">{result["rsi"]:.2f} / {status}</span>',
            unsafe_allow_html=True,
        )
        st.write(f"Latest signal: `{result['signal']}`")
    else:
        st.info("Click Run Now or enable auto monitoring to load market data.")

with right_col:
    st.subheader("AI Deep Insight")
    if result:
        st.markdown(f"**AI final_score:** `{result['final_score']:.4f}`")
        st.write(result["market_view"])
        with st.expander("AI selected news and scores", expanded=True):
            for index, item in enumerate(result["news"], 1):
                st.markdown(f"**{index}. {item.get('title', '')}**")
                st.write(item.get("summary", ""))
                st.write(
                    f"sentiment: `{item.get('sentiment', 0)}` | "
                    f"weight: `{item.get('weight', 0)}`"
                )
                st.caption(item.get("reason", ""))
    else:
        st.info("AI market view will appear after the first run.")

st.subheader("Trading Control & Logs")
status_col, account_col = st.columns(2)
with status_col:
    st.write(f"Auto monitor: `{'ON' if auto_enabled else 'OFF'}`")
    if st.session_state.last_run_at:
        st.write(f"Last run: `{st.session_state.last_run_at.isoformat()}`")
with account_col:
    st.write(f"Cash balance: `{st.session_state.balance:.2f} USDT`")
    st.write(f"Position: `{st.session_state.position:.8f} BTC`")
    st.write(f"Entry price: `{st.session_state.entry_price:.2f}`")

recent_trades = load_recent_trades()
st.dataframe(recent_trades, use_container_width=True, hide_index=True)

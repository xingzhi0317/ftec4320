from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import pandas_ta as ta
import requests


BTC_SYMBOL = "BTC/USD"
CRYPTOCOMPARE_HISTOHOUR_URL = "https://min-api.cryptocompare.com/data/v2/histohour"
FEE_RATE = 0.001
STOP_LOSS_RATE = 0.02
TRAIL_PROFIT_TRIGGER = 0.03
TRAIL_PULLBACK_RATE = 0.01


Signal = Literal["BUY", "SELL", "HOLD"]


@dataclass
class Account:
    balance: float = 10_000.0
    position: float = 0.0
    entry_price: float = 0.0
    highest_price: float = 0.0

    @property
    def in_position(self) -> bool:
        return self.position > 0

    def total_assets(self, current_price: float) -> float:
        return self.balance + self.position * current_price

    def buy_all(self, price: float) -> None:
        if self.in_position or self.balance <= 0:
            return

        available_after_fee = self.balance * (1 - FEE_RATE)
        self.position = available_after_fee / price
        self.balance = 0.0
        self.entry_price = price
        self.highest_price = price

    def sell_all(self, price: float) -> None:
        if not self.in_position:
            return

        gross_value = self.position * price
        self.balance += gross_value * (1 - FEE_RATE)
        self.position = 0.0
        self.entry_price = 0.0
        self.highest_price = 0.0

    def update_highest_price(self, price: float) -> None:
        if self.in_position:
            self.highest_price = max(self.highest_price, price)


def fetch_btc_hourly_klines(hours: int = 100) -> pd.DataFrame:
    params = {
        "fsym": "BTC",
        "tsym": "USD",
        "limit": hours,
    }
    response = requests.get(CRYPTOCOMPARE_HISTOHOUR_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    if payload.get("Response") != "Success":
        message = payload.get("Message", "Unknown CryptoCompare error.")
        raise RuntimeError(f"CryptoCompare request failed: {message}")

    rows = payload.get("Data", {}).get("Data", [])
    if not rows:
        raise RuntimeError("CryptoCompare returned no BTC hourly kline data.")

    data = pd.DataFrame(rows)
    required_columns = ["time", "open", "high", "low", "close"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise RuntimeError(f"CryptoCompare data missing columns: {missing_columns}")

    data = data[required_columns].copy()
    data["time"] = pd.to_datetime(data["time"], unit="s", utc=True)
    data = data.set_index("time")
    data[["open", "high", "low", "close"]] = data[
        ["open", "high", "low", "close"]
    ].astype(float)
    return data.tail(hours)


def add_rsi(data: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    if "close" not in data.columns:
        raise ValueError("Kline data must contain a close column.")

    result = data.copy()
    result[f"RSI_{length}"] = ta.rsi(result["close"], length=length)
    return result


def latest_price_and_rsi(data: pd.DataFrame) -> tuple[float, float]:
    data_with_rsi = add_rsi(data)
    latest = data_with_rsi.dropna(subset=["close", "RSI_14"]).iloc[-1]
    return float(latest["close"]), float(latest["RSI_14"])


def should_stop_loss(account: Account, current_price: float) -> bool:
    return account.in_position and current_price <= account.entry_price * (1 - STOP_LOSS_RATE)


def should_trailing_stop(account: Account, current_price: float) -> bool:
    if not account.in_position:
        return False

    profit_reached = account.highest_price >= account.entry_price * (1 + TRAIL_PROFIT_TRIGGER)
    pulled_back = current_price <= account.highest_price * (1 - TRAIL_PULLBACK_RATE)
    return profit_reached and pulled_back


def decide_signal(final_score: float, rsi: float, account: Account, current_price: float) -> Signal:
    account.update_highest_price(current_price)

    if should_stop_loss(account, current_price):
        return "SELL"
    if should_trailing_stop(account, current_price):
        return "SELL"
    if account.in_position and (final_score < -0.3 or rsi > 80):
        return "SELL"
    if not account.in_position and final_score > 0.4 and rsi < 70:
        return "BUY"
    return "HOLD"


def execute_strategy(final_score: float, market_view: str, account: Account | None = None) -> dict:
    account = account or Account()
    data = fetch_btc_hourly_klines()
    current_price, rsi = latest_price_and_rsi(data)
    signal = decide_signal(final_score, rsi, account, current_price)

    if signal == "BUY":
        account.buy_all(current_price)
    elif signal == "SELL":
        account.sell_all(current_price)

    return {
        "symbol": BTC_SYMBOL,
        "current_price": round(current_price, 2),
        "rsi": round(rsi, 2),
        "final_score": final_score,
        "market_view": market_view,
        "signal": signal,
        "balance": round(account.balance, 2),
        "position": account.position,
        "entry_price": round(account.entry_price, 2),
        "highest_price": round(account.highest_price, 2),
        "total_assets": round(account.total_assets(current_price), 2),
        "in_position": account.in_position,
    }


def print_strategy_result(result: dict) -> None:
    position_status = "IN_POSITION" if result["in_position"] else "FLAT"
    print(f"BTC price: {result['current_price']}")
    print(f"RSI(14): {result['rsi']}")
    print(f"AI final_score: {result['final_score']}")
    print(f"Signal: {result['signal']}")
    print(f"Position status: {position_status}")
    print(f"Balance: {result['balance']}")
    print(f"Position: {result['position']}")
    print(f"Entry price: {result['entry_price']}")
    print(f"Total assets: {result['total_assets']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BTC strategy execution.")
    parser.add_argument("--final-score", type=float, default=0.0)
    parser.add_argument("--market-view", default="")
    args = parser.parse_args()

    print_strategy_result(execute_strategy(args.final_score, args.market_view))

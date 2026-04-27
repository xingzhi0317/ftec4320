from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
import requests
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CRYPTOCOMPARE_HISTOHOUR_URL = "https://min-api.cryptocompare.com/data/v2/histohour"


@dataclass
class LSTMPrediction:
    trend_signal: int
    last_price: float
    forecast_end_price: float
    forecast_return: float
    prediction_frame: pd.DataFrame
    train_loss: float


class BTCPriceLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.output(out[:, -1, :])


def fetch_btc_60d_hourly() -> pd.DataFrame:
    params = {"fsym": "BTC", "tsym": "USD", "limit": 60 * 24}
    response = requests.get(CRYPTOCOMPARE_HISTOHOUR_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("Response") != "Success":
        raise RuntimeError(payload.get("Message", "CryptoCompare request failed."))

    rows = payload.get("Data", {}).get("Data", [])
    if not rows:
        raise RuntimeError("CryptoCompare returned no hourly BTC data.")

    data = pd.DataFrame(rows)
    data = data[["time", "open", "high", "low", "close"]].copy()
    data["time"] = pd.to_datetime(data["time"], unit="s", utc=True)
    data = data.set_index("time")
    data[["open", "high", "low", "close"]] = data[["open", "high", "low", "close"]].astype(float)
    return data.tail(60 * 24)


def minmax_scale(values: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    min_value = float(values.min())
    max_value = float(values.max())
    denominator = max(max_value - min_value, 1e-9)
    return (values - min_value) / denominator, min_value, max_value


def inverse_minmax(value: float, min_value: float, max_value: float) -> float:
    return value * (max_value - min_value) + min_value


def make_sequences(values: torch.Tensor, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    targets = []
    for index in range(len(values) - sequence_length):
        sequences.append(values[index : index + sequence_length])
        targets.append(values[index + sequence_length])
    x = torch.stack(sequences).unsqueeze(-1)
    y = torch.stack(targets).unsqueeze(-1)
    return x.float(), y.float()


def forecast_next_24h(
    model: BTCPriceLSTM,
    scaled_values: torch.Tensor,
    min_value: float,
    max_value: float,
    sequence_length: int,
    horizon: int = 24,
) -> list[float]:
    model.eval()
    window = scaled_values[-sequence_length:].clone().float()
    predictions: list[float] = []

    with torch.no_grad():
        for _ in range(horizon):
            prediction = model(window.view(1, sequence_length, 1)).view(())
            prediction = prediction.clamp(0, 1)
            predictions.append(inverse_minmax(float(prediction), min_value, max_value))
            window = torch.cat([window[1:], prediction.view(1)])

    return predictions


def classify_trend(forecast_return: float, neutral_band: float = 0.003) -> int:
    if forecast_return > neutral_band:
        return 1
    if forecast_return < -neutral_band:
        return -1
    return 0


def train_and_predict_lstm(
    sequence_length: int = 48,
    epochs: int = 8,
    batch_size: int = 64,
    hidden_size: int = 32,
) -> LSTMPrediction:
    torch.manual_seed(42)
    data = fetch_btc_60d_hourly()
    close_values = torch.tensor(data["close"].values, dtype=torch.float32)
    scaled_values, min_value, max_value = minmax_scale(close_values)
    x, y = make_sequences(scaled_values, sequence_length)

    model = BTCPriceLSTM(hidden_size=hidden_size, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    last_loss = 0.0

    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch_x)
        last_loss = epoch_loss / len(loader.dataset)

    forecast = forecast_next_24h(
        model=model,
        scaled_values=scaled_values,
        min_value=min_value,
        max_value=max_value,
        sequence_length=sequence_length,
        horizon=24,
    )
    last_price = float(data["close"].iloc[-1])
    forecast_end_price = float(forecast[-1])
    forecast_return = forecast_end_price / last_price - 1
    trend_signal = classify_trend(forecast_return)

    future_index = [
        data.index[-1] + timedelta(hours=step)
        for step in range(1, 25)
    ]
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
        trend_signal=trend_signal,
        last_price=last_price,
        forecast_end_price=forecast_end_price,
        forecast_return=forecast_return,
        prediction_frame=prediction_frame,
        train_loss=last_loss,
    )


if __name__ == "__main__":
    result = train_and_predict_lstm()
    print(f"trend_signal={result.trend_signal}")
    print(f"last_price={result.last_price:.2f}")
    print(f"forecast_end_price={result.forecast_end_price:.2f}")
    print(f"forecast_return={result.forecast_return * 100:.4f}%")
    print(f"train_loss={result.train_loss:.8f}")

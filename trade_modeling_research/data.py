import os
import json
import re
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
from zoneinfo import ZoneInfo


def cast_timeframe(timeframe):
    pattern = re.compile(r'(\d+)([mhdwM])')
    match = pattern.match(timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe: {timeframe}")
    number = int(match.group(1))
    letter = match.group(2)
    match letter:
        case 'm':
            return TimeFrame(number, TimeFrameUnit.Minute)
        case 'h':
            return TimeFrame(number, TimeFrameUnit.Hour)
        case 'd':
            return TimeFrame(number, TimeFrameUnit.Day)
        case 'w':
            return TimeFrame(number, TimeFrameUnit.Week)
        case 'M':
            return TimeFrame(number, TimeFrameUnit.Month)


def get_stock_data(symbol, start_date: datetime, end_date: datetime, tf: str, cached=True, extended=True):
    load_dotenv()
    stock_client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    timeframe = cast_timeframe(tf)

    cache_dir = "../data"
    os.makedirs(cache_dir, exist_ok=True)

    base_filename = f"{symbol}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{tf}_{'ext' if extended else 'reg'}"
    csv_path = os.path.join(cache_dir, f"{base_filename}.csv")
    json_path = os.path.join(cache_dir, f"{base_filename}.json")

    # Load cached CSV if available
    if cached and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df

    # Fallback to JSON cache
    if cached and os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df["datetime"] = pd.to_datetime(df["timestamp"])
            df.set_index("datetime", inplace=True)
            df.sort_index(inplace=True)
            return df

    # Fetch fresh data
    params = StockBarsRequest(
        start=start_date,
        end=end_date,
        timeframe=timeframe,
        symbol_or_symbols=symbol,
        datafeed=DataFeed.SIP if extended else DataFeed.IEX,
        adjustment=Adjustment.SPLIT
    )

    df = stock_client.get_stock_bars(params).df

    # Convert timestamp to Pacific time
    timestamps = df.index.get_level_values("timestamp").tz_convert(ZoneInfo("US/Pacific"))
    df.index = df.index.set_levels(timestamps, level="timestamp")
    df = df.reset_index()

    # Rename and reorder columns
    df.rename(columns={
        "timestamp": "datetime",
        "volume": "volume",         # ✅ Must be "volume"
        "vwap": "vol_$",
        "trade_count": "trades"
    }, inplace=True)

    column_order = ["datetime", "close", "open", "high", "low", "volume", "vol_$", "trades"]
    df = df[[col for col in column_order if col in df.columns]]

    # Set index and sort for Gym-Trading-Env compatibility
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)

    # Try saving as CSV
    try:
        df.to_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not save CSV, using JSON fallback: {e}")
        json_string = df.to_json(orient="records", indent=4, date_format="iso")
        with open(json_path, "w") as f:
            json.dump(json.loads(json_string), f, indent=4)

    return df



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
from rich import print
import warnings 
import pandas_market_calendars as mcal

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
    stock_client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    )
    timeframe = cast_timeframe(tf)

    cache_dir = "../cache"
    os.makedirs(cache_dir, exist_ok=True)

    base_filename = f"{symbol}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{tf}_{'ext' if extended else 'reg'}"
    csv_path = os.path.join(cache_dir, f"{base_filename}.csv")

    # Load cached CSV if available
    if cached and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        df["datetime"] = df.index  # Keep datetime as column too
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

    if df.empty:
        warnings.warn(f"\nCOULD NOT GET DATA FOR {symbol} FROM {start_date} TO {end_date}")
        return df

    # Flatten MultiIndex and reset
    df = df.reset_index()

    # Create 'datetime' column from 'timestamp'
    df["datetime"] = df["timestamp"]

    # Rename and reorder columns
    df.rename(columns={
        "volume": "volume",         # ✅ Must be "volume"
        "vwap": "vol_$",
        "trade_count": "trades"
    }, inplace=True)

    column_order = ["datetime", "close", "open", "high", "low", "volume", "vol_$", "trades"]
    df = df[[col for col in column_order if col in df.columns]]

    # Set index and keep datetime as a column too
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    df["datetime"] = df.index  # Keep datetime as column again

    # Try saving as CSV
    try:
        df.to_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not save CSV: {e}")

    return df


def get_next_market_open_day(start: datetime, end: datetime) -> datetime | None:
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d")
    )
    if schedule.empty:
        return None 
    first_open_day = schedule.index[0].to_pydatetime()
    return first_open_day.replace(tzinfo=ZoneInfo("US/Eastern")).astimezone(ZoneInfo("US/Pacific"))



def get_dates(tf="4h", cac=True, ext=True):
    with open("../data/processed/dates.json", 'r') as f:
        stocks = json.load(f)
    total_pairs = sum(len(pairs) for pairs in stocks.values())
    count = 0
    for stock in stocks:
        for date_pair in stocks[stock]:
            count+=1
            original_start = datetime.strptime(date_pair[0], '%Y-%m-%d').replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=ZoneInfo("US/Pacific"))
            end_pacific = datetime.strptime(date_pair[1], '%Y-%m-%d').replace(
                hour=23, minute=59, second=59, microsecond=0, tzinfo=ZoneInfo("US/Pacific"))

            adjusted_start = get_next_market_open_day(original_start, end_pacific)
            if adjusted_start is None or adjusted_start > end_pacific:
                continue

            used_start_str = adjusted_start.strftime('%Y-%m-%d')
            used_end_str = end_pacific.strftime('%Y-%m-%d')

            out_file = f"../data/raw_data/{stock}_{used_start_str}_{used_end_str}.pkl"

            if os.path.exists(out_file):
                print(f"[green]File exists: {out_file}, skipping...[/green]")
                continue

            start_utc = adjusted_start.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
            end_utc = end_pacific.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

            df = get_stock_data(stock, start_utc, end_utc, tf, cached=cac, extended=ext)
            print(f"Downloaded {count}/{total_pairs}")
            if not df.empty:
                df.to_pickle(out_file)



def clear_cache():
    folder_path = "../cache"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Only delete files, not subfolders
            os.remove(file_path)

def clear_raw_data():
    folder_path = "../data/raw_data"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Only delete files, not subfolders
            os.remove(file_path)

def clear_render_logs():
    folder_path = "../render_logs"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Only delete files, not subfolders
            os.remove(file_path)

def clear_filtered_out():
    folder_path = "../data/filtered_out"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Only delete files, not subfolders
            os.remove(file_path)

if __name__ == "__main__":
    clear_raw_data()
    get_dates()


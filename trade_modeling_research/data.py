import os
import json
from datetime import datetime
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed, Adjustment
import re
from zoneinfo import ZoneInfo

from datetime import datetime, timedelta

import pandas as pd


def get_stock_data(symbol, start_date: datetime, end_date:datetime, tf:str, cached=True, extended=True):
    #init client
    load_dotenv()
    stock_client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))

    timeframe = cast_timeframe(tf)

    if not os.path.exists('../cache'):
            os.makedirs('../cache')

    filename = f"{symbol}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}_{tf}_{'ext' if extended else 'reg'}.json"

    if cached and os.path.exists("../cache/" + filename):
        with open("../cache/" + filename, "r") as f:
            data = json.load(f)
            df = pd.DataFrame(data)
        return df

    params = StockBarsRequest(
        start=start_date,
        end=end_date,
        timeframe=timeframe,
        symbol_or_symbols=symbol,
        datafeed = DataFeed.SIP if extended else DataFeed.IEX,
        adjustment=Adjustment.SPLIT
    )

    # make req
    df = stock_client.get_stock_bars(params).df
    # Convert the 'timestamp' level of the MultiIndex to Pacific Time
    timestamps = df.index.get_level_values("timestamp")  # Get the timestamp level
    timestamps = timestamps.tz_convert(ZoneInfo('US/Pacific'))  # Convert to Pacific Time
    # timestamps = timestamps.tz_convert(pytz.timezone('US/Pacific'))  # For Python 3.8 and below

    # Reassign the converted timestamps back to the MultiIndex
    df.index = df.index.set_levels(timestamps, level="timestamp")
    json_string = df.to_json()
    with open(f'../cache/{filename}', 'w') as f:
        json.dump(json.loads(json_string), f, indent=4)
    return df


def cast_timeframe(timeframe):
    pattern = re.compile(r'(\d+)([mhdwM])')
    match = pattern.match(timeframe)
    if not match:
        raise ValueError (f"Invalid timeframe: {timeframe}")
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


if __name__ == "__main__":
    # Define the symbol and timeframe
    symbol = "PLTR"
    timeframe = "1h"

    # Define the date range (past 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)

    # Fetch stock data with extended hours
    print(f"Fetching {symbol} data for the past 7 days with a {timeframe} timeframe (extended hours)...")
    get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date, tf=timeframe, extended=True, cached=False)
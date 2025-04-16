import pandas as pd
from datetime import datetime, timedelta
import json
import os
from rich import print


def get_top_10(days=5):
    df = pd.read_csv("../data/quiver_csv/wsb-all(Sheet1).csv")
    print(df.head())
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    df['Mentions'] = pd.to_numeric(df['Mentions'], errors='coerce')
    
    df = df.drop(columns=["YOLO", "Rocket", "Stimulus", "FOMO", "Diamonds", "Moon"])

    end_date = datetime(2025, 2, 21)
    top_10_list = []

    for day in range(days):
        today = end_date - timedelta(days=days - day)
        filtered_df = df[df['Datetime'].dt.date == today.date()]

        if not filtered_df.empty:
            sorted_df = filtered_df.sort_values(by="Mentions", ascending=False)
            sorted_df['Rank'] = sorted_df['Mentions'].rank(ascending=False, method='min').astype(int)

            top_10_list.append(sorted_df.head(10))

    final_top_10_df = pd.concat(top_10_list, ignore_index=True)

    final_top_10_df.to_csv("../data/processed/top_10.csv", index=False)
    

def get_dates(spike=True):
    df = pd.read_csv("../data/processed/top_10.csv")
    print(df.head())

    df['Datetime'] = pd.to_datetime(df['Datetime'])

    stocks = df['Ticker'].unique()
    dates = {stock: [] for stock in stocks}

    for stock in stocks:
        stock_data = sorted(df[df['Ticker'] == stock]['Datetime'].dt.date.unique())
        inv = [None, None]

        for day in stock_data:
            if inv[0] is None:
                inv[0] = day
                inv[1] = day
            elif (day - inv[1]).days <= 4:
                inv[1] = day  # Extend current interval
            else:
                dates[stock].append(inv)
                inv = [day, day]

        if inv[0] is not None:
            dates[stock].append(inv)

    # Convert date objects to strings
    dates_str = {
        stock: [[str(start), str(end)] for start, end in date_ranges]
        for stock, date_ranges in dates.items()
    }

    with open("../data/processed/dates.json", 'w') as f:
        json.dump(dates_str, f, indent=4)

    if spike:
        add_spike_data()


def add_spike_data():
    file_path = "../data/processed/dates.json"

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    
    key_dates = [['2021-01-10', '2021-05-31']]
    stocks = {
        'GME': key_dates,  
        'AMC': key_dates,
        'BB':  key_dates,  
    }
    data.update(stocks)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
 

if __name__ == "__main__":
    get_top_10(5)
    get_dates() 
    add_spike_data()




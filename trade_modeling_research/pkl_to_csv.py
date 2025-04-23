import pandas as pd

df = pd.read_pickle('../data/filtered_out\QUBT_2024-12-18_2024-12-19.pkl')

df.to_csv('./output.csv', index=False) 

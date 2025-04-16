import pandas as pd

# Load the pickle file (replace 'your_file.pkl' with your pickle file path)
df = pd.read_pickle('../render_logs\PLTR_2025-04-10_19-56-23.pkl')

# Save as CSV (replace 'output.csv' with the desired output file path)
df.to_csv('../render_logs/output.csv', index=False)  # Use `index=False` if you don't want to save the index column

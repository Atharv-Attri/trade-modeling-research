import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.style as style

# Use dark background for plots
style.use('dark_background')

# --- Config ---
LOG_DIR = "../render_logs/training_logs"
st.set_page_config(page_title="Training Log Visualizer", layout="wide")

st.title("RL Training Log Visualizer")

# --- Sidebar ---
st.sidebar.header("Select Log File")
log_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".csv")]

if not log_files:
    st.sidebar.warning("No log files found in the directory.")
    st.stop()

selected_file = st.sidebar.selectbox("Available Logs", sorted(log_files))
file_path = os.path.join(LOG_DIR, selected_file)

st.sidebar.markdown("---")
show_native = st.sidebar.checkbox("Use Interactive Streamlit Charts", value=True)

# --- Load and Clean Data ---
@st.cache_data
def load_log(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=["episode_len"], errors="ignore")
    df = df.dropna(axis=1, how='all')
    return df

df = load_log(file_path)

if "timesteps" not in df.columns:
    st.error("This file doesn't contain a 'timesteps' column.")
    st.stop()

x = df["timesteps"]
metrics = [col for col in df.columns if col not in ["timesteps", "learning_rate"]]

st.markdown(f"### Metrics from: `{selected_file}`")
st.markdown("---")

# --- Plotting ---
for metric in metrics:
    with st.container():
        st.markdown(f"#### {metric.replace('_', ' ').title()}")

        if show_native:
            st.line_chart(df.set_index("timesteps")[[metric]])
        else:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, df[metric], linewidth=2, color='cyan')
            ax.set_facecolor('#111111')
            fig.patch.set_facecolor('#111111')
            ax.set_xlabel("Timesteps", color='white')
            ax.set_ylabel(metric, color='white')
            ax.set_title(metric.replace('_', ' ').title(), color='white')
            ax.tick_params(colors='white')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            st.pyplot(fig)

        st.markdown("---")

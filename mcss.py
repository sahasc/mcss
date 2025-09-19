# mcss.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
from io import StringIO

st.set_page_config(page_title="MCSS - Monte Carlo Stock Simulator", layout="wide")
st.title("MCSS — Monte Carlo Stock Price Simulator")

# ----------------- Sidebar controls -----------------
with st.sidebar:
    st.header("Simulation settings")
    ticker = st.text_input("Ticker (e.g. AAPL, RBLX)", value="AAPL").upper().strip()
    simulation_days = st.slider("Days to simulate", min_value=30, max_value=730, value=252, step=1)
    num_simulations = st.slider("Number of simulations", min_value=100, max_value=2000, value=1000, step=100)
    training_days = st.slider("Days to estimate drift/vol", min_value=30, max_value=1000, value=252, step=1)
    top_paths_to_plot = st.slider("Number of highlighted likely paths", min_value=3, max_value=50, value=20, step=1)
    show_percentile_bands = st.checkbox("Show percentile cones (10/25/50/75/90)", value=True)
    run_button = st.button("Run simulation")

# ----------------- Helper: fetch & cache historical data -----------------
@st.cache_data(ttl=3600)
def fetch_history(ticker, start):
    df = yf.download(ticker, start=start, end=str(datetime.date.today()), auto_adjust=True)
    return df

# ----------------- Run when user clicks -----------------
if run_button:
    if not ticker:
        st.error("Please enter a ticker symbol.")
        st.stop()

    # fetch history (use training_days + a bit of headroom)
    start_date = (datetime.date.today() - datetime.timedelta(days=training_days * 2)).isoformat()
    hist = fetch_history(ticker, start=start_date)
    if hist is None or hist.empty:
        st.error(f"No historical data found for {ticker}. Try another ticker.")
        st.stop()

    close = hist["Close"].dropna()
    if len(close) < 10:
        st.error("Not enough data to estimate parameters.")
        st.stop()

    # prepare returns and parameters (use most recent `training_days`)
    log_returns = np.log(close / close.shift(1)).dropna()
    recent_returns = log_returns[-training_days:]
    if recent_returns.empty:
        st.error("Not enough recent returns to estimate drift/volatility.")
        st.stop()

    annual_drift = float(recent_returns.mean() * 252)
    annual_volatility = float(recent_returns.std() * np.sqrt(252))
    S0 = float(close.iloc[-1])

    st.markdown(f"**Current price ({ticker})**: ${S0:,.2f}")
    st.markdown(f"**Estimated annual drift:** {annual_drift:.2%} • **volatility:** {annual_volatility:.2%}")

    # ----------------- Vectorized Monte Carlo -----------------
    dt = 1 / 252
    # We'll build increments of shape (simulation_days, num_simulations)
    # Generate Z ~ N(0,1)
    Z = np.random.normal(size=(simulation_days, num_simulations))
    # Compute increments for GBM: (mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z_t
    increments = (annual_drift - 0.5 * annual_volatility**2) * dt + annual_volatility * np.sqrt(dt) * Z
    # Set first row increments to 0 so day 0 = S0
    increments[0, :] = 0.0
    # Cumulative log price: log(S0) + cumsum(increments)
    logS = np.log(S0) + np.cumsum(increments, axis=0)
    all_paths = np.exp(logS)  # shape: (simulation_days, num_simulations)

    # ----------------- Statistics & selection -----------------
    final_prices = all_paths[-1, :]
    mean_final = np.mean(final_prices)
    dist_from_mean = np.abs(final_prices - mean_final)

    # Choose top likely paths (closest final price to mean)
    top_k = min(top_paths_to_plot, num_simulations)
    most_likely_idx = np.argsort(dist_from_mean)[:top_k]
    highlight_paths = all_paths[:, most_likely_idx]

    # compute colors: normalized distances among chosen top_k
    chosen_dist = dist_from_mean[most_likely_idx]
    # smaller distance => more likely => greener -> val close to 0 -> green
    normalized = (chosen_dist - chosen_dist.min()) / (chosen_dist.max() - chosen_dist.min() + 1e-9)
    colors_rgb = [f"rgb({int(255*v)},{int(255*(1-v))},0)" for v in normalized]  # red->green mapping

    # mean path and std path for bands
    mean_path = np.mean(all_paths, axis=1)
    std_path = np.std(all_paths, axis=1)

    # percentiles
    percentiles = [10, 25, 50, 75, 90]
    percentile_paths = np.percentile(all_paths, percentiles, axis=1)  # shape (len(percentiles), days)

    # ----------------- Build interactive Plotly figure -----------------
    days = np.arange(simulation_days)

    fig = go.Figure()

    # optional: percentile cones (outer first then inner so shading stacks nicely)
    if show_percentile_bands:
        # top 90 then 10 fill
        fig.add_trace(go.Scatter(
            x=days, y=percentile_paths[4], mode='lines',
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=days, y=percentile_paths[0], mode='lines',
            line=dict(width=0), fill='tonexty',
            fillcolor='rgba(200,200,200,0.15)', showlegend=True, name='10-90 percentile', hoverinfo='skip'
        ))
        # 75-25 band
        fig.add_trace(go.Scatter(x=days, y=percentile_paths[3], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=days, y=percentile_paths[1], mode='lines', line=dict(width=0), fill='tonexty',
                                 fillcolor='rgba(150,150,150,0.2)', showlegend=True, name='25-75 percentile', hoverinfo='skip'))

    # highlight top likely paths (green to red)
    for i in range(top_k):
        fig.add_trace(go.Scatter(
            x=days,
            y=highlight_paths[:, i],
            mode='lines',
            line=dict(color=colors_rgb[i], width=2),
            name=f'Likely path {i+1}',
            hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
        ))

    # mean path dashed
    fig.add_trace(go.Scatter(
        x=days, y=mean_path, mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Expected mean path',
        hovertemplate='Day %{x}: $%{y:.2f}<extra></extra>'
    ))

    # ±1 std band as shading
    fig.add_trace(go.Scatter(x=days, y=mean_path + std_path, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=days, y=mean_path - std_path, mode='lines', line=dict(width=0), fill='tonexty',
                             fillcolor='rgba(100,100,100,0.15)', name='±1 Std Dev', hoverinfo='skip'))

    # current price horizontal
    fig.add_trace(go.Scatter(x=days, y=[S0] * simulation_days, mode='lines', line=dict(color='blue', dash='dash'),
                             name='Current price', hovertemplate='Price: $%{y:.2f}<extra></extra>'))

    fig.update_layout(title=f"Monte Carlo Future Price Prediction — {ticker}",
                      xaxis_title="Days from today",
                      yaxis_title="Price",
                      hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    st.plotly_chart(fig, use_container_width=True)

    # ----------------- Show summary statistics -----------------
    st.markdown("### Summary statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Simulated mean final price", f"${mean_final:,.2f}")
    col2.metric("Simulated median final price", f"${np.median(final_prices):,.2f}")
    col3.metric("Simulated std of final price", f"${np.std(final_prices):,.2f}")

    # Percentage of simulated final prices within ±X% of mean (example)
    pct_within_10 = np.mean(np.abs(final_prices - mean_final) <= 0.10 * mean_final) * 100
    st.write(f"Percent of simulated finals within ±10% of mean: {pct_within_10:.1f}%")

    # ----------------- Download simulated paths CSV -----------------
    all_paths_df = pd.DataFrame(all_paths)
    all_paths_df.index.name = "day"
    csv = all_paths_df.to_csv(index=True)
    st.download_button(
        label="Download all simulated paths (CSV)",
        data=csv,
        file_name=f"{ticker}_simulated_paths.csv",
        mime="text/csv"
    )

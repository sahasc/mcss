# monte_carlo_streamlit.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Monte Carlo Stock Price Predictor")

# ------------------------------
# User input
# ------------------------------
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, RBLX, TSLA):", value="AAPL").upper()
num_simulations = st.slider("Number of simulations:", 50, 500, 200)
days = st.slider("Number of trading days to simulate:", 30, 504, 252)

if ticker:
    # ------------------------------
    # Fetch historical data
    # ------------------------------
    data = yf.download(ticker, period="3y")
    if data.empty:
        st.error(f"No historical data found for {ticker}. Try another ticker.")
    else:
        close_prices = data["Close"]
        log_returns = np.log(1 + close_prices.pct_change().dropna())
        mu = log_returns.mean()
        sigma = log_returns.std()
        S0 = float(close_prices.iloc[-1])

        st.write(f"Starting price for {ticker}: {S0:.2f}")
        st.write(f"Estimated drift: {mu*252:.2%}, volatility: {sigma*np.sqrt(252):.2%}")

        # ------------------------------
        # Monte Carlo Simulation
        # ------------------------------
        simulations = np.zeros((days, num_simulations))
        simulations[0] = S0

        for t in range(1, days):
            Z = np.random.standard_normal(num_simulations)
            simulations[t] = simulations[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

        # Percentiles
        percentiles = np.percentile(simulations, [10, 50, 90], axis=1)

        # ------------------------------
        # Plot Monte Carlo Paths
        # ------------------------------
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        # Color paths based on final percentile
        final_prices = simulations[-1, :]
        ranks = pd.qcut(final_prices, num_simulations, labels=False)

        for i in range(num_simulations):
            color = plt.cm.coolwarm(ranks[i] / num_simulations)  # red = low, blue = high
            ax1.plot(simulations[:, i], color=color, alpha=0.2)

        # Overlay percentile bands
        ax1.plot(percentiles[1], "k--", linewidth=2, label="Median")
        ax1.plot(percentiles[0], "g--", linewidth=1.5, label="10th Percentile")
        ax1.plot(percentiles[2], "r--", linewidth=1.5, label="90th Percentile")
        ax1.axhline(y=S0, color="blue", linestyle="--", linewidth=1, label="Starting Price")

        ax1.set_title(f"Monte Carlo Simulation for {ticker} ({num_simulations} paths, {days} days)")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Price")
        ax1.legend()
        st.pyplot(fig1)  # ✅ Important: this makes the plot appear in Streamlit

        # ------------------------------
        # Histogram of final prices
        # ------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.hist(simulations[-1, :], bins=30, color="skyblue", edgecolor="black")
        ax2.axvline(S0, color="blue", linestyle="--", label="Starting Price")
        ax2.set_title(f"Distribution of Final Prices after {days} days ({ticker})")
        ax2.set_xlabel("Price")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)  # ✅ Important: this makes the histogram appear
        
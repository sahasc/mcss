# monte_carlo_streamlit_interactive_page_annotation.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.title("Monte Carlo Stock Price Predictor (Interactive & Enhanced Visualization)")

# ------------------------------
# User input
# ------------------------------
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, RBLX, TSLA):", value="AAPL").upper()
num_simulations = st.slider("Number of simulations:", 50, 500, 200)
days = st.slider("Number of trading days to simulate:", 30, 504, 252)
training_years = st.slider("Years of historical data for drift/volatility:", 1, 10, 3)

chart_bg_color = "#f5f5f5"  # light gray background

if ticker:
    try:
        # ------------------------------
        # Fetch historical data
        # ------------------------------
        data = yf.download(ticker, period=f"{training_years}y")
        if data.empty:
            st.error(f"No historical data found for {ticker}. Try another ticker.")
        else:
            close_prices = data["Close"]
            log_returns = np.log(1 + close_prices.pct_change().dropna())
            mu = float(log_returns.mean())
            sigma = float(log_returns.std())
            S0 = float(close_prices.tail(1).iloc[0])

            st.write(f"Starting price for {ticker}: {S0:.2f}")
            st.write(f"Estimated annualized drift: {mu*252:.2%}, volatility: {sigma*np.sqrt(252):.2%}")

            # ------------------------------
            # Monte Carlo Simulation
            # ------------------------------
            simulations = np.zeros((days, num_simulations))
            simulations[0] = S0

            for t in range(1, days):
                Z = np.random.standard_normal(num_simulations)
                simulations[t] = simulations[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * Z)

            percentiles = np.percentile(simulations, [10, 50, 90], axis=1)

            # ------------------------------
            # Backtest Accuracy
            # ------------------------------
            backtest_days = 252
            if len(close_prices) > backtest_days:
                hist_start = float(close_prices.iloc[-backtest_days])
                hist_end = float(close_prices.iloc[-1])
                hist_return = (hist_end / hist_start - 1) * 100
                sim_end_prices = simulations[-1, :]
                avg_sim_return = (np.mean(sim_end_prices) / S0 - 1) * 100
                st.write(f"Historical 1Y return: {hist_return:.2f}%")
                st.write(f"Simulated 1Y avg return: {avg_sim_return:.2f}%")

            # ------------------------------
            # Monte Carlo Paths Plot (Stacked)
            # ------------------------------
            fig_paths = go.Figure()
            final_prices = simulations[-1, :]
            try:
                ranks = pd.qcut(final_prices, num_simulations, labels=False)
            except ValueError:
                ranks = np.linspace(0, num_simulations-1, num_simulations)

            for i in range(num_simulations):
                color = f"rgba({int(255*(1-ranks[i]/num_simulations))}, {int(255*(ranks[i]/num_simulations))}, 0, 0.3)"
                fig_paths.add_trace(go.Scatter(y=simulations[:, i], mode='lines', line=dict(color=color), showlegend=False))

            # Percentile lines
            fig_paths.add_trace(go.Scatter(y=percentiles[1], mode='lines', line=dict(color='black', dash='dash'), name='Median'))
            fig_paths.add_trace(go.Scatter(y=percentiles[0], mode='lines', line=dict(color='green', dash='dash'), name='10th Percentile'))
            fig_paths.add_trace(go.Scatter(y=percentiles[2], mode='lines', line=dict(color='red', dash='dash'), name='90th Percentile'))
            fig_paths.add_hline(y=S0, line_dash="dash", line_color="blue", annotation_text="Starting Price")

            fig_paths.update_layout(title=f"Monte Carlo Simulation ({num_simulations} paths)",
                                    xaxis_title="Days", yaxis_title="Price",
                                    hovermode="x unified",
                                    plot_bgcolor=chart_bg_color, paper_bgcolor=chart_bg_color,
                                    font=dict(color="black"))
            st.plotly_chart(fig_paths, use_container_width=True)

            # ------------------------------
            # Histogram Plot
            # ------------------------------
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=simulations[-1, :], nbinsx=30, name='Final Prices', marker_color='skyblue'))
            fig_hist.add_vline(x=S0, line_dash="dash", line_color="blue", annotation_text="Starting Price")
            fig_hist.update_layout(title=f"Distribution of Final Prices",
                                   xaxis_title="Price", yaxis_title="Frequency",
                                   plot_bgcolor=chart_bg_color, paper_bgcolor=chart_bg_color,
                                   font=dict(color="black"))
            st.plotly_chart(fig_hist, use_container_width=True)

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# ------------------------------
# Bottom-right page annotation
# ------------------------------
st.markdown(
    "<div style='text-align: right; color: gray; font-size:12px;'>Made by Sahas Chekuri 2025©</div>",
    unsafe_allow_html=True
)

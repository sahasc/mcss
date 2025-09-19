import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Monte Carlo Stock Predictor
# ------------------------------

def monte_carlo_simulation(ticker, num_simulations=200, days=252, training_years=3):
    """
    Runs Monte Carlo simulation with backtest accuracy and percentile bands.
    """

    # Download historical data
    data = yf.download(ticker, period=f"{training_years}y")
    if data.empty:
        print(f"No historical data found for {ticker}. Try another ticker.")
        return

    close_prices = data["Close"]
    log_returns = np.log(1 + close_prices.pct_change().dropna())

    # Estimate drift and volatility
    mu = log_returns.mean()
    sigma = log_returns.std()

    # Starting price
    S0 = float(close_prices.iloc[-1])
    print(f"Starting price for {ticker}: {S0:.2f}")
    print(f"Estimated drift: {mu*252:.2%}, volatility: {sigma*np.sqrt(252):.2%}")

    # Run simulations
    simulations = np.zeros((days, num_simulations))
    for i in range(num_simulations):
        prices = [S0]
        for d in range(1, days):
            shock = np.random.normal(mu, sigma)
            prices.append(prices[-1] * np.exp(shock))
        simulations[:, i] = prices

    # Calculate percentiles
    percentiles = np.percentile(simulations, [10, 50, 90], axis=1)

    # ------------------------------
    # Backtest Accuracy (optional)
    # ------------------------------
    backtest_days = 252  # compare 1 year back
    if len(close_prices) > backtest_days:
        hist_start = float(close_prices.iloc[-backtest_days])
        hist_end = float(close_prices.iloc[-1])
        hist_return = (hist_end / hist_start - 1) * 100

        sim_end_prices = simulations[-1, :]
        avg_sim_return = (np.mean(sim_end_prices) / S0 - 1) * 100

        print(f"Historical 1Y return: {hist_return:.2f}%")
        print(f"Simulated 1Y avg return: {avg_sim_return:.2f}%")

    # ------------------------------
    # Plotting
    # ------------------------------
    plt.figure(figsize=(12, 7))

    # Color paths by final percentile
    final_prices = simulations[-1, :]
    ranks = pd.qcut(final_prices, num_simulations, labels=False)

    for i in range(num_simulations):
        color = plt.cm.coolwarm(ranks[i] / num_simulations)  # red = low, blue = high
        plt.plot(simulations[:, i], color=color, alpha=0.2)

    # Overlay percentile bands
    plt.plot(percentiles[1], "k--", linewidth=2, label="Median")
    plt.plot(percentiles[0], "g--", linewidth=1.5, label="10th Percentile")
    plt.plot(percentiles[2], "r--", linewidth=1.5, label="90th Percentile")

    plt.axhline(y=S0, color="blue", linestyle="--", linewidth=1, label="Starting Price")

    plt.title(f"Monte Carlo Simulation for {ticker} ({num_simulations} paths, {days} days)")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Histogram of final simulated prices
    plt.figure(figsize=(10, 5))
    plt.hist(simulations[-1, :], bins=30, color="skyblue", edgecolor="black")
    plt.axvline(S0, color="blue", linestyle="--", label="Starting Price")
    plt.title(f"Distribution of Final Prices after {days} days ({ticker})")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# ------------------------------
# Run the Simulation
# ------------------------------
if __name__ == "__main__":
    ticker = input("Enter a stock ticker (e.g., AAPL, RBLX, TSLA): ").upper()
    monte_carlo_simulation(ticker)

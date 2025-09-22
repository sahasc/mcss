import os
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import openai

# ==============================
# 🔑 Setup API Keys
# ==============================
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY. Please set it in your .env file.")

# ==============================
# 📊 Data Fetching
# ==============================
def fetch_data(ticker):
    df = yf.download(ticker, start="2023-01-01", end="2025-01-01")

    # Moving Averages
    df["MA_10"] = df["Close"].rolling(10).mean().ffill()
    df["MA_50"] = df["Close"].rolling(50).mean().ffill()

    # RSI Calculation
    delta = df["Close"].diff().dropna()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(14).mean().ffill()
    avg_loss = pd.Series(loss).rolling(14).mean().ffill()

    rs = avg_gain / avg_loss
    df.loc[delta.index, "RSI"] = 100 - (100 / (1 + rs))

    return df

# ==============================
# 📈 Simulation
# ==============================
def simulate_strategy(data):
    last_price = data["Close"].iloc[-1]
    returns = data["Close"].pct_change().dropna()

    avg_return = returns.mean()
    volatility = returns.std()

    return {
        "last_price": last_price,
        "avg_return": avg_return,
        "volatility": volatility,
    }

# ==============================
# 🤖 GPT Analysis
# ==============================
def analyze_with_gpt(sim_data):
    prompt = f"""
    Analyze the stock simulation data:
    - Last Price: {sim_data['last_price']:.2f}
    - Average Return: {sim_data['avg_return']:.2%}
    - Volatility: {sim_data['volatility']:.2%}

    Provide a concise professional summary of the stock's performance.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Lightweight GPT model
        messages=[
            {"role": "system", "content": "You are a financial analyst."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()

# ==============================
# 🌐 Streamlit App
# ==============================
def main():
    st.title("📊 Market Condition Simulation System (MCSS)")

    ticker = st.text_input("Enter Stock Ticker:", "AAPL")

    if st.button("Run Analysis"):
        with st.spinner("Fetching data and running simulation..."):
            df = fetch_data(ticker)
            sim_data = simulate_strategy(df)
            analysis_text = analyze_with_gpt(sim_data)

        st.subheader("📈 Simulation Data")
        st.write(sim_data)

        st.subheader("🧠 AI Analysis")
        st.write(analysis_text)

        st.subheader("📊 Price Chart")
        st.line_chart(df[["Close", "MA_10", "MA_50"]])

if __name__ == "__main__":
    main()

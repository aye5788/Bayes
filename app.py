import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import time

# Function to fetch intraday data
def fetch_intraday_data(ticker: str, interval: str = '1m'):
    """
    Fetches intraday data for the given ticker using yfinance.
    Args:
        ticker (str): The stock ticker symbol.
        interval (str): Data interval ('1m', '2m', '5m', etc.).
    Returns:
        DataFrame: A DataFrame containing intraday price data.
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d', interval=interval)
    return data

# Function to perform Bayesian posterior updates
def update_posterior(prior_alpha, prior_beta, observed_returns):
    """
    Updates the posterior probability of finishing higher or lower based on observed returns.
    Args:
        prior_alpha (float): Prior alpha parameter for Beta distribution.
        prior_beta (float): Prior beta parameter for Beta distribution.
        observed_returns (list): List of observed intraday returns.
    Returns:
        dict: Updated posterior probabilities and parameters.
    """
    successes = sum(1 for r in observed_returns if r > 0)  # Positive returns
    failures = sum(1 for r in observed_returns if r <= 0)  # Non-positive returns

    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + failures

    prob_higher = beta.mean(posterior_alpha, posterior_beta)
    prob_lower = 1 - prob_higher

    return {
        "posterior_alpha": posterior_alpha,
        "posterior_beta": posterior_beta,
        "prob_higher": prob_higher,
        "prob_lower": prob_lower
    }

# Function to plot the posterior distribution
def plot_posterior(alpha, beta_param):
    """
    Plots the Beta distribution (posterior) based on alpha and beta parameters.
    """
    x = np.linspace(0, 1, 100)
    y = beta.pdf(x, alpha, beta_param)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=f'Beta({alpha}, {beta_param})')
    plt.title('Posterior Probability Distribution')
    plt.xlabel('Probability of Closing Higher')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot(plt.gcf())

# Streamlit app starts here
st.title("Real-Time Bayesian Probability Tracker")
st.markdown("### Estimate the probability of a stock finishing higher or lower in a daily session.")

# Ticker input
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, TSLA):", "AAPL")

# Parameters for Bayesian inference
alpha = st.sidebar.number_input("Prior Alpha (Successes)", min_value=1, value=1)
beta_param = st.sidebar.number_input("Prior Beta (Failures)", min_value=1, value=1)

# Update interval (in seconds)
update_interval = st.sidebar.number_input("Update Interval (seconds)", min_value=10, value=60)

# Run the tracker
if st.button("Start Tracking"):
    st.write(f"Fetching intraday data for `{ticker}`...")
    while True:
        try:
            # Fetch data and calculate returns
            data = fetch_intraday_data(ticker)
            returns = data['Close'].pct_change().dropna()

            # Update posterior probabilities
            posterior = update_posterior(alpha, beta_param, returns)
            alpha, beta_param = posterior["posterior_alpha"], posterior["posterior_beta"]

            # Display real-time updates
            st.write(f"Time: {pd.Timestamp.now()}")
            st.write(f"**Current Price:** ${data['Close'].iloc[-1]:.2f}")
            st.write(f"**Probability of Closing Higher:** {posterior['prob_higher']:.2%}")
            st.write(f"**Probability of Closing Lower:** {posterior['prob_lower']:.2%}")

            # Plot posterior distribution
            plot_posterior(alpha, beta_param)

            # Wait for the next update
            time.sleep(update_interval)

        except Exception as e:
            st.error(f"Error fetching data or updating probabilities: {e}")
            break

import yfinance as yf
import streamlit as st
from datetime import datetime
import time

# Initialize a global counter for tracking data pulls
if 'yf_count' not in st.session_state:
    st.session_state.yf_count = 0
if 'api_call_count' not in st.session_state:
    st.session_state.api_call_count = 0

@st.cache_data(ttl=86400)  # Time-to-live in seconds (24 hours)
def yahoo_finance_load(ticker):
    ticker_obj = yf.Ticker(ticker)
    
    # Increment the API call counter for actual API calls
    st.session_state.api_call_count += 1
    st.write(f"API call count (not cached): {st.session_state.api_call_count}")

    # Get stats from Yahoo Finance
    stats = ticker_obj.info

    return stats

# List of tickers
tickers_list = ['AAPL', 'ABEV']

# Initialize the company facts metrics dictionary
companyfacts_metrics_yahoo_dict = {}

# Append data to the dictionary based on each ticker
for ticker in tickers_list:
    yahoo_company_data = yahoo_finance_load(ticker)
    companyfacts_metrics_yahoo_dict[ticker] = {}
    for key, value in yahoo_company_data.items():
        if key != 'companyOfficers':
            companyfacts_metrics_yahoo_dict[ticker][key] = value
    time.sleep(0.1)  # Delay of 1/10 second

# Display the updated dictionary
st.write(companyfacts_metrics_yahoo_dict)

# Display the API call count
st.write(f"Total API calls made: {st.session_state.api_call_count}")


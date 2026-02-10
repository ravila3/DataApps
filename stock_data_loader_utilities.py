import psycopg2
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from icecream import ic
from streamlit import session_state as ss
# from snowflake.snowpark import Session
# from snowflake.snowpark.context import get_active_session
# from snowflake.snowpark.functions import col
from datetime import datetime,timedelta
import time
import requests
import os
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# from sec_api import MappingApi
# List of CIKs
ciks = ['21344', '789019', '1090872']  # Example CIKs

@st.cache_data(ttl="24h")
def retrieve_snowflake_data(sql):
    start_time = time.time()
    # connection_parameters = st.secrets["snowflake"]
    # session = Session.builder.configs(connection_parameters).create()
    conn = snowflake.connector.connect(**st.secrets["snowflake"])
    # st.write(f"sql = {sql}") #debug
    df = pd.read_sql(sql,conn)
    conn.close()
    end_time = time.time() # End the timer
    elapsed_time = end_time - start_time
    print(f"Time taken to complete the function: {elapsed_time:.2f} seconds")
    return df

def write_snowflake_data(df,table_name,create_table_query,):
    # connection_params = st.secrets["snowflake_write"]
    # st.write(connection_params)

    # session = Session.builder.configs(connection_params).create()
    # conn = snowflake.connector.connect(**st.secrets["snowflake_write"])

    # Load private key details from secrets
    private_key_passphrase = st.secrets["snowflake_write"]["private_key_passphrase"]
    private_key_path = st.secrets["snowflake_write"]["private_key_path"]

    # Load the private key
    with open(private_key_path, 'rb') as key_file:
        p_key = serialization.load_pem_private_key(
            key_file.read(),
            password=private_key_passphrase.encode(),
            backend=default_backend()
        )
        pkb = p_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

    # Create a new dictionary that includes the private key
    connection_params = {
        "user": st.secrets["snowflake_write"]["user"],
        "account": st.secrets["snowflake_write"]["account"],
        "warehouse": st.secrets["snowflake_write"]["warehouse"],
        "database": st.secrets["snowflake_write"]["database"],
        "schema": st.secrets["snowflake_write"]["schema"],
        "role": st.secrets["snowflake_write"]["role"],
        "private_key": pkb
    }

    # st.write(connection_params) #debug
    # Connect to Snowflake
    conn = snowflake.connector.connect(**connection_params)

    # # Add the private key to the connection parameters
    # connection_params = st.secrets["snowflake_write"]
    # connection_params["private_key"] = pkb

    # # Connect to Snowflake
    # conn = snowflake.connector.connect(**connection_params)

    # Write the DataFrame to Snowflake
    conn.cursor().execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.cursor().execute(create_table_query)
    
    st.write(create_table_query) #debug
    df.columns = df.columns.str.upper()
    df.replace('Infinity', np.nan, inplace=True)

    success, nchunks, nrows, _ = write_pandas(conn, df, table_name.upper()) 
    
    # Check the result
    if success:
        st.write(f"Successfully wrote {nrows} rows to Snowflake.")
        conn.cursor().execute(f"GRANT SELECT ON TABLE {table_name.upper()} TO ROLE READ_ONLY_ROLE")
    else:
        st.write("Failed to write data to Snowflake.")
        
    # Close the connection
    conn.close()
    return

@st.cache_data(ttl="24h")
def get_tickers_marketstack(ticker_list):

    # Replace with your Marketstack API key - free key only allows 100 api requests
    api_key = '641f5f1d463000e9d6215fba76652251'
    base_url = "https://api.marketstack.com/v1/tickers"

    tickers = {}
    offset = 0
    limit = 100  # Number of results per page
    
    for exchange in ['NYSE','NASDAQ']:
        
        while True:
            url = f"{base_url}?access_key={api_key}&exchange={exchange}&limit={limit}&offset={offset}"
            response = requests.get(url)
            data = response.json()

            # Debugging information
            st.write('first stock in data node:', data['data'][0])

            # Extract symbols and names
            if 'data' in data:
                for stock in data['data']:
                    try:
                        symbol = stock['symbol']
                        name = stock['name']
                        exchange_acronym = stock.get('stock_exchange', {}).get('acronym', '')
                        tickers[symbol] = {'name': name, 'exchange_acronym': exchange_acronym}
                    except (KeyError, TypeError) as e:
                        st.write(f"Error: {e}")

            # Check if there's another page
            if 'pagination' in data:
                total = data['pagination'].get('total',None)
                limit = data['pagination'].get('limit',None)
                offset += limit
            else:
                break

    return tickers


# Function to get ticker symbols SEC EDGAR using CIKs with rate limiting
def get_tickers_sec(ciks, delay=0.2):
    tickers = []
    headers = {'User-Agent': 'AI Analytics & Development (rafaelavila3@gmail.com)'}

    for cik in ciks:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            company_data = response.json()
            ticker = company_data.get('ticker', None)
            if ticker:
                tickers.append(ticker)
        # Introduce a delay between API calls
        time.sleep(delay)
    return tickers

@st.cache_data(ttl=86400) # 86400 is 24 hrs in seconds 
def yahoo_finance_load(ticker):
    
    if 'yf_count' not in ss:
        ss.yf_count=0

    today = datetime.today().date()
    try:
        ticker_obj = yf.Ticker(ticker)
        stats=ticker_obj.info
        time.sleep(0.2)  # brief pause to avoid rate limiting
    except ValueError as e:
        stats = None 
        st.write(f"Error fetching info: {e}")

    # st.write(ticker_obj.info) #debug

    # historical_prices=ticker_obj.history(period='1d',start='2019-01-01',end=today) # gets following fields: Date, High, Low, Close, Volume, Dividends, Stock Splits
    # # st.write(f"historical_prices={historical_prices}") #debug
    # historical_prices.reset_index(inplace=True)
    # selected_columns = historical_prices[["Date", "Close", "Dividends", "Stock Splits"]].copy()
    # historical_prices_df = selected_columns
    # historical_prices_df.rename(columns={"Date":"date","Close": "stock_price","Dividends":"dividends","Stock Splits":"stock_splits"}, inplace=True)
    # historical_prices_df["ticker"] = ticker
    # historical_prices_df['date'] = historical_prices_df['date'].dt.date  # Extract the date part
    # historical_prices_df = historical_prices_df.reindex(columns=['ticker','date',"stock_price","dividends","stock_splits"])
    
    ss.yf_count += 1 # track how many times the data was pulled from Yahoo Finance, not cached
    # print(f"ss.yf_count = {ss.yf_count}") #debug

    # st.write(ticker_obj.info) #debug

    return stats #, historical_prices_df, companyfacts_metrics_yahoo_dict

def yahoo_finance_df_format():

    CompanyInfoYahoo = [
    ('symbol','text'),
    ('shortName','text'),
    ('longName','text'),
    ('firstTradeDateEpochUtc','epoch'),
    ('website','text'),
    ('industry','text'),
    ('industryKey','text'),
    ('industryDisp','text'),
    ('sector','text'),
    ('longBusinessSummary','text'),
    ('fullTimeEmployees','integer'),
    ('irWebsite','text'),
    ('previousClose','decimal' ),
    ('beta','decimal' ),
    ('trailingPE','decimal' ),
    ('forwardPE','decimal' ),
    ('volume','integer'),
    ('averageDailyVolume10Day','integer'),
    ('fiftyTwoWeekLow','decimal' ),
    ('fiftyTwoWeekHigh','decimal' ),
    ('priceToSalesTrailing12Months','decimal' ),
    ('enterpriseValue','integer'),
    ('profitMargins','decimal' ),
    ('sharesOutstanding','integer'),
    ('sharesShort','integer'),
    ('sharesShortPriorMonth','integer'),
    ('heldPercentInsiders','decimal' ),
    ('heldPercentInstitutions','decimal' ),
    ('shortRatio','decimal'),
    ('bookValue','integer'),
    ('priceToBook','decimal' ),
    ('lastFiscalYearEnd','epoch'),
    ('nextFiscalYearEnd','epoch'),
    ('trailingEps','decimal' ),
    ('forwardEps','decimal' ),
    ('enterpriseToRevenue','decimal' ),
    ('enterpriseToEbitda','decimal' ),
    ('52WeekChange','decimal' ),
    ('numberOfAnalystOpinions','integer'),
    ('recommendationKey','text'),
    ('totalCash','integer'),
    ('totalCashPerShare','decimal' ),
    ('ebitda','integer'),
    ('totalDebt','integer'),
    ('quickRatio','decimal' ),
    ('currentRatio','decimal' ),
    ('totalRevenue','integer'),
    ('debtToEquity','decimal' ),
    ('revenuePerShare','decimal' ),
    ('returnOnAssets','decimal' ),
    ('returnOnEquity','decimal' ),
    ('revenueGrowth','decimal' ),
    ('freeCashflow','integer'),
    ('operatingCashflow','integer'),
    ('grossMargins','decimal' ),
    ('ebitdaMargins','decimal' ),
    ('operatingMargins','decimal' ),
    ('trailingPegRatio','decimal' )
    ]

    companyfacts_metrics_yahoo_dict = {metric[0]: {'format': metric[1]} for metric in CompanyInfoYahoo}

    return companyfacts_metrics_yahoo_dict

def postgresql_write(df,table_name,create_table_query):
    # connection_params = st.secrets["postgresql"]
    # st.write(connection_params)

    # # Load private key details from secrets
    # private_key_passphrase = st.secrets["postgresql"]["private_key_passphrase"]
    # private_key_path = st.secrets["postgresql"]["private_key_path"]

    # # Load the private key
    # with open(private_key_path, 'rb') as key_file:
    #     p_key = serialization.load_pem_private_key(
    #         key_file.read(),
    #         password=private_key_passphrase.encode(),
    #         backend=default_backend()
    #     )
    #     pkb = p_key.private_bytes(
    #         encoding=serialization.Encoding.DER,
    #         format=serialization.PrivateFormat.PKCS8,
    #         encryption_algorithm=serialization.NoEncryption()
    #     )

    # Create a new dictionary that includes the private key
    connection_params = {
        "user": st.secrets["postgresql"]["user"],
        "host": st.secrets["postgresql"]["host"],
        "port": st.secrets["postgresql"]["port"],
        "database": st.secrets["postgresql"]["database"],
        # "private_key": pkb
    }

    # st.write(connection_params) #debug
    # Connect to PostgreSQL
    conn = psycopg2.connect(**connection_params)

    # Write the DataFrame to PostgreSQL
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(create_table_query)
    
    st.write(create_table_query) #debug
    df.columns = df.columns.str.lower()  # PostgreSQL typically uses lowercase for column names
    df.replace('Infinity', np.nan, inplace=True)

    for index, row in df.iterrows():
        columns = ', '.join(row.index)
        values = ', '.join(['%s'] * len(row))
        insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
        cursor.execute(insert_query, tuple(row))

    conn.commit()
    
    st.write(f"Successfully wrote {len(df)} rows to PostgreSQL.")
        
    # Close the connection
    cursor.close()
    conn.close()


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
from sqlalchemy import create_engine

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

# @st.cache_data(ttl="24h")
# def get_tickers_marketstack(ticker_list):

#     # Replace with your Marketstack API key - free key only allows 100 api requests
#     api_key = '641f5f1d463000e9d6215fba76652251'
#     base_url = "https://api.marketstack.com/v1/tickers"

#     tickers = {}
#     offset = 0
#     limit = 100  # Number of results per page
    
#     for exchange in ['NYSE','NASDAQ']:
        
#         while True:
#             url = f"{base_url}?access_key={api_key}&exchange={exchange}&limit={limit}&offset={offset}"
#             response = requests.get(url)
#             data = response.json()

#             # Debugging information
#             st.write('first stock in data node:', data['data'][0])

#             # Extract symbols and names
#             if 'data' in data:
#                 for stock in data['data']:
#                     try:
#                         symbol = stock['symbol']
#                         name = stock['name']
#                         exchange_acronym = stock.get('stock_exchange', {}).get('acronym', '')
#                         tickers[symbol] = {'name': name, 'exchange_acronym': exchange_acronym}
#                     except (KeyError, TypeError) as e:
#                         st.write(f"Error: {e}")

#             # Check if there's another page
#             if 'pagination' in data:
#                 total = data['pagination'].get('total',None)
#                 limit = data['pagination'].get('limit',None)
#                 offset += limit
#             else:
#                 break

#     return tickers


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

@st.cache_data(ttl=43200) # 43200 is 12 hrs in seconds 
def yahoo_finance_load(ticker):
    
    today = datetime.today().date()
    x_years_ago = today - timedelta(days=366)
    start_date = x_years_ago.replace(month=1, day=1) #set start date to Jan 1 of the day 4 years ago
    today = datetime.today().date() #datetime.today().strftime('%Y-%m-%d')

    if 'yf_count' not in ss:
        yf_count=0

    today = datetime.today().date()
    
    try:
        ticker = yf.Ticker(ticker)
        # st.write(ticker.info) # debug
        # historical_prices=ticker.history(start=x_years_ago,end=today) # gets following fields: Date, High, Low, Close, Volume, Dividends, Stock Splits
        # sp=historical_prices.reset_index()
        # sp=sp.rename(columns={'Close':'Stock Price'})
        stats=ticker.info
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
    
    # yf_count += 1 # track how many times the data was pulled from Yahoo Finance, not cached
    # print(f"ss.yf_count = {ss.yf_count}") #debug

    # st.write(ticker_obj.info) #debug

    return stats #, historical_prices #, historical_prices_df, companyfacts_metrics_yahoo_dict

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


def postgres_update(df, table_name, primary_key_columns=None):
    import re

    def sanitize_column_name(col: str) -> str:
        col = col.lower()
        col = re.sub(r"[^a-z0-9]+", "_", col)
        col = re.sub(r"_+", "_", col)
        return col.strip("_")

    # Build connection params
    connection_params = {
        "user": st.secrets["postgres_financial_data"]["user"],
        "host": st.secrets["postgres_financial_data"]["host"],
        "port": st.secrets["postgres_financial_data"]["port"],
        "database": st.secrets["postgres_financial_data"]["database"],
        "password": st.secrets["postgres_financial_data"]["password"]
    }
    # st.write(connection_params) #debug
    conn = psycopg2.connect(**connection_params)
    cursor = conn.cursor()

    # Sanitize column names
    df = df.copy()
    df.columns = [sanitize_column_name(c) for c in df.columns]

    # Replace Infinity
    df.replace('Infinity', np.nan, inplace=True)

    # Add last_modified
    df["last_modified"] = pd.Timestamp.utcnow()

    # Validate PK columns
    if primary_key_columns:
        for pk in primary_key_columns:
            if pk not in df.columns:
                raise ValueError(f"Primary key column '{pk}' not found in DataFrame")

    # Type inference
    dtype_map = {
        "int64": "BIGINT",
        "float64": "DOUBLE PRECISION",
        "object": "TEXT",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
    }

    # Build CREATE TABLE
    column_defs = []
    for col, dtype in df.dtypes.items():
        pg_type = dtype_map.get(str(dtype), "TEXT")
        column_defs.append(f"{col} {pg_type}")

    # Composite PK
    if primary_key_columns:
        pk_clause = f"PRIMARY KEY ({', '.join(primary_key_columns)})"
        column_defs.append(pk_clause)

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {", ".join(column_defs)}
        );
    """
    st.write(create_table_sql) #debug
    cursor.execute(create_table_sql)

    # Grant permissions
    cursor.execute(f"GRANT ALL PRIVILEGES ON TABLE {table_name} TO ravila3;")

    # Build UPSERT
    columns = list(df.columns)
    col_names = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    update_assignments = ", ".join([
        f"{col} = EXCLUDED.{col}"
        for col in columns
        if col not in primary_key_columns and col != "last_modified"
    ])

    conflict_cols = ", ".join(primary_key_columns)

    upsert_query = f"""
        INSERT INTO {table_name} ({col_names})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_cols})
        DO UPDATE SET
            {update_assignments},
            last_modified = NOW();
    """

    # Execute UPSERT
    for _, row in df.iterrows():
        cursor.execute(upsert_query, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()

    # st.write(f"Upsert complete: {len(df)} rows processed.")
    return

def postgres_write(df, table_name, primary_key_column=None):
    # Build connection params
    connection_params = {
        "user": st.secrets["postgres_financial_data"]["user"],
        "host": st.secrets["postgres_financial_data"]["host"],
        "port": st.secrets["postgres_financial_data"]["port"],
        "database": st.secrets["postgres_financial_data"]["database"],
        "password": st.secrets["postgres_financial_data"]["password"]  
    }

    conn = psycopg2.connect(**connection_params)
    cursor = conn.cursor()

    # Normalize DataFrame
    def sanitize_column_name(col: str) -> str:
        return (
            col.lower()
            .replace("/", "_")
            .replace(" ", "_")
            .replace("&", "and")
            .replace("-", "_")
        )

    # Sanitize column names
    df.columns = [sanitize_column_name(c) for c in df.columns]

    # Replace Infinity
    df.replace('Infinity', np.nan, inplace=True)

    # Add last_modified
    df["last_modified"] = pd.Timestamp.utcnow()

    # Validate PK
    if primary_key_column not in df.columns:
        raise ValueError(f"Primary key column '{primary_key_column}' not found in DataFrame")

    # Type inference
    dtype_map = {
        "int64": "BIGINT",
        "int32": "BIGINT",
        "float64": "DOUBLE PRECISION",
        "float32": "DOUBLE PRECISION",
        "object": "TEXT",
        "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP",
        "datetime64[ns, UTC]": "TIMESTAMP",
    }

    column_defs = []
    for col, dtype in df.dtypes.items():
        pg_type = dtype_map.get(str(dtype), "TEXT")
        if col == primary_key_column:
            column_defs.append(f"{col} {pg_type} PRIMARY KEY")
        else:
            column_defs.append(f"{col} {pg_type}")

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {", ".join(column_defs)}
        );
    """
    cursor.execute(create_table_sql)

    GRANT_ALL_PRIVILEGES_SQL = f"GRANT ALL PRIVILEGES ON TABLE {table_name} TO ravila3;"
    cursor.execute(GRANT_ALL_PRIVILEGES_SQL)

    # Build INSERT query
    columns = list(df.columns)
    col_names = ", ".join(columns)
    placeholders = ", ".join(["%s"] * len(columns))

    insert_query = f"""
        INSERT INTO {table_name} ({col_names})
        VALUES ({placeholders});
    """

    st.write(insert_query) #debug   
    # Execute INSERT for each row
    for _, row in df.iterrows():
        cursor.execute(insert_query, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()

    st.write(f"Insert complete: {len(df)} rows inserted.")
    return

def postgres_read(table_name, where_clause=None, params=None):
    """
    Read data from PostgreSQL into a pandas DataFrame.

    Args:
        table_name (str): Name of the table to read from.
        where_clause (str, optional): SQL WHERE clause without the 'WHERE' keyword.
        params (tuple/list, optional): Parameters for the WHERE clause.
        sanitize_cols (bool): If True, sanitize column names to Python-friendly format.

    Returns:
        pd.DataFrame: The resulting data.
    """
    
    engine = create_engine(
        f"postgresql+psycopg2://{st.secrets['postgres_financial_data']['user']}:{st.secrets['postgres_financial_data']['password']}@{st.secrets['postgres_financial_data']['host']}:{st.secrets['postgres_financial_data']['port']}/{st.secrets['postgres_financial_data']['database']}"
    )
# # Build connection params
    # connection_params = {
    #     "user": st.secrets["postgres_financial_data"]["user"],
    #     "host": st.secrets["postgres_financial_data"]["host"],
    #     "port": st.secrets["postgres_financial_data"]["port"],
    #     "database": st.secrets["postgres_financial_data"]["database"],
    #     "password": st.secrets["postgres_financial_data"]["password"]
    # }

    # conn = psycopg2.connect(**connection_params)

    # Build SQL query
    if where_clause:
        sql = f"SELECT * FROM {table_name} WHERE {where_clause};"
    else:
        sql = f"SELECT * FROM {table_name};"

    # Execute query
    df = pd.read_sql(sql, engine, params=params)

    # conn.close()

    # # Optional: sanitize column names to match your write function
    # if sanitize_cols:
    #     def sanitize_column_name(col: str) -> str:
    #         return (
    #             col.lower()
    #             .replace("/", "_")
    #             .replace(" ", "_")
    #             .replace("&", "and")
    #             .replace("-", "_")
    #         )
    #     df.columns = [sanitize_column_name(c) for c in df.columns]

    return df

#format values for Yahoo Metrics based on declared format
def _format_value(fmt, value):
    """Format a raw value according to the declared fmt for display in the UI."""
    if value is None:
        return ""
    try:
        if fmt == 'epoch':
            v = int(value)
            if v > 1e12:  # milliseconds
                return datetime.utcfromtimestamp(v / 1000).strftime('%Y-%m-%d')
            else:
                return datetime.utcfromtimestamp(v).strftime('%Y-%m-%d')
        if fmt == 'integer':
            return f"{int(float(value)):,}"
        if fmt == 'decimal':
            return f"{float(value):,.2f}"
        if fmt == 'percent':
            try:
                v = float(value)
                # If value is provided as a fraction (-1..1), convert to percent
                if -1 <= v <= 1:
                    v = v * 100
                return f"{v:,.2f}%"
            except Exception:
                return str(value)
        # text or default
        s = str(value)
        # truncate very long text for table display
        return s if len(s) <= 400 else s[:400] + '...'
    except Exception:
        return str(value)

# def Yahoo_Loader(ticker):
    
#     today = datetime.today().date()
#     x_years_ago = today - timedelta(days=366)
#     start_date = x_years_ago.replace(month=1, day=1) #set start date to Jan 1 of the day 4 years ago
#     today = datetime.today().date() #datetime.today().strftime('%Y-%m-%d')
    
#     # ss.filings_df=ss.filings_df[ss.filings_df['filed_date']>=start_date]
#     # ss.quarterly_financials['end_date'] = pd.to_datetime(ss.quarterly_financials['end_date']).dt.date
#     # st.write(f"ss.ticker={ss.ticker}") #debug

#     ticker=yf.Ticker(ss.ticker)
#     historical_prices=ticker.history(start=x_years_ago,end=today) # gets following fields: Date, High, Low, Close, Volume, Dividends, Stock Splits
#     sp=historical_prices.reset_index()
#     sp['Ticker']=ss.ticker
#     sp=sp.rename(columns={'Close':'Stock Price'})
#     # st.write('stock prices',ss.sp) #debug

#     stats=ticker.info
#     # st.write('yahoo stats',stats) #debug write out yahoo dict
#     ss.company_name = stats.get('shortName', 'Unknown Company Name')

#     # CompanyInfoYahoo: tuples of (field_name, format) OR (field_name, format, thresholds)
#     # thresholds is an optional dict with numeric 'low' and 'high' values used for coloring.
#     # The thresholds are illustrative defaults you can tune per-company or per-field.
#     CompanyInfoYahoo = [
#     # ('symbol','text'),
#     # ('shortName','text'),
#     ('regularMarketPrice','decimal' ),
#     # ('previousClose','decimal' ),
#     # ('open','decimal' ),
#     ('regularMarketDayRange','text' ),
#     ('fiftyTwoWeekRange','text' ),
#     ('volume','integer'),
#     # PE ratios: low/high are illustrative (PE > 30 often considered high growth; adjust as needed)
#     ('trailingPE','decimal', {'low': 12, 'high':25, 'direction': 'higher_is_bad'}),
#     ('forwardPE','decimal', {'low': 12, 'high': 25, 'direction': 'higher_is_bad'}),
#     ('dividendYield','percent', {'low': 1.0, 'high': 3.0, 'direction': 'higher_is_good'} ),
#     ('fullTimeEmployees','integer'),
#     ('priceToSalesTrailing12Months','decimal', {'low': 2, 'high': 5, 'direction': 'higher_is_bad'}),
#     ('enterpriseToRevenue','decimal', {'low': 2, 'high': 5, 'direction': 'higher_is_bad'}),
#     ('enterpriseToEbitda','decimal', {'low': 2, 'high': 25, 'direction': 'higher_is_bad'}),
#     ('revenueGrowth','percent', {'low': 0.0, 'high': 10.0, 'direction': 'higher_is_good'}),
#     ('earningsGrowth','percent', {'low': 0.0, 'high': 20.0, 'direction': 'higher_is_good'}),
#     ('earningsQuarterlyGrowth','percent', {'low': 0.0, 'high': 20.0, 'direction': 'higher_is_good'}),
#     ('trailingPegRatio','decimal', {'low': 1, 'high': 1.5, 'direction': 'higher_is_bad'} ),
#     ('marketCap','integer'),
#     ('averageDailyVolume10Day','integer'),
#     ('quickRatio','decimal', {'low': 1.0, 'high': 1.5, 'direction': 'higher_is_good'} ),
#     ('currentRatio','decimal', {'low': 1.0, 'high': 2.0, 'direction': 'higher_is_good'} ),
#     ('debtToEquity','decimal', {'low': 0.0, 'high': 1.0, 'direction': 'higher_is_bad'}),
#     ('revenuePerShare','decimal' ),
#     ('heldPercentInsiders','percent' , {'low': 5.0, 'high': 20.0, 'direction': 'higher_is_bad'}),
#     ('heldPercentInstitutions','percent' , {'low': 5.0, 'high': 20.0, 'direction': 'higher_is_good'}),
#     ('shortPercentOfFloat', 'percent', {'low': 5.0, 'high': 15.0, 'direction': 'higher_is_bad'}),
#     ('grossMargins','percent' , {'low': 10.0, 'high': 40.0, 'direction': 'higher_is_good'}),
#     ('ebitdaMargins','percent' , {'low': 3.0, 'high': 30.0, 'direction': 'higher_is_good'}),
#     ('operatingMargins','percent' , {'low': 5.0, 'high': 25.0, 'direction': 'higher_is_good'}),
#     ('profitMargins','percent', {'low': 3.0, 'high': 15.0, 'direction': 'higher_is_good'}),
#     ('returnOnAssets','percent', {'low': 3.0, 'high': 10.0, 'direction': 'higher_is_good'}),
#     ('returnOnEquity','percent', {'low': 3.0, 'high': 15.0, 'direction': 'higher_is_good'}),
#     ('shortRatio','decimal'),
#     ('enterpriseValue','integer'),
#     ('numberOfAnalystOpinions','integer'),
#     ('recommendationKey','text'),
#     # ('fullExchangeName','text'),
#     ('earningsTimestamp','epoch'),
#     # ('website','text'),
#     # ('industryDisp','text'),
#     # ('sector','text'),
#     # omit longBusinessSummary here (renders separately)
#     ('beta','decimal' ),
#     ('sharesOutstanding','integer'),
#     ('sharesShort','integer'),
#     ('sharesShortPriorMonth','integer'),
#     ('bookValue','integer'),
#     ('priceToBook','decimal' ),
#     ('lastFiscalYearEnd','epoch'),
#     # ('nextFiscalYearEnd','epoch'),
#     ('trailingEps','decimal' ),
#     ('forwardEps','decimal' ),
#     ('totalCash','integer'),
#     ('totalCashPerShare','decimal' ),
#     ('ebitda','integer'),
#     ('totalDebt','integer'),
#     ('totalRevenue','integer'),
#     ('freeCashflow','integer'),
#     ('operatingCashflow','integer'),
#     ]

#     st.write(f"""{stats.get('longBusinessSummary', 'Unknown Company')}, website: {stats.get('website', 'N/A')}
#                 , exchange: {stats.get('fullExchangeName', 'N/A')}, sector: {stats.get('sector', 'N/A')}, industry: {stats.get('industryDisp', 'N/A')}""")
#     # Build a dict with both the expected format and the actual value pulled from yfinance
#     ss.companyfacts_metrics_yahoo_dict = {}
#     rows = []

#     for item in CompanyInfoYahoo:
#         # allow entries to be (name, fmt) or (name, fmt, thresholds)
#         if isinstance(item, (list, tuple)) and len(item) == 2:
#             name, fmt = item
#             thresholds = {}
#         else:
#             name, fmt = item[0], item[1]
#             thresholds = item[2] if len(item) > 2 else {}
#         try:
#             raw_value = stats.get(name)
#         except Exception:
#             raw_value = None

#         display_value = _format_value(fmt, raw_value)

#         ss.companyfacts_metrics_yahoo_dict[name] = {'format': fmt, 'value': raw_value, 'display': display_value, 'thresholds': thresholds}
#         rows.append({'field': name, 'format': fmt, 'display': display_value, 'raw_value': raw_value, 'thresholds': thresholds})
# return  
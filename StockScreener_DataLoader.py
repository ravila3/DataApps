import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf
import yfinance as yf
import time
from icecream import ic
# import snowflake as snowflake
from streamlit import session_state as ss
from altair.expr import *
# from snowflake.snowpark import Session
# from snowflake.snowpark.context import get_active_session
# from snowflake.snowpark.functions import col
# import snowflake.snowpark.modin.plugin
from datetime import datetime,timedelta
from stock_data_loader_utilities import retrieve_snowflake_data, write_snowflake_data, yahoo_finance_load

# connection_parameters = st.secrets["snowflake"]
# session = Session.builder.configs(connection_parameters).create()

if 'cik' not in ss:
    ss.cik=''
    ss.ticker=''
    ss.company_name=''
    ss.company_and_ticker=''
    ss.llm_model=''
    ss.df=pd.DataFrame()
    ss.filings_df=pd.DataFrame()
    ss.quarterly_financials=pd.DataFrame()
    ss.annual_financials=pd.DataFrame()
    # ss.sp=pd.DataFrame(columns=['Column1'])
    ss.analysis_result=''
    ss.messages=[]
    ss.counter=0

# set page config and title
st.set_page_config( page_title="Stock Screener", layout="wide" )
st.markdown('<h2 style="color:#3894f0;">Stock Screener for Publically Traded Stocks</h2>', unsafe_allow_html=True)
st.write('Created by Rafael Avila leveraging Snowflake & Streamlit, using stock tickers from Marketwatch and stock details from yfinance. Analysis Summary and AI chat powered by mistral-large2 AI model')

def main():

    sql = f"""
select ticker, asset_class, date, primary_exchange_code
, max(case when variable='post-market_close' then value end) as closing_price
, max(case when variable='nasdaq_volume' then value end) as volume
from FINANCE__ECONOMICS.CYBERSYN.STOCK_PRICE_TIMESERIES
where date=current_date-3 and asset_class not like 'ETF%'
group by 1,2,3,4
order by volume desc
--having volume>1000000
--ticker='T'
--and ticker in (select ticker from FINANCE__ECONOMICS.CYBERSYN.STOCK_PRICE_TIMESERIES where variable='nasdaq_volume' and date=current_date-1 and value>10000)
--order by ticker -- date
limit 350
    """

    tickers_df = retrieve_snowflake_data(sql)
    # st.dataframe(tickers_df) #debug
    tickers_list = tickers_df['TICKER'].tolist() #['AAL','AAPL','ABEV','ABNB','ACHR']
    st.write(f"Snowflake pulled {len(tickers_list)} stock tickers that met volume criteria")
    # tickers_list = ['PBR'] #,'AAPL','ABEV' #debug limit yahoo finance calls when debugging
    # st.write(f"tickers_list = {tickers_list}") #debug purposes only

    companyfacts_metrics_yahoo_dict = {}
    historical_prices_df2 = pd.DataFrame()
    
    # tickers_list.remove('PBR.A') # Remove value causing an error with Yahoo load

    # Append data to the dictionary based on each ticker
    runningcount = 0
    
    for ticker in tickers_list:
        try:
            yahoo_company_data = yahoo_finance_load(ticker) #,historical_prices_df_temp
            # historical_prices_df2 = pd.concat([historical_prices_df2, historical_prices_df_temp], ignore_index=True)
            # st.dataframe(historical_prices_df2) #debug
            companyfacts_metrics_yahoo_dict[ticker] = {}
            # st.write(yahoo_company_data)
            for key, value in yahoo_company_data.items():
                if key not in ('companyOfficers','longBusinessSummary'):
                    companyfacts_metrics_yahoo_dict[ticker][key] = value
            time.sleep(0.1) # delay 1/10 second
            runningcount += 1
            print(f"Received yahoo data for {ticker} #{runningcount}")
        except ValueError as e:
            st.write(f'error pulling data for {ticker} due to {e}')

    # Flatten the dictionary into a list of dictionaries
    flattened_data = []
    for ticker, metrics in companyfacts_metrics_yahoo_dict.items():
        try:
            flattened_row = {'ticker': ticker}
            flattened_row.update(metrics)
            flattened_data.append(flattened_row)
        except ValueError as e:
            st.write(f'error pulling data for {ticker} due to {e}')
            continue

    # ic(flattened_data) #debug
    # Create a DataFrame from the list of dictionaries
    companyfacts_metrics_yahoo_df = pd.DataFrame(flattened_data)

    # Convert epoch to datetime
    try:
        companyfacts_metrics_yahoo_df['firstTradeDate'] = pd.to_datetime(companyfacts_metrics_yahoo_df['firstTradeDateEpochUtc'], unit='s').dt.date
    except Exception as e:
        print(f"Error processing 'firstTradeDate': {e}")
            
    companyfacts_metrics_yahoo_df['lastFiscalYearEndDate'] = pd.to_datetime(companyfacts_metrics_yahoo_df['lastFiscalYearEnd'], unit='s').dt.date
    companyfacts_metrics_yahoo_df['nextFiscalYearEndDate'] = pd.to_datetime(companyfacts_metrics_yahoo_df['nextFiscalYearEnd'], unit='s').dt.date

    # Drop the original epoch columns if not needed
    # companyfacts_metrics_yahoo_df.drop(columns=['firstTradeDateEpochUtc', 'lastFiscalYearEnd', 'nextFiscalYearEnd'], inplace=True)

    column_order = [ 'ticker', 'currentPrice', 'shortName', 'longName', 'industry', 'sector', 'symbol', 'website', 'fullTimeEmployees', 'irWebsite'
                    , 'previousClose', 'beta', 'dividendYield', 'trailingPE', 'forwardPE', 'volume', 'averageDailyVolume10Day', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 'priceToSalesTrailing12Months'
                    , 'marketCap', 'enterpriseValue', 'profitMargins', 'sharesOutstanding', 'sharesShort', 'sharesShortPriorMonth', 'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio'
                    , 'bookValue', 'priceToBook', 'lastFiscalYearEndDate', 'nextFiscalYearEndDate', 'trailingEps', 'forwardEps', 'enterpriseToRevenue', 'enterpriseToEbitda'
                    , '52WeekChange', 'numberOfAnalystOpinions', 'recommendationKey', 'totalCash', 'totalCashPerShare', 'ebitda', 'totalDebt', 'quickRatio', 'currentRatio', 'totalRevenue'
                    , 'debtToEquity', 'revenuePerShare', 'returnOnAssets', 'returnOnEquity', 'revenueGrowth', 'freeCashflow', 'operatingCashflow', 'grossMargins', 'ebitdaMargins', 'operatingMargins'
                    , 'trailingPegRatio', 'firstTradeDate'] #
    # companyfacts_metrics_yahoo_df=companyfacts_metrics_yahoo_df[column_order]
    companyfacts_metrics_yahoo_df = companyfacts_metrics_yahoo_df.reindex(columns=column_order)
    companyfacts_metrics_yahoo_df.rename(columns={"52WeekChange": "Change52Weeks"}, inplace=True)

    # Display the DataFrame
    st.dataframe(companyfacts_metrics_yahoo_df)
    table_name="Yahoo_Company_Data"

    create_table_query = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    ticker STRING,
    currentPrice FLOAT,
    shortName STRING,
    longName STRING,
    industry STRING,
    sector STRING,
    symbol STRING,
    website STRING,
    firstTradeDate DATE,
    fullTimeEmployees INTEGER,
    irWebsite STRING,
    previousClose FLOAT,
    beta FLOAT,
    dividendYield FLOAT,
    trailingPE FLOAT,
    forwardPE FLOAT,
    volume INTEGER,
    averageDailyVolume10Day INTEGER,
    fiftyTwoWeekLow FLOAT,
    fiftyTwoWeekHigh FLOAT,
    priceToSalesTrailing12Months FLOAT,
    marketCap INTEGER,
    enterpriseValue FLOAT,
    profitMargins FLOAT,
    sharesOutstanding INTEGER,
    sharesShort INTEGER,
    sharesShortPriorMonth INTEGER,
    heldPercentInsiders FLOAT,
    heldPercentInstitutions FLOAT,
    shortRatio FLOAT,
    bookValue FLOAT,
    priceToBook FLOAT,
    lastFiscalYearEndDate DATE,
    nextFiscalYearEndDate DATE,
    trailingEps FLOAT,
    forwardEps FLOAT,
    enterpriseToRevenue FLOAT,
    enterpriseToEbitda FLOAT,
    Change52Weeks FLOAT,
    numberOfAnalystOpinions INTEGER,
    recommendationKey STRING,
    totalCash FLOAT,
    totalCashPerShare FLOAT,
    ebitda FLOAT,
    totalDebt FLOAT,
    quickRatio FLOAT,
    currentRatio FLOAT,
    totalRevenue FLOAT,
    debtToEquity FLOAT,
    revenuePerShare FLOAT,
    returnOnAssets FLOAT,
    returnOnEquity FLOAT,
    revenueGrowth FLOAT,
    freeCashflow FLOAT,
    operatingCashflow FLOAT,
    grossMargins FLOAT,
    ebitdaMargins FLOAT,
    operatingMargins FLOAT,
    trailingPegRatio FLOAT
);
    """
    write_snowflake_data(companyfacts_metrics_yahoo_df,table_name,create_table_query)

    
    table_name = "yahoo_stock_prices"
    # st.dataframe(historical_prices_df2) #debug
    # Filter rows where 'stock_price' is less than or equal to 10,000
    # historical_prices_df2 = historical_prices_df2[historical_prices_df2['stock_price'] <= 10000]

    create_table_query = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    ticker STRING,
    date DATE,
    stock_price DECIMAL(10,2),
    dividends DECIMAL(10,2),
    stock_splits DECIMAL(10,2)
    );
    """
    
    # write_snowflake_data(historical_prices_df2,table_name,create_table_query)
    # st.dataframe(historical_prices_df2)
    
    return()

main()


import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf
import yfinance as yf
import time
from icecream import ic
from streamlit import session_state as ss
from altair.expr import *
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import pandas as pd
from datetime import datetime,timedelta
from stock_data_loader_utilities import yahoo_finance_load, retrieve_snowflake_data,write_snowflake_data


def maincontrol():

    st.set_page_config( page_title="Financial Trends", layout="wide" )
    st.markdown('<h2 style="color:#3894f0;">Value Stock Screener Leveraging AI Capabilities</h2>', unsafe_allow_html=True)
    st.write('Created by Rafael Avila leveraging Snowflake & Streamlit, using Yahoo Finance data with Analysis Summary and AI chat powered by mistral-large2 AI model')

    sql = f"""
select ticker
, shortname as company_name
, industry
, sector
, fulltimeemployees as full_time_employees
, previousclose as stock_price
, fiftytwoweeklow as low_52w
, fiftytwoweekhigh as high_52w
, averagedailyvolume10day as avg_volume_10d
, beta::decimal(10,1) as beta
, (dividendYield*100)::decimal(10,1) as dividend_yield
, trailingpe::decimal(10,1) as trailing_pe
, forwardpe::decimal(10,1) as forward_pe
, pricetosalestrailing12months::decimal(10,1) as price_to_sales_last_12m
, enterprisevalue as enterprise_value
, (profitmargins*100)::decimal(10,1) as profit_margin_pct
, (heldpercentinsiders*100)::decimal(10,1) as insider_ownership_pct
, (heldpercentinstitutions*100)::decimal(10,1) as institution_ownership_pct
, shortratio::decimal(10,1) as short_ratio
, pricetobook::decimal(10,1) as price_to_book
, trailingeps::decimal(10,1) as trailing_eps
, forwardeps::decimal(10,1) as forward_eps
, marketCap as market_cap
, enterprisetorevenue::decimal(10,1) as enterprise_value_to_revenue
, enterprisetoebitda::decimal(10,1) as enterprise_value_to_ebitda
, totalcashpershare::decimal(10,1) as cash_per_share
, quickratio::decimal(10,1) as quick_ratio
, currentratio::decimal(10,1) as current_ratio
, debttoequity::decimal(10,1) as debt_to_equity
, revenuepershare::decimal(10,1) as revenue_per_share
, (revenuegrowth*100)::decimal(10,1) as revenue_growth
, (returnonassets*100)::decimal(10,1) as return_on_assets
, (returnonequity*100)::decimal(10,1) as return_on_equity
, (grossmargins*100)::decimal(10,1) as gross_margins
, (ebitdamargins*100)::decimal(10,1) as ebitda_margins
, (operatingmargins*100)::decimal(10,1) as operating_margins
, trailingpegratio::decimal(10,1) as trailing_peg_ratio
from NOTEBOOK.PUBLIC.YAHOO_COMPANY_DATA
limit 250
    """
    yahoo_company_data_df = retrieve_snowflake_data(sql)

    st.write("Data compiled from Yahoo Finance for top volume stocks:")
    st.dataframe(yahoo_company_data_df) 

    if 'messages' not in ss: #initialize messages
        ss.messages=[]
    if "analysis_result" not in ss:
        ss.analysis_result=""
    
    if ss.messages==[]: # Initialize the chat message history if no chat yet
        # data_json = yahoo_company_data_df.dropna().to_json(orient='records')

        # Function to filter out null values
        # def filter_nulls_and_format_date(row):
        #     # row = {k: v for k, v in row.items() if pd.notna(v)}
        #     row = {k: (int(v) if isinstance(v, (int, float)) else v) for k, v in row.items() if pd.notna(v)}
        #     # row['end_date'] = row['end_date'].strftime('%b %Y')
        #     return row

        def filter_nulls_and_format_date(row):
            row = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row.items() if pd.notna(v)}
            # row['end_date'] = row['end_date'].strftime('%b %Y')
            return row

        fields_for_LLM = [
            #'TICKER',
            'COMPANY_NAME',
            'INDUSTRY',
            #'SECTOR',
            #'FULL_TIME_EMPLOYEES',
            'STOCK_PRICE',
            #'AVG_VOLUME_10D',
            #'BETA',
            'DIVIDEND_YIELD',
            'TRAILING_PE',
            'FORWARD_PE',
            #'LOW_52W',
            #'HIGH_52W',
            'PRICE_TO_SALES_LAST_12M',
            #'ENTERPRISE_VALUE',
            'PROFIT_MARGIN_PCT',
            #'INSIDER_OWNERSHIP_PCT',
            'INSTITUTION_OWNERSHIP_PCT',
            'SHORT_RATIO',
            'PRICE_TO_BOOK',
            'TRAILING_EPS',
            #'FORWARD_EPS',
            #'MARKET_CAP',
            'ENTERPRISE_VALUE_TO_REVENUE',
            'ENTERPRISE_VALUE_TO_EBITDA',
            #'CASH_PER_SHARE',
            'QUICK_RATIO',
            'CURRENT_RATIO',
            'DEBT_TO_EQUITY',
            #'REVENUE_PER_SHARE',
            'REVENUE_GROWTH',
            #'RETURN_ON_ASSETS',
            #'RETURN_ON_EQUITY',
            'GROSS_MARGINS',
            'EBITDA_MARGINS',
            'OPERATING_MARGINS',
            #'TRAILING_PEG_RATIO'
        ]

        filtered_df_for_LLM = yahoo_company_data_df[fields_for_LLM]

         # Convert each row to a dictionary excluding null values
        data_json = filtered_df_for_LLM.apply(lambda x: filter_nulls_and_format_date(x), axis=1).to_json(orient='records')
        # st.write('data_json',data_json)
        
        ss.messages=[{"role":"user","content":f"""Please recommend 15 stocks as the best potential investments, ensuring you review the full list and don't give preference to the ones at the top of the list. For each of the
                      recommended stocks, provide a good level of rationale including specific metrics with full decimal precision. Please use trailing PE instead of forward PE since I don't trust analyst estimates that include high growth expectations.
                      I'd consider companies with a low PE ratio, good liquidity, a decent dividend growth rate, and good earnings prospects to be good values.
                      Please include a summary of the rationale for each of your selections.Double check your results to ensure that your analysis is correct.
                      Utilize this json with financial data, with each record being a company: {data_json}"""}]
        ss.messages.append({"role":"system","content":"""Limit the responses to only questions that are relevant to this company's performance
                            If the user asks about performance relative to other competitors, do not respond with generic comparison frameworks, 
                            but rather answer with any data you do know about the industry performance or specific competitors and their performance
                            """})
        ss.llm_model='mistral-large2'

        # st.write(sql) #debug
        # ss.messages.append({"role":"assistant","content":"message to test without LLM"}) #debug
    
       # Pull the data based on the messages array
        
        prompt = "\n".join([
            "{}: {}".format(msg['role'], msg['content'].replace("'", "").replace('"', ''))
            for idx, msg in enumerate(ss.messages)
        ])

        # st.write(prompt) #debug
        sql = f"""
        select snowflake.cortex.complete('{ss.llm_model}', 
            '{prompt}, temperature=0.1'
            ) as response;
        """ 

    if len(ss.messages)==2 and len(ss.analysis_result)==0:
        with st.spinner('Running LLM Analysis to provide a summary...'):
            response_df = ic(retrieve_snowflake_data(sql))
        ss.analysis_result=response_df.iloc[0,0]
        ss.analysis_result=ss.analysis_result.replace('$', '\\$')
        print(f"Retrieved LLM response, length of response = {len(ss.analysis_result)}")
        ss.messages.append({"role":"assistant","content":ss.analysis_result})

    # Print the summary and the dataframe
    if len(ss.messages)>=3:
        st.markdown('<h3 style="color:#3894f0;">Summary of potential value stocks from LLM Analysis:</h3>', unsafe_allow_html=True)
        st.markdown(ss.messages[2]["content"])

        # st.write(analysis_result_text) ######## debug purposes only

    #if the summary is complete, prep for the chat interactions
    # if len(ss.messages)==3:
    #     ss.messages.append({"role": "assistant", "content": 
    #         f"""Welcome to Cortex Chat, powered by the Mistral-Large2 LLM Model. Please let me know if you have any questions about these metrics for {ss.company_name}. I may be able to provide some limited data on industry and competitors based on the public knowledge I was trained on"""})
    #     ss.messages.append({"role":"system","content":"""
    #         If the user asks about performance relative to other competitors, do not respond with generic comparison frameworks, 
    #         but rather answer with any data you do know about the industry performance or specific competitors and their performance. Do not state your role in the response.
    #         """})

maincontrol()
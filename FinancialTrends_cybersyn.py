# Items to do:
# 1) Ensure stock prices reflect splits (MSTR was way off)
# 2) look at 8-K summarization?

# Import python packages
import sec_edgar.financial_statements
import sec_edgar.utils
import streamlit as st
import pandas as pd
import altair as alt
import yfinance as yf
import sec_edgar
import snowflake.connector
from icecream import ic
# import snowflake as snowflake
from streamlit import session_state as ss
from altair.expr import *
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col
from datetime import datetime,timedelta

connection_parameters = st.secrets["snowflake"]
session = Session.builder.configs(connection_parameters).create()

if 'cik' not in ss:
    ss.cik=''
    ss.ticker=''
    ss.company_name=''
    ss.company_and_ticker=''
    ss.llm_model=''
    ss.df=pd.DataFrame(columns=['Column1'])
    ss.sp=pd.DataFrame(columns=['Column1'])
    ss.mdf=pd.DataFrame(columns=['Column1'])
    ss.analysis_result=''
    ss.messages=[]
    ss.counter=0

# set page config and title
st.set_page_config( page_title="Financial Trends", layout="wide" )
st.markdown('<h2 style="color:#3894f0;">Financial Trends for Publically Traded Stocks</h2>', unsafe_allow_html=True)
st.write('Created by Rafael Avila leveraging Snowflake & Streamlit, using SEC Filings data provided by Cybersyn. Analysis Summary and AI chat powered by mistral-large2 AI model')

@st.cache_data(ttl="24h")
def retrieve_data(sql):
    conn = snowflake.connector.connect(**st.secrets["snowflake"])
    # st.write(f"sql = {sql}") #debug
    df = pd.read_sql(sql,conn)
    conn.close()
    return df

def add_user_message():
    # Chat input
    ss.counter=ss.counter+1 # `DEBUG`
    sanitized_user_input=ss.user_input.replace("'","").replace('"',"")
    ss.messages.append({"role": "user", "content": sanitized_user_input})
    #ss.user_input="" # clear message after sending
    # st.write(f"just added messaage: {sanitized_user_input}") #debug

def add_assistant_message(response):
    ss.messages.append({"role": "assistant", "content": response})

def get_line_chart(tdf,date,metric_name,value_field,width,height):

    hover = alt.selection_point(
        fields=[date, metric_name],
        nearest=True,
        on="mouseover",
        empty=False) #"none")
    legend_selection = alt.selection_point(fields=[metric_name], bind='legend')
    
    color_encoding = alt.Color(metric_name, legend=alt.Legend(title=metric_name, labelLimit=400), sort=alt.EncodingSortField('total_for_order', order='descending'))
    
    lines = (
        alt.Chart(tdf)
        .mark_line(interpolate="linear")
        .encode(
            x=alt.X(date, type='temporal', title='Date (PST)', axis=alt.Axis(format='%b %Y')),
            y=alt.Y(value_field, type='quantitative', title=value_field, axis=alt.Axis(format='$,d')),
            color=color_encoding,
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.1)),
        ).add_params(legend_selection)
    ).properties(width=width, height=height)
    
    points = alt.Chart(tdf).mark_point().encode(
        x=alt.X(date, type='temporal'),
        y=alt.Y(value_field, type='quantitative'), #metric_name,
        color=color_encoding,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip(date, type='temporal', format='%m/%d/%y(%a) %I%p', title="Date (PST)"),
            metric_name,
            alt.Tooltip(value_field, type='quantitative', format='$,d', title=value_field)
        ]
    ).add_params(hover)  #.interactive()
    
    return (lines + points) #  + tooltips

def main():
    print('#### Starting at top of main') #debug
    # Load company lookup table
    sql= """Select case when primary_ticker is not null then company_name||' ('||primary_ticker||')' else company_name END as company_and_ticker
            , company_name, primary_ticker, cik, last_filing_date
            from Notebook.Public.cybersyn_company_lookup order by 1 """
    company_lookup_df=retrieve_data(sql)
    # st.write(company_lookup_df) #debug

    with st.form("Company Lookup"):
        company_and_ticker=st.selectbox('Select which company/stock ticker:',company_lookup_df['COMPANY_AND_TICKER'],index=None,placeholder='Start typing to narrow company name or ticker options')
        submit_button = st.form_submit_button(label='Submit')
    if submit_button and company_and_ticker==None:
        st.subheader(':red[Please select a company]')
    if submit_button and (company_and_ticker!=None and company_and_ticker!=ss.company_and_ticker):
        ss.company_and_ticker=company_and_ticker
        ss.cik = company_lookup_df.loc[company_lookup_df['COMPANY_AND_TICKER'] == ss.company_and_ticker, 'CIK'].values[0]
        ss.ticker = company_lookup_df.loc[company_lookup_df['COMPANY_AND_TICKER'] == ss.company_and_ticker, 'PRIMARY_TICKER'].values[0]
        ss.company_name = company_lookup_df.loc[company_lookup_df['COMPANY_AND_TICKER'] == ss.company_and_ticker, 'COMPANY_NAME'].values[0]
        ss.df=ss.sp=ss.mdf=pd.DataFrame(columns=['Column1']) # Clear out the dataframe
        ss.analysis_result=''
        ss.messages=[] # Clear out messages
        ss.counter=0 # Reset counter
        # st.write(f"cik={ss.cik}, ticker={ss.ticker}, company_name={ss.company_name}, company_and_ticker={ss.company_and_ticker}\n") #debug
        # st.subheader('Dataframe Reset due to submit button') #debug

        with st.spinner('Pulling 10-Q Financial Data...'):
            sql = f"""
with cf as
(
SELECT distinct
-- , r.variable
--r.metadata, r.metadata:BusinessSegments::string as BusinessSegments, r.metadata:Subsegments::string as Subsegments, r.metadata:ConsolidationItems::string as ConsolidationItems, r.metadata:ProductOrService::string as ProductOrService
r.cik
, r.adsh
, ri.form_type
, ri.filed_date
, c.primary_ticker
, initcap(i.company_name) as company_name
, r.period_start_date
, r.period_end_date
, r.covered_qtrs
, r.statement
, r.tag
, case
    when statement = 'Income Statement'  and r.tag in ('RevenueFromContractWithCustomerExcludingAssessedTax','RevenueFromContractWithCustomerIncludingAssessedTax','Revenues','InvestmentIncomeInterestAndDividend','InterestAndDividendIncomeOperating') then 'Sales/Revenue'
    when statement = 'Income Statement'  and r.tag in ('OperatingIncomeLoss','IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments') then 'Operating Income' --,'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
    when statement = 'Income Statement'  and r.tag in ('NetIncomeLoss','NetIncomeLossAvailableToCommonStockholdersBasic') then 'Net Income'
    when statement = 'Income Statement'  and r.tag='CostOfRevenue' then 'Cost of Sales'
    when statement = 'Income Statement'  and r.tag in ('CostsAndExpenses','BenefitsLossesAndExpenses') then 'Operating Costs'
    when statement = 'Income Statement'  and r.tag='InterestAndDividendIncomeOperating' then 'Interest and Dividend Income'
    when statement = 'Income Statement'  and r.tag in ('InterestExpense','InterestExpenseOperating','InterestExpenseNonoperating') then 'Interest Expense'
    when statement = 'Income Statement'  and r.tag='InterestIncomeExpenseNet' then 'Interest Net Income'
    when statement = 'Income Statement'  and r.tag='NoninterestIncome' then 'Non-Interest Income'
    when r.tag='StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest' then 'Stockholders Equity'
    when r.tag='LiabilitiesCurrent' then 'Current Liabilities'
    when r.tag='Liabilities' then 'Total Liabilities'
    when r.tag='AssetsCurrent' then 'Current Assets'
    when r.tag='Assets' then 'Total Assets'
    when r.tag in ('CommonStockSharesIssued','CommonStockSharesOutstanding') then 'Common Shares Outstanding'
    -- when r.tag='Revenues' then 'Revenues'
    else 'Other'
  end as Metric_Name
, r.measure_description
, (TRUNC(TO_NUMERIC(r.value),0)) AS value
, ROW_NUMBER() OVER (PARTITION BY r.cik, r.period_start_date, r.period_end_date, r.covered_qtrs, metric_name ORDER BY ri.filed_date, (TRUNC(TO_NUMERIC(r.value),0)) DESC) AS rn
--, ROW_NUMBER() OVER (PARTITION BY r.cik, r.period_start_date, r.period_end_date, r.covered_qtrs, metric_name ORDER BY ri.filed_date, r.adsh DESC) AS rn
FROM SEC_FILINGS.cybersyn.sec_cik_index AS i
JOIN SEC_FILINGS.cybersyn.company_index as c on (c.cik=i.cik)
JOIN SEC_FILINGS.cybersyn.sec_report_attributes AS r ON (r.cik = i.cik)
JOIN SEC_FILINGS.cybersyn.sec_report_index as ri on (ri.adsh=r.adsh)
WHERE 
  c.cik='{ss.cik}'
--  c.primary_ticker='CMCSA'
--  i.company_name like '%AT&T%' --'AMR CORP'
--  AND i.sic_code_description = 'AIR TRANSPORTATION, SCHEDULED'
  AND r.period_end_date >= '2020-01-01'
  -- AND r.period_end_date = '2023-07-31'
  AND (r.covered_qtrs = 1 or statement='Balance Sheet')
  AND TRY_CAST(r.value AS NUMBER) IS NOT NULL
  AND r.statement in ('Income Statement','Balance Sheet','Stockholder Equity')-- ,'Balance Sheet','Cash Flow'
  AND form_type='10-Q'
  AND (r.metadata is null or tag='CommonStockSharesIssued')-- businesssegments in ('Communications') -- and subsegments is null and productorservice is null --    ,'CorporateAndOther','LatinAmericaBusinessSegment'
  AND (r.tag in ('RevenueFromContractWithCustomerExcludingAssessedTax','RevenueFromContractWithCustomerIncludingAssessedTax','OperatingIncomeLoss'
        ,'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest','InterestExpenseOperating','NetIncomeLoss','CostsAndExpenses','BenefitsLossesAndExpenses'
        ,'InterestExpense','InterestExpenseNonoperating','Revenues','InterestIncomeExpenseNet','NoninterestIncome','InterestAndDividendIncomeOperating'
        ,'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments'
        ,'CommonStockSharesOutstanding','CommonStockSharesIssued', 'AssetsCurrent', 'Assets', 'Liabilities', 'LiabilitiesCurrent', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')
  --or r.tag like '%Revenue%' or r.tag like '%Income%' 
  )
)
  
select form_type, cik, primary_ticker, company_name, period_end_date, statement, Metric_Name, max(cast(value as integer)) as value  --, tag, measure_description, rn, businesssegments, subsegments, productorservice, ConsolidationItems, metadata --, max(Value) as Value
from cf
where value<>0 and rn=1 and Metric_Name<>'Other' --and tag not in ('RevenueFromContractWithCustomerExcludingAssessedTax','CostOfRevenue')
--AND period_end_date >= '2024-01-01' -- LIMIT SCOPE FOR DEVELOPMENT EFFICIENCY
group by 1,2,3,4,5,6,7
order by period_end_date desc
--limit 100
            """
            ss.df = ic(retrieve_data(sql))
            print(f"#### 10-Q data retrieved, row count = {len(ss.df)}")
            sql = f"""select ticker, date, primary_exchange_code, value as closing_price from FINANCE__ECONOMICS.CYBERSYN.STOCK_PRICE_TIMESERIES where variable='post-market_close' and ticker='{ss.ticker}' order by date"""
            # ss.sp = ic(retrieve_data(sql))
            today=datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=4*365)).strftime('%Y-%m-%d')
            ticker=yf.Ticker(ss.ticker)
            historical_prices=ticker.history(period='1d',start='2017-01-01',end=today) # gets following fields: Date, High, Low, Close, Volume, Dividends, Stock Splits
            ss.sp=historical_prices.reset_index()
            ss.sp['Ticker']=ss.ticker

            # income_statement=pd.DataFrame(ticker.quarterly_financials.sort_index(axis=1).loc[:,start_date:])
            # income_statement=pd.DataFrame(ticker.quarterly_financials.loc)

            edgar = sec_edgar.utils

            # Set the company and filing type
            edgar.set_company(ss.ticker)
            edgar.set_filing_type('10-Q', '10-K')
            edgar_results = edgar.get_filings(start_date=start_date)

            # Extract income statements
            income_statements = []
            for filing in edgar_results:
                if 'income statement' in filing['description'].lower():
                    summary = {
                        'Date Filed': filing['date'],
                        'Form Type': filing['form_type'],
                        'Description': filing['description']
                    }
                    income_statements.append(summary)

            # Convert to DataFrame
            income_statements_df = pd.DataFrame(income_statements)

            # Display the income statements
            st.write("Income Statements from SEC EDGAR:")
            st.write(income_statements_df)

            income_statement=ticker.get_income_stmt(freq="yearly")
            st.write(income_statement)

            balance_sheet=pd.DataFrame(ticker.quarterly_balance_sheet)
            cashflow=pd.DataFrame(ticker.quarterly_cashflow)
            ratios=pd.DataFrame(ticker.info)

            st.write(f"ss.sp of type {type(ss.sp)}: ")
            st.write(ss.sp)
            st.write(f"balance_sheet of type {type(balance_sheet)}:")
            st.write(balance_sheet)

            st.write(f"ratios of type {type(cashflow)}")
            st.write(cashflow)
            st.write(f"ratios of type {type(ratios)}")
            st.write(ratios)
            # print(f"#### stock price data retrieved, row count = {len(ss.sp)}, and last price from {ss.sp['Date'].max()}")
            # ss.df=df
            # df = conn.query(sql)
            if len(ss.df)==0:
                st.write(f"No 10-Q Data Retrieved for '{ss.company_and_ticker}'")
            if len(ss.sp)==0:
                st.write(f"No Stock Price Data Retrieved for '{ss.company_and_ticker}'")                
    
    # if not df.empty:
    # st.write(f"did it meet condition for chart? len(ss.df)={len(ss.df)}") # debug
    if len(ss.df)!=0:
        # st.write(f"about to write chart (ss.df)={ss.df}") # debug
        ss.df['VALUE'] = ss.df['VALUE'].astype(int)
        ss.df['PERIOD_END_DATE'] = pd.to_datetime(ss.df['PERIOD_END_DATE'])
        # st.write(ss.df) ################ debug purposes only
        filtered_df=ss.df[ss.df['STATEMENT']=='Income Statement']
        chart_income_statement=get_line_chart(filtered_df,'PERIOD_END_DATE','METRIC_NAME','VALUE',400,300)

        filtered_df = ss.df[(ss.df['STATEMENT'] != 'Income Statement') & (ss.df['METRIC_NAME'] != 'Common Shares Outstanding')]

        chart_balance_sheet=get_line_chart(filtered_df,'PERIOD_END_DATE','METRIC_NAME','VALUE',400,300)
        
        if len(ss.mdf)==0:
            pivoted_df = ic(ss.df.pivot_table(index=['PERIOD_END_DATE', 'CIK', 'PRIMARY_TICKER', 'COMPANY_NAME'], columns='METRIC_NAME', values='VALUE').reset_index())
            ss.mdf = ic(pivoted_df[pivoted_df['Sales/Revenue'].notna()])
            # st.write(ss.mdf) #debug

            if len(ss.sp)==0: # if no stock prices retrieved, set merged dataframe to pivoted data
                ss.mdf['ROA'] = ss.mdf['Net Income'] / ss.mdf['Total Assets']
                ss.mdf['EPS'] = ss.mdf['Net Income'] / ss.mdf['Common Shares Outstanding']
                ss.mdf['Market_Cap'] = None
                ss.mdf['PS'] = None
                ss.mdf['PE'] = None
                ss.mdf[['Market Cap', 'ROA', 'PS', 'PE']] = ss.mdf[['Market_Cap', 'ROA', 'PS', 'PE']].apply(pd.to_numeric, errors='coerce')

            else: # if stock prices were retrieved, run the rest of the calculations
                # .... and get summary of 8-K updates
                    ss.mdf['PERIOD_END_DATE'] = pd.to_datetime(ss.mdf['PERIOD_END_DATE']).dt.tz_localize("America/New_York")
                    temp_sp = ss.sp.sort_values(by='Date').set_index('Date').asfreq('D', method='ffill').reset_index() #fill in missing dates since there are no stock prices on weekends
                    ss.mdf = pd.merge(ss.mdf, temp_sp[['Date', 'Close']], left_on='PERIOD_END_DATE', right_on='Date', how='left').drop(columns=['Date'])
                    ss.mdf['ROA'] = ss.mdf['Net Income'] / ss.mdf['Total Assets']
                    ss.mdf['EPS'] = ss.mdf['Net Income'] / ss.mdf['Common Shares Outstanding']
                    ss.mdf['Market_Cap'] = ss.mdf['Close'] * ss.mdf['Common Shares Outstanding']
                    ss.mdf['PS'] = ss.mdf['Market_Cap'] / ss.mdf['Sales/Revenue']
                    ss.mdf['PE'] = ss.mdf['Close'] / ss.mdf['EPS']
                    ss.mdf[['Market_Cap', 'ROA', 'PS', 'PE']] = ss.mdf[['Market_Cap', 'ROA', 'PS', 'PE']].apply(pd.to_numeric, errors='coerce')

        # st.write(ss.mdf) #debug
        st.write(f":blue[Chart of Key Financials for {ss.company_name}, stock ticker '{ss.ticker}']")

        # Add CSS for min-width columns
        st.markdown("""
            <style>
                [data-testid="stColumn"] {
                    flex: 1 1 500px; 
                    min-width: 500px;
                }
                .centered-title {
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)

        # Use Streamlit columns to arrange charts side by side
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<p class="centered-title">Income Statement</p>', unsafe_allow_html=True)
            st.altair_chart(chart_income_statement, use_container_width=True)

        with col2:
            st.markdown('<p class="centered-title">Balance Sheet</p>', unsafe_allow_html=True)
            st.altair_chart(chart_balance_sheet, use_container_width=True)

        
        if len(ss.sp)!=0:
            with col3:
                # st.write(ss.sp) #debug
                chart_stock_price=get_line_chart(ss.sp,'Date','Ticker','Close',400,300)
                title=f'Stock Price for {ss.ticker}'
                st.markdown(f'<p class="centered-title">{title}</p>', unsafe_allow_html=True)
                st.altair_chart(chart_stock_price, use_container_width=True)
        
        # Prep for pulling LLM Summaries
        if ss.messages==[]: # Initialize the chat message history if no chat yet
            ss.df['PERIOD_END_DATE'] = pd.to_datetime(ss.df['PERIOD_END_DATE']).dt.strftime('%Y-%m-%d')
            temp_df = ss.df[ss.df['STATEMENT'] == 'Income Statement']
            data_json = temp_df[['PERIOD_END_DATE','METRIC_NAME','VALUE']].to_json(orient='records')
            ss.messages=[{"role":"user","content":f"""Put together a few bullets to summarize the trends of financial performance for {ss.df['COMPANY_NAME'].iloc[0].replace("'","")},
                with each bullet including average annual growth rates and any major changes in trend. Use the following data from SEC 10-Q reports,
                where the period_end_date is time, and the metric_name tells us what financial metric the value represents.
                Also include trends related to operating margin and net margin, and ensure calculations are correct, and trend analysis is accurate.
                Please verify the summary against the data and note any discrepancies: {data_json}"""}]
            ss.messages.append({"role":"system","content":"""Limit the responses to only questions that are relevant to this company's performance
                                If the user asks about performance relative to other competitors, do not respond with generic comparison frameworks, 
                                but rather answer with any data you do know about the industry performance or specific competitors and their performance
                                """})
            ss.llm_model='mistral-large2'

        # Pull the data based on the messages array
        prompt = "\n".join([
                f"{msg['role']}: {msg['content']}" if idx == 0 else "{}: {}".format(msg['role'], msg['content'].replace("'", "").replace('"', ''))
                for idx, msg in enumerate(ss.messages)
            ])

        # st.write(prompt) #debug
        sql = f"""
        select snowflake.cortex.complete('{ss.llm_model}', 
            '{prompt}, temperature=0.5'
            ) as response;
        """ 
        # st.write(sql) #debug
        if len(ss.messages)==2 and len(ss.analysis_result)==0:
            with st.spinner('Running LLM Analysis to provide a summary...'):
                response_df = ic(retrieve_data(sql))
            ss.analysis_result=response_df.iloc[0,0]
            ss.analysis_result=ss.analysis_result.replace('$', '\\$')
            print(f"Retrieved LLM response, length of response = {len(ss.analysis_result)}")
            ss.messages.append({"role":"assistant","content":ss.analysis_result})

        # Print the summary and the dataframe
        if len(ss.messages)>=3:
            st.markdown('<h3 style="color:#3894f0;">Summary of key financial trends from LLM Analysis:</h3>', unsafe_allow_html=True)
            st.markdown(ss.messages[2]["content"])

            # And write out the dataframe
            st.markdown('<h3 style="color:#3894f0;">SEC 10-Q data collected from Cybersyn:</h3>', unsafe_allow_html=True)            
            st.dataframe(ss.mdf.sort_values(by="PERIOD_END_DATE",ascending=False))
            # st.write(analysis_result_text) ######## debug purposes only

        #if the summary is complete, prep for the chat interactions
        if len(ss.messages)==3:
            ss.messages.append({"role": "assistant", "content": 
                f"""Welcome to Cortex Chat, powered by the Mistral-Large2 LLM Model. Please let me know if you have any questions about these metrics for {ss.company_name}. I may be able to provide some limited data on industry and competitors based on the public knowledge I was trained on"""})
            ss.messages.append({"role":"system","content":"""
                If the user asks about performance relative to other competitors, do not respond with generic comparison frameworks, 
                but rather answer with any data you do know about the industry performance or specific competitors and their performance. Do not state your role in the response.
                """})

        # st.markdown(f"""len(ss.messages)={len(ss.messages)} <br> ss.messages = {ss.messages}""", unsafe_allow_html=True) #debug                    
            
        # st.write(f"""Just before msg print. counter = {ss.counter}, ss.messages[-1]["role"]={ss.messages[-1]["role"]}""") #debug
        for message in [msg for msg in ss.messages[3:] if msg['role']!='system']: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # st.write(f"""Just before prompt. counter = {ss.counter}, ss.messages[-1]["role"]={ss.messages[-1]["role"]}""") #debug
        if ss.messages[-1]["role"] != "user":
            with st.chat_message('user'):
                st.chat_input('Enter question:', key='user_input', on_submit=add_user_message)

        #if the user entered a chat last, pull result from LLM
        if len(ss.messages)>=3 and ss.messages[-1]["role"] == "user":
            with st.chat_message('assistant'):
                with st.spinner("Thinking..."):
                    ss.llm_model='mistral-large'
                    response_df=ic(retrieve_data(sql))
            response_string=response_df.iloc[0,0]
            response_string=response_string.replace('"', '').replace('$', '\\$')
            print(f"Chat response received, response length={len(response_string)}")
            # st.write(f"""sending this to add_assistant_mesage: {response_string}""") #debug
            add_assistant_message(response_string)
            st.rerun()

            # st.write(f"""at end ss.messages[-1]["role"]={ss.messages[-1]["role"]}""") #debug
            print('#### at end of main') #debug

main()

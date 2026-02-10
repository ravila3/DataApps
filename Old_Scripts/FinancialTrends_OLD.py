# Import python packages
import streamlit as st
import pandas as pd
import altair as alt
import snowflake.connector
# import snowflake as snowflake
from streamlit import session_state as ss
from altair.expr import *
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col

connection_parameters = st.secrets["snowflake"]
session = Session.builder.configs(connection_parameters).create()

if 'ticker' not in ss:
    ss.ticker=''
if 'df' not in ss:
    ss.df=pd.DataFrame(columns=['Column1'])
if 'analysis_result' not in ss:
    ss.analysis_result=''
if 'counter' not in ss:
    ss.counter=0
if 'messages' not in ss:
    ss.messages=[]

# set page config and title
st.set_page_config( page_title="Financial Trends", layout="wide" )
st.markdown('<h2 style="color:#3894f0;">Financial Trends for Publically Traded Stocks</h2>', unsafe_allow_html=True)
st.write('Created by Rafael Avila leveraging Snowflake & Streamlit, using SEC Filings data provided by Cybersyn')

@st.cache_data(ttl="60m")
def retrieve_data(sql):
    conn = snowflake.connector.connect(**st.secrets["snowflake"])
    df = pd.read_sql(sql,conn)
    conn.close()
    return df

def get_line_chart(tdf,date,metric_name,value_field,width,height):

    hover = alt.selection_point(
        fields=[date, metric_name],
        nearest=True,
        on="mouseover",
        empty=False) #"none")
    legend_selection = alt.selection_point(fields=[metric_name], bind='legend')
    
    color_encoding = alt.Color(metric_name, legend=alt.Legend(title=metric_name, labelLimit=400), sort=alt.EncodingSortField('total_for_order', order='descending'))
    
    lines = (
        alt.Chart(ss.df)
        .mark_line(interpolate="linear")
        .encode(
            x=alt.X(date, type='temporal', title='Date (PST)', axis=alt.Axis(format='%b %Y')),
            y=alt.Y(value_field, type='quantitative', title=value_field, axis=alt.Axis(format='$,d')),
            color=color_encoding,
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.1)),
        ).add_params(legend_selection)
    ).properties(width=width, height=height)
    
    points = alt.Chart(ss.df).mark_point().encode(
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

    company_name = ss.df['COMPANY_NAME'].iloc[0] if not ss.df.empty else 'Unknown Company'
    st.write(f"Chart of Key Financials for {company_name}, stock ticker '{ss.ticker}'")
    st.altair_chart(lines + points, use_container_width=True)
    
    return (lines + points) #  + tooltips

def main():

    with st.form("ticker_form"):
        ss.ticker = st.text_input('Enter Stock Ticker', value='MSFT')
        submit_button = st.form_submit_button(label='Submit')

    if len(ss.df) == 0:
        df=pd.DataFrame(columns=['Column1']) # Initialize the dataframe
    
    if submit_button and (ticker:=ss.ticker and ss.ticker!=''):
        ticker_cleaned=ss.ticker.replace(" ","").upper()
        ss.ticker=ticker_cleaned
        ss.df=pd.DataFrame(columns=['Column1']) # Clear out the dataframe
        ss.messages=[] # Clear out messages
        ss.counter=0 # Reset counter
        # st.subheader('Dataframe Reset due to submit button') #debug
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
, i.company_name
, r.period_start_date
, r.period_end_date
, r.covered_qtrs
, r.statement
, r.tag
, case
    when r.tag in ('RevenueFromContractWithCustomerExcludingAssessedTax','RevenueFromContractWithCustomerIncludingAssessedTax','Revenues','InvestmentIncomeInterestAndDividend','InterestAndDividendIncomeOperating') then 'Sales/Revenue'
    when r.tag in ('OperatingIncomeLoss','IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest','IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments') then 'Operating Income'
    when r.tag in ('NetIncomeLoss','NetIncomeLossAvailableToCommonStockholdersBasic') then 'Net Income'
    when r.tag='CostOfRevenue' then 'Cost of Sales'
    when r.tag in ('CostsAndExpenses','BenefitsLossesAndExpenses') then 'Operating Costs'
    when r.tag='InterestAndDividendIncomeOperating' then 'Interest and Dividend Income'
    when r.tag in ('InterestExpense','InterestExpenseOperating','InterestExpenseNonoperating') then 'Interest Expense'
    when r.tag='InterestIncomeExpenseNet' then 'Interest Net Income'
    when r.tag='NoninterestIncome' then 'Non-Interest Income'
    -- when r.tag='Revenues' then 'Revenues'
    else 'Other'
  end as Metric_Name
, r.measure_description
, (TRUNC(TO_NUMERIC(r.value),0)) AS value
, ROW_NUMBER() OVER (PARTITION BY r.cik, r.period_start_date, r.period_end_date, r.covered_qtrs, tag ORDER BY ri.filed_date, r.adsh DESC) AS rn
FROM SEC_FILINGS.cybersyn.sec_cik_index AS i
JOIN SEC_FILINGS.cybersyn.company_index as c on (c.cik=i.cik)
JOIN SEC_FILINGS.cybersyn.sec_report_attributes AS r ON (r.cik = i.cik)
JOIN SEC_FILINGS.cybersyn.sec_report_index as ri on (ri.adsh=r.adsh)
WHERE 
  c.primary_ticker='{ticker_cleaned}'
--  i.company_name like '%AT&T%' --'AMR CORP'
--  AND i.sic_code_description = 'AIR TRANSPORTATION, SCHEDULED'
  AND r.period_end_date >= '2010-01-01'
  -- AND r.period_end_date = '2023-07-31'

  AND r.covered_qtrs = 1
  AND TRY_CAST(r.value AS NUMBER) IS NOT NULL
  AND r.statement in ('Income Statement')-- ,'Balance Sheet','Cash Flow'
  AND form_type='10-Q'
  AND r.metadata is null -- businesssegments in ('Communications') -- and subsegments is null and productorservice is null --    ,'CorporateAndOther','LatinAmericaBusinessSegment'
  AND (r.tag in ('RevenueFromContractWithCustomerExcludingAssessedTax','RevenueFromContractWithCustomerIncludingAssessedTax','OperatingIncomeLoss'
        ,'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest','InterestExpenseOperating','NetIncomeLoss','CostsAndExpenses','BenefitsLossesAndExpenses'
        ,'InterestExpense','InterestExpenseNonoperating','Revenues','InterestIncomeExpenseNet','NoninterestIncome','InterestAndDividendIncomeOperating'
        ,'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments')
  or r.tag like '%Revenue%' or r.tag like '%Income%' )
)
  
select form_type, primary_ticker, company_name, period_end_date, statement, Metric_Name, max(cast(value as integer)) as value  --, tag, measure_description, rn, businesssegments, subsegments, productorservice, ConsolidationItems, metadata --, max(Value) as Value
from cf
where value<>0 and rn=1 and Metric_Name<>'Other' --and tag not in ('RevenueFromContractWithCustomerExcludingAssessedTax','CostOfRevenue')
--AND period_end_date >= '2024-01-01' -- LIMIT SCOPE FOR DEVELOPMENT EFFICIENCY
group by 1,2,3,4,5,6
order by period_end_date desc
--limit 100
            """
        
        with st.spinner('Pulling 10-Q Financial Data...'):
            ss.df = retrieve_data(sql)
            # ss.df=df
            # df = conn.query(sql)
            if len(ss.df)==0:
                st.write(f"No Data Retrieved for stock ticker '{ticker_cleaned}'")
    
    # if not df.empty:
    # st.write(f"did it meet condition for chart? len(ss.df)={len(ss.df)}") # debug
    if len(ss.df)!=0:
        # st.write(f"about to write chart (ss.df)={ss.df}") # debug
        ss.df['VALUE'] = ss.df['VALUE'].astype(int)
        # st.write(ss.df) ################ debug purposes only
        get_line_chart(ss.df,'PERIOD_END_DATE','METRIC_NAME','VALUE',700,300)
        
        if ss.messages==[]: # Initialize the chat message history - old logic was "messages" not in ss.keys()
            ss.df['PERIOD_END_DATE'] = pd.to_datetime(ss.df['PERIOD_END_DATE']).dt.strftime('%Y-%m-%d')
            data_json = ss.df[['PERIOD_END_DATE','METRIC_NAME','VALUE']].to_json(orient='records')
            ss.messages=[{"role":"user","content":f"""Put together a few bullets to summarize the trends of financial performance for {ss.df['COMPANY_NAME'].iloc[0].replace("'","")},
                with each bullet including average annual growth rates and any major changes in trend. Use the following data from SEC 10-Q reports,
                where the period_end_date is time, and the metric_name tells us what financial metric the value represents.
                Also include trends related to operating margin and net margin, and ensure calculations are correct.
                Please verify the summary against the data and note any discrepancies: {data_json}"""}]
            ss.messages.append({"role":"system","content":"Limit the responses to only questions that are relevant to this company's performance"})
        
            # prompt = ss.messages[-1]["content"]
            # st.write(prompt)
        # Call the Cortex `COMPLETE` function
        analysis_query = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2',
            '{ss.messages[0]["content"]}, temperature=0.5')  
        """  #,guardrails=True

        with st.spinner('Running LLM Analysis to provide a summary...'):
            ss.analysis_result = retrieve_data(analysis_query)
            # ss.analysis_result = pd.DataFrame([["For testing, Net Income = $10000, Sales = $40000"]], columns=["Column1"]) #### DEBUG ONLY
       
        if len(ss.messages)==2 and len(ss.analysis_result)!=0:
            analysis_result_text=ss.analysis_result.iloc[0,0]
            analysis_result_text = analysis_result_text.replace('$', '\\$')
            ss.messages.append({"role":"assistant","content":analysis_result_text})
            # st.write(analysis_result_text) ######## debug purposes only

        # st.write(f"len(ss.messages)={len(ss.messages)}") # debug
        # st.write(f"len(ss.df)={len(ss.df)}") # debug
        if len(ss.messages)>=3:
            # Print the summary
            st.markdown('<h3 style="color:#3894f0;">Summary of key financial trends from LLM Analysis:</h3>', unsafe_allow_html=True)
            st.markdown(ss.messages[2]["content"])

            # And write out the dataframe
            st.markdown('<h3 style="color:#3894f0;">Raw SEC 10-Q data collected from Cybersyn:</h3>', unsafe_allow_html=True)            
            st.dataframe(ss.df)

            def add_user_message():
                # Chat input
                ss.counter=ss.counter+1 # DEBUG
                if ss.user_input:
                    sanitized_user_input=ss.user_input.replace("'","")
                    ss.messages.append({"role": "user", "content": sanitized_user_input})
                    ss.user_input="" # clear message after sending
                    # st.write(f"just added messaage: {sanitized_user_input}") #debug
            
            def add_response(response):
                ss.messages.append({"role": "assistant", "content": response})

            if len(ss.messages)==3:
                ss.messages.append({"role": "assistant", "content": 
                    f"""Welcome to Cortex Chat, powered by the Mistral-Large2 LLM Model. Please let me know if you have any questions about these metrics for {ss.df['COMPANY_NAME'].iloc[0]}."""})

            # st.write(f"ss.messages[-1]={ss.messages[-1]}") # debug
            # st.write(f"""Just before querying LLM. counter = {ss.counter}, ss.messages[-1]["role"]={ss.messages[-1]["role"]}""") # DEBUG

            # process user questions and get an LLM response
            if ss.messages[-1]["role"] == "user":
                with st.spinner("Thinking..."):
                    prompt = "\n".join([
                            f"{msg['role']}: {msg['content']}" if idx == 0 else "{}: {}".format(msg['role'], msg['content'].replace("'", "").replace('"', ''))
                            for idx, msg in enumerate(ss.messages)
                        ])
                    # st.write(prompt) #debug
                    sql = f"""
                    select snowflake.cortex.complete(
                        'mistral-large2', 
                        '{{"prompt": "{prompt}"}}, temperature=0.5'
                        ) as response;
                    """ 
                    response_df=retrieve_data(sql)
                    response_string=response_df.iloc[0,0]
                    response_string=response_string.replace('"', '').replace('$', '\\$')
                    # st.write(f"response_string={response_string}") #debug
                    add_response(response_string)
            
            # st.write(f"""Just before msg print. counter = {ss.counter}, ss.messages[-1]["role"]={ss.messages[-1]["role"]}""") #debug
            for message in ss.messages[3:]: # Display the prior chat messages
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            # st.write(f"""Just before prompt. counter = {ss.counter}, ss.messages[-1]["role"]={ss.messages[-1]["role"]}""") #debug
            if ss.messages[-1]["role"] == "assistant":
                st.text_input('Enter question:', key='user_input', on_change=add_user_message)
            
            # st.write(f"""at end ss.messages[-1]["role"]={ss.messages[-1]["role"]}""") #debug

main()

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

if 'cik' not in ss:
    ss.cik=''
    ss.ticker=''
    ss.company_name=''
    ss.company_and_ticker=''
    ss.llm_model=''
    ss.df=pd.DataFrame(columns=['Column1'])
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

    if len(ss.df) == 0:
        df=pd.DataFrame(columns=['Column1']) # Initialize the dataframe
    if submit_button and company_and_ticker==None:
        st.subheader(':red[Please select a company]')
    if submit_button and (company_and_ticker!=None and company_and_ticker!=ss.company_and_ticker):
        ss.company_and_ticker=company_and_ticker
        ss.cik = company_lookup_df.loc[company_lookup_df['COMPANY_AND_TICKER'] == ss.company_and_ticker, 'CIK'].values[0]
        ss.ticker = company_lookup_df.loc[company_lookup_df['COMPANY_AND_TICKER'] == ss.company_and_ticker, 'PRIMARY_TICKER'].values[0]
        ss.company_name = company_lookup_df.loc[company_lookup_df['COMPANY_AND_TICKER'] == ss.company_and_ticker, 'COMPANY_NAME'].values[0]
        ss.df=pd.DataFrame(columns=['Column1']) # Clear out the dataframe
        ss.analysis_result=''
        ss.messages=[] # Clear out messages
        ss.counter=0 # Reset counter
        # st.write(f"cik={ss.cik}, ticker={ss.ticker}, company_name={ss.company_name}, company_and_ticker={ss.company_and_ticker}\n") #debug
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
, initcap(i.company_name) as company_name
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
  c.cik='{ss.cik}'
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
  
select form_type, cik, primary_ticker, company_name, period_end_date, statement, Metric_Name, max(cast(value as integer)) as value  --, tag, measure_description, rn, businesssegments, subsegments, productorservice, ConsolidationItems, metadata --, max(Value) as Value
from cf
where value<>0 and rn=1 and Metric_Name<>'Other' --and tag not in ('RevenueFromContractWithCustomerExcludingAssessedTax','CostOfRevenue')
--AND period_end_date >= '2024-01-01' -- LIMIT SCOPE FOR DEVELOPMENT EFFICIENCY
group by 1,2,3,4,5,6,7
order by period_end_date desc
--limit 100
            """
        
        with st.spinner('Pulling 10-Q Financial Data...'):
            ss.df = retrieve_data(sql)
            # ss.df=df
            # df = conn.query(sql)
            if len(ss.df)==0:
                st.write(f"No Data Retrieved for company '{ss.company_and_ticker}'")

    
    # if not df.empty:
    # st.write(f"did it meet condition for chart? len(ss.df)={len(ss.df)}") # debug
    if len(ss.df)!=0:
        # st.write(f"about to write chart (ss.df)={ss.df}") # debug
        ss.df['VALUE'] = ss.df['VALUE'].astype(int)
        # st.write(ss.df) ################ debug purposes only
        get_line_chart(ss.df,'PERIOD_END_DATE','METRIC_NAME','VALUE',700,300)
        
        # Prep for pulling LLM Summaries
        if ss.messages==[]: # Initialize the chat message history if no chat yet
            ss.df['PERIOD_END_DATE'] = pd.to_datetime(ss.df['PERIOD_END_DATE']).dt.strftime('%Y-%m-%d')
            data_json = ss.df[['PERIOD_END_DATE','METRIC_NAME','VALUE']].to_json(orient='records')
            ss.messages=[{"role":"user","content":f"""Put together a few bullets to summarize the trends of financial performance for {ss.df['COMPANY_NAME'].iloc[0].replace("'","")},
                with each bullet including average annual growth rates and any major changes in trend. Use the following data from SEC 10-Q reports,
                where the period_end_date is time, and the metric_name tells us what financial metric the value represents.
                Also include trends related to operating margin and net margin, and ensure calculations are correct.
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

        # sql = f"""
        # select snowflake.cortex.complete('{ss.llm_model}', 
        #     '{{"prompt": "{prompt}"}}, temperature=0.5'
        #     ) as response;
        # """ 

        # sql = f"""
        #             SELECT SNOWFLAKE.CORTEX.COMPLETE({ss.llm_model},
        #             '{ss.messages[0]["content"]}, temperature=0.5')  
        #         """  #,guardrails=True

        # st.write(f"response_string={response_string}") #debug
        

        
        # st.markdown(f"""before pulling summary data: len(ss.messages)={len(ss.messages)} <br> ss.messages = {ss.messages}""", unsafe_allow_html=True) #debug

        # If no analysis results yet, pull, analysis summary

        # st.write(f"Just before pulling LLM Analysis, len(ss.messages)={len(ss.messages)} and len(ss.analysis_result)={len(ss.analysis_result)}, ss.analysis_result = {ss.analysis_result}") #debug

        if len(ss.messages)==2 and len(ss.analysis_result)==0:
            with st.spinner('Running LLM Analysis to provide a summary...'):
                response_df = retrieve_data(sql)
            ss.analysis_result=response_df.iloc[0,0]
            ss.analysis_result=ss.analysis_result.replace('$', '\\$')
            ss.messages.append({"role":"assistant","content":ss.analysis_result})

        # Print the summary and the dataframe
        if len(ss.messages)>=3:
            st.markdown('<h3 style="color:#3894f0;">Summary of key financial trends from LLM Analysis:</h3>', unsafe_allow_html=True)
            st.markdown(ss.messages[2]["content"])

            # And write out the dataframe
            st.markdown('<h3 style="color:#3894f0;">Raw SEC 10-Q data collected from Cybersyn:</h3>', unsafe_allow_html=True)            
            st.dataframe(ss.df)
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
                    response_df=retrieve_data(sql)
            response_string=response_df.iloc[0,0]
            response_string=response_string.replace('"', '').replace('$', '\\$')
            # st.write(f"""sending this to add_assistant_mesage: {response_string}""") #debug
            add_assistant_message(response_string)
            st.rerun()

            # st.write(f"""at end ss.messages[-1]["role"]={ss.messages[-1]["role"]}""") #debug
            print('#### at end of main') #debug

main()

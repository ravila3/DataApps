# Import python packages
import streamlit as st
import pandas as pd
import altair as alt
import snowflake.snowpark as snowpark
# import json

from snowflake.snowpark.functions import call_builtin
from snowflake.snowpark.context import get_active_session
from altair.expr import format

# Write directly to the app
st.markdown('<h2 style="color:#3894f0;">Financial Trends for Publically Traded Stocks</h2>', unsafe_allow_html=True)

# Get the current credentials
session = get_active_session()

def get_line_chart(df,date,field_name,metric_name,width,height):
    
    hover = alt.selection_point(
        fields=[date, field_name],
        nearest=True,
        on="mouseover",
        empty=False) #"none")
    legend_selection = alt.selection_point(fields=[field_name], bind='legend')
    
    color_encoding = alt.Color(field_name, legend=alt.Legend(title=field_name, labelLimit=400), sort=alt.EncodingSortField('total_for_order', order='descending'))
    
    lines = (
        alt.Chart(df)
        .mark_line(interpolate="linear")
        .encode(
            x=alt.X(date, title='Date (PST)', axis=alt.Axis(format='%b %Y')),
            y=alt.Y(metric_name, title=metric_name),
            color=color_encoding,
            opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.1)),
        ).add_params(legend_selection)
    ).properties(width=width, height=height)
    
    points = alt.Chart(df).mark_point().encode(
        x=date,
        y=metric_name,
        color=color_encoding,
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip(date, format='%m/%d/%y(%a) %I%p', title="Date (PST)"),
            field_name,
            alt.Tooltip(metric_name, format=',.0f', title=metric_name)
        ]
    ).add_params(hover)  #.interactive()
    
    return (lines + points) #  + tooltips

def main():

    with st.form("ticker_form"):
        ticker = st.text_input('Enter Stock Ticker', value='SNOW')
        submit_button = st.form_submit_button(label='Submit')
    
    if submit_button and ticker:
        query = f"""
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
    when r.tag in ('RevenueFromContractWithCustomerExcludingAssessedTax','RevenueFromContractWithCustomerIncludingAssessedTax') then 'Sales/Revenue'
    when r.tag in ('OperatingIncomeLoss','InterestExpenseOperating') then 'Operating Income'
    when r.tag in ('NetIncomeLoss','NetIncomeLossAvailableToCommonStockholdersBasic') then 'Net Income'
    when r.tag='CostOfRevenue' then 'Cost of Sales'
    when r.tag in ('CostsAndExpenses','BenefitsLossesAndExpenses') then 'Operating Costs'
    when r.tag='InterestAndDividendIncomeOperating' then 'Interest and Dividend Income'
    when r.tag='InterestExpense' then 'Interest Expense'
    when r.tag='InterestIncomeExpenseNet' then 'Interest Net Income'
    when r.tag='NoninterestIncome' then 'Non-Interest Income'
    when r.tag='Revenues' then 'Revenues'
    else 'Other'
  end as Metric_Name
, r.measure_description
, (TO_NUMERIC(r.value)) AS value
, ROW_NUMBER() OVER (PARTITION BY r.cik, r.period_start_date, r.period_end_date, r.covered_qtrs, tag ORDER BY ri.filed_date, r.adsh DESC) AS rn
FROM SEC_FILINGS.cybersyn.sec_cik_index AS i
JOIN SEC_FILINGS.cybersyn.company_index as c on (c.cik=i.cik)
JOIN SEC_FILINGS.cybersyn.sec_report_attributes AS r ON (r.cik = i.cik)
JOIN SEC_FILINGS.cybersyn.sec_report_index as ri on (ri.adsh=r.adsh)
WHERE 
  c.primary_ticker='{ticker.upper()}'
--  i.company_name like '%AT&T%' --'AMR CORP'
--  AND i.sic_code_description = 'AIR TRANSPORTATION, SCHEDULED'
  AND r.period_end_date >= '2010-01-01'
  -- AND r.period_end_date = '2023-07-31'

  AND r.covered_qtrs = 1
  AND TRY_CAST(r.value AS NUMBER) IS NOT NULL
  AND r.statement in ('Income Statement')-- ,'Balance Sheet','Cash Flow'
  AND form_type='10-Q'
  AND r.metadata is null -- businesssegments in ('Communications') -- and subsegments is null and productorservice is null --    ,'CorporateAndOther','LatinAmericaBusinessSegment'
  AND (r.tag in ('RevenueFromContractWithCustomerExcludingAssessedTax','RevenueFromContractWithCustomerIncludingAssessedTax','OperatingIncomeLoss','InterestExpenseOperating','NetIncomeLoss','CostsAndExpenses','BenefitsLossesAndExpenses','InterestExpense','Revenues','InterestIncomeExpenseNet','NoninterestIncome','InterestAndDividendIncomeOperating')
  or r.tag like '%Revenue%' or r.tag like '%Income%' )
)
  
select form_type, primary_ticker, company_name, period_end_date, statement, tag, measure_description, Metric_Name, value, rn--, businesssegments, subsegments, productorservice, ConsolidationItems, metadata --, sum(Value) as Value
from cf
where value<>0 and rn=1 and Metric_Name<>'Other' --and tag not in ('RevenueFromContractWithCustomerExcludingAssessedTax','CostOfRevenue')
--group by 1,2,3,4,5,6 --,7,8,9,10
order by period_end_date, tag desc
--limit 100
            """
        
        with st.spinner('Pulling 10-Q Financial Data...'):
            df = session.sql(query).to_pandas()
            if df.empty:
                st.write('No Data Retrieved for that Ticker')
    
        if not df.empty:
            
            chart=get_line_chart(df,'PERIOD_END_DATE','METRIC_NAME','VALUE',700,600)
            
            company_name = df['COMPANY_NAME'].iloc[0] if not df.empty else 'Unknown Company'
            st.write(f"Chart of Key Financials for {company_name}, ticker '{ticker}'")
            st.altair_chart(chart, use_container_width=True)

            # Now create the LLM Summary
            
            # Convert the pandas DataFrame to JSON
            df['PERIOD_END_DATE'] = pd.to_datetime(df['PERIOD_END_DATE']).dt.strftime('%Y-%m-%d')
            # df['PERIOD_END_DATE'] = df['PERIOD_END_DATE'].dt.strftime('%Y-%m-%d')
            data_json = df[['PERIOD_END_DATE','METRIC_NAME','VALUE']].to_json(orient='records')
            # escaped_data_json = data_json.replace("'", "''")
            
            # Call the Cortex `COMPLETE` function
            analysis_query = f"""
                SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2','Put together a few bullets to summarize the trends of financial performance, with each bullet including average annual growth rates and any major changes in trend. Use the following data from SEC 10-Q reports, where the period_end_date is time, and the metric_name tells us what financial metric the value represents.
                Please verify the summary against the data and note any discrepancies: {data_json},temperature=0.5' 
                )  
            """  #,guardrails=True
            
            # Execute the query
            with st.spinner('Running LLM Analysis to provide a summary...'):
                analysis_result = session.sql(analysis_query).collect()
            
            # Print the result
            st.markdown('<h3 style="color:#3894f0;">Summary of key financial trends from LLM Analysis:</h3>', unsafe_allow_html=True)
        
            for row in analysis_result:
                # st.markdown(f"**Summary of key trends:**\n\n{row[0]}")
                summary_text = row[0]
                summary_text = summary_text.replace('$', '\\$')

                st.markdown(summary_text)

            # And write out the dataframe
            st.markdown('<h3 style="color:#3894f0;">Raw SEC 10-Q data collected from Cybersyn:</h3>', unsafe_allow_html=True)            
            st.dataframe(df)

    return()


main()

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
import yfinance as yf
import snowflake.connector
from icecream import ic
# import snowflake as 
from streamlit import session_state as ss
from altair.expr import *
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col
from datetime import datetime,timedelta
from SEC_Edgar_Loader import sec_edgar_financial_load

connection_parameters = st.secrets["snowflake"]
session = Session.builder.configs(connection_parameters).create()

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

def get_line_chart(tdf,date,metric_name,value_field,precision,width,height):

    # st.write(tdf) #debug
   # Aggregate the DataFrame to allow sort by for chart legend
    totals_by_metric_df = tdf.groupby(metric_name, as_index=False)[value_field].sum()
    totals_by_metric_df.rename(columns={value_field: 'total_value'}, inplace=True)
    tdf = tdf.merge(totals_by_metric_df, on=metric_name, how='left')
    tdf['total_for_order'] = tdf['total_value']
    tdf.drop(columns=['total_value'], inplace=True)

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
            alt.Tooltip(value_field, type='quantitative', format=f'$,.{precision}f', title=value_field)
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
        ss.filings_df=ss.annual_financials=ss.quarterly_financials=ss.df=ss.sp=ss.companyfacts_metrics_yahoo_dict=pd.DataFrame() #columns=['Column1']) # Clear out the dataframe
        ss.analysis_result=''
        ss.messages=[] # Clear out messages
        ss.counter=0 # Reset counter
        # st.write(f"cik={ss.cik}, ticker={ss.ticker}, company_name={ss.company_name}, company_and_ticker={ss.company_and_ticker}\n") #debug
        # st.subheader('Dataframe Reset due to submit button') #debug

        with st.spinner('Pulling 10-Q Financial Data...'):
            ss.filings_df,ss.quarterly_financials,ss.annual_financials=sec_edgar_financial_load(ss.cik)
            
            ss.ticker=ss.filings_df['primary_ticker'].iloc[0]

            today = datetime.today().date()
            x_years_ago = today - timedelta(days=4*365)
            start_date = x_years_ago.replace(month=1, day=1) #set start date to Jan 1 of the day 4 years ago
            today = datetime.today().date() #datetime.today().strftime('%Y-%m-%d')
            
            # ss.filings_df=ss.filings_df[ss.filings_df['filed_date']>=start_date]
            # ss.quarterly_financials['end_date'] = pd.to_datetime(ss.quarterly_financials['end_date']).dt.date
            ss.quarterly_financials=ss.quarterly_financials[ss.quarterly_financials['end_date']>=start_date]

            # st.write('filings_df',ss.filings_df,'annual_financials',ss.annual_financiFals,'quarterly_financials',ss.quarterly_financials)

            ticker=yf.Ticker(ss.ticker)
            historical_prices=ticker.history(period='1d',start='2017-01-01',end=today) # gets following fields: Date, High, Low, Close, Volume, Dividends, Stock Splits
            ss.sp=historical_prices.reset_index()
            ss.sp['Ticker']=ss.ticker
            ss.sp=ss.sp.rename(columns={'Close':'Stock Price'})
            # st.write('stock prices',ss.sp) #debug

            stats=ticker.info
            # st.write('yahoo stats',stats) #debug write out yahoo dict

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

            ss.companyfacts_metrics_yahoo_dict = {metric[0]: {'format': metric[1]} for metric in CompanyInfoYahoo}

            if len(ss.quarterly_financials)==0:
                st.write(f"No 10-Q Data Retrieved for '{ss.company_and_ticker}'")
            if len(ss.sp)==0:
                st.write(f"No Stock Price Data Retrieved for '{ss.company_and_ticker}'")                
    
    # if not df.empty:
    # st.write(f"did it meet condition for chart? len(ss.df)={len(ss.df)}") # debug
    if len(ss.quarterly_financials)!=0:
        # st.write(f"about to write chart (ss.df)={ss.df}") # debug
        # st.write(ss.df) ################ debug purposes only
        # filtered_df=ss.df[ss.df['STATEMENT']=='Income Statement']

        # Step 1: Filter columns by "Income Statement" category
        # st.write('ss.companyfacts_metrics_dict',ss.companyfacts_metrics_dict) #debug
        # income_statement_columns = [metric for metric,details in ss.companyfacts_metrics_dict.items() if details['category'] == 'Income Statement']
        
        
        # income_statement_columns_for_chart=['end_date']+ss.income_statement_columns_for_chart  
        # income_statement_columns_for_chart=['end_date']+income_statement_columns_for_chart
        # income_statement_columns_for_chart.insert(0, 'end_date') # Include 'end_date' for melting
        # ss.quarterly_financials['end_date'] = pd.to_datetime(ss.quarterly_financials['end_date']).dt.date
        # st.write(f"ss.quarterly_financials:",ss.quarterly_financials.dtypes) #debug
        # st.write(f"income_statement_columns_for_chart: {income_statement_columns_for_chart}") #debug
        chart_df=ss.quarterly_financials[['end_date']+ss.income_statement_columns_for_chart].melt(id_vars=['end_date'], var_name='metric', value_name='value')
        chart_df=chart_df[pd.notna(chart_df['value'])]

        # Apply labels from dict, if there's a conflict take the max value
        # label_map = {metric: details['label'] for metric, details in ss.companyfacts_metrics_dict.items()}
        # chart_df['label'] = chart_df['metric'].map(label_map)
        # chart_df.drop(columns='metric',inplace=True)
        # chart_df.rename(columns={'label': 'metric'}, inplace=True)
        chart_df_max = chart_df.groupby(['end_date', 'metric']).agg({'value': 'max'}).reset_index()
        chart_income_statement=get_line_chart(chart_df_max,'end_date','metric','value',0,400,300)

        # balance_sheet_columns=ss.balance_sheet_columns.insert(0, 'end_date')  # Include 'end_date' for melting
        chart_df=ss.quarterly_financials[['end_date']+ss.balance_sheet_columns].melt(id_vars=['end_date'], var_name='metric', value_name='value')
        chart_df=chart_df[pd.notna(chart_df['value'])]
        chart_balance_sheet=get_line_chart(chart_df,'end_date','metric','value',0,400,300)
        
        # st.altair_chart(chart_income_statement, use_container_width=True) #temp
        # st.altair_chart(chart_balance_sheet, use_container_width=True) #temp

        try:
            # If no stock prices retrieved, set merged dataframe to pivoted data
            ss.quarterly_financials['Operating Margin'] = ss.quarterly_financials['Operating Income'] / ss.quarterly_financials['Revenue/Sales']
            ss.quarterly_financials['Net Margin'] = ss.quarterly_financials['Net Income'] / ss.quarterly_financials['Revenue/Sales']
            ss.quarterly_financials['ROA'] = ss.quarterly_financials['Net Income'] / ss.quarterly_financials['Total Assets']
            ss.quarterly_financials['EPS'] = ss.quarterly_financials['Net Income'] / ss.quarterly_financials['Common Shares Outstanding']
            ss.quarterly_financials['EBITDA'] = (
                    ss.quarterly_financials['Net Income'] + ss.quarterly_financials['Interest Expense'] + 
                    ss.quarterly_financials['Income Tax'] + 
                    ss.quarterly_financials['Depreciation'] + ss.quarterly_financials['Amortization'])
            
            if len(ss.sp) == 0:
                ss.quarterly_financials['Market_Cap'] = None
                ss.quarterly_financials['PS'] = None
                ss.quarterly_financials['PE'] = None
                ss.quarterly_financials[['Market_Cap', 'ROA', 'PS', 'PE']] = ss.quarterly_financials[['Market_Cap', 'ROA', 'PS', 'PE']].apply(pd.to_numeric, errors='coerce')

            else:
                if 'Stock Price' not in ss.quarterly_financials.columns:
                    temp_sp = ss.sp.sort_values(by='Date').set_index('Date').asfreq('D', method='ffill').reset_index()
                    ss.quarterly_financials['end_date'] = pd.to_datetime(ss.quarterly_financials['end_date']).dt.date #.dt.tz_localize(None)
                    temp_sp['Date'] = pd.to_datetime(temp_sp['Date']).dt.date #.tz_localize(None)

                    ss.quarterly_financials = pd.merge(
                        ss.quarterly_financials, 
                        temp_sp[['Date', 'Stock Price']], 
                        left_on='end_date', 
                        right_on='Date', 
                        how='left'
                    ).drop(columns=['Date'])
                
                # ss.quarterly_financials=ss.quarterly_financials.rename(columns={'Close':'Stock Price'})

                ss.quarterly_financials['Market_Cap'] = (ss.quarterly_financials['Stock Price'] * ss.quarterly_financials['Common Shares Outstanding']).astype(int)
                ss.quarterly_financials['PS'] = ss.quarterly_financials['Market_Cap'] / ss.quarterly_financials['Revenue/Sales']
                ss.quarterly_financials['PE'] = ss.quarterly_financials['Stock Price'] / ss.quarterly_financials['EPS Basic']
                ss.quarterly_financials[['Market_Cap', 'ROA', 'PS', 'PE']] = ss.quarterly_financials[['Market_Cap', 'ROA', 'PS', 'PE']].apply(pd.to_numeric, errors='coerce')

        except ValueError as e:
            print(f"Failed to calculate metrics: {e}")

        # st.write('ss.quarterly_financials with ratios added',ss.quarterly_financials) #debug
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
                chart_stock_price=get_line_chart(ss.sp,'Date','Ticker','Stock Price',2,400,300)
                title=f'Stock Price for {ss.ticker}'
                st.markdown(f'<p class="centered-title">{title}</p>', unsafe_allow_html=True)
                st.altair_chart(chart_stock_price, use_container_width=True)

        # Prep for pulling LLM Summaries
        if ss.messages==[]: # Initialize the chat message history if no chat yet
            income_metrics_df=ss.quarterly_financials[['end_date']+ss.income_statement_columns_for_chart]
            data_json = ss.quarterly_financials[['end_date']+ss.income_statement_columns].dropna().to_json(orient='records')

            # Function to filter out null values
            def filter_nulls_and_format_date(row):
                # row = {k: v for k, v in row.items() if pd.notna(v)}
                row = {k: (int(v) if isinstance(v, (int, float)) else v) for k, v in row.items() if pd.notna(v)}
                row['end_date'] = row['end_date'].strftime('%b %Y')
                return row

            # Convert each row to a dictionary excluding null values
            data_json = income_metrics_df.apply(lambda x: filter_nulls_and_format_date(x), axis=1).to_json(orient='records')
            # st.write('data_json',data_json)
            
            # st.write('data_json',data_json)
            ss.messages=[{"role":"user","content":f"""Put together 2-3 bullets on each income statement metric summarizing the trends of financial performance for {ss.company_name.replace("'","")}.
                Include average annual growth rates and any years with major changes in growth rates year over year, and also note an observation on trend changes for the last few quarters, and the last quarterly result.
                When comparing quarterly performance year over year, please compare against the same quarter in the previous years due to seasonality, and include the percent change if it is helpful.
                Use the following data from SEC 10-Q reports, where the end_date is the end of quarter (MM-YYYY), and the metric names reference the income statement line items.
                Also include operating margin and net margin as key metrics and always state the specific margins, and ensure calculations are correct, and trend analysis is accurate.
                When stating specific values, please note them in millions of $ where appropriate. Please verify the summary against the data and note any discrepancies: {data_json}"""}]
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
            '{prompt}, temperature=0.4'
            ) as response;
        """ 
        # st.write(sql) #debug
        # ss.messages.append({"role":"assistant","content":"message to test without LLM"}) #debug

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
            st.markdown('<h3 style="color:#3894f0;">SEC 10-Q data collected from SEC Edgar:</h3>', unsafe_allow_html=True)
            filtered_df=ss.quarterly_financials.set_index('end_date').dropna(axis=1, how='all')

            # Format the dataframe
            formatted_df = filtered_df.style.format({
                'Revenue/Sales': '${:,.0f}',
                'Cost of Sales': '${:,.0f}',
                'Gross Profit': '${:,.0f}',
                'Total Expenses': '${:,.0f}',
                'Operating Expenses': '${:,.0f}',
                'SG&A Exp': '${:,.0f}',
                'R&D Exp': '${:,.0f}',
                'Operating Income': '${:,.0f}',
                'Pretax Income': '${:,.0f}',
                'Net Income': '${:,.0f}',
                'Interest Expense': '${:,.0f}',
                'EPS Basic': '{:.2f}',
                'EPS Diluted': '{:.2f}',
                'Total Assets': '${:,.0f}',
                'Curr Assets': '${:,.0f}',
                'Deposits': '${:,.0f}',
                'Liabilities': '${:,.0f}',
                'Curr Liabilities': '${:,.0f}',
                'Long Term Debt': '${:,.0f}',
                'Stockholder Equity': '${:,.0f}',
                'Common Shares Outstanding': '{:,.0f}',
                'Income Tax': '${:,.0f}',
                'Depreciation': '${:,.0f}',
                'Amortization': '${:,.0f}',
                'Operating Margin': '{:.2%}',
                'Net Margin': '{:.2%}',
                'ROA': '{:.2%}',
                'EPS': '{:.2f}',
                'EBITDA': '${:,.2f}',
                'Stock Price': '${:,.2f}',
                'Market_Cap': '${:,.0f}',
                'PS': '{:.2f}',
                'PE': '{:.2f}'
            })

            st.dataframe(formatted_df) #.sort_values(by="end_date",ascending=False
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

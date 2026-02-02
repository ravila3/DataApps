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
from icecream import ic
from streamlit import session_state as ss
from altair.expr import *
import snowflake.connector
from snowflake.snowpark import Session
# from snowflake.snowpark.context import get_active_session
# from snowflake.snowpark.functions import col
from datetime import datetime,timedelta
from SEC_Edgar_Loader import sec_edgar_financial_load,get_company_tickers_df

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
st.set_page_config( page_title="Financial Trends", layout="wide" )
st.markdown('<h2 style="color:#3894f0;">Financial Trends for Publically Traded Stocks</h2>', unsafe_allow_html=True)
st.write('Created by Rafael Avila leveraging Snowflake & Streamlit, using SEC Filings data provided by SEC Edgar platform. Analysis Summary and AI chat powered by mistral-large2 AI model')

@st.cache_data(ttl="24h")
def retrieve_data(sql):
    # st.write(st.secrets["snowflake"]) #debug
    conn = snowflake.connector.connect(**st.secrets["snowflake"])
    # session = conn.Session()
    # st.write(f"sql = {sql}") #debug
    df = pd.read_sql(sql,conn)
    conn.close()
    return df


def retrieve_llm_data():
    prompt = "\n".join([
        "{}: {}".format(msg['role'], msg['content'].replace("'", "").replace('"', ''))
        for idx, msg in enumerate(ss.messages)
    ])

    # st.write(prompt) #debug
    sql = f"""
    select snowflake.cortex.complete('{ss.llm_model}', 
        '{prompt}, temperature=0.4'
        ) as response;
    """ 
    
    response_df = retrieve_data(sql)
    return response_df

def add_user_message(user_input):
    # Chat input
    ss.counter=ss.counter+1 # `DEBUG`
    sanitized_user_input=user_input.replace("'","").replace('"',"")
    ss.messages.append({"role": "user", "content": sanitized_user_input})
    #ss.user_input="" # clear message after sending
    # st.write(f"just added messaage: {sanitized_user_input}") #debug

def add_assistant_message(response):
    ss.messages.append({"role": "assistant", "content": response})

def get_line_chart(tdf,date,metric_name,value_field,precision,width,height,growth_field=None):

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
        tooltip=(
            [
                alt.Tooltip(date, type='temporal', format='%m/%d/%y(%a) %I%p', title="Date (PST)"),
                metric_name,
                alt.Tooltip(value_field, type='quantitative', format=f'$,.{precision}f', title=value_field)
            ] + ([alt.Tooltip(growth_field, type='quantitative', format='.2%', title='12m growth')] if growth_field is not None else [])
        )
    ).add_params(hover)  #.interactive()
    
    return (lines + points) #  + tooltips

def main():
    print('#### Starting at top of main') #debug
    # Load company lookup table
    # sql= """Select case when primary_ticker is not null then company_name||' ('||primary_ticker||')' else company_name END as company_and_ticker
    #         , company_name, primary_ticker, cik, last_filing_date
    #         from Notebook.Public.cybersyn_company_lookup order by 1 """
    company_lookup_df=get_company_tickers_df()
    # company_lookup_df=retrieve_data(sql)
    # st.write(company_lookup_df) #debug

    # Use a selectbox that triggers processing immediately when a selection is made
    # Prepend an empty option so the app doesn't auto-run on first load
    options = [''] + company_lookup_df['company_and_ticker'].tolist()
    company_and_ticker = st.selectbox('Select which company/stock ticker:', options, index=0, key='company_selectbox', help='Start typing to narrow company name or ticker options')
    if not company_and_ticker:
        st.subheader(':red[Please select a company]')
    if company_and_ticker:
        if company_and_ticker != ss.company_and_ticker: # only reset if selection changed
            ss.company_and_ticker = company_and_ticker
            ss.cik = company_lookup_df.loc[company_lookup_df['company_and_ticker'] == ss.company_and_ticker, 'cik'].values[0]
            ss.ticker = company_lookup_df.loc[company_lookup_df['company_and_ticker'] == ss.company_and_ticker, 'ticker'].values[0]
            ss.company_name = company_lookup_df.loc[company_lookup_df['company_and_ticker'] == ss.company_and_ticker, 'company_name'].values[0]
            # Reset session state explicitly (avoid chained assignment aliasing objects)
            ss.filings_df = pd.DataFrame()
            ss.quarterly_financials = pd.DataFrame()
            ss.annual_financials = pd.DataFrame()
            ss.df = pd.DataFrame()
            ss.sp = pd.DataFrame()
            ss.companyfacts_metrics_yahoo_dict = {}
            ss.analysis_result=''
            ss.messages=[] # Clear out messages
            ss.counter=0 # Reset counter
            ss.llm_analysis_requested=False
            # st.write(f"cik={ss.cik}, ticker={ss.ticker}, company_name={ss.company_name}, company_and_ticker={ss.company_and_ticker}\n") #debug
            # st.subheader('Dataframe Reset due to submit button') #debug

        with st.spinner('Pulling 10-Q Financial Data...'):
            today = datetime.today().date()
            x_years_ago = today - timedelta(days=4*365)
            start_date = x_years_ago.replace(month=1, day=1) #set start date to Jan 1 of the day 4 years ago
            today = datetime.today().date() #datetime.today().strftime('%Y-%m-%d')
            
            # ss.filings_df=ss.filings_df[ss.filings_df['filed_date']>=start_date]
            # ss.quarterly_financials['end_date'] = pd.to_datetime(ss.quarterly_financials['end_date']).dt.date
            # st.write(f"ss.ticker={ss.ticker}") #debug

            ticker=yf.Ticker(ss.ticker)
            historical_prices=ticker.history(start='2017-01-01',end=today) # gets following fields: Date, High, Low, Close, Volume, Dividends, Stock Splits
            ss.sp=historical_prices.reset_index()
            ss.sp['Ticker']=ss.ticker
            ss.sp=ss.sp.rename(columns={'Close':'Stock Price'})
            # st.write('stock prices',ss.sp) #debug

            stats=ticker.info
            # st.write('yahoo stats',stats) #debug write out yahoo dict
            ss.company_name = stats.get('shortName', 'Unknown Company Name')

            # CompanyInfoYahoo: tuples of (field_name, format) OR (field_name, format, thresholds)
            # thresholds is an optional dict with numeric 'low' and 'high' values used for coloring.
            # The thresholds are illustrative defaults you can tune per-company or per-field.
            CompanyInfoYahoo = [
            # ('symbol','text'),
            # ('shortName','text'),
            ('regularMarketPrice','decimal' ),
            # ('previousClose','decimal' ),
            # ('open','decimal' ),
            ('regularMarketDayRange','text' ),
            ('fiftyTwoWeekRange','text' ),
            ('volume','integer'),
            # PE ratios: low/high are illustrative (PE > 30 often considered high growth; adjust as needed)
            ('trailingPE','decimal', {'low': 12, 'high':25, 'direction': 'higher_is_bad'}),
            ('forwardPE','decimal', {'low': 12, 'high': 25, 'direction': 'higher_is_bad'}),
            ('dividendYield','percent', {'low': 1.0, 'high': 3.0, 'direction': 'higher_is_good'} ),
            ('fullTimeEmployees','integer'),
            ('priceToSalesTrailing12Months','decimal', {'low': 2, 'high': 5, 'direction': 'higher_is_bad'}),
            ('enterpriseToRevenue','decimal', {'low': 2, 'high': 5, 'direction': 'higher_is_bad'}),
            ('enterpriseToEbitda','decimal', {'low': 2, 'high': 25, 'direction': 'higher_is_bad'}),
            ('revenueGrowth','percent', {'low': 0.0, 'high': 10.0, 'direction': 'higher_is_good'}),
            ('earningsGrowth','percent', {'low': 0.0, 'high': 20.0, 'direction': 'higher_is_good'}),
            ('earningsQuarterlyGrowth','percent', {'low': 0.0, 'high': 20.0, 'direction': 'higher_is_good'}),
            ('trailingPegRatio','decimal', {'low': 1, 'high': 1.5, 'direction': 'higher_is_bad'} ),
            ('marketCap','integer'),
            ('averageDailyVolume10Day','integer'),
            ('quickRatio','decimal', {'low': 1.0, 'high': 1.5, 'direction': 'higher_is_good'} ),
            ('currentRatio','decimal', {'low': 1.0, 'high': 2.0, 'direction': 'higher_is_good'} ),
            ('debtToEquity','decimal', {'low': 0.0, 'high': 1.0, 'direction': 'higher_is_bad'}),
            ('revenuePerShare','decimal' ),
            ('heldPercentInsiders','percent' , {'low': 5.0, 'high': 20.0, 'direction': 'higher_is_bad'}),
            ('heldPercentInstitutions','percent' , {'low': 5.0, 'high': 20.0, 'direction': 'higher_is_good'}),
            ('shortPercentOfFloat', 'percent', {'low': 5.0, 'high': 15.0, 'direction': 'higher_is_bad'}),
            ('grossMargins','percent' , {'low': 10.0, 'high': 40.0, 'direction': 'higher_is_good'}),
            ('ebitdaMargins','percent' , {'low': 3.0, 'high': 30.0, 'direction': 'higher_is_good'}),
            ('operatingMargins','percent' , {'low': 5.0, 'high': 25.0, 'direction': 'higher_is_good'}),
            ('profitMargins','percent', {'low': 3.0, 'high': 15.0, 'direction': 'higher_is_good'}),
            ('returnOnAssets','percent', {'low': 3.0, 'high': 10.0, 'direction': 'higher_is_good'}),
            ('returnOnEquity','percent', {'low': 3.0, 'high': 15.0, 'direction': 'higher_is_good'}),
            ('shortRatio','decimal'),
            ('enterpriseValue','integer'),
            ('numberOfAnalystOpinions','integer'),
            ('recommendationKey','text'),
            # ('fullExchangeName','text'),
            ('earningsTimestamp','epoch'),
            # ('website','text'),
            # ('industryDisp','text'),
            # ('sector','text'),
            # omit longBusinessSummary here (renders separately)
            ('beta','decimal' ),
            ('sharesOutstanding','integer'),
            ('sharesShort','integer'),
            ('sharesShortPriorMonth','integer'),
            ('bookValue','integer'),
            ('priceToBook','decimal' ),
            ('lastFiscalYearEnd','epoch'),
            # ('nextFiscalYearEnd','epoch'),
            ('trailingEps','decimal' ),
            ('forwardEps','decimal' ),
            ('totalCash','integer'),
            ('totalCashPerShare','decimal' ),
            ('ebitda','integer'),
            ('totalDebt','integer'),
            ('totalRevenue','integer'),
            ('freeCashflow','integer'),
            ('operatingCashflow','integer'),
            ]

            st.write(f"""{stats.get('longBusinessSummary', 'Unknown Company')}, website: {stats.get('website', 'N/A')}
                     , exchange: {stats.get('fullExchangeName', 'N/A')}, sector: {stats.get('sector', 'N/A')}, industry: {stats.get('industryDisp', 'N/A')}""")
            # Build a dict with both the expected format and the actual value pulled from yfinance
            ss.companyfacts_metrics_yahoo_dict = {}
            rows = []
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

            def _evaluate_color(fmt, raw_value, thresholds):
                """Decide a text color (red/green) based on thresholds and direction.
                Returns a CSS color string (hex) for the value text, or empty string for no color.
                """
                if not thresholds or raw_value is None:
                    return ""
                try:
                    v = raw_value
                    # normalize percent fields to percent numbers (e.g., 0.12 -> 12.0)
                    if fmt == 'percent':
                        v = float(v)
                        if -1 <= v <= 1:
                            v = v * 100.0
                    else:
                        v = float(v)

                    low = thresholds.get('low')
                    high = thresholds.get('high')
                    direction = thresholds.get('direction', 'higher_is_good')

                    # text color constants (darker colors for readability)
                    red_text = '#8b0000'    # dark red
                    green_text = '#006400'  # dark green

                    if direction == 'higher_is_bad':
                        if high is not None and v >= high:
                            return red_text
                        if low is not None and v <= low:
                            return green_text
                        return ""
                    else:  # higher_is_good
                        if high is not None and v >= high:
                            return green_text
                        if low is not None and v <= low:
                            return red_text
                        return ""
                except Exception:
                    return ""

            for item in CompanyInfoYahoo:
                # allow entries to be (name, fmt) or (name, fmt, thresholds)
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    name, fmt = item
                    thresholds = {}
                else:
                    name, fmt = item[0], item[1]
                    thresholds = item[2] if len(item) > 2 else {}
                try:
                    raw_value = stats.get(name)
                except Exception:
                    raw_value = None

                display_value = _format_value(fmt, raw_value)

                ss.companyfacts_metrics_yahoo_dict[name] = {'format': fmt, 'value': raw_value, 'display': display_value, 'thresholds': thresholds}
                rows.append({'field': name, 'format': fmt, 'display': display_value, 'raw_value': raw_value, 'thresholds': thresholds})

            # Show a compact, readable table in Streamlit using the formatted display column
            try:
                df_rows = pd.DataFrame(rows)

                # Layout the Yahoo fields in three visual columns. Each column lists
                # multiple entries with the field name (bold) and the formatted display value below it.
                entries = list(df_rows[['field', 'display']].itertuples(index=False, name=None))
                if len(entries) == 0:
                    st.write("No Yahoo fields available")
                else:
                    # Put the controls and display inside an expander so users can collapse it.
                    with st.expander("Yahoo data (4-column layout)", expanded=True):
                        # Fixed 4-column layout with roughly equal rows per column
                        num_cols = 4
                        # show_format = st.checkbox("Show format inline (e.g. percent/decimal)", value=False)

                        # split entries into num_cols roughly equal chunks
                        n = len(entries)
                        per_col = (n + num_cols - 1) // num_cols  # ceiling division
                        cols = st.columns(num_cols)
                        for i, col in enumerate(cols):
                            start = i * per_col
                            end = start + per_col
                            with col:
                                for field, display in entries[start:end]:
                                    try:
                                        fmt = df_rows.loc[df_rows['field'] == field, 'format'].iloc[0]
                                        thresholds = df_rows.loc[df_rows['field'] == field, 'thresholds'].iloc[0]
                                    except Exception:
                                        fmt = ''
                                        thresholds = {}

                                    color = _evaluate_color(fmt, df_rows.loc[df_rows['field'] == field, 'raw_value'].iloc[0], thresholds)
                                    # render single-line; apply background color inline if present
                                    label = f"{field}: {display}"

                                    if color:
                                        html = f"<div style=\"background-color: {color}; padding:6px; border-radius:4px\">{label}</div>"
                                        st.markdown(html, unsafe_allow_html=True)
                                    else:
                                        st.markdown(label)
            except Exception:
                                    if color:
                                        # color only the value text; keep the field label bold and uncolored
                                        # label contains the bold field and optional format; we need to insert span around the display part
                                        # build safe HTML: bold field + optional format, then colored span for the value
                                        try:
                                            prefix, sep, val = label.partition(': ')
                                            if sep:
                                                html = f"{prefix}: <span style=\"color:{color}\">{val}</span>"
                                            else:
                                                html = f"<span style=\"color:{color}\">{label}</span>"
                                        except Exception:
                                            html = f"<span style=\"color:{color}\">{label}</span>"
                                        st.markdown(html, unsafe_allow_html=True)
                                    else:
                                        st.markdown(label)
        # Load SEC EDGAR financials for the selected CIK (guarded)
        try:
            res = sec_edgar_financial_load(ss.cik)
            if not isinstance(res, (list, tuple)) or len(res) != 3:
                st.error("Unexpected return from sec_edgar_financial_load; expected (filings_df, quarterly_df, annual_df).")
                ss.filings_df = pd.DataFrame()
                ss.quarterly_financials = pd.DataFrame()
                ss.annual_financials = pd.DataFrame()
                st.stop()
            ss.filings_df, ss.quarterly_financials, ss.annual_financials = res

            # If quarterly data isn't present, surface a message and stop further processing
            if not isinstance(ss.quarterly_financials, pd.DataFrame) or ss.quarterly_financials.empty:
                st.warning("No quarterly financials returned from SEC for this company.")
                if isinstance(ss.filings_df, pd.DataFrame) and not ss.filings_df.empty:
                    st.write("Filings (sample):")
                    st.dataframe(ss.filings_df.head())
                st.stop()
        except Exception as e:
            st.error(f"Failed to load SEC data: {e}")
            try:
                import traceback
                st.code(traceback.format_exc())
            except Exception:
                pass
            ss.filings_df = pd.DataFrame()
            ss.quarterly_financials = pd.DataFrame()
            ss.annual_financials = pd.DataFrame()
            st.stop()

        # st.write(f"income_statement_columns_for_chart just prior to melt: {ss.income_statement_columns_for_chart}") #debug
        # st.write('ss.quarterlyfinancials just prior to melt',ss.quarterly_financials) #debug
        chart_df = ss.quarterly_financials[['end_date'] + ss.income_statement_columns_for_chart].melt(id_vars=['end_date'], var_name='metric', value_name='value')
        chart_df = chart_df[pd.notna(chart_df['value'])]

        # Aggregate to one value per (end_date, metric)
        chart_df_max = chart_df.groupby(['end_date', 'metric']).agg({'value': 'max'}).reset_index()
        # compute 12-month (4-quarter) growth per metric as fraction (e.g., 0.10 == +10%)
        chart_df_max = chart_df_max.sort_values(['metric', 'end_date'])
        chart_df_max['growth_12m'] = chart_df_max.groupby('metric')['value'].pct_change(periods=4)

        chart_income_statement = get_line_chart(chart_df_max, 'end_date', 'metric', 'value', 0, 400, 300, growth_field='growth_12m')

        # balance_sheet_columns=ss.balance_sheet_columns.insert(0, 'end_date')  # Include 'end_date' for melting
        chart_df = ss.quarterly_financials[['end_date'] + ss.balance_sheet_columns].melt(id_vars=['end_date'], var_name='metric', value_name='value')
        chart_df = chart_df[pd.notna(chart_df['value'])]
        # Aggregate for balance sheet and compute 12-month growth
        chart_df_bs = chart_df.groupby(['end_date', 'metric']).agg({'value': 'max'}).reset_index()
        chart_df_bs = chart_df_bs.sort_values(['metric', 'end_date'])
        chart_df_bs['growth_12m'] = chart_df_bs.groupby('metric')['value'].pct_change(periods=4)
        chart_balance_sheet = get_line_chart(chart_df_bs, 'end_date', 'metric', 'value', 0, 400, 300, growth_field='growth_12m')
        
        # st.altair_chart(chart_income_statement, use_container_width=True) #temp
        # st.altair_chart(chart_balance_sheet, use_container_width=True) #temp

        try:
            # If no stock prices retrieved, set merged dataframe to pivoted data
            ss.quarterly_financials['Operating Margin'] = ss.quarterly_financials['Operating Income'] / ss.quarterly_financials['Revenue/Sales']
            ss.quarterly_financials['Net Margin'] = ss.quarterly_financials['Net Income'] / ss.quarterly_financials['Revenue/Sales']
            ss.quarterly_financials['ROA'] = ss.quarterly_financials['Net Income'] / ss.quarterly_financials['Total Assets']
            ss.quarterly_financials['EPS'] = ss.quarterly_financials['Net Income'] / ss.quarterly_financials['Common Shares Outstanding']
            
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

                # Market cap (stock price * shares)
                ss.quarterly_financials['Market_Cap'] = (ss.quarterly_financials['Stock Price'] * ss.quarterly_financials['Common Shares Outstanding']).astype(float)

                # Ensure end_date is datetime and DataFrame is sorted by date (oldest -> newest)
                ss.quarterly_financials['end_date'] = pd.to_datetime(ss.quarterly_financials['end_date'])
                ss.quarterly_financials.sort_values('end_date', inplace=True)

                # Compute denominators as the sum of the prior 4 quarters (exclude the current quarter)
                # Require full 4 prior quarters (min_periods=4) so early rows without 4 prior quarters will be NaN
                revenue_series = ss.quarterly_financials['Revenue/Sales'].astype(float)
                ss.quarterly_financials['revenue_prior4'] = revenue_series.shift(0).rolling(window=4, min_periods=4).sum()

                eps_col = 'EPS Diluted'
                ss.quarterly_financials[eps_col] = ss.quarterly_financials.get(eps_col, ss.quarterly_financials.get('EPS'))
                eps_series = ss.quarterly_financials[eps_col].astype(float)
                ss.quarterly_financials['eps_prior4'] = eps_series.shift(0).rolling(window=4, min_periods=4).sum()

                # Compute ratios using prior-4 denominators; guard against division by zero or missing data
                ss.quarterly_financials['PS'] = ss.quarterly_financials['Market_Cap'] / ss.quarterly_financials['revenue_prior4']
                ss.quarterly_financials.loc[ss.quarterly_financials['revenue_prior4'].isna() | (ss.quarterly_financials['revenue_prior4'] == 0), 'PS'] = pd.NA

                ss.quarterly_financials['PE'] = ss.quarterly_financials['Stock Price'] / ss.quarterly_financials['eps_prior4']
                ss.quarterly_financials.loc[ss.quarterly_financials['eps_prior4'].isna() | (ss.quarterly_financials['eps_prior4'] == 0), 'PE'] = pd.NA

                # Convert types and clean up temporary columns
                ss.quarterly_financials[['Market_Cap', 'ROA', 'PS', 'PE']] = ss.quarterly_financials[['Market_Cap', 'ROA', 'PS', 'PE']].apply(pd.to_numeric, errors='coerce')
                ss.quarterly_financials.drop(columns=['revenue_prior4', 'eps_prior4'], inplace=True, errors='ignore') #debug
                # st.write(f"Price/Earnings = {ss.quarterly_financials['PE']}, Price/Sales = {ss.quarterly_financials['PS']}") #debug
                ss.quarterly_financials.sort_values('end_date', inplace=True, ascending=False) # sort back to newest -> oldest

        except ValueError as e:
            print(f"Failed to calculate metrics: {e}")

        # st.write('ss.quarterly_financials with ratios added',ss.quarterly_financials) #debug
        st.write(f":blue[Chart of Key Financials for {ss.company_name}, stock ticker '{ss.ticker}', cik = {ss.cik}]")

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
            st.altair_chart(chart_income_statement, width='stretch')

        with col2:
            st.markdown('<p class="centered-title">Balance Sheet</p>', unsafe_allow_html=True)
            st.altair_chart(chart_balance_sheet, width='stretch')

        
        if len(ss.sp)!=0:
            with col3:
                # st.write(ss.sp) #debug
                chart_stock_price=get_line_chart(ss.sp,'Date','Ticker','Stock Price',2,400,300)
                title=f'Stock Price for {ss.ticker}'
                st.markdown(f'<p class="centered-title">{title}</p>', unsafe_allow_html=True)
                st.altair_chart(chart_stock_price, width='stretch')
            
        # Format the dataframe
        filtered_df=ss.quarterly_financials.set_index('end_date').dropna(axis=1, how='all')
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
            'Accumulated Depreciation': '${:,.0f}',
            'Common Shares Outstanding': '{:,.0f}',
            'Income Tax': '${:,.0f}',
            'Depreciation': '${:,.0f}',
            'Amortization': '${:,.0f}',
            'Depreciation & Amortization': '${:,.0f}',
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

        st.markdown('<h3 style="color:#3894f0;">SEC 10-Q data collected from SEC Edgar:</h3>', unsafe_allow_html=True)
        st.dataframe(formatted_df) #.sort_values(by="end_date",ascending=False
        # st.write(analysis_result_text) ######## debug purposes only

        # Prep for pulling LLM Summaries
        
        if ss.messages==[]:
            llm_analysis_requested=st.button(label='Click for LLM analysis')
            ss.llm_analysis_requested=llm_analysis_requested
        
        if ss.llm_analysis_requested == True:
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
                    but rather answer with any data you do know about the industry performance or specific competitors and their performance."""})
                ss.llm_model='mistral-large2'
                response_df = retrieve_llm_data()

                # Pull the data based on the messages array
                # prompt = "\n".join([
                #         f"{msg['role']}: {msg['content']}" if idx == 0 else "{}: {}".format(msg['role'], msg['content'].replace("'", "").replace('"', ''))
                #         for idx, msg in enumerate(ss.messages)
                #     ])
                
                # st.write(sql) #debug
                # ss.messages.append({"role":"assistant","content":"message to test without LLM"}) #debug

            if len(ss.messages)==2 and len(ss.analysis_result)==0:
                with st.spinner('Running LLM Analysis to provide a summary...'):
                    response_df = ic(retrieve_llm_data())
                ss.analysis_result=response_df.iloc[0,0]
                ss.analysis_result=ss.analysis_result.replace('$', '\\$')
                print(f"Retrieved LLM response, length of response = {len(ss.analysis_result)}")
                ss.messages.append({"role":"assistant","content":ss.analysis_result})

            # Print the summary and the dataframe

                # And write out the dataframe
                # st.markdown('<h3 style="color:#3894f0;">SEC 10-Q data collected from SEC Edgar:</h3>', unsafe_allow_html=True)
                # filtered_df=ss.quarterly_financials.set_index('end_date').dropna(axis=1, how='all')

                # Format the dataframe
                # st.write(analysis_result_text) ######## debug purposes only

            #if the summary is complete, prep for the chat interactions
            if len(ss.messages)==3:
                ss.messages.append({"role": "assistant", "content": 
                    f"""Welcome to Cortex Chat, powered by the Mistral-Large2 LLM Model. Please let me know if you have any questions about these metrics for {ss.company_name}. I may be able to provide some limited data on industry and competitors based on the public knowledge I was trained on"""})
                ss.messages.append({"role":"system","content":"""If the user asks about performance relative to other competitors, do not respond with generic comparison frameworks, 
                    but rather answer with any data you do know about the industry performance or specific competitors and their performance. Do not state your role in the response. Do not restate the question, and I already know you're the assistant so do not add role statements in your answers"""})
            
            for message in [msg for msg in ss.messages[2:] if msg['role']!='system']: # Display the prior chat messages
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            user_input=st.chat_input('Enter question:') #, key='user_input' on_submit=add_user_message
            if user_input:
                add_user_message(user_input)
                with st.chat_message(ss.messages[-1]["role"]):
                    st.write(ss.messages[-1]["content"])
                with st.chat_message('assistant'):
                    with st.spinner("Thinking..."):
                        ss.llm_model='mistral-large'
                        response_df=ic(retrieve_llm_data())
                        # st.write(response_df) #debug

                        response_string=response_df["RESPONSE"].iloc[0]
                        response_string=response_string.replace('"', '').replace('$', '\\$')
                        # st.write(response_string) #debug
                        # st.stop() #debug
                        print(f"Chat response received, response length={len(response_string)}")
                # st.write(f"""sending this to add_assistant_mesage: {response_string}""") #debug
                        add_assistant_message(response_string)
                        st.rerun()

        print('#### at end of main') #debug

        

main()

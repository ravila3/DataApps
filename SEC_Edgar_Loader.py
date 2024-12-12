import requests
import streamlit as st
import pandas as pd
from icecream import ic
from streamlit import session_state as ss
# from bs4 import BeautifulSoup
from datetime import datetime
# from dateutil.relativedelta import relativedelta
from collections import OrderedDict

def show_all_metrics(companyfacts_data): #debug this function is for viewing all categorizations on a filing

    data_list = []
    frame_criteria='CY2024Q3'

    for key in companyfacts_data['facts']['us-gaap'].keys():
        metric_units = companyfacts_data['facts']['us-gaap'][key]['units']
        
        for unit, unit_data in metric_units.items():
            for item in unit_data:                    
                try:
                    metric = ss.companyfacts_metrics_dict.get(key, {}).get('metric', '')
                    frame=item['frame']
                    value=item['val']
                    if frame_criteria in frame:
                        data_list.append({'frame': frame, 'key': key, 'value': value, 'metric':metric})

                except KeyError as e:
                    if item['end']>'2024-06-30':
                        data_list.append({'frame': 'no frame', 'key': key, 'value': value, 'metric':metric})
                        
                        # st.write(f":red[No frame:] {key}, end= {item['end']}, value= :blue[{item['val']:,}]")
                except TypeError as e:
                    print(f"TypeError: {e} - Check if the expected dictionary is actually a list or other type.")

    items_df = pd.DataFrame(data_list)
    # items_df['metric']=ss.companyfacts_metrics_dict[metric]['metric']
    # st.write(f"items_df = ",items_df) #debug

    # Sort the DataFrame by value in descending order
    items_df['value'] = items_df['value'].astype(int)  # Ensure value is treated as integer
    items_df = items_df.sort_values(by='value', ascending=False)

    # Filter and write out the messages

    # Extract unique frames from the items_df
    frames = sorted(items_df['frame'].unique())

    # Loop through each frame
    for frame in frames:
        st.write(f"Messages for frame: {frame}:")
        # if frame != 'no frame':
        #     st.write(f"{frame} Messages:")
        # else:
        #     st.write("No Frame Messages:")

        # Filter the DataFrame based on the current frame
        frame_df = items_df[items_df['frame'] == frame]
        
        # Set color based on frame
        color = 'green' if frame == 'CY2024Q3' else 'blue' if frame == 'CY2024Q3I' else 'red'
        
        # Write out the rows with the specified color, only if metric is not ''
        for _, row in frame_df.iterrows():
            if row['metric'] != '':
                st.write(f"metric = {row['metric']}, :{color}[{row['key']}, value = {row['value']:,}]")
            else:
                st.write(f":{color}[{row['key']}, value = {row['value']:,}]")

    return()


def sec_edgar_financial_load(cik):
    
    metrics_df=pd.DataFrame()

    # cik = '0000732717' #936528 1050446 1821806
    cik_str = cik.zfill(10)

    # Define the metrics and their corresponding categories and units
    metrics_data = [
        ('Revenues', 'Income Statement', 'USD','Revenue/Sales','I'),
        ('RevenueFromContractWithCustomerExcludingAssessedTax', 'Income Statement', 'USD','Revenue/Sales','I'),
        ('RevenueFromContractWithCustomerIncludingAssessedTax', 'Income Statement', 'USD','Revenue/Sales','I'),
        ('RevenuesNetOfInterestExpense', 'Income Statement', 'USD','Revenue/Sales','I'),
        ('InvestmentIncomeInterestAndDividend', 'Income Statement', 'USD','Revenue/Sales','I'),
        # ('InterestIncomeOperating', 'Income Statement', 'USD','Revenue/Sales','I'),
        # ('InterestAndDividendIncomeOperating', 'Income Statement', 'USD','Revenue/Sales','I'),
        ('CostOfRevenue', 'Income Statement', 'USD','Cost of Sales','I'),
        ('CostOfGoodsAndServicesSold', 'Income Statement', 'USD','Cost of Sales','I'),
        ('GrossProfit', 'Income Statement', 'USD','Gross Profit','I'),
        ('OperatingCostsAndExpenses', 'Income Statement', 'USD','Operating Expenses','I'),
        ('OperatingExpenses', 'Income Statement', 'USD','Operating Expenses','I'),
        ('BenefitsLossesAndExpenses', 'Income Statement', 'USD','Operating Expenses','I'),
        ('SellingGeneralAndAdministrativeExpense', 'Income Statement', 'USD','SG&A Exp','I'),
        ('MarketingAndAdvertisingExpense', 'Income Statement', 'USD','SG&A Exp','I'),
        ('SellingAndMarketingExpense', 'Income Statement', 'USD','SG&A Exp','I'),
        ('ResearchAndDevelopmentExpense', 'Income Statement', 'USD','R&D Exp','I'),
        ('ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost', 'Income Statement', 'USD','R&D Exp','I'),
        ('OperatingIncomeLoss', 'Income Statement', 'USD','Operating Income','I'),
        ('IncomeLossFromContinuingOperations', 'Income Statement', 'USD','Operating Income','I'),
        ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments', 'Income Statement', 'USD','Operating Income','I'),
        ('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 'Income Statement', 'USD','Pretax Income','I'),
        ('NetIncomeLoss', 'Income Statement', 'USD','Net Income','I'),
        ('NetIncomeLossAvailableToCommonStockholdersBasic', 'Income Statement', 'USD','Net Income','I'),
        ('InterestExpense', 'Income Statement', 'USD','Interest Expense','E'),
        ('InterestExpenseOperating', 'Income Statement', 'USD','Interest Expense','E'),
        ('InterestExpenseNonoperating', 'Income Statement', 'USD','Interest Expense','E'),
        ('CostsAndExpenses', 'Income Statement', 'USD','Total Expenses','E'),
        # ('ProfitLoss', 'Income Statement', 'USD',''),
        # ('GrossProfit', 'Income Statement', 'USD',''),
        # ('IncomeLossFromContinuingOperations', 'Income Statement', 'USD',''),
        # ('InterestIncomeExpenseNonoperatingNet', 'Income Statement', 'USD',''),
        ('SubscriptionRevenue', 'Income Statement', 'USD','Subscription Revenue','E'),
        ('EarningsPerShareBasic', 'Per Share Metrics', 'USD/shares', 'EPS Basic','E'),
        ('EarningsPerShareDiluted', 'Per Share Metrics', 'USD/shares', 'EPS Diluted','E'),
        ('Assets', 'Balance Sheet', 'USD','Total Assets','I'),
        ('AssetsCurrent', 'Balance Sheet', 'USD','Curr Assets','I'),
        ('Liabilities', 'Balance Sheet', 'USD','Liabilities','I'),
        ('LiabilitiesCurrent', 'Balance Sheet', 'USD','Curr Liabilities','I'),
        ('LongTermDebt', 'Balance Sheet', 'USD','Long Term Debt','I'),
        ('StockholdersEquity', 'Balance Sheet', 'USD','Stockholder Equity','I'),
        # ('TreasuryStockCommonShares', 'Balance Sheet', 'shares','Treasury Shares','E'),
        ('CommonStockSharesIssued', 'Shares Outstanding', 'shares','Common Shares Outstanding','E'),
        ('CommonStockSharesOutstanding', 'Shares Outstanding', 'shares','Common Shares Outstanding','E'),
        ('Deposits', 'Balance Sheet', 'USD','Deposits','I'),
        ('IncomeTaxExpenseBenefit', 'Income Statement', 'USD','Income Tax','E'),
        ('DepreciationDepletionAndAmortization', 'Income Statement', 'USD','Depreciation','E'),
        ('Depreciation', 'Income Statement', 'USD','Depreciation','E'),
        ('AmortizationOfIntangibleAssets', 'Income Statement', 'USD','Amortization','E'),
        ('GainLossOnSaleOfBusiness', 'Income Statement', 'USD','Sale of Business','E')
    ]

    # Convert the list of tuples to a dictionary
    ss.companyfacts_metrics_dict = {metric[0]: {'category': metric[1], 'unit': metric[2], 'metric': metric[3],'chart_include':metric[4]} for metric in metrics_data}

    @st.cache_data(ttl="1h")
    def get_edgar_data(url):
        headers = {
            'User-Agent': 'AI Analytics & Development (rafaelavila3@gmail.com)'
        }
        response = requests.get(url, headers=headers)
        # st.write(f'Retrieved data for {url}')
        return response
    
    # pull the list of filings from SEC Edgar
    filings_url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
    response=get_edgar_data(filings_url)
    if response.status_code == 200:
        try:
            filings_data = response.json()
            # st.write("filings_data:", filings_data) #debug
            company_info={key: value for key, value in filings_data.items() if not isinstance(value, (list, dict))}
            # st.write('company_info',company_info) #debug
            # Extract recent filings
            filings = filings_data.get('filings', {}).get('recent', [])
            filings_df = pd.DataFrame(filings)
            filings_df['primary_ticker']=filings_data['tickers'][0]
            filings_df['primary_exchange']=filings_data['exchanges'][0]

            filings_df['cik']=company_info['cik']
            filings_df['company_name']=company_info['name']
            filings_df['sic']=company_info['sic']
            filings_df['sicDescription']=company_info['sicDescription']
            filings_df['fiscalYearEnd']=company_info['fiscalYearEnd']

            filings_df=filings_df[['cik','company_name','primary_ticker','primary_exchange','sic','sicDescription','fiscalYearEnd','accessionNumber','filingDate','reportDate','form','primaryDocument','fileNumber']]
            filings_10q_df = filings_df[filings_df['form'] == '10-Q']
            # st.write(f"{filings_df['form']} Filings DataFrame: {filings_10q_df}")

            # Display URLs for all recent filings
            for index, filing in filings_df.iterrows():
                form = filing.get('form')
                primary_document = filing.get('primaryDocument')
                accession_number = filing.get('accessionNumber').replace('-', '')
                if primary_document and accession_number: #and form in ['10-Q','10-K']
                    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{primary_document}"
                    filings_df['filing_url']=filing_url
                    # st.write(f"Filing URL for form {form}, Accession Number {accession_number}: {filing_url}")  #debug shows a list of all filings

            # st.write('filings_df',filings_df) #debug

        except ValueError:
            st.write("Error parsing JSON response.")
    else:
        st.error(f"Request failed with status code {response.status_code}")
        st.write(response.text)

    # pull the financial details from the filings from SEC Edgar
    companyfacts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_str}.json"
    response = get_edgar_data(companyfacts_url)
    
    if response.status_code == 200:
        try:
            companyfacts_data = response.json()
            # st.write("CompanyFacts JSON:", companyfacts_data)
            # st.write("CompanyFacts Keys:", companyfacts_data['facts']['us-gaap'].keys()) #debug
            # st.write("CompanyFacts Response:", companyfacts_data['facts']['us-gaap']['Revenues']['units']['USD'])  #.keys()

            # show_all_metrics(companyfacts_data) #debug

            for metric in ss.companyfacts_metrics_dict:
                try:
                    if metric in companyfacts_data['facts']['us-gaap'].keys():
                        units = ss.companyfacts_metrics_dict[metric]['unit']
                        # st.write(f"metric in loop = {metric}, units = {units}") #debug
                        # st.write(companyfacts_data['facts']['us-gaap'][metric]['units']) #debug
                        temp_df=(pd.DataFrame(companyfacts_data['facts']['us-gaap'][metric]['units'][units]))
                        # if metric=='SellingGeneralAndAdministrativeExpense':
                        #     st.write(companyfacts_data['facts']['us-gaap'][metric]['units'][units]) #debug
                        # st.write(f'just created df for {metric}')
                        temp_df=temp_df[temp_df['frame'].notna()]
                        # temp_df=temp_df[temp_df['form'].isin(['10-Q','10-K'])]
                        temp_df.sort_values(by='frame',axis=0, ascending=False, inplace=True)
                        temp_df['metric_label']=metric
                        temp_df['metric']=ss.companyfacts_metrics_dict[metric]['metric']
                        temp_df['category']=ss.companyfacts_metrics_dict[metric]['category']
                        temp_df['chart_include']=ss.companyfacts_metrics_dict[metric]['chart_include']
                        metrics_df=pd.concat([metrics_df,temp_df],ignore_index=True)
                    else:
                        continue
                except KeyError as e:
                    st.write(f"KeyError: {e} not found in metric {metric}")
                except Exception as e:
                    st.write(f"An error occurred: {e}")
            metrics_df['cik']=cik
            metrics_df.rename(columns={'val':'value'},inplace=True)
            metrics_df.rename(columns={'end':'end_date'},inplace=True)
            metrics_df['end_date'] = pd.to_datetime(metrics_df['end_date']).dt.date
            metrics_df=metrics_df[['cik','start','end_date','category','metric','metric_label','chart_include','value','frame','fy','fp','filed']]
            # st.write('initial metrics_df right after sourcing from SEC Edgar',metrics_df) #debug

            # Extract the quarter part from the frame column with error handling
            def extract_quarter(frame):
                try:
                    return frame[7]
                except IndexError:
                    return 0

            # Add the 'frame_quarter' column only if it doesn't exist
            if 'frame_quarter' not in metrics_df.columns:
                metrics_df.loc[metrics_df['category'] == 'Income Statement', 'frame_quarter'] = metrics_df['frame'].apply( lambda x: extract_quarter(x) if len(x) > 7 else 0 )
                # metrics_df['frame_quarter'] = metrics_df['frame'].apply(extract_quarter)

            quarter_counts = metrics_df[metrics_df['category'] == 'Income Statement']['frame_quarter'].value_counts()

            # Create a dictionary with all quarters and initialize counts to zero
            all_quarters_counts = {str(q): 0 for q in range(1, 5)}

            # Update the dictionary with the actual counts
            for quarter, count in quarter_counts.items():
                if quarter in all_quarters_counts:
                    all_quarters_counts[quarter] = count

            # Convert the dictionary back to a Series for compatibility
            final_quarter_counts = pd.Series(all_quarters_counts)
            ss.least_count_quarter = int(final_quarter_counts.idxmin())
            # st.write(f'ss.least_count_quarter = {ss.least_count_quarter} and final_quarter_counts = ',final_quarter_counts) #debug

            def extract_fiscal_timeframe(row):
                try:
                    fiscal_year_end_month = int(company_info['fiscalYearEnd'][:2])

                    quarters_adjust = 4-ss.least_count_quarter  #(fiscal_year_end_month // 3)
                    
                    if len(row['frame']) < 7:
                        year = row['end_date'].year #int(row['frame'][2:6])
                        fiscal_timeframe = f"{year}"
                    else:
                        year = int(row['frame'][2:6])
                        quarter = int(row['frame'][7])
                    
                        # Adjust the quarter and year
                        quarter += quarters_adjust
                        if quarter > 4:
                            quarter -= 4
                            year += 1
                    
                        fiscal_timeframe = f"{year} Q{quarter}"
                
                    return fiscal_timeframe
                
                except Exception as e:
                    st.write(f"An error occurred in extract_timeframe: {e}, row:",row)
                    return None
                
                except Exception as e:
                    st.write(f"An error occurred: {e} on row:",row)

                return fiscal_timeframe

            metrics_df['fiscal_timeframe'] = metrics_df.apply(extract_fiscal_timeframe,axis=1)

            metrics_df.sort_values(by='fiscal_timeframe',axis=0, ascending=False, inplace=True)
            # st.write('metrics_df after applying fiscal timeframe',metrics_df) #debug show metrics df after applying fiscal timeframes just before assigning to quarter and annual dataframes

            metrics_qtr_df=metrics_df[(metrics_df['category']=='Balance Sheet') | (metrics_df['fiscal_timeframe'].str.len()>5)]
            metrics_qtr_df = metrics_qtr_df[~((metrics_qtr_df['category'] == 'Income Statement') & (metrics_qtr_df['fp'] == 'FY'))]
            
            # st.write('metrics_qtr_df right before pivoting', metrics_qtr_df) #debug [metrics_qtr_df['metric'] == 'RevenueFromContractWithCustomerExcludingAssessedTax']) #debug
            
            pivoted_qtr_df = ic(metrics_qtr_df.pivot_table(index=['fiscal_timeframe','cik'], columns='metric', values='value', aggfunc='max').reset_index())
            pivoted_qtr_df = pivoted_qtr_df.merge( metrics_qtr_df[['fiscal_timeframe', 'end_date']], on='fiscal_timeframe', how='left' ).drop_duplicates() # Display the pivoted DataFrame st.write('pivoted_qtr_df:'

            # st.write('pivoted_qtr_df right after creation',pivoted_qtr_df) #debug

            def dedup_list_preserving_order(list):
                deduped_list=[]
                seen = set()
                for metric in list:
                    if metric not in seen:
                        deduped_list.append(metric)
                        seen.add(metric)
                return(deduped_list)    

            ss.unique_metric_list=[]
            for item in ss.companyfacts_metrics_dict.values():
                    ss.unique_metric_list.append(item['metric'])

            ss.unique_metric_list=dedup_list_preserving_order(ss.unique_metric_list)

            ss.income_statement_columns=[]
            for item in ss.companyfacts_metrics_dict.values():
                if item['category'] == 'Income Statement':
                    ss.income_statement_columns.append(item['metric'])

            ss.income_statement_columns=dedup_list_preserving_order(ss.income_statement_columns)

            ss.income_statement_columns_for_chart=[]
            for item in ss.companyfacts_metrics_dict.values():
                if item['category'] == 'Income Statement' and item['chart_include']=='I':
                    ss.income_statement_columns_for_chart.append(item['metric'])
                    # st.write(f"item['metric']={item['metric']}, item['category']={item['category']}, item['chart_include']={item['chart_include']}")
           
            ss.income_statement_columns_for_chart=dedup_list_preserving_order(ss.income_statement_columns_for_chart)
            # st.write(f"ss.income_statement_columns_for_chart = {ss.income_statement_columns_for_chart}") #debug

            ss.balance_sheet_columns=[]
            for item in ss.companyfacts_metrics_dict.values():
                if item['category'] == 'Balance Sheet':
                    ss.balance_sheet_columns.append(item['metric'])

            ss.balance_sheet_columns=dedup_list_preserving_order(ss.balance_sheet_columns)

            # st.write(ss.income_statement_metrics) #debug

            # unique_metric_list = list(set(item['metric'] for item in ss.companyfacts_metrics_dict.values()))
            # st.write(f"unique_metric_list: {ss.unique_metric_list}") #debug
            desired_order = ['cik','fiscal_timeframe','end_date'] + ss.unique_metric_list
            pivoted_qtr_df=pivoted_qtr_df.reindex(columns=desired_order)
            pivoted_qtr_df.sort_values(by='fiscal_timeframe', ascending=False, inplace=True)
            # st.write('pivoted_qtr_df',pivoted_qtr_df) #debug

            def remove_q4_from_annual_timeframes(fiscal_timeframe_value):
                year = fiscal_timeframe_value[0:4]
                return f"{year}"


            # st.write('metrics_df just before metrics_annual_df created',metrics_df) #debug
            metrics_annual_df=metrics_df[((metrics_df['category']=='Balance Sheet') & (metrics_df['fiscal_timeframe'].str.endswith('Q4'))) | (metrics_df['fiscal_timeframe'].str.len()<=5)]
            
            metrics_annual_df['fiscal_timeframe']=metrics_annual_df['fiscal_timeframe'].apply(remove_q4_from_annual_timeframes)
            # st.write('metrics_annual_df just before pivot:',metrics_annual_df) #[metrics_qtr_df['metric'] == 'Revenues']) #debug

            pivoted_annual_df = ic(metrics_annual_df.pivot_table(index=['fiscal_timeframe','cik'], columns='metric', values='value', aggfunc='max').reset_index())
            max_end_dates = metrics_annual_df.groupby(['fiscal_timeframe', 'cik'])['end_date'].max().reset_index()
            pivoted_annual_df = pd.merge(pivoted_annual_df, max_end_dates, on=['fiscal_timeframe', 'cik'])
            # st.write('pivoted_annual_df:',pivoted_annual_df) #debug

            # pivoted_annual_df = pivoted_annual_df.merge( metrics_annual_df[['fiscal_timeframe', 'end_date']], on='fiscal_timeframe', how='left' ).drop_duplicates() # Display the pivoted DataFrame st.write('pivoted_qtr_df:'
            pivoted_annual_df=pivoted_annual_df.reindex(columns=desired_order)
            pivoted_annual_df.sort_values(by='fiscal_timeframe', ascending=False, inplace=True)
            # st.write('pivoted_annual_df:',pivoted_annual_df) #debug

            # Function to calculate missing quarter values
            def calculate_missing_quarters(annual_df, qtr_df):
                for metric in annual_df.columns:
                    if metric == 'fiscal_timeframe':
                        continue
                    for year in annual_df['fiscal_timeframe']:
                        try:
                            annual_value = annual_df.loc[annual_df['fiscal_timeframe'] == year, metric].values[0]
                            q_values = []
                            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                                q_value = qtr_df.loc[qtr_df['fiscal_timeframe'] == f'{year} {q}', metric].values
                                if pd.isna(q_value[0]):
                                    q_values.append(None)
                                else:
                                    q_values.append(q_value[0])

                            # Calculate the missing quarter
                            if q_values.count(None) == 1:
                                missing_index = q_values.index(None)
                                calculated_value = annual_value - sum([q for q in q_values if q is not None])
                                qtr_df.loc[qtr_df['fiscal_timeframe'] == f'{year} Q{missing_index + 1}', metric] = calculated_value
                                # st.write(f"metric={metric}, year={year}, annual_value={annual_value}, q_values={q_values}, calculated_value={calculated_value}") #debug
                            
                        except Exception as e:
                            print(f"An error occurred calculating missing values: {e}, metric={metric}, year={year}")

            # Run the calculation
            calculate_missing_quarters(pivoted_annual_df, pivoted_qtr_df)
              
            # st.write('Updated pivoted_qtr_df with calculated Q4 values:', pivoted_qtr_df) #debug

        except ValueError:
            st.write("Error parsing JSON response.")
    else:
        st.error(f"Request failed with status code {response.status_code}")
        st.write(response.text)

    return filings_df, pivoted_qtr_df, pivoted_annual_df


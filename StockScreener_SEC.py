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
# import snowflake.connector
# from snowflake.snowpark import Session
# from snowflake.snowpark.context import get_active_session
# from snowflake.snowpark.functions import col
from datetime import datetime,timedelta
from SEC_Edgar_Loader import sec_edgar_financial_load,get_company_tickers_df
import numpy as np

# ===== Growth Analysis Functions =====
def analyze_yoy_growth(quarterly_df):
    """
    Analyze year-over-year growth consistency for a company.
    
    Args:
        quarterly_df: DataFrame with quarterly financials (includes 'fiscal_timeframe', 'Revenue/Sales', 'Net Income')
    
    Returns:
        tuple: (updated_quarterly_df, metrics_dict)
               - updated_quarterly_df: Original df with growth columns added
               - metrics_dict: Growth metrics including regression analysis
    """
    if quarterly_df.empty or len(quarterly_df) < 8:  # Need at least 2 years of data
        return quarterly_df, None
    
    # Ensure fiscal_timeframe column exists
    if 'fiscal_timeframe' not in quarterly_df.columns:
        return quarterly_df, None
    
    # Create a working dataframe with fiscal_timeframe
    df = quarterly_df.reset_index(drop=True).copy()
    df['fiscal_timeframe'] = df['fiscal_timeframe'].astype(str)
    df=df.sort_values(by='fiscal_timeframe', ascending=True).reset_index(drop=True)
       
    # Parse fiscal_timeframe to extract year and quarter
    def parse_fiscal_timeframe(tf_str):
        """Parse '2025 Q1' format to (year, quarter)"""
        try:
            parts = tf_str.strip().split()
            year = int(parts[0])
            quarter = int(parts[1].replace('Q', ''))
            return (year, quarter)
        except:
            return None
    
    df['year_quarter'] = df['fiscal_timeframe'].apply(parse_fiscal_timeframe)
    df = df[df['year_quarter'].notna()]  # Remove rows with invalid fiscal_timeframe
    
    if len(df) < 8:
        return quarterly_df, None
    
    metrics = {
        'revenue_growth': {},
        'revenue_growth_list': [],
        'income_growth': {},
        'income_growth_list': [],
        'margin_growth': {},
        'margin_growth_list': [],
        'margin_values': [],
        'revenue_consistency_score': 0,
        'income_consistency_score': 0,
        'last3_quarters_revenue_growth': 0,
        'last3_quarters_income_growth': 0,
        'last3_quarters_margin_growth': 0,
        'last3_quarters_income_positive_count': 0,
        'median_revenue_growth': 0,
        'median_income_growth': 0,
        'median_margin_growth': 0,
        'median_margin': 0,
        'last3_quarters_median_margin': 0,
        'revenue_growth_slope': 0,
        'revenue_r2': 0,
        'income_growth_slope': 0,
        'income_r2': 0,
        'margin_growth_slope': 0,
        'margin_r2': 0
    }
    
    # Add growth columns to the original dataframe
    result_df = quarterly_df.copy()
    result_df['Revenue_YoY_Growth_%'] = np.nan
    result_df['Income_YoY_Growth_%'] = np.nan
    result_df['Margin_YoY_Growth_%'] = np.nan
    result_df['Margin_%'] = np.nan
    
    min_year=2020
    
    # Calculate and store margin values for all quarters
    if 'Net Income' in df.columns and 'Revenue/Sales' in df.columns:
        income_values = pd.to_numeric(df['Net Income'], errors='coerce')
        revenue_values = pd.to_numeric(df['Revenue/Sales'], errors='coerce')
        
        for i in range(len(df)):
            if pd.notna(income_values.iloc[i]) and pd.notna(revenue_values.iloc[i]) and revenue_values.iloc[i] > 0:
                margin = (income_values.iloc[i] / revenue_values.iloc[i]) * 100
                fiscal_str = df.loc[i, 'fiscal_timeframe']
                metrics['margin_values'].append(margin)
                result_df.loc[result_df['fiscal_timeframe'] == fiscal_str, 'Margin_%'] = margin
    
    # Calculate YoY growth for Revenue/Sales
    if 'Revenue/Sales' in df.columns:
        revenue_values = pd.to_numeric(df['Revenue/Sales'], errors='coerce')
        
        for i in range(len(df)):
            current_year, current_q = df.loc[i, 'year_quarter']
            prior_year = current_year - 1
            
            # Find the corresponding quarter from prior year
            prior_row = df[(df['year_quarter'].apply(lambda x: x[0] == prior_year and x[1] == current_q))]
            
            if not prior_row.empty and pd.notna(revenue_values.iloc[i]):
                prior_idx = prior_row.index[0]
                prior_value = pd.to_numeric(df.loc[prior_idx, 'Revenue/Sales'], errors='coerce')
                current_value = revenue_values.iloc[i]
                
                if pd.notna(prior_value) and prior_value > 0:
                    growth = ((current_value - prior_value) / prior_value) * 100
                    fiscal_str = df.loc[i, 'fiscal_timeframe']
                    if int(fiscal_str[:4])>=min_year:
                        metrics['revenue_growth'][fiscal_str] = growth
                        metrics['revenue_growth_list'].append(growth)
                    # Update the result dataframe
                    result_df.loc[result_df['fiscal_timeframe'] == fiscal_str, 'Revenue_YoY_Growth_%'] = growth
    
    # Calculate YoY growth for Net Income
    if 'Net Income' in df.columns:
        income_values = pd.to_numeric(df['Net Income'], errors='coerce')
        
        for i in range(len(df)):
            current_year, current_q = df.loc[i, 'year_quarter']
            prior_year = current_year - 1
            
            # Find the corresponding quarter from prior year
            prior_row = df[(df['year_quarter'].apply(lambda x: x[0] == prior_year and x[1] == current_q))]
            
            if not prior_row.empty and pd.notna(income_values.iloc[i]):
                prior_idx = prior_row.index[0]
                prior_value = pd.to_numeric(df.loc[prior_idx, 'Net Income'], errors='coerce')
                current_value = income_values.iloc[i]
                
                if pd.notna(prior_value) and prior_value != 0:
                    growth = ((current_value - prior_value) / abs(prior_value)) * 100
                    fiscal_str = df.loc[i, 'fiscal_timeframe']
                    if int(fiscal_str[:4])>=min_year:
                        metrics['income_growth'][fiscal_str] = growth
                        metrics['income_growth_list'].append(growth)
                    # Update the result dataframe
                    result_df.loc[result_df['fiscal_timeframe'] == fiscal_str, 'Income_YoY_Growth_%'] = growth
    
    # Calculate YoY growth for Margin (Net Income / Revenue)
    if 'Net Income' in df.columns and 'Revenue/Sales' in df.columns:
        income_values = pd.to_numeric(df['Net Income'], errors='coerce')
        revenue_values = pd.to_numeric(df['Revenue/Sales'], errors='coerce')
        
        for i in range(len(df)):
            current_year, current_q = df.loc[i, 'year_quarter']
            prior_year = current_year - 1
            
            # Find the corresponding quarter from prior year
            prior_row = df[(df['year_quarter'].apply(lambda x: x[0] == prior_year and x[1] == current_q))]
            
            if not prior_row.empty and pd.notna(income_values.iloc[i]) and pd.notna(revenue_values.iloc[i]):
                prior_idx = prior_row.index[0]
                prior_revenue = pd.to_numeric(df.loc[prior_idx, 'Revenue/Sales'], errors='coerce')
                prior_income = pd.to_numeric(df.loc[prior_idx, 'Net Income'], errors='coerce')
                current_revenue = revenue_values.iloc[i]
                current_income = income_values.iloc[i]
                
                if pd.notna(prior_revenue) and prior_revenue > 0 and pd.notna(prior_income):
                    prior_margin = (prior_income / prior_revenue) * 100
                    current_margin = (current_income / current_revenue) * 100
                    margin_growth = current_margin - prior_margin
                    
                    fiscal_str = df.loc[i, 'fiscal_timeframe']
                    if int(fiscal_str[:4])>=min_year:
                        metrics['margin_growth'][fiscal_str] = margin_growth
                        metrics['margin_growth_list'].append(margin_growth)
                    # Update the result dataframe
                    result_df.loc[result_df['fiscal_timeframe'] == fiscal_str, 'Margin_YoY_Growth_%'] = margin_growth
    
    # Calculate consistency metrics
    if metrics['revenue_growth_list']:
        revenue_growths = metrics['revenue_growth_list']
        result_df = result_df[result_df["fiscal_timeframe"].str[:4].astype(int) >= min_year] #filter to min_year

        # st.write(metrics) #debug
        metrics['median_revenue_growth'] = np.median(revenue_growths)
        # Consistency score: positive growth with low volatility
        metrics['revenue_consistency_score'] = len([g for g in revenue_growths if g > 0]) / len(revenue_growths) * 100
        
        # Last 3 quarters average
        if len(revenue_growths) >= 3:
            metrics['last3_quarters_revenue_growth'] = np.median(revenue_growths[-3:])
        elif len(revenue_growths) > 0:
            metrics['last3_quarters_revenue_growth'] = np.median(revenue_growths)
        
        # Linear regression for revenue growth trend
        if len(revenue_growths) >= 2:
            X = np.arange(len(revenue_growths))
            y = np.array(revenue_growths)
            # Calculate slope and R² using numpy polyfit
            coeffs = np.polyfit(X, y, 1)
            # st.write('X:',X,'y:',y,'coeffs:',coeffs) #debug
            
            metrics['revenue_growth_slope'] = coeffs[0]
            # Calculate R² manually
            y_pred = np.polyval(coeffs, X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics['revenue_r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    if metrics['income_growth_list']:
        income_growths = metrics['income_growth_list']
        metrics['median_income_growth'] = np.median(income_growths)
        # Income growth consistency
        metrics['income_consistency_score'] = len([g for g in income_growths if g > 0]) / len(income_growths) * 100
        metrics['last3_quarters_income_positive_count'] = len([g for g in income_growths[-3:] if g > 0]) if len(income_growths) >= 3 else len([g for g in income_growths if g > 0])
        
        # Last 3 quarters average
        if len(income_growths) >= 3:
            metrics['last3_quarters_income_growth'] = np.median(income_growths[-3:])
        elif len(income_growths) > 0:
            metrics['last3_quarters_income_growth'] = np.median(income_growths)
        
        # Linear regression for income growth trend
        if len(income_growths) >= 2:
            X = np.arange(len(income_growths))
            y = np.array(income_growths)
            # Calculate slope and R² using numpy polyfit
            coeffs = np.polyfit(X, y, 1)
            metrics['income_growth_slope'] = coeffs[0]
            # Calculate R² manually
            y_pred = np.polyval(coeffs, X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics['income_r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        # st.write(metrics)  # debug
    
    if metrics['margin_growth_list']:
        margin_growths = metrics['margin_growth_list']
        metrics['median_margin_growth'] = np.median(margin_growths)
        
        # Last 3 quarters average
        if len(margin_growths) >= 3:
            metrics['last3_quarters_margin_growth'] = np.median(margin_growths[-3:])
        elif len(margin_growths) > 0:
            metrics['last3_quarters_margin_growth'] = np.median(margin_growths)
        
        # Linear regression for margin growth trend
        if len(margin_growths) >= 2:
            X = np.arange(len(margin_growths))
            y = np.array(margin_growths)
            # Calculate slope and R² using numpy polyfit
            coeffs = np.polyfit(X, y, 1)
            metrics['margin_growth_slope'] = coeffs[0]
            # Calculate R² manually
            y_pred = np.polyval(coeffs, X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            metrics['margin_r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Calculate average margin and last 3 quarters average margin
    if metrics['margin_values']:
        margin_values = metrics['margin_values']
        metrics['median_margin'] = np.median(margin_values)
        
        # Last 3 quarters average margin
        if len(margin_values) >= 3:
            metrics['last3_quarters_median_margin'] = np.median(margin_values[-3:])
        elif len(margin_values) > 0:
            metrics['last3_quarters_median_margin'] = np.median(margin_values)
        
        # st.write(metrics)  # debug
    
    return result_df, metrics


def rank_companies_by_growth(companies_dict):
    """
    Rank companies by growth consistency.
    
    Args:
        companies_dict: Dictionary with company names as keys and quarterly DataFrames as values
    
    Returns:
        tuple: (ranking_df, updated_companies_dict)
    """
    results = []
    updated_companies = {}
    
    for cik, quarterly_df in companies_dict.items():
        updated_df, metrics = analyze_yoy_growth(quarterly_df)
        updated_companies[cik] = updated_df
        
        if metrics:
            results.append({
                'cik': cik,
                'Consolidated_Score': (
                    -100 if len(metrics['revenue_growth']) < 15
                    else round(
                        (
                            metrics['median_revenue_growth'] * metrics['revenue_consistency_score'] / 100
                            + metrics['last3_quarters_median_margin'] * metrics['income_consistency_score'] / 100
                        ) / 2
                        - (3 - metrics['last3_quarters_income_positive_count'] * 10),
                        1
                    )
                ),
                'Revenue_Consistency_Score': round(metrics['revenue_consistency_score'], 1),
                'Income_Consistency_Score': round(metrics['income_consistency_score'], 1),
                'Median__Revenue_Growth_%': round(metrics['median_revenue_growth'], 2),
                'Median__Income_Growth_%': round(metrics['median_income_growth'], 2),
                'Median__Margin_Growth_%': round(metrics['median_margin_growth'], 2),
                'Median__Margin_%': round(metrics['median_margin'], 2),
                'Last3Q_Revenue_Growth_%': round(metrics['last3_quarters_revenue_growth'], 2),
                'Last3Q_Income_Growth_%': round(metrics['last3_quarters_income_growth'], 2),
                'Last3Q_Margin_Growth_%': round(metrics['last3_quarters_margin_growth'], 2),
                'Last3Q_Median__Margin_%': round(metrics['last3_quarters_median_margin'], 2),
                'Last3Q_Income_Positive': int(metrics['last3_quarters_income_positive_count']),
                'Revenue_Growth_Count': len(metrics['revenue_growth']),
                'Income_Growth_Count': len(metrics['income_growth']),
                'Margin_Growth_Count': len(metrics['margin_growth']),
                'Revenue_Growth_Slope': round(metrics['revenue_growth_slope'], 4),
                'Revenue_R2': round(metrics['revenue_r2'], 4),
                'Income_Growth_Slope': round(metrics['income_growth_slope'], 4),
                'Income_R2': round(metrics['income_r2'], 4),
                'Margin_Growth_Slope': round(metrics['margin_growth_slope'], 4),
                'Margin_R2': round(metrics['margin_r2'], 4)
            })
    
    results_df = pd.DataFrame(results)
    # st.dataframe(results_df)  # debug
    # st.json(updated_companies)  # debug

    results_df = results_df.sort_values('Consolidated_Score', ascending=False)
    
    return results_df, updated_companies

# set page config and title
st.set_page_config( page_title="Stock Screener", layout="wide" )
st.markdown('<h2 style="color:#3894f0;">Stock Screener for Publically Traded Stocks</h2>', unsafe_allow_html=True)
st.write('Created by Rafael Avila leveraging Snowflake & Streamlit, using SEC Filings data provided by SEC Edgar platform. Analysis Summary and AI chat powered by mistral-large2 AI model')

def main():
    ss.company_lookup_df=get_company_tickers_df()
    ss.company_lookup_df = (
        ss.company_lookup_df
        .sort_values("company_and_ticker")
        .drop_duplicates(subset="cik", keep="first")
    )
    st.write(f"company_lookup_df = ",ss.company_lookup_df)  #debug
    
    #debug this only selects a few CIKs for testing
    # cik_list={'0000066740'} #'0001368514','0001070494','0000318306','0000002488','0001018724','0000789019','0001045810','0000034782'} #,'0001652044','0000320193','0001018724','0001326801','0001730168','0001318605','0001067983','0000059478'} #debug
    
    cik_list = ss.company_lookup_df['cik'].tolist() #.head(5)
    
    companies_data = {}  # Store company data for analysis
    
    progress_bar = st.progress(0)
    status = st.empty()
    total = len(cik_list)
    
    for i, cik in enumerate(cik_list):
        # Load SEC EDGAR financials for the selected CIK (guarded)
        try:
            # st.write(f"Loading SEC data for CIK: {cik}...")  # debug
            status.write(f"Processing CIK {cik} ({i+1}/{total})")
            res = sec_edgar_financial_load(cik)
            progress_bar.progress((i + 1) / total)
            # st.write(res)  # debug
            if not isinstance(res, (list, tuple)) or len(res) != 3:
                print("Unexpected return from sec_edgar_financial_load; expected (filings_df, quarterly_df, annual_df).")
                ss.filings_df = pd.DataFrame()
                ss.quarterly_financials = pd.DataFrame()
                ss.annual_financials = pd.DataFrame()
                st.stop()
                
            ss.filings_df, ss.quarterly_financials, ss.annual_financials = res
            ss.quarterly_financials = ss.quarterly_financials.merge(
                ss.company_lookup_df[['cik', 'company_and_ticker']],
                on='cik',
                how='left')
            col = ss.quarterly_financials.pop('company_and_ticker')
            ss.quarterly_financials.insert(0, 'company_and_ticker', col)
            
            # st.dataframe(ss.quarterly_financials.head())  # debug

            # If quarterly data isn't present, surface a message and stop further processing
            if not isinstance(ss.quarterly_financials, pd.DataFrame) or ss.quarterly_financials.empty:
                print(f"No quarterly financials returned from SEC for {cik} company.")
                # if isinstance(ss.filings_df, pd.DataFrame) and not ss.filings_df.empty:
                #     st.write("Filings (sample):")
                #     st.dataframe(ss.filings_df.head())
                continue
            
            # Store company data for growth analysis
            # company_cik = ss.company_lookup_df.iloc[0]['company_and_ticker'] #ss.filings_df.iloc[0]['Company Name'] if not ss.filings_df.empty and 'Company Name' in ss.filings_df.columns else f"CIK_{cik}"
            companies_data[cik] = ss.quarterly_financials
            # st.write(companies_data)  # debug
            
        except Exception as e:
            print(f"Failed to load SEC data: {e}")
            # try:
            #     import traceback
            #     st.code(traceback.format_exc())
            # except Exception:
            #     pass
            ss.filings_df = pd.DataFrame()
            ss.quarterly_financials = pd.DataFrame()
            ss.annual_financials = pd.DataFrame()
            continue # Skip to next CIK    
    # Analyze and rank companies by growth consistency
    if companies_data:
        st.markdown("---")
        st.subheader("📊 Growth Consistency Analysis")
        
        # st.write(companies_data)  # debug
        ranking_df, updated_companies = rank_companies_by_growth(companies_data)
        
        if not ranking_df.empty:
            # st.write(ranking_df)  # debug
            ranking_df = ranking_df.merge(
                ss.company_lookup_df[['cik', 'company_and_ticker']],
                on='cik',
                how='left')
            col = ranking_df.pop('company_and_ticker')
            ranking_df.insert(0, 'company_and_ticker', col)

            st.dataframe(
                ranking_df.style.format({
                    'Revenue_Consistency_Score': '{:.1f}',
                    'Income_Consistency_Score': '{:.1f}',
                    'Median__Revenue_Growth_%': '{:.2f}',
                    'Median__Income_Growth_%': '{:.2f}',
                    'Median__Margin_Growth_%': '{:.2f}',
                    'Median__Margin_%': '{:.2f}',
                    'Last3Q_Revenue_Growth_%': '{:.2f}',
                    'Last3Q_Income_Growth_%': '{:.2f}',
                    'Last3Q_Margin_Growth_%': '{:.2f}',
                    'Last3Q_Median__Margin_%': '{:.2f}',
                    'Revenue_Growth_Slope': '{:.4f}',
                    'Revenue_R2': '{:.4f}',
                    'Income_Growth_Slope': '{:.4f}',
                    'Income_R2': '{:.4f}',
                    'Margin_Growth_Slope': '{:.4f}',
                    'Margin_R2': '{:.4f}'
                }),
                use_container_width=True
            )
            # st.write(ranking_df)  # debug
            
            # Highlight top performer
            top_company = ranking_df.iloc[0]
            st.success(f"🏆 Top Performer: **{top_company['company_and_ticker']}** (Score: {top_company['Revenue_Consistency_Score']:.1f})")
            # st.write(f"**Revenue Analysis:**")
            # st.write(f"- Avg Growth: {top_company['median_Revenue_Growth_%']:.2f}%")
            # st.write(f"- Trend Slope: {top_company['Revenue_Growth_Slope']:.4f} (% per quarter)")
            # st.write(f"- Consistency (R²): {top_company['Revenue_R2']:.4f}")
            
            # st.write(f"**Income Analysis:**")
            # st.write(f"- Avg Growth: {top_company['median_Income_Growth_%']:.2f}%")
            # st.write(f"- Trend Slope: {top_company['Income_Growth_Slope']:.4f} (% per quarter)")
            # st.write(f"- Consistency (R²): {top_company['Income_R2']:.4f}")
            
            # Display top company's quarterly data with growth columns
            # st.write(f"**{top_company['company_and_ticker']} - Quarterly Financials with YoY Growth:**")
            # top_company_df = updated_companies[top_company['cik']]
            # st.dataframe(
            #     top_company_df[['fiscal_timeframe', 'Revenue/Sales', 'Revenue_YoY_Growth_%', 'Net Income', 'Income_YoY_Growth_%']],
            #     use_container_width=True
            # )

            quarterly_updated_df = pd.concat(
                updated_companies.values(), 
                ignore_index=True
            )
            
            # Merge company metrics into the quarterly dataframe
            # metrics_cols = [
            #     'cik',
            #     'Revenue_Consistency_Score',
            #     'Income_Consistency_Score',
            #     'Median__Revenue_Growth_%',
            #     'Median__Income_Growth_%',
            #     'Median__Margin_Growth_%',
            #     'Median__Margin_%',
            #     'Last3Q_Revenue_Growth_%',
            #     'Last3Q_Income_Growth_%',
            #     'Last3Q_Margin_Growth_%',
            #     'Last3Q_Median__Margin_%',
            #     'Last3Q_Income_Positive',
            #     'Revenue_Growth_Slope',
            #     'Revenue_R2',
            #     'Income_Growth_Slope',
            #     'Income_R2',
            #     'Margin_Growth_Slope',
            #     'Margin_R2'
            # ]
            # quarterly_updated_df = quarterly_updated_df.merge(
            #     ranking_df[metrics_cols],
            #     on='cik',
            #     how='left'
            # )
            
            cols = [
                'company_and_ticker',
                'cik',
                'fiscal_timeframe',
                'Revenue/Sales',
                'Revenue_YoY_Growth_%',
                'Net Income',
                'Income_YoY_Growth_%',
                'Margin_%',
                'Margin_YoY_Growth_%'
            ]
            #     'Revenue_Consistency_Score',
            #     'Income_Consistency_Score',
            #     'Median__Revenue_Growth_%',
            #     'Median__Income_Growth_%',
            #     'Median__Margin_Growth_%',
            #     'Median__Margin_%',
            #     'Last3Q_Revenue_Growth_%',
            #     'Last3Q_Income_Growth_%',
            #     'Last3Q_Margin_Growth_%',
            #     'Last3Q_Median__Margin_%',
            #     'Revenue_Growth_Slope',
            #     'Revenue_R2',
            #     'Income_Growth_Slope',
            #     'Income_R2',
            #     'Margin_Growth_Slope',
            #     'Margin_R2'
            # ]
            # quarterly_updated_df = quarterly_updated_df[cols]
            
            st.write("**All Companies - Quarterly Data with Metrics:**")
            st.dataframe(quarterly_updated_df[cols],use_container_width=True)
            
main()
import streamlit as st
import streamlit.elements.widgets.data_editor as de
import pyarrow as pa
import pandas as pd
import altair as alt
from icecream import ic
from altair.expr import *
from SEC_Edgar_Loader import sec_edgar_financial_load,get_company_tickers_df
import numpy as np
import sys, linecache, traceback
from stock_data_loader_utilities import postgres_update, postgres_read,yahoo_finance_load,yahoo_finance_df_format

ss = st.session_state

if "quarterly_financials" not in ss:
    ss.quarterly_financials = pd.DataFrame()
    ss.ranking_df = pd.DataFrame()
    ss.editable_stock_growth_analysis_df = pd.DataFrame()
    ss.styled_editable_stock_growth_analysis_df = pd.DataFrame()
    ss.metrics={}
    ss.updated_companies={}
    ss.value_btn=0
    ss.selected_company=None
    ss.hide_menu=False

def plot_regression_line(name, var_name, X, y, y_pred_plot, slope, r2, end_date):

    # Build DataFrame for Altair
    plot_df = pd.DataFrame({
        "Quarter Index": X,
        "x_label": pd.to_datetime(end_date),
        var_name: y,
        "Fitted": y_pred_plot,
    })
    
    # st.write(plot_df) #debug

    # Chart
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("x_label:T", title="End Date"),
            y=alt.Y(f"{var_name}:Q", title=var_name.replace("_", " ")),
            tooltip=[
                alt.Tooltip("x_label:T", title="End Date"),
                alt.Tooltip(f"{var_name}:Q", title=var_name.replace("_", " "), format=",.0f"),
                alt.Tooltip("Fitted:Q", title="Trend", format=",.0f")
            ]            
        )
        .properties(
            width=800,
            height=400,
            title=f"{var_name.replace('_', ' ')} Regression for {name}, Slope: ${slope/1000:,.1f}k, R²: {r2:.4f}"
        )
    )

    # Add regression line
    reg_line = (
        alt.Chart(plot_df)
        .mark_line(color="red")
        .encode(
            x="x_label:T",
            y="Fitted:Q"
        )
    )
    
    # st.altair_chart(chart + reg_line, width='content')              
    return chart + reg_line

def sanitize(values):
    # Convert everything to float, invalid values become np.nan
    arr = pd.to_numeric(values, errors='coerce').astype(float)

    # If all values are NaN, return None
    if np.isnan(arr).all():
        return None

    return arr

def perform_regression(name, var_name, values, end_date, plot_regression_bin):
    
    # 1. Sanitize input
    y = sanitize(values)
    if y is None:
        return None  # nothing to regress on
    
    # st.write(f"values: {len(values)}, y: {len(y)}")
    end_date = np.array(end_date)
    
    # st.write(len(y),len(end_date)) #debug
    
    assert len(y) == len(end_date), f"Length mismatch: y={len(y)}, end_date={len(end_date)}"

    # y already length-preserving
    median = np.median(y[~np.isnan(y)])  # ignore NaNs for median
    mad = np.median(np.abs(y[~np.isnan(y)] - median))


    if mad == 0:
        mask = ~np.isnan(y)
    else:
        z = 0.6745 * (y - median) / mad
        mask = (np.abs(z) < 20.0) & ~np.isnan(y)

    y_clean = y[mask]
    # st.write(y_clean) #debug

    end_date_clean = np.array(end_date)[mask]
    # same for fiscal_str if you use it
    
    # 4. Bail early if mask wipes out too much
    if len(y_clean) < 2 or np.isnan(y_clean).any():
        return None

    # 5. Regression-ready arrays
    X_clean = np.arange(len(y_clean))
    y_clean = np.array(y_clean, dtype=float)

    # 6. Compute stats safely
    mean = np.mean(y_clean)
    median = np.median(y_clean)
    std = np.std(y_clean)

    # 7. Count outliers
    n = len(y)
    n_outliers = n - len(y_clean)
    
    # debug_rows = []
    # for i, v in enumerate(y):
    #     debug_rows.append({
    #         "index": i,
    #         "value": float(v),
    #         "z_score": float(z[i]),
    #         "kept": bool(mask[i])
    #     })
    # st.write(f"Outlier debug for {var_name} (company {name}):", debug_rows) # debug

    # Need at least 6 points after filtering
    if len(y_clean) >= 6:
        coeffs = np.polyfit(X_clean, y_clean, 1)
        slope, intercept = coeffs

        y_pred = np.polyval(coeffs, X_clean)
        y_pred_plot = slope * X_clean + intercept

        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        if plot_regression_bin==1:
            chart=plot_regression_line(name, var_name, X_clean, y_clean, y_pred_plot, slope, r2, end_date_clean) #debug
        else:
            chart=None

        residuals = y_clean - y_pred
        last_n = min(3, len(residuals)) # if less than 3 points, take all
        avg_residual_last3 = residuals[-last_n:].mean()
    return slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart

# ===== Growth Analysis Functions =====
def analyze_yoy_growth(quarterly_df, name, plot_regression_bin):
    """
    Analyze year-over-year growth consistency for a company.
    
    Args: quarterly_df: DataFrame with quarterly financials (includes 'fiscal_timeframe', 'Revenue/Sales', 'Net Income')
    
    Returns: tuple: (updated_quarterly_df, metrics_dict)
               - updated_quarterly_df: Original df with growth columns added
               - metrics_dict: Growth metrics including regression analysis
    """
    
    if quarterly_df.empty or len(quarterly_df) < 8:  # Need at least 2 years of data
        return quarterly_df, None
    
    # Ensure fiscal_timeframe column exists
    if 'fiscal_timeframe' not in quarterly_df.columns:
        return quarterly_df, None
    
    # Create a working dataframe with fiscal_timeframe
    quarterly_df_wc = quarterly_df.reset_index(drop=True).copy()
    quarterly_df_wc['fiscal_timeframe'] = quarterly_df_wc['fiscal_timeframe'].astype(str)
    quarterly_df_wc = quarterly_df_wc.sort_values(by='fiscal_timeframe', ascending=True).reset_index(drop=True)
       
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
    
    quarterly_df_wc['year_quarter'] = quarterly_df_wc['fiscal_timeframe'].apply(parse_fiscal_timeframe)
    quarterly_df_wc = quarterly_df_wc[quarterly_df_wc['year_quarter'].notna()]  # Remove rows with invalid fiscal_timeframe
    
    # st.write(quarterly_df_wc) #debug
    
    if len(quarterly_df_wc) < 8:
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
        'revenue_growth_median': 0,
        'revenue_n': 0,
        'revenue_n_outliers': 0,
        'revenue_r2': 0,
        'revenue_avg_residual_last3': 0,
        'income_growth_slope': 0,
        'income_growth_median': 0,
        'income_n': 0,
        'income_n_outliers': 0,
        'income_r2': 0,
        'income_avg_residual_last3': 0,
        'margin_growth_slope': 0,
        'margin_growth_median': 0,
        'margin_n': 0,
        'margin_n_outliers': 0,
        'margin_r2': 0,
        'margin_avg_residual_last3': 0
    }

    # st.dataframe(quarterly_df_wc)  # debug

    # Add growth columns to the original dataframe
    result_df = quarterly_df.copy()
    result_df['Revenue_YoY_Growth_PCT'] = np.nan
    result_df['Income_YoY_Growth_PCT'] = np.nan
    result_df['Margin_YoY_Growth_PCT'] = np.nan
    result_df['Margin_PCT'] = np.nan
    
    min_year=2020
    
    # Calculate and store margin values for all quarters
    if 'Net Income' in quarterly_df_wc.columns and 'Revenue/Sales' in quarterly_df_wc.columns:
        fiscal_str = quarterly_df_wc['fiscal_timeframe'].tolist()
        end_date = quarterly_df_wc['end_date'].tolist()
        income_values = pd.to_numeric(quarterly_df_wc['Net Income'], errors='coerce')
        revenue_values = pd.to_numeric(quarterly_df_wc['Revenue/Sales'], errors='coerce')
        
        for i in range(len(quarterly_df_wc)):
            if pd.notna(income_values.iloc[i]) and pd.notna(revenue_values.iloc[i]) and revenue_values.iloc[i] > 0:
                margin = (income_values.iloc[i] / revenue_values.iloc[i]) * 100
                fiscal_str = quarterly_df_wc.loc[i, 'fiscal_timeframe']
                metrics['margin_values'].append(margin)
                result_df.loc[result_df['fiscal_timeframe'] == fiscal_str, 'Margin_PCT'] = margin
    
    # Calculate YoY growth for Revenue/Sales
    if 'Revenue/Sales' in quarterly_df_wc.columns:
        revenue_values = pd.to_numeric(quarterly_df_wc['Revenue/Sales'], errors='coerce')
        
        for i in range(len(quarterly_df_wc)):
            current_year, current_q = quarterly_df_wc.loc[i, 'year_quarter']
            prior_year = current_year - 1
            
            # Find the corresponding quarter from prior year
            prior_row = quarterly_df_wc[(quarterly_df_wc['year_quarter'].apply(lambda x: x[0] == prior_year and x[1] == current_q))]
            
            if not prior_row.empty and pd.notna(revenue_values.iloc[i]):
                prior_idx = prior_row.index[0]
                prior_value = pd.to_numeric(quarterly_df_wc.loc[prior_idx, 'Revenue/Sales'], errors='coerce')
                current_value = revenue_values.iloc[i]
                
                if pd.notna(prior_value) and prior_value > 0:
                    growth = ((current_value - prior_value) / prior_value) * 100
                    fiscal_str = quarterly_df_wc.loc[i, 'fiscal_timeframe']
                    if int(fiscal_str[:4])>=min_year:
                        metrics['revenue_growth'][fiscal_str] = growth
                        metrics['revenue_growth_list'].append(growth)
                    # Update the result dataframe
                    result_df.loc[result_df['fiscal_timeframe'] == fiscal_str, 'Revenue_YoY_Growth_PCT'] = growth
                    # st.write(f"Revenue Growth for {fiscal_str}: {growth:.2f}%")  # debug
    
    # Calculate YoY growth for Net Income
    if 'Net Income' in quarterly_df_wc.columns:
        income_values = pd.to_numeric(quarterly_df_wc['Net Income'], errors='coerce')
        
        for i in range(len(quarterly_df_wc)):
            current_year, current_q = quarterly_df_wc.loc[i, 'year_quarter']
            prior_year = current_year - 1
            
            # Find the corresponding quarter from prior year
            prior_row = quarterly_df_wc[(quarterly_df_wc['year_quarter'].apply(lambda x: x[0] == prior_year and x[1] == current_q))]
            
            if not prior_row.empty and pd.notna(income_values.iloc[i]):
                prior_idx = prior_row.index[0]
                prior_value = pd.to_numeric(quarterly_df_wc.loc[prior_idx, 'Net Income'], errors='coerce')
                current_value = income_values.iloc[i]
                
                if pd.notna(prior_value) and prior_value != 0:
                    growth = ((current_value - prior_value) / abs(prior_value)) * 100
                    fiscal_str = quarterly_df_wc.loc[i, 'fiscal_timeframe']
                    if int(fiscal_str[:4])>=min_year:
                        metrics['income_growth'][fiscal_str] = growth
                        metrics['income_growth_list'].append(growth)
                    # Update the result dataframe
                    result_df.loc[result_df['fiscal_timeframe'] == fiscal_str, 'Income_YoY_Growth_PCT'] = growth
    
    # Calculate YoY growth for Margin (Net Income / Revenue)
    if 'Net Income' in quarterly_df_wc.columns and 'Revenue/Sales' in quarterly_df_wc.columns:
        income_values = pd.to_numeric(quarterly_df_wc['Net Income'], errors='coerce')
        revenue_values = pd.to_numeric(quarterly_df_wc['Revenue/Sales'], errors='coerce')
        margin_values = (income_values / revenue_values * 100).fillna(0)
        
        for i in range(len(quarterly_df_wc)):
            current_year, current_q = quarterly_df_wc.loc[i, 'year_quarter']
            prior_year = current_year - 1
            
            # Find the corresponding quarter from prior year
            prior_row = quarterly_df_wc[(quarterly_df_wc['year_quarter'].apply(lambda x: x[0] == prior_year and x[1] == current_q))]
            
            if not prior_row.empty and pd.notna(income_values.iloc[i]) and pd.notna(revenue_values.iloc[i]):
                prior_idx = prior_row.index[0]
                prior_revenue = pd.to_numeric(quarterly_df_wc.loc[prior_idx, 'Revenue/Sales'], errors='coerce')
                prior_income = pd.to_numeric(quarterly_df_wc.loc[prior_idx, 'Net Income'], errors='coerce')
                current_revenue = revenue_values.iloc[i]
                current_income = income_values.iloc[i]
                def normalize_margin(x):
                    if x is None or (isinstance(x, float) and np.isnan(x)):
                        return np.nan
                    return float(x)

                def safe_margin(income, revenue):
                    if revenue is None or revenue == 0 or np.isnan(revenue):
                        return None
                    return (income / revenue) * 100
                
                current_margin = None
                prior_margin = None
                
                if pd.notna(prior_revenue) and prior_revenue > 0 and pd.notna(prior_income):
                    current_margin = safe_margin(current_income, current_revenue)
                    prior_margin   = safe_margin(prior_income, prior_revenue)
                    current_margin = normalize_margin(current_margin)
                    prior_margin   = normalize_margin(prior_margin)

                    if np.isnan(current_margin) or np.isnan(prior_margin):
                        margin_growth = np.nan
                    else:
                        margin_growth = current_margin - prior_margin

                    margin_growth = current_margin - prior_margin
                    
                    fiscal_str = quarterly_df_wc.loc[i, 'fiscal_timeframe']
                    if int(fiscal_str[:4])>=min_year:
                        metrics['margin_growth'][fiscal_str] = margin_growth
                        metrics['margin_growth_list'].append(margin_growth)
                    # Update the result dataframe
                    result_df.loc[result_df['fiscal_timeframe'] == fiscal_str, 'Margin_YoY_Growth_PCT'] = margin_growth
    
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
        # st.write(revenue_values)  # debug
    # st.json(metrics) # debug
    if len(revenue_values) >= 6:
        try:
            # st.write(len(revenue_values),len(end_date)) #debug
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_revenue = perform_regression(name, "Revenue", revenue_values, end_date, plot_regression_bin)
            metrics['revenue_r2'] = r2
            metrics['revenue_growth_slope'] = round(slope, 1)
            metrics['revenue_growth_median'] = median
            metrics['revenue_n'] = n
            metrics['revenue_n_outliers'] = n_outliers
            metrics['revenue_avg_residual_last3'] = avg_residual_last3
        except Exception as e:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_revenue = 0, 0, 0, 0, 0, 0, 0, 0, None
            print(f"Regression failed for Revenue on company {name} due to {e}")  # debug
            
   
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
    if len(income_values) >= 6:  # Need at least 6 quarters to do a meaningful regression
        try:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_income = perform_regression(name, "Income", income_values, end_date, plot_regression_bin)
            metrics['income_growth_slope'] = round(slope, 1)
            metrics['income_r2'] = r2
            metrics['income_growth_median'] = median
            metrics['income_n'] = n
            metrics['income_n_outliers'] = n_outliers
            metrics['income_avg_residual_last3'] = avg_residual_last3
        except Exception as e:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_income = 0, 0, 0, 0, 0, 0, 0, 0, None
            print(f"Regression failed for Income for company {name} due to {e}")  # debug
        
    if metrics['margin_growth_list']:
        margin_growths = metrics['margin_growth_list']
        metrics['median_margin_growth'] = np.median(margin_growths)
        
        # Last 3 quarters average
        if len(margin_growths) >= 3:
            metrics['last3_quarters_margin_growth'] = np.median(margin_growths[-3:])
        elif len(margin_growths) > 0:
            metrics['last3_quarters_margin_growth'] = np.median(margin_growths)
        
    # Linear regression for margin growth trend
    if len(margin_values) >= 6:  # Need at least 6 quarters to do a meaningful regression
        try:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_margin = perform_regression(name, "Margin", margin_values, end_date, plot_regression_bin)
            metrics['margin_r2'] = r2
            metrics['margin_growth_slope'] = round(slope, 1)
            metrics['margin_avg_residual_last3'] = avg_residual_last3
            metrics['margin_growth_median'] = median
            metrics['margin_n'] = n
            metrics['margin_n_outliers'] = n_outliers
        except Exception as e:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_margin = 0, 0, 0, 0, 0, 0, 0, 0, None
            print(f"Regression failed for Margin for company {name} due to {e}")  # debug
    
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
    # st.write(ss.selected_company)  # debug
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
        
    if plot_regression_bin==1:
        try:
            col1, col2, col3 = st.columns(3)
            if chart_revenue is not None:
                with col1:
                    # st.markdown(f'<p class="centered-title">Income Statement {name}</p>', unsafe_allow_html=True)
                    placeholder=st.empty()
                    placeholder.altair_chart(chart_revenue, width='stretch') #, width='stretch'
            if chart_income is not None:
                with col2:
                    placeholder=st.empty()
                    placeholder.altair_chart(chart_income, width='stretch')
            if chart_margin is not None:
                with col3:
                    placeholder=st.empty()
                    placeholder.altair_chart(chart_margin, width='stretch')
        except Exception as e:
            st.write(f"Could not render regression charts due to {e}")
    
    return result_df, metrics


def rank_companies_by_growth():
    """
    Rank companies by growth consistency.
    Args: companies_dict: Dictionary with company names as keys and quarterly DataFrames as values
    Returns: tuple: (ranking_df, updated_companies_dict)
    """
    
    # st.write(ss.quarterly_financials)  # debug
    results = []
    updated_companies = {}
    # st.write(list(companies_dict.items())[:5])  # debug   
    
    cik_list=ss.quarterly_financials['cik'].unique() #{'0000066740','0001368514'}#,'0001070494','0000318306','0001018724','0000789019','0000002488','0001045810','0000034782','0001652044','0000320193','0001018724','0001326801','0001730168','0001318605','0001067983','0000059478'} #debug
    progress_bar = st.progress(0)
    total = len(cik_list)
    status = st.empty()
    # error=0
    
    for i, cik in enumerate(cik_list): # :#companies_dict.items():
        status.write(f"Processing CIK {cik} ({i+1}/{total})")
        progress_bar.progress((i + 1) / total)
        ticker = (ss.quarterly_financials.loc[ss.quarterly_financials['cik'] == cik, 'ticker'].iloc[0])
        max_filing_date = ss.quarterly_financials.loc[ss.quarterly_financials['cik'] == cik, 'max_filing_date'].max()
        max_report_date = ss.quarterly_financials.loc[ss.quarterly_financials['cik'] == cik, 'max_report_date'].max()
        company_and_ticker = (ss.quarterly_financials.loc[ss.quarterly_financials['cik'] == cik, 'company_and_ticker'].iloc[0])
        quarterly_df_wc = ss.quarterly_financials[ss.quarterly_financials['cik'] == cik].reset_index(drop=True).copy()
        updated_df, metrics = analyze_yoy_growth(quarterly_df_wc, company_and_ticker,plot_regression_bin=0)
        updated_companies[cik] = updated_df
        
        def extract_yahoo_fields(stats):
            if not stats:
                return {k: None for k in [
                    "stock_price", "price_range_52wks",
                    "pct_chg_from_52wk_high", "pct_chg_from_52wk_low",
                    "trailing_pe", "trailing_ps", "forward_pe"
                ]}

            return {
                "stock_price": stats.get("regularMarketPrice"),
                "price_range_52wks": stats.get("fiftyTwoWeekRange"),
                "pct_chg_from_52wk_high": stats.get("fiftyTwoWeekHighChangePercent"),
                "pct_chg_from_52wk_low": stats.get("fiftyTwoWeekLowChangePercent"),
                "trailing_pe": stats.get("trailingPE"),
                "trailing_ps": stats.get("priceToSalesTrailing12Months"),
                "forward_pe": stats.get("forwardPE"),
            }
            
        stats = yahoo_finance_load(ticker)
        yf_data = extract_yahoo_fields(stats)

        stock_price = yf_data["stock_price"]
        price_range_52wks = yf_data["price_range_52wks"]
        pct_chg_from_52wk_high = yf_data["pct_chg_from_52wk_high"]
        pct_chg_from_52wk_low = yf_data["pct_chg_from_52wk_low"]
        trailing_pe = yf_data["trailing_pe"]
        trailing_ps = yf_data["trailing_ps"]
        forward_pe = yf_data["forward_pe"]
        
        def safe_round(x, n=1):
            return round(x, n) if isinstance(x, (int, float)) else None

        def safe_multiply(x, n=1):
            return x*n if isinstance(x, (int, float)) else None
                
        if metrics:
            try:
                results.append({
                'cik': cik,
                'ticker': ticker,
                'company_and_ticker': company_and_ticker,
                'stock_price' : stock_price,
                'price_range_52wks' : price_range_52wks,
                'pct_chg_from_52wk_high' : safe_multiply(safe_round(pct_chg_from_52wk_high,3),100),
                'pct_chg_from_52wk_low' : safe_multiply(safe_round(pct_chg_from_52wk_low,3),100),
                'trailing_pe' : safe_round(trailing_pe,1),
                'forward_pe' : safe_round(forward_pe,1),
                'trailing_ps' : safe_round(trailing_ps,1),
                'Consolidated_Score': (
                    -100 if len(metrics['revenue_growth']) < 15
                    else safe_round(
                        (
                            metrics['median_revenue_growth'] * metrics['revenue_consistency_score'] / 100
                            + metrics['last3_quarters_median_margin'] * metrics['income_consistency_score'] / 100
                        ) / 2
                        - (3 - metrics['last3_quarters_income_positive_count'] * 10),
                        1
                    )
                ),
                'Revenue_Consistency_Score': safe_round(metrics['revenue_consistency_score'], 1),
                'Income_Consistency_Score': safe_round(metrics['income_consistency_score'], 1),
                'Median_Revenue_Growth_PCT': safe_round(metrics['median_revenue_growth'], 2),
                'Median_Income_Growth_PCT': safe_round(metrics['median_income_growth'], 2),
                'Median_Margin_Growth_PCT': safe_round(metrics['median_margin_growth'], 2),
                'Median_Margin_PCT': safe_round(metrics['median_margin'], 2),
                'Last3Q_Revenue_Growth_PCT': safe_round(metrics['last3_quarters_revenue_growth'], 2),
                'Last3Q_Income_Growth_PCT': safe_round(metrics['last3_quarters_income_growth'], 2),
                'Last3Q_Margin_Growth_PCT': safe_round(metrics['last3_quarters_margin_growth'], 2),
                'Last3Q_Median_Margin_PCT': safe_round(metrics['last3_quarters_median_margin'], 2),
                'Last3Q_Income_Positive': int(metrics['last3_quarters_income_positive_count']),
                'Revenue_Growth_Count': len(metrics['revenue_growth']),
                'Income_Growth_Count': len(metrics['income_growth']),
                'Margin_Growth_Count': len(metrics['margin_growth']),
                'Revenue_Growth_Slope': safe_round(metrics['revenue_growth_slope'] / 1000, 2),
                'Revenue_Growth_Median': safe_round(metrics['revenue_growth_median'] / 1000, 2),
                'Revenue_Growth_PCT': safe_round(metrics['revenue_growth_slope'] * 100 / (metrics['revenue_growth_median'] if metrics['revenue_growth_median'] != 0 else 1), 2) if 'revenue_growth_slope' in metrics and 'revenue_growth_median' in metrics else 0,
                'Revenue_Growth_N': metrics['revenue_n'],
                'Revenue_Growth_N_Outliers': metrics['revenue_n_outliers'],
                'Revenue_Growth_Outlier_PCT': safe_round(metrics['revenue_n_outliers'] / metrics['revenue_n'] * 100, 2) if metrics['revenue_n'] > 0 else 0,
                'Revenue_R2': safe_round(metrics['revenue_r2'], 4),
                'Revenue_Avg_Residual_Last3': safe_round(metrics['revenue_avg_residual_last3']/1000, 2),
                'Income_Growth_Slope': safe_round(metrics['income_growth_slope'] / 1000, 2),
                'Income_Growth_Median': safe_round(metrics['income_growth_median'] / 1000, 2),
                'Income_Growth_PCT': safe_round(metrics['income_growth_slope'] * 100 / abs(metrics['income_growth_median'] if metrics['income_growth_median'] != 0 else 1), 2) if 'income_growth_slope' in metrics and 'income_growth_median' in metrics else 0,
                'Income_Growth_N': metrics['income_n'],
                'Income_Growth_N_Outliers': metrics['income_n_outliers'],
                'Income_Growth_Outlier_PCT': safe_round(metrics['income_n_outliers'] / metrics['income_n'] * 100, 2) if metrics['income_n'] > 0 else 0,
                'Income_R2': safe_round(metrics['income_r2'], 4),
                'Income_Avg_Residual_Last3': safe_round(metrics['income_avg_residual_last3']/1000, 2),
                'Margin_Growth_Slope': safe_round(metrics['margin_growth_slope'], 4),
                'Margin_Growth_Median': safe_round(metrics['margin_growth_median'], 2),
                'Margin_Growth_N': metrics['margin_n'],
                'Margin_Growth_N_Outliers': metrics['margin_n_outliers'],
                'Margin_Growth_Outlier_PCT': safe_round(metrics['margin_n_outliers'] / metrics['margin_n'] * 100, 2) if metrics['margin_n'] > 0 else 0,
                'Margin_R2': safe_round(metrics['margin_r2'], 4),
                'Margin_Avg_Residual_Last3': safe_round(metrics['margin_avg_residual_last3'], 2),
                'max_filing_date': max_filing_date,
                'max_report_date': max_report_date
            })
            except Exception as e:
                exc_type, exc_obj, tb = sys.exc_info()
                filename = tb.tb_frame.f_code.co_filename
                line_no = tb.tb_lineno
                code_line = linecache.getline(filename, line_no).strip()
                st.write(f"Failed to append data due to {e}, file: {filename}, line #{line_no}, code line: {code_line}")
    
    results_df = pd.DataFrame(results)
    # st.dataframe(results_df)  # debug
    # st.json(updated_companies)  # debug

    results_df = results_df.sort_values('Revenue_Growth_Slope', ascending=False)
    
    return results_df, updated_companies

def get_column_specs_ranking_df():
    column_specs= {
        "cik": {"pg_name": "cik", "fmt": "{}"},
        "ticker": {"pg_name": "ticker", "fmt": "{}"},
        "company_and_ticker": {"pg_name": "company_and_ticker", "fmt": "{}"},
        "Consolidated_Score": {"pg_name": "consolidated_score", "fmt": "{:.1f}"},
        "Revenue_Consistency_Score": {"pg_name": "revenue_consistency_score", "fmt": "{:.1f}"},
        "Income_Consistency_Score": {"pg_name": "income_consistency_score", "fmt": "{:.1f}"},
        "Median_Revenue_Growth_PCT": {"pg_name": "median_revenue_growth_pct", "fmt": "{:.2f}%"},
        "Median_Income_Growth_PCT": {"pg_name": "median_income_growth_pct", "fmt": "{:.2f}%"},
        "Median_Margin_Growth_PCT": {"pg_name": "median_margin_growth_pct", "fmt": "{:.2f}%"},
        "Median_Margin_PCT": {"pg_name": "median_margin_pct", "fmt": "{:.2f}%"},
        "Last3Q_Revenue_Growth_PCT": {"pg_name": "last3q_revenue_growth_pct", "fmt": "{:.2f}%"},
        "Last3Q_Income_Growth_PCT": {"pg_name": "last3q_income_growth_pct", "fmt": "{:.2f}%"},
        "Last3Q_Margin_Growth_PCT": {"pg_name": "last3q_margin_growth_pct", "fmt": "{:.2f}%"},
        "Last3Q_Median_Margin_PCT": {"pg_name": "last3q_median_margin_pct", "fmt": "{:.2f}%"},
        "Last3Q_Income_Positive": {"pg_name": "last3q_income_positive", "fmt": "{}"},
        "Revenue_Growth_Count": {"pg_name": "revenue_growth_count", "fmt": "{}"},
        "Income_Growth_Count": {"pg_name": "income_growth_count", "fmt": "{}"},
        "Margin_Growth_Count": {"pg_name": "margin_growth_count", "fmt": "{}"},
        "Revenue_Growth_Slope": {"pg_name": "revenue_growth_slope", "fmt": "{:.2f}K"},
        "Revenue_Growth_Median": {"pg_name": "revenue_growth_median", "fmt": "{:.2f}K"},
        "Revenue_Growth_PCT": {"pg_name": "revenue_growth_pct", "fmt": "{:.2f}%"},
        "Revenue_Growth_N": {"pg_name": "revenue_growth_n", "fmt": "{}"},
        "Revenue_Growth_N_Outliers": {"pg_name": "revenue_growth_n_outliers", "fmt": "{}"},
        "Revenue_Growth_Outlier_PCT": {"pg_name": "revenue_growth_outlier_pct", "fmt": "{:.2f}%"},
        "Revenue_R2": {"pg_name": "revenue_r2", "fmt": "{:.4f}"},
        "Revenue_Avg_Residual_Last3": {"pg_name": "revenue_avg_residual_last3", "fmt": "{:.2f}K"},
        "Income_Growth_Slope": {"pg_name": "income_growth_slope", "fmt": "{:.2f}K"},
        "Income_Growth_Median": {"pg_name": "income_growth_median", "fmt": "{:.2f}K"},
        "Income_Growth_PCT": {"pg_name": "income_growth_pct", "fmt": "{:.2f}%"},
        "Income_Growth_N": {"pg_name": "income_growth_n", "fmt": "{}"},
        "Income_Growth_N_Outliers": {"pg_name": "income_growth_n_outliers", "fmt": "{}"},
        "Income_Growth_Outlier_PCT": {"pg_name": "income_growth_outlier_pct", "fmt": "{:.2f}%"},
        "Income_R2": {"pg_name": "income_r2", "fmt": "{:.4f}"},
        "Income_Avg_Residual_Last3": {"pg_name": "income_avg_residual_last3", "fmt": "{:.2f}K"},
        "Margin_Growth_Slope": {"pg_name": "margin_growth_slope", "fmt": "{:.4f}"},
        "Margin_Growth_Median": {"pg_name": "margin_growth_median", "fmt": "{:.2f}"},
        "Margin_Growth_N": {"pg_name": "margin_growth_n", "fmt": "{}"},
        "Margin_Growth_N_Outliers": {"pg_name": "margin_growth_n_outliers", "fmt": "{}"},
        "Margin_Growth_Outlier_PCT": {"pg_name": "margin_growth_outlier_pct", "fmt": "{:.2f}%"},
        "Margin_R2": {"pg_name": "margin_r2", "fmt": "{:.4f}"},
        "Margin_Avg_Residual_Last3": {"pg_name": "margin_avg_residual_last3", "fmt": "{:.2f}"}
    }
    return column_specs

def get_column_specs_quarterly_df():
    column_specs = {
    "cik": {"pg_name": "cik", "fmt": "{}", "label": "CIK", "width": 80, "disabled": True},
    "ticker": {"pg_name": "ticker", "fmt": "{}", "label": "Ticker", "width": 80, "disabled": True},
    "company_and_ticker": {"pg_name": "company_and_ticker", "fmt": "{}", "label": "Company and Ticker", "width": 180, "disabled": True},
    "fiscal_timeframe": {"pg_name": "fiscal_timeframe", "fmt": "{}", "label": "Fiscal Timeframe", "width": 120, "disabled": True},

    "Revenue/Sales": {"pg_name": "revenue_sales", "fmt": "${:,.0f}", "label": "Revenue / Sales", "width": 120, "disabled": True},
    "Cost of Sales": {"pg_name": "cost_of_sales", "fmt": "${:,.0f}", "label": "Cost of Sales", "width": 120, "disabled": True},
    "Gross Profit": {"pg_name": "gross_profit", "fmt": "${:,.0f}", "label": "Gross Profit", "width": 120, "disabled": True},
    "Total Expenses": {"pg_name": "total_expenses", "fmt": "${:,.0f}", "label": "Total Expenses", "width": 120, "disabled": True},
    "Operating Expenses": {"pg_name": "operating_expenses", "fmt": "${:,.0f}", "label": "Operating Expenses", "width": 140, "disabled": True},
    "SG&A Exp": {"pg_name": "sga_exp", "fmt": "${:,.0f}", "label": "SG&A Expense", "width": 120, "disabled": True},
    "R&D Exp": {"pg_name": "rnd_exp", "fmt": "${:,.0f}", "label": "R&D Expense", "width": 120, "disabled": True},
    "Operating Income": {"pg_name": "operating_income", "fmt": "${:,.0f}", "label": "Operating Income", "width": 140, "disabled": True},
    "Pretax Income": {"pg_name": "pretax_income", "fmt": "${:,.0f}", "label": "Pretax Income", "width": 120, "disabled": True},
    "Net Income": {"pg_name": "net_income", "fmt": "${:,.0f}", "label": "Net Income", "width": 120, "disabled": True},
    "Interest Expense": {"pg_name": "interest_expense", "fmt": "${:,.0f}", "label": "Interest Expense", "width": 140, "disabled": True},
    "EBITDA": {"pg_name": "ebitda", "fmt": "${:,.0f}", "label": "EBITDA", "width": 120, "disabled": True},

    "EPS Basic": {"pg_name": "eps_basic", "fmt": "{:.2f}", "label": "EPS Basic", "width": 100, "disabled": True},
    "EPS Diluted": {"pg_name": "eps_diluted", "fmt": "{:.2f}", "label": "EPS Diluted", "width": 100, "disabled": True},

    "Total Assets": {"pg_name": "total_assets", "fmt": "${:,.0f}", "label": "Total Assets", "width": 140, "disabled": True},
    "Curr Assets": {"pg_name": "current_assets", "fmt": "${:,.0f}", "label": "Current Assets", "width": 140, "disabled": True},
    "Deposits": {"pg_name": "deposits", "fmt": "${:,.0f}", "label": "Deposits", "width": 120, "disabled": True},
    "Liabilities": {"pg_name": "liabilities", "fmt": "${:,.0f}", "label": "Liabilities", "width": 120, "disabled": True},
    "Curr Liabilities": {"pg_name": "current_liabilities", "fmt": "${:,.0f}", "label": "Current Liabilities", "width": 150, "disabled": True},
    "Long Term Debt": {"pg_name": "long_term_debt", "fmt": "${:,.0f}", "label": "Long Term Debt", "width": 140, "disabled": True},
    "Stockholder Equity": {"pg_name": "stockholder_equity", "fmt": "${:,.0f}", "label": "Stockholder Equity", "width": 150, "disabled": True},
    "Accumulated Depreciation": {"pg_name": "accumulated_depreciation", "fmt": "${:,.0f}", "label": "Accumulated Depreciation", "width": 180, "disabled": True},
    "Common Shares Outstanding": {"pg_name": "common_shares_outstanding", "fmt": "{:,.0f}", "label": "Common Shares Outstanding", "width": 180, "disabled": True},

    "Income Tax": {"pg_name": "income_tax", "fmt": "${:,.0f}", "label": "Income Tax", "width": 120, "disabled": True},
    "Depreciation": {"pg_name": "depreciation", "fmt": "${:,.0f}", "label": "Depreciation", "width": 120, "disabled": True},
    "Amortization": {"pg_name": "amortization", "fmt": "${:,.0f}", "label": "Amortization", "width": 120, "disabled": True},
    "Depreciation & Amortization": {"pg_name": "depreciation_and_amortization", "fmt": "${:,.0f}", "label": "Depreciation & Amortization", "width": 180, "disabled": True}
}
    return column_specs

def write_sec_data_into_db():
    ss.company_lookup_df=get_company_tickers_df()
    ss.company_lookup_df = (
        ss.company_lookup_df
        .sort_values("company_and_ticker")
        .drop_duplicates(subset="cik", keep="first")
    )
    # st.write(f"company_lookup_df = ",ss.company_lookup_df)  #debug
    
    #debug this only selects a few CIKs for testing
    # cik_list={'0000066740','0001368514'}#,'0001070494','0000318306','0001018724','0000789019','0000002488','0001045810','0000034782','0001652044','0000320193','0001018724','0001326801','0001730168','0001318605','0001067983','0000059478'} #debug
    
    cik_list = ss.company_lookup_df['cik'].tolist() #[:2] - limit to first 2 CIKs for testing
    
    companies_data = {}  # Store company data for analysis
    
    progress_bar = st.progress(0)
    status = st.empty()
    total = len(cik_list)
    error=0
    
    for i, cik in enumerate(cik_list):
        # Load SEC EDGAR financials for the selected CIK (guarded)
        try:
            # st.write(f"Loading SEC data for CIK: {cik}...")  # debug
            company_and_ticker = ss.company_lookup_df[ss.company_lookup_df['cik'] == cik]['company_and_ticker'].iloc[0] if not ss.company_lookup_df[ss.company_lookup_df['cik'] == cik].empty else cik
            status.write(f"Processing CIK {cik} ({i+1}/{total}), {error/i*100 if i > 0 else 0:.1f}% errors so far")
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
                ss.company_lookup_df[['cik', 'ticker','company_and_ticker']],
                on='cik',
                how='left')
            col = ss.quarterly_financials.pop('company_and_ticker')
            ss.quarterly_financials.insert(0, 'company_and_ticker', col)
            filings_10q_10k_df = ss.filings_df[ss.filings_df['form'].isin(['10-Q', '10-K'])]
            # st.dataframe(filings_10q_10k_df)  # debug
            max_filing_date= filings_10q_10k_df['filingDate'].max() if not filings_10q_10k_df.empty else None
            max_report_date = filings_10q_10k_df['reportDate'].max() if not filings_10q_10k_df.empty else None
            ss.quarterly_financials['max_filing_date'] = max_filing_date
            ss.quarterly_financials['max_report_date'] = max_report_date

            # If quarterly data isn't present, surface a message and stop further processing
            if not isinstance(ss.quarterly_financials, pd.DataFrame) or ss.quarterly_financials.empty:
                print(f"No quarterly financials returned from SEC for {cik} company.")
                # if isinstance(ss.filings_df, pd.DataFrame) and not ss.filings_df.empty:
                #     st.write("Filings (sample):")
                #     st.dataframe(ss.filings_df.head())
                continue
            
            # Store company data for growth analysis
            # st.dataframe(ss.quarterly_financials)  # debug
            
            column_specs=get_column_specs_quarterly_df()        
            df_pg = ss.quarterly_financials.rename(columns={
                pretty: spec["pg_name"]
                for pretty, spec in column_specs.items()
            })
            
            postgres_update(df_pg, 'stock_quarterly_financials_sec', ['cik','fiscal_timeframe'])  # Save quarterly data with metrics to PostgreSQL
        except Exception as e:
            error=error+1
            st.write(f"Failed to load and write SEC data for {company_and_ticker}: {e}")
            ss.filings_df = pd.DataFrame()
            ss.quarterly_financials = pd.DataFrame()
            ss.annual_financials = pd.DataFrame()
            continue # Skip to next CIK    
    return

def load_sec_data_from_db():
    status = st.empty()
    df = postgres_read('stock_quarterly_financials_sec')
    # st.dataframe(df)  # debug
    column_specs=get_column_specs_quarterly_df()
    pg_to_pretty = {v["pg_name"]: k for k, v in column_specs.items()}
    df_pretty=df.rename(columns={pg_col: pg_to_pretty.get(pg_col, pg_col) for pg_col in df.columns})
    return df_pretty #, companies_data

def load_stock_growth_analysis_data_from_db():
    status = st.empty()
    df = postgres_read('stock_growth_analysis_results')
    # st.dataframe(df)  # debug
    # column_specs=get_column_specs_ranking_df()
    # pg_to_pretty = {v["pg_name"]: k for k, v in column_specs.items()}
    # df_pretty=df.rename(columns={pg_col: pg_to_pretty.get(pg_col, pg_col) for pg_col in df.columns})
    return df #, companies_data

def read_or_create_editable_table():
    # Try to read the editable table from the database
    try:
        editable_df = postgres_read('editable_stock_data')
        if editable_df.empty:
            raise ValueError("Editable table is empty")
    except Exception as e:
        st.write(f"Editable table not found or empty, creating new one. Error: {e}")
        # Create a new DataFrame with the same CIKs and default values for editable fields
        editable_df = pd.DataFrame({
            'cik': ss.rankings_df['cik'],
            'ticker': ss.rankings_df['ticker'],
            'company_and_ticker': ss.rankings_df['company_and_ticker'],
            'category': [''] * len(ss.rankings_df),
            'notes': [''] * len(ss.rankings_df)
        })
        # st.write(editable_df)  # debug
        # Save the new DataFrame to the database
        postgres_update(editable_df, 'editable_stock_data', ['cik'])
    
    return editable_df

def quantile_color(s):
    bottom_q = s.quantile(0.2)
    top_q = s.quantile(0.8)

    def color(v):
        if v <= bottom_q:
            return "background-color: red"   # red
        elif v >= top_q:
            return "background-color: green"   # green
        # else:
        #     return "background-color: yellow"   # yellow

    return s.apply(color)

def display_analysis_summary(stock_growth_analysis_df):
        return

# set page config and title
st.set_page_config( page_title="Stock Screener", layout="wide" )
st.markdown('<h2 style="color:#3894f0;">Stock Screener for Publically Traded Stocks</h2>', unsafe_allow_html=True)
st.write('Created by Rafael Avila leveraging Snowflake & Streamlit, using SEC Filings data provided by SEC Edgar platform. Analysis Summary and AI chat powered by mistral-large2 AI model')

def main():

    load_btn =analysis_btn = value_btn = qtr_data_btn = return_menu_btn = False

    if ss.hide_menu==False:
        st.write("Choose an action:")
        load_btn = st.button("Load Financial Data from SEC")
        analysis_btn = st.button("Perform Fundamental Analysis")
        value_btn = st.button("Perform Value Analysis")
        qtr_data_btn = st.button("Show Quarterly Financial Data")
    else:
        return_menu_btn=st.button('Return to Menu')

    if return_menu_btn:
        ss.hide_menu=False
        ss.value_btn=False
        ss.selected_company=None
        st.rerun()

    if load_btn:
        ss.value_btn=0
        write_sec_data_into_db()

    if analysis_btn:
        ss.value_btn=0
        # column_specs=get_column_specs()
        ss.quarterly_financials = load_sec_data_from_db()
        # st.write(companies_data['0000353184'])  # debug
        
        # Analyze and rank companies by growth consistency
        if ss.quarterly_financials is not None and not ss.quarterly_financials.empty:
            st.markdown("---")
            st.subheader("📊 Growth Consistency Analysis")
            
            if ss.ranking_df.empty or ss.updated_companies=={}:
                ss.ranking_df, ss.updated_companies = rank_companies_by_growth()
                postgres_update(ss.ranking_df, 'stock_growth_analysis_results', ['cik'])  # Save results to PostgreSQL
                            
                # Highlight top performer
                top_company = ss.ranking_df.iloc[0]
                # st.dataframe(top_company) # debug
                st.success(f"🏆 Top Performer: **{top_company['company_and_ticker']}** (Score: {top_company['Revenue_Consistency_Score']:.1f})")
                display_analysis_summary(stock_growth_analysis_df)
                st.button("Commit Changes to Database", on_click=lambda: postgres_update(ss.editable_stock_growth_analysis_df[['cik','category','notes']], 'editable_stock_data', ['cik']))

    if value_btn:
        ss.value_btn=True
        ss.hide_menu=True
        st.rerun()
        
    if ss.value_btn==True:
        stock_growth_analysis_df = load_stock_growth_analysis_data_from_db()
        styled=display_analysis_summary(stock_growth_analysis_df)
        editable_columns = ['category', 'notes']
        stock_growth_analysis_df=stock_growth_analysis_df[stock_growth_analysis_df['revenue_growth_slope'] > 0] # Filter to only show companies with positive revenue growth slope

        columns = [ 'cik', 'company_and_ticker'] + editable_columns + ['stock_price', 'price_range_52wks', 'pct_chg_from_52wk_high', 'pct_chg_from_52wk_low', 'trailing_pe', 'forward_pe', 'trailing_ps', 
            'Revenue_Growth_Slope','Revenue_R2','Revenue_Growth_PCT','Revenue_Avg_Residual_Last3','Revenue_Growth_N','Revenue_Growth_Outlier_PCT','Revenue_Growth_Median',
            'Income_Growth_Slope','Income_R2','Income_Growth_PCT','Income_Avg_Residual_Last3','Income_Growth_N','Income_Growth_Outlier_PCT',
            'Margin_Growth_Slope','Margin_R2','Margin_Avg_Residual_Last3','Margin_Growth_N','Margin_Growth_N_Outliers',
            'Last3Q_Revenue_Growth_PCT', 'Last3Q_Income_Growth_PCT', 'Last3Q_Margin_Growth_PCT', 'Last3Q_Median_Margin_PCT', 'Last3Q_Income_Positive',
            ]

        st.write("**Stock Growth Analysis Data:**")

        if ss.editable_stock_growth_analysis_df.empty:
            column_specs=get_column_specs_ranking_df()
            rename_map = {
                spec["pg_name"]: pretty
                for pretty, spec in column_specs.items()
            }
            ss.rankings_df = stock_growth_analysis_df.rename(columns=rename_map)

            editable_stock_data=read_or_create_editable_table()
            ss.editable_stock_growth_analysis_df = ss.rankings_df.merge(editable_stock_data[['cik'] + editable_columns], on='cik', how='left')
        
            ss.editable_stock_growth_analysis_df = ss.editable_stock_growth_analysis_df.reindex(columns=columns)
            # st.dataframe(editable_stock_growth_analysis_df) # debug     
            ss.editable_stock_growth_analysis_df = ss.editable_stock_growth_analysis_df.sort_values(by='Revenue_Growth_Slope', ascending=False)
                
        @st.cache_data
        def compute_quantiles(df, cols):
            breakpoints = [0.2, 0.4, 0.6, 0.8]
            return {
                col: tuple(df[col].quantile(q) for q in breakpoints)
                for col in cols
                if col in df.columns
            }

        @st.cache_data
        def compute_low_quantiles(df, cols):
            return {
                col: df[col].quantile(0.4)
                for col in cols
                if col in df.columns
            }
        
        def build_styler(df):
            styled = df.style
            cols_for_color_inc=[
                    'Revenue_Growth_Slope', 'Income_Growth_Slope', 'Margin_Growth_Slope'
                    ,'Revenue_R2','Income_R2','Margin_R2'
                    ,'Revenue_Growth_PCT','Income_Growth_PCT'
                    ,'Revenue_Avg_Residual_Last3','Income_Avg_Residual_Last3','Margin_Avg_Residual_Last3'
                    ,'Last3Q_Revenue_Growth_PCT', 'Last3Q_Income_Growth_PCT', 'Last3Q_Margin_Growth_PCT', 'Last3Q_Median_Margin_PCT'
                ]
            cols_red_bottom_quintile = ['Revenue_Growth_N','Income_Growth_N','Margin_Growth_N']
            cols_for_color_dec = ['trailing_pe','trailing_ps','forward_pe','pct_chg_from_52wk_high']

            q_inc = compute_quantiles(df, cols_for_color_inc)
            q_dec = compute_quantiles(df, cols_for_color_dec)
            q_bottom = compute_low_quantiles(df, cols_red_bottom_quintile)
            
            for col, (q20, q40, q60, q80) in q_inc.items():
                def color_func(s, q20=q20, q40=q40, q60=q60, q80=q80):
                    return [
                        'background-color: green' if v >= q80 else
                        'background-color: red' if v <= q40 else
                        ''
                        for v in s
                    ]
                styled = styled.apply(color_func, subset=[col])

            for col, (q20, q40, q60, q80) in q_dec.items():
                def color_func(s, q20=q20, q40=q40, q60=q60, q80=q80):
                    return [
                        'background-color: red' if v >= q80 else
                        'background-color: green' if v <= q40 else
                        'background-color: mediumseagreen' if v<=q60
                        else ''
                        for v in s
                    ]
                styled = styled.apply(color_func, subset=[col])

            # apply bottom-quintile red
            for col, q40 in q_bottom.items():
                def red_func(s, q40=q40):
                    return ['background-color: red' if v <= q40 else '' for v in s]
                styled = styled.apply(red_func, subset=[col])
            return styled
    
        disabled_cols = list(set(columns) - set(editable_columns) )
        
        if 'chart' not in ss.editable_stock_growth_analysis_df.columns: 
            ss.editable_stock_growth_analysis_df.insert(0, "chart", False)
        
        filtered_df = ss.editable_stock_growth_analysis_df[
            (ss.editable_stock_growth_analysis_df['Revenue_Growth_Median'] <100000000000000000000000)
            # & (ss.editable_stock_growth_analysis_df['Revenue_R2'] >= .60)
            # & (ss.editable_stock_growth_analysis_df['Revenue_Growth_N'] >= 25)
            # & (ss.editable_stock_growth_analysis_df['Revenue_Growth_PCT'] >= 5)
            # & (ss.editable_stock_growth_analysis_df['Income_Growth_PCT'] >= 5)
            # # & (ss.editable_stock_growth_analysis_df['trailing_ps'] <= 8)
            # & (ss.editable_stock_growth_analysis_df['category']=="")
        ]
            
        filtered_df=filtered_df.reset_index(drop=True)
        ss.df=filtered_df.copy()        
        styled = build_styler(filtered_df)
        # st.write("styled df:",styled) #debug
        
        def handle():
            if "my_editor" not in ss:
                return
            
            editor_state = ss['my_editor']
            
            if "edited_rows" not in editor_state:
                return
            
            edited = editor_state["edited_rows"]
            # st.write(edited) #debug
            # Always use your own stored DataFrame
            df = ss.get("df")
            # arrow_table = pa.Table.from_pandas('df')

            if df is None:
                return
            
            if "prev_edits" not in ss:
                ss.prev_edits={}
                
            prev = ss.prev_edits
            
            new_changes = {
                idx: changes
                for idx, changes in edited.items()
                if idx not in prev or prev[idx] != changes
            }

            # st.write(ss.prev_edits) #debug
            # st.write(edited) #debug
            # st.write(new_changes) #debug
            
            ss.prev_edits = edited
            # st.write(ss.prev_edits) #debug

            # Process checkbox clicks even if "data" is missing
            for row_index, changes in new_changes.items():
                if changes.get("chart") is True:
                    # st.write(f"processing row {row_index}") #debug
                    cik = df.loc[row_index, "cik"]
                    ss.selected_company = cik
                    # st.write(ss.selected_company) #debug

                    # df.at[row_index, "chart"] = False   # Reset checkbox in your underlying df
                    # changes["chart"] = False
                    # edited.pop(row_index,None)
                    # ss.df=df
                    # st.write("editor state after:",editor_state)
                    # st.write(f"deleted row {row_index}") #debug
            
            # st.write(edited)
            
        ss.categories_list = ['Buy Now','Owned','Strong Rev & Income Growth', 'Strong Rev, Neg Income','Inconsistent Growth', 'Declining Growth', 'Other']
        st.data_editor(styled, key="my_editor", on_change=handle, width='stretch', disabled=disabled_cols
                    #,hide_index=True
                    ,column_config= {
                    "company_and_ticker": st.column_config.TextColumn(label='Company and Ticker',pinned=True),
                    'chart':st.column_config.CheckboxColumn(label='Charts', width="small", pinned=True),
                    "category": st.column_config.SelectboxColumn(label="Category", pinned=True, options=ss.categories_list, width="small"),
                    'stock_price': st.column_config.NumberColumn(label="Stock Price", format='dollar'),
                    'price_range_52wks': st.column_config.TextColumn(),
                    'pct_chg_from_52wk_high':st.column_config.NumberColumn(label="% from 52wk High²", format='%.1f', width="small"),
                    'pct_chg_from_52wk_low':st.column_config.NumberColumn(label="% from 52wk Low", format='%.1f', width="small"),
                    'trailing_pe':st.column_config.NumberColumn(label="P/E Trailing", format='%.1f', width="small"),
                    'forward_pe':st.column_config.NumberColumn(label="P/E Fwd", format='%.1f', width="small"),
                    'trailing_ps':st.column_config.NumberColumn(label="P/S Trailing", format='%.1f', width="small"),
                    "Revenue_Growth_Slope": st.column_config.NumberColumn(label="Revenue Growth Slope", format='dollar', step='int'),
                    "Income_Growth_Slope": st.column_config.NumberColumn(label="Income Growth Slope", format='dollar', step='int'),
                    "Margin_Growth_Slope": st.column_config.NumberColumn(label="Margin Growth Slope", format='dollar'),
                    "Revenue_R2": st.column_config.NumberColumn(label="Revenue R²", format='%.2f', width="small"),
                    "Income_R2": st.column_config.NumberColumn(label="Income R²", format='%.2f', width="small"),
                    "Margin_R2": st.column_config.NumberColumn(label="Margin R²", format='%.2f', width="small"),
                    "Revenue_Avg_Residual_Last3": st.column_config.NumberColumn(label="Last 3Q Avg Residual - Revenue", format='dollar',step='int', width="small"),
                    "Income_Avg_Residual_Last3": st.column_config.NumberColumn(label="Last 3Q Avg Residual - Income", format='dollar',step='int', width="small"),
                    "Margin_Avg_Residual_Last3": st.column_config.NumberColumn(label="Last 3Q Avg Residual - Margin", format='dollar',step='int', width="small"),
                    "Revenue_Growth_PCT": st.column_config.NumberColumn(label="Growth % - Revenue", format='%.1f', width="small"),
                    "Income_Growth_PCT": st.column_config.NumberColumn(label="Growth % - Income", format='%.1f', width="small"),
                    "Revenue_Growth_N": st.column_config.NumberColumn(label="N Count - Revenue", step='int', width="small"),
                    "Income_Growth_N": st.column_config.NumberColumn(label="N Count - Income", step='int', width="small"),
                    "Margin_Growth_N": st.column_config.NumberColumn(label="N Count - Margin", step='int', width="small"),
                    "Revenue_Growth_Outlier_PCT": st.column_config.NumberColumn(label="Outlier %", format='%.1f', width="small"),
                    "Income_Growth_Outlier_PCT": st.column_config.NumberColumn(label="Outlier %", format='%.1f', width="small"),
                    "Margin_Growth_Outlier_PCT": st.column_config.NumberColumn(label="Outlier %", format='%.1f', width="small"),
                    'Revenue_Growth_Median':st.column_config.NumberColumn(label="Revenue Median", format='dollar', step='int'),
                    "Last3Q_Revenue_Growth_PCT": st.column_config.NumberColumn(label="Last 3Q Revenue Growth %", format='%.1f', width="small"),
                    "Last3Q_Income_Growth_PCT": st.column_config.NumberColumn(label="Last 3Q Income Growth %", format='%.1f', width="small"),
                    "Last3Q_Margin_Growth_PCT": st.column_config.NumberColumn(label="Last 3Q Margin Growth %", format='%.1f', width="small"),
                    "Last3Q_Median_Margin_PCT": st.column_config.NumberColumn(label="Last 3Q Median Margin %", format='%.1f', width="small"),
                    },
                    )
        # st.write(ss.selected_company) #debug
        # st.write("REGISTERED CALLBACK:", handle)
                
        if ss.selected_company is not None:
            try: 
                cik = ss.selected_company
                quarterly_df=postgres_read('stock_quarterly_financials_sec',f"cik='{cik}'")
                company_and_ticker=quarterly_df['company_and_ticker'].iloc[0]
                column_specs = get_column_specs_quarterly_df()
                pg_to_pretty = {spec["pg_name"]: pretty for pretty, spec in column_specs.items()}
                quarterly_df = quarterly_df.rename(columns=pg_to_pretty)
                # st.dataframe(quarterly_df) # debug
                analyze_yoy_growth(quarterly_df,company_and_ticker,plot_regression_bin=1)
            except Exception as e:
                st.warning(f"Regression failed: {e}")

    if qtr_data_btn:
        # st.write(updated_companies)  # debug
        quarterly_updated_df = pd.concat(
            ss.updated_companies.values(), 
            ignore_index=True
        )
                    
        cols = [
            'company_and_ticker',
            'cik',
            'fiscal_timeframe',
            'Revenue/Sales',
            'Revenue_YoY_Growth_PCT',
            'Net Income',
            'Income_YoY_Growth_PCT',
            'Margin_PCT',
            'Margin_YoY_Growth_PCT'
        ]
            
        st.write("**All Companies - Quarterly Data with Metrics:**")
        st.dataframe(quarterly_updated_df[cols],width='stretch')
        # postgres_update2(quarterly_updated_df[cols], 'stock_quarterly_financials', ['cik','fiscal_timeframe'])  # Save quarterly data with metrics to PostgreSQL
                
main()
import streamlit as st
import streamlit.elements.widgets.data_editor as de
from streamlit_extras.stylable_container import stylable_container
import pyarrow as pa
import pandas as pd
import altair as alt
from icecream import ic
from altair.expr import *
from SEC_Edgar_Loader import sec_edgar_financial_load,get_company_tickers_df, load_daily_SEC_submission_index
import numpy as np
import sys, linecache, traceback
from stock_data_loader_utilities import postgres_update, postgres_read,yahoo_finance_load,yahoo_finance_df_format
from datetime import date
import uuid
import math
import logging
from datetime import datetime,timedelta

logging.basicConfig(
    filename="sec_loader_errors.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

ss = st.session_state

if "quarterly_financials" not in ss:
    ss.quarterly_financials = pd.DataFrame()
    ss.results_df = pd.DataFrame()
    ss.editable_stock_growth_analysis_df = pd.DataFrame()
    ss.styled_editable_stock_growth_analysis_df = pd.DataFrame()
    ss.company_lookup_df = pd.DataFrame()
    ss.transactions_table = pd.DataFrame()
    ss.rankings_df = pd.DataFrame()
    ss.filtered_df = pd.DataFrame()
    ss.metrics={}
    ss.view_stock_analysis_form=False
    ss.qtr_data_form=False
    ss.process_yahoo_and_statistics=False
    ss.load_sec_incremental_filings=False
    ss.load_sec_full_filings=False
    ss.selected_company=None
    ss.hide_menu=False
    ss.show_transaction_form=False
    ss.investment_returns_form=False
    ss.rerun_the_application=False
    ss.filter_category = ['Buy Now','Owned','Strong Rev & Income Growth', 'Strong Rev, Neg Income']
    ss.filter_min_revenue_growth=0
    ss.filter_max_revenue_median=0
    ss.filter_min_revenue_n_count=10
    ss.filter_max_trailing_pe = 0
    ss.filter_max_trailing_ps = 0
    ss.filter_min_revenue_r2 = 0.0
    ss.filter_min_revenue_growth = 0
    ss.filter_min_income_growth = 0
    ss.filter_min_last3_income_positive = 0
    ss.filter_max_rev_outlier_pct = 0
    ss.filter_min_last_filing_date = ss.temp_filter_min_last_filing_date= (datetime.today() + timedelta(days=1)).date()
    ss.filter_industry = None
    ss.filter_sector = None
        
def safe_round(x, n=1):
    return round(x, n) if isinstance(x, (int, float)) else None

def safe_multiply(x, n=1):
    return x*n if isinstance(x, (int, float)) else None

def clean_number(x):
    # Reject None
    if x is None:
        return None

    # Reject non-numeric types
    if not isinstance(x, (int, float)):
        return None

    # Reject NaN or Infinity
    if math.isnan(x) or math.isinf(x):
        return None

    return x

def sanitize(values):
    # Convert everything to float, invalid values become np.nan
    arr = pd.to_numeric(values, errors='coerce').astype(float)

    # If all values are NaN, return None
    if np.isnan(arr).all():
        return None

    return arr

def plot_regression_line(name, var_name, X, y, y_pred_plot, slope, r2, end_date, median):

    # Build DataFrame for Altair
    plot_df = pd.DataFrame({
        "Quarter Index": X,
        "x_label": pd.to_datetime(end_date),
        var_name: y,
        "Fitted": y_pred_plot,
    })
    
    plot_df = plot_df.sort_values("x_label")  # ensure chronological order
    plot_df["Growth_12m"] =  (plot_df[var_name] / plot_df[var_name].shift(4) - 1)
    plot_df["MA4"] = plot_df[var_name].rolling(window=4).mean()
    
    # st.write(plot_df) #debug
    # base = alt.Chart(plot_df).encode(x=alt.X("x_label:T",title=None))
    base = alt.Chart(plot_df).encode(
        x=alt.X("x_label:T", title=None,axis=alt.Axis(grid=False, tickCount="year")) #, axis=alt.Axis(grid=False)
        # y=alt.Y(f"{var_name}:Q") #, axis=alt.Axis(grid=False)
    )
    
    line = base.mark_line(color="#4C78A8").encode(
            # x="x_label:T",
            y=alt.Y(f"{var_name}:Q") #,axis=alt.Axis(grid=False)
        )

    nearest = alt.selection_point(
        fields=["x_label"],
        nearest=True,
        on="pointerover", # pointerdown, pointermove, mouseover, click
        empty='none' #False
    )

    selectors = base.mark_point().add_params(nearest).encode(
            # x="x_label:T",
            opacity=alt.value(0),
            tooltip=[
                alt.Tooltip("x_label:T", title="Date"),
                alt.Tooltip(f"{var_name}:Q", title=var_name.replace("_", " "), format=",.0f"),
                alt.Tooltip("Fitted:Q", title="Trend", format=",.0f"),
                alt.Tooltip("Growth_12m:Q", title="Growth vs 12m Ago", format=",.1%")
            ]
        )

    points = (base.mark_point(size=20).encode(
            x=alt.X('x_label:T',scale=alt.Scale(padding=10),axis=alt.Axis(grid=False)),
            y=f"{var_name}:Q",
            opacity=alt.condition(nearest, alt.value(1), alt.value(0.2))
            # tooltip=[
            #    alt.Tooltip("x_label:T", title="End Date"),
            #    alt.Tooltip(f"{var_name}:Q", title=var_name.replace("_", " "), format=",.0f"),
            #    alt.Tooltip("Fitted:Q", title="Trend", format=",.0f"),
            #    alt.Tooltip("Growth_12m:Q", title="Growth vs 12m Ago", format=",.1%")
            # ]
        )
        .transform_filter(nearest)
    )
    
    # points = points.add_params(nearest)
    rule = base.mark_rule(color="gray").encode(
            x=alt.X("x_label:T"),
            opacity=alt.condition(nearest, alt.value(0.3), alt.value(0))
        )
    
    ma_line = base.mark_line(color="#FFA500", strokeDash=[4, 4]).encode(
        x=alt.X("x_label:T"),
        y="MA4:Q"
    )
    
    # Add regression line
    reg_line = base.mark_line(color="red").encode(
            x=alt.X("x_label:T"),
            y="Fitted:Q"
        )
    
    chart = (
        line + points + rule + ma_line + selectors
    ).properties(
        width=800,
        height=400,
        title=alt.TitleParams(
            f"{var_name.replace('_', ' ')} Regression for {name}",
            anchor="middle"
        ),
        padding={"bottom": 0}
    )
    
    # st.altair_chart(chart + reg_line, width='content')              
    return chart + reg_line

def perform_regression(name, var_name, values, end_date, plot_regression_bin, remove_outliers):
    
    # 1. Sanitize input
    # st.write(values) #debug

    y = sanitize(values)
    if y is None:
        return None  # nothing to regress on
    
    # st.write(f"values: {len(values)}, y: {len(y)}")
    end_date = np.array(end_date)
    
    # st.write(len(y),len(end_date)) #debug
    
    assert len(y) == len(end_date), f"Length mismatch: y={len(y)}, end_date={len(end_date)}"

    # y already length-preserving
    # st.write(f"remove_outliers = {remove_outliers}, len(y) = {len(y)}") #debug
    if remove_outliers==True:
        median = np.median(y[~np.isnan(y)])  # ignore NaNs for median
        mad = np.median(np.abs(y[~np.isnan(y)] - median))

        if mad == 0:
            mask = ~np.isnan(y)
        else:
            z = 0.6745 * (y - median) / mad
            mask = (np.abs(z) < 6.0) & ~np.isnan(y)

        y_clean = y[mask]
        # st.write(y_clean) #debug

        end_date_clean = np.array(end_date)[mask]
    else:
        mask = ~np.isnan(y)
        y_clean = y[mask]
        # st.write(y_clean) #debug
        end_date_clean = np.array(end_date)[mask]

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
    # st.write(y) #debug

    num_nulls = sum(
        (v is None) or (isinstance(v, float) and math.isnan(v))
        for v in y
    )
    n = len(y)
    n_outliers = n - len(y_clean)-num_nulls
    
    # debug_rows = []
    # for i, v in enumerate(y):
    #     debug_rows.append({
    #         "index": i,
    #         "end_date": end_date[i],
    #         "value": float(v),
    #         "z_score": float(z[i]),
    #         "kept": bool(mask[i])
    #     })

    # debug_df = pd.DataFrame(debug_rows)
    # st.write(f"Debug dataframe for {name}, {var_name} variable")
    # st.dataframe(debug_df)    

    # Need at least 6 points after filtering
    if len(y_clean) >= 6:
        coeffs = np.polyfit(X_clean, y_clean, 1)
        slope, intercept = coeffs

        y_pred = np.polyval(coeffs, X_clean)
        y_pred_plot = slope * X_clean + intercept

        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        # st.write(f"len(y_clean) = {len(y_clean)}") #debug
        if plot_regression_bin==1:
            chart=plot_regression_line(name, var_name, X_clean, y_clean, y_pred_plot, slope, r2, end_date_clean, median) #debug
        else:
            chart=None

        residuals = y_clean - y_pred
        last_n = min(3, len(residuals)) # if less than 3 points, take all
        avg_residual_last3 = residuals[-last_n:].mean()
    return slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart

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
        'income_values_list': [],
        'margin_growth': {},
        'margin_growth_list': [],
        'margin_values': [],
        'revenue_consistency_score': 0,
        'income_consistency_score': 0,
        'last_3q_revenue_growth': 0,
        'last_3q_income_growth': 0,
        'last_3q_margin_growth': 0,
        'last_3q_income_positive_count': 0,
        'last_3q_income_growth_positive_count': 0,
        'last_6q_revenue_median': 0,
        'last_6q_income_median': 0,
        'median_revenue_growth': 0,
        'median_income_growth': 0,
        'median_margin_growth': 0,
        'last_3q_median_margin': 0,
        'revenue_median': 0,
        'revenue_growth_slope': 0,
        'revenue_growth_median': 0,
        'revenue_growth_pct': 0,
        'revenue_n': 0,
        'revenue_n_outliers': 0,
        'revenue_outlier_pct': 0,
        'revenue_r2': 0,
        'revenue_avg_residual_last3': 0,
        'income_median': 0,
        'income_growth_slope': 0,
        'income_growth_median': 0,
        'income_growth_pct': 0,
        'income_n': 0,
        'income_n_outliers': 0,
        'income_outlier_pct': 0,
        'income_r2': 0,
        'income_avg_residual_last3': 0,
        'margin_median': 0,
        'margin_growth_slope': 0,
        'margin_growth_median': 0,
        'margin_slope_pct': 0,
        'margin_n': 0,
        'margin_n_outliers': 0,
        'margin_outlier_pct': 0,
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
        revenue_values = pd.to_numeric(quarterly_df_wc['Revenue/Sales'], errors='coerce')
        income_values = pd.to_numeric(quarterly_df_wc['Net Income'], errors='coerce')
        metrics['last_6q_revenue_median'] = revenue_values.tail(6).median()
        metrics['last_6q_income_median'] = income_values.tail(6).median()
        metrics['revenue_median'] = revenue_values.median()
        metrics['income_median'] = income_values.median()
        
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
                        metrics['income_values_list'].append(current_value)
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
            metrics['last_3q_revenue_growth'] = np.median(revenue_growths[-3:])
        elif len(revenue_growths) > 0:
            metrics['last_3q_revenue_growth'] = np.median(revenue_growths)
        
    if plot_regression_bin==1:
        col1, col2 = st.columns([1,2])
        with col1:
            ticker=quarterly_df['ticker'].iloc[0]
            url = f"https://finance.yahoo.com/quote/{ticker}/"
            st.write("Click this for the Yahoo Finance Page (%s)" % url)
        with col2:
            remove_outliers = st.toggle("Remove outliers", value=False)
    else:
        remove_outliers=True

        # Linear regression for revenue growth trend
        # st.write(revenue_values)  # debug
    if len(revenue_values) >= 6:
        try:
            # st.write(len(revenue_values),len(end_date)) #debug
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_revenue = perform_regression(name, "Revenue", revenue_values, end_date, plot_regression_bin,remove_outliers)
            metrics['revenue_r2'] = r2
            metrics['revenue_growth_slope'] = safe_round(slope, 1)
            metrics['revenue_growth_median'] = median
            metrics['revenue_growth_pct'] = safe_round(metrics['revenue_growth_slope'] * 100 / (metrics['last_6q_revenue_median'] if metrics['last_6q_revenue_median'] != 0 else 1), 2) if 'revenue_growth_slope' in metrics and 'last_6q_revenue_median' in metrics else 0
            metrics['revenue_n'] = n
            metrics['revenue_n_outliers'] = n_outliers
            metrics['revenue_outlier_pct'] = safe_round(metrics['revenue_n_outliers'] / metrics['revenue_n'] * 100, 2) if metrics['revenue_n'] > 0 else 0
            metrics['revenue_avg_residual_last3'] = avg_residual_last3
        except Exception as e:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_revenue = 0, 0, 0, 0, 0, 0, 0, 0, None
            st.write(f"Regression failed for Revenue on company {name} due to {e}")
            
   
    if metrics['income_growth_list']:
        # st.write(metrics['income_values_list'],metrics['income_growth_list']) #debug
        income_growths = metrics['income_growth_list']
        metrics['median_income_growth'] = np.median(income_growths)
        # income_values_list = metrics['income_values_list']
        # Income growth consistency
        metrics['income_consistency_score'] = len([g for g in income_growths if g > 0]) / len(income_growths) * 100
        metrics['last_3q_income_growth_positive_count'] = len([g for g in income_growths[-3:] if g > 0]) if len(income_growths) >= 3 else len([g for g in income_growths if g > 0])
        metrics['last_3q_income_positive_count'] = len([g for g in income_values[-3:] if g > 0]) if len(income_values) >= 3 else len([g for g in income_values if g > 0])
        
        # Last 3 quarters average
        if len(income_growths) >= 3:
            metrics['last_3q_income_growth'] = np.median(income_growths[-3:])
        elif len(income_growths) > 0:
            metrics['last_3q_income_growth'] = np.median(income_growths)
        
        # Linear regression for income growth trend
    if len(income_values) >= 6:  # Need at least 6 quarters to do a meaningful regression
        try:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_income = perform_regression(name, "Income", income_values, end_date, plot_regression_bin, remove_outliers)
            metrics['income_growth_slope'] = safe_round(slope, 1)
            metrics['income_r2'] = r2
            metrics['income_growth_median'] = median
            metrics['income_growth_pct'] = safe_round(metrics['income_growth_slope'] * 100 / max(abs(metrics['last_6q_income_median']),abs(metrics['income_median'])) if metrics['last_6q_income_median'] != 0 else 1, 2) if 'income_growth_slope' in metrics and 'last_6q_income_median' in metrics else 0
            metrics['income_n'] = n
            metrics['income_n_outliers'] = n_outliers
            metrics['income_outlier_pct'] = safe_round(metrics['income_n_outliers'] / metrics['income_n'] * 100, 2) if metrics['income_n'] > 0 else 0
            metrics['income_avg_residual_last3'] = avg_residual_last3
        except Exception as e:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_income = 0, 0, 0, 0, 0, 0, 0, 0, None
            st.write(f"Regression failed for Income for company {name} due to {e}")
        
    if metrics['margin_growth_list']:
        margin_growths = metrics['margin_growth_list']
        metrics['median_margin_growth'] = np.median(margin_growths)
        
        # Last 3 quarters average
        if len(margin_growths) >= 3:
            metrics['last_3q_margin_growth'] = np.median(margin_growths[-3:])
        elif len(margin_growths) > 0:
            metrics['last_3q_margin_growth'] = np.median(margin_growths)
        
    # Linear regression for margin growth trend
    if len(margin_values) >= 6:  # Need at least 6 quarters to do a meaningful regression
        try:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_margin = perform_regression(name, "Margin", margin_values, end_date, plot_regression_bin, remove_outliers)
            metrics['margin_r2'] = r2
            metrics['margin_growth_slope'] = safe_round(slope, 1)
            metrics['margin_avg_residual_last3'] = avg_residual_last3
            metrics['margin_growth_median'] = median
            metrics['margin_growth_pct'] = safe_round(metrics['margin_growth_slope'] * 100 / abs(metrics['margin_growth_median'] if metrics['margin_growth_median'] != 0 else 1), 2) if 'margin_growth_slope' in metrics and 'margin_growth_median' in metrics else 0
            metrics['margin_n'] = n
            metrics['margin_n_outliers'] = n_outliers
            metrics['margin_outlier_pct'] = safe_round(metrics['margin_n_outliers'] / metrics['margin_n'] * 100, 2) if metrics['margin_n'] > 0 else 0
            
        except Exception as e:
            slope, intercept, r2, median, mean, n, n_outliers, avg_residual_last3, chart_margin = 0, 0, 0, 0, 0, 0, 0, 0, None
            st.write(f"Regression failed for Margin for company {name} due to {e}")
    
    # Calculate average margin and last 3 quarters average margin
    if metrics['margin_values']:
        margin_values = metrics['margin_values']
        metrics['margin_median'] = np.median(margin_values)
        
        # Last 3 quarters average margin
        if len(margin_values) >= 3:
            metrics['last_3q_median_margin'] = np.median(margin_values[-3:])
        elif len(margin_values) > 0:
            metrics['last_3q_median_margin'] = np.median(margin_values)
        
        
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

    def centered_text(text: str):
        st.markdown(f"<div style='text-align:center;'>{text}</div>", unsafe_allow_html=True)
        
    if plot_regression_bin==1:
        try:
            col1, col2, col3 = st.columns(3)
            if chart_revenue is not None:
                with col1:
                    # st.markdown(f'<p class="centered-title">Income Statement {name}</p>', unsafe_allow_html=True)
                    # revenue_slope_pct = (metrics['revenue_growth_slope'] / metrics['revenue_growth_median'] * 100) if metrics['revenue_growth_median'] != 0 else 0
                    placeholder=st.empty()
                    placeholder.altair_chart(chart_revenue, width='stretch') #, width='stretch'
                    annualized_growth = ((1+(metrics['revenue_growth_pct'])/100) **4 - 1)* 100 * np.sign(metrics['revenue_growth_pct'])
                    # st.write(f"metrics['revenue_growth_pct']={metrics['revenue_growth_pct']}, annualized={annualized_growth}")
                    centered_text(f"Annual Growth (reg line): {annualized_growth:,.1f}%, R²: {metrics['revenue_r2']:.2f}, Outlier: {metrics['revenue_outlier_pct']:.1f}%")
            if chart_income is not None:
                with col2:
                    # income_slope_pct = (metrics['income_growth_slope'] / metrics['income_growth_median'] * 100) if metrics['income_growth_median'] != 0 else 0
                    placeholder=st.empty()
                    placeholder.altair_chart(chart_income, width='stretch')
                    annualized_growth = ((1+(metrics['income_growth_pct'])/100) **4 - 1)* 100 * np.sign(metrics['income_growth_pct'])
                    centered_text(f"Annual Growth (reg line): {annualized_growth:,.1f}%, R²: {metrics['income_r2']:.2f}, Outlier: {metrics['income_outlier_pct']:.1f}%")
            if chart_margin is not None:
                with col3:
                    # margin_slope_pct = (metrics['margin_growth_slope'] / metrics['margin_growth_median'] * 100) if metrics['margin_growth_median'] != 0 else 0
                    placeholder=st.empty()
                    placeholder.altair_chart(chart_margin, width='stretch')
                    annualized_growth = ((1+(metrics['margin_growth_pct'])/100) **4 - 1)* 100 * np.sign(metrics['margin_growth_pct'])
                    centered_text   (f"Annual Growth (reg line): {annualized_growth:,.1f}%, R²: {metrics['margin_r2']:.2f}, Outlier: {metrics['margin_outlier_pct']:.1f}%")
        except Exception as e:
            st.write(f"Could not render regression charts due to {e}")
    
    return metrics

def compute_value_score(company_and_ticker, m, trailing_pe, forward_pe, trailing_ps):
    """
    Compute a growth-adjusted valuation score using:
    - Growth Quality
    - Recent Momentum
    - Stability
    - Valuation Pressure
    """
    # st.write(company_and_ticker, trailing_pe, forward_pe, trailing_ps, revenue_growth_PCT, income_growth_PCT, revenue_growth_outlier_PCT, income_growth_outlier_PCT) # debug

    # Hard fail for insufficient history
    try:
        if m['revenue_n'] < 8:
            st.write(f"Not enough data points for {company_and_ticker}")
            return -100.0, -100.0, -100.0, -100.0, -100.0

        # --- Growth Quality ---
        capped_rev_growth_PCT=min(m['revenue_growth_pct'],100)
        capped_inc_growth_PCT=min(m['income_growth_pct'],2*capped_rev_growth_PCT)
        capped_margin_growth_slope=min(m['margin_growth_pct'],10)
        
        GQ = safe_round((
            1.4 * (capped_rev_growth_PCT * m['revenue_r2'])
            + 1.4 * (capped_inc_growth_PCT * m['income_r2'])
            + 0.5 * (capped_margin_growth_slope * m['margin_r2'])
            - 1.0 * (m['revenue_outlier_pct'] + m['income_outlier_pct'])
        ),1)

        # --- Recent Momentum ---
        rev = m['last_3q_revenue_growth']
        inc = m['last_3q_income_growth']

        # Cap income growth to 2x revenue growth
        capped_rev = min(rev, 100)
        capped_inc = min(inc, 2 * capped_rev)

        RM = safe_round(
            (
                0.10 * capped_rev
                + 0.10 * capped_inc
                - 3 * (3 - m['last_3q_income_positive_count'])
            ),
            1
        )

        # --- Stability (softened + capped) ---

        rev_norm_raw = m['revenue_avg_residual_last3'] / max(abs(m['revenue_growth_median']), 1)
        inc_norm_raw = m['income_avg_residual_last3'] / max(abs(m['income_growth_median']), 1)

        # Cap extreme spikes at 25%
        rev_norm = min(abs(rev_norm_raw), 0.25)
        inc_norm = min(abs(inc_norm_raw), 0.25)

        ST = safe_round(
            1.5 * m['revenue_r2']
            + 1.5 * m['income_r2']
            - 1.5 * rev_norm
            - 1.5 * inc_norm,
            1
        )
        
        ST = max(ST,0) #if negative, make it 0

        # --- Valuation Pressure ---
        try:
            tpe = float(trailing_pe) if trailing_pe not in (None, "") and trailing_pe>=0 else 0.0
            tpe = min(tpe,500)
            fpe = float(forward_pe)  if forward_pe  not in (None, "") and forward_pe>=0 else 0.0
            tps = float(trailing_ps) if trailing_ps not in (None, "") else 0.0
        except Exception as e:
            st.write(f"Error on Value_Pressure calc for {company_and_ticker}: {e}")
            return -100.0, -100.0, -100.0, -100.0, -100.0

        VP = safe_round(
            2 * math.log1p(tpe)
            + 2* math.log1p(fpe)
            + 4 * math.log1p(tps),
            1
        )
        
        if VP<2:
            VP=2.0
            
        # --- Final Score ---
        score = safe_round(((GQ + RM) * ST) / VP,1)
    
    except Exception as e:
        st.write(f"{company_and_ticker} had error on value calc: {e}")
        return -100.0, -100.0, -100.0, -100.0, -100.0

        
    # st.write(m) #debug
    # st.write(f"{company_and_ticker}: Value Score: {score}, Growth Qual: {GQ}, Recent Momentum: {RM}, Stability: {ST}, Value Pressure: {VP}") #, PE: {trailing_pe}, fwd PE:{forward_pe:.1f}, PS:{trailing_ps:.1f}")
    # st.stop() #debug
    return score, GQ, RM, ST, VP

#Note that ss.quarterly_financials must be loaded for this to function
def collect_data_for_company(cik):

    try:
        ticker = (ss.quarterly_financials.loc[ss.quarterly_financials['cik'] == cik, 'ticker'].iloc[0])
        max_filing_date = ss.quarterly_financials.loc[ss.quarterly_financials['cik'] == cik, 'max_filing_date'].max()
        max_report_date = ss.quarterly_financials.loc[ss.quarterly_financials['cik'] == cik, 'max_report_date'].max()
        company_and_ticker = (ss.quarterly_financials.loc[ss.quarterly_financials['cik'] == cik, 'company_and_ticker'].iloc[0])
        quarterly_df_wc = ss.quarterly_financials[ss.quarterly_financials['cik'] == cik].reset_index(drop=True).copy()
        metrics = analyze_yoy_growth(quarterly_df_wc, company_and_ticker,plot_regression_bin=0)

        # st.write('metrics:',metrics) #debug

        yahoo_stats = yahoo_finance_load(ticker)
        industry = yahoo_stats.get('industry')
        sector = yahoo_stats.get('sector')
        company_desc = yahoo_stats.get('longBusinessSummary')
        stock_price = yahoo_stats.get("currentPrice")
        price_range_52wks = yahoo_stats.get("fiftyTwoWeekRange")
        pct_chg_from_52wk_high = yahoo_stats.get("fiftyTwoWeekHighChangePercent")
        pct_chg_from_52wk_low = yahoo_stats.get("fiftyTwoWeekLowChangePercent")
        trailing_pe = clean_number(yahoo_stats.get("trailingPE"))
        trailing_ps = clean_number(yahoo_stats.get("priceToSalesTrailing12Months"))
        forward_pe = clean_number(yahoo_stats.get("forwardPE"))

        # st.write(company_and_ticker, trailing_pe,forward_pe,trailing_ps) #debug
        # st.json(yahoo_stats) #debug

        result = compute_value_score(company_and_ticker, metrics, trailing_pe,forward_pe,trailing_ps)
        value_score, growth_quality, recent_momentum, stability_trend, value_pressure = result
        
        # st.write(result) #debug

        results=({
        'cik': cik,
        'ticker': ticker,
        'company_and_ticker': company_and_ticker,
        'industry': industry,
        'sector': sector,
        'stock_price' : stock_price,
        'price_range_52wks' : price_range_52wks,
        'pct_chg_from_52wk_high' : safe_multiply(safe_round(pct_chg_from_52wk_high,3),100),
        'pct_chg_from_52wk_low' : safe_multiply(safe_round(pct_chg_from_52wk_low,3),100),
        'trailing_pe' : safe_round(trailing_pe,1),
        'forward_pe' : safe_round(forward_pe,1),
        'trailing_ps' : safe_round(trailing_ps,1),
        'Consolidated_Score': value_score,
        'Growth_Quality': growth_quality,
        'Recent_Momentum': recent_momentum,
        'Stability_Trend': stability_trend,
        'Value_Pressure': value_pressure,
        'Revenue_Consistency_Score': safe_round(metrics['revenue_consistency_score'], 1),
        'Income_Consistency_Score': safe_round(metrics['income_consistency_score'], 1),
        'Median_Revenue_Growth_PCT': safe_round(metrics['median_revenue_growth'], 2),
        'Median_Income_Growth_PCT': safe_round(metrics['median_income_growth'], 2),
        'Median_Margin_Growth_PCT': safe_round(metrics['median_margin_growth'], 2),
        'Median_Margin_PCT': safe_round(metrics['margin_median'], 2),
        'Last3Q_Revenue_Growth_PCT': safe_round(metrics['last_3q_revenue_growth'], 2),
        'Last3Q_Income_Growth_PCT': safe_round(metrics['last_3q_income_growth'], 2),
        'Last3Q_Margin_Growth_PCT': safe_round(metrics['last_3q_margin_growth'], 2),
        'Last3Q_Median_Margin_PCT': safe_round(metrics['last_3q_median_margin'], 2),
        'Last3Q_Income_Positive': int(metrics['last_3q_income_positive_count']),
        'Revenue_Growth_Count': len(metrics['revenue_growth']),
        'Income_Growth_Count': len(metrics['income_growth']),
        'Margin_Growth_Count': len(metrics['margin_growth']),
        'Revenue_Growth_Slope': safe_round(metrics['revenue_growth_slope'] / 1000, 2),
        'Revenue_Growth_Median': safe_round(metrics['revenue_growth_median'] / 1000, 2),
        'Revenue_Growth_PCT': safe_round(metrics['revenue_growth_pct'],2),
        'Revenue_Growth_N': metrics['revenue_n'],
        'Revenue_Growth_N_Outliers': metrics['revenue_n_outliers'],
        'Revenue_Growth_Outlier_PCT': safe_round(metrics['revenue_outlier_pct'],2),
        'Revenue_R2': safe_round(metrics['revenue_r2'], 4),
        'Revenue_Avg_Residual_Last3': safe_round(metrics['revenue_avg_residual_last3']/1000, 2),
        'Income_Growth_Slope': safe_round(metrics['income_growth_slope'] / 1000, 2),
        'Income_Growth_Median': safe_round(metrics['income_growth_median'] / 1000, 2),
        'Income_Growth_PCT': safe_round(metrics['income_growth_pct'],2),
        'Income_Growth_N': metrics['income_n'],
        'Income_Growth_N_Outliers': metrics['income_n_outliers'],
        'Income_Growth_Outlier_PCT': safe_round(metrics['income_outlier_pct'],2),
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
        'max_report_date': max_report_date,
        'company_desc': company_desc
        })
        # st.write(results) #debug 
        
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        filename = tb.tb_frame.f_code.co_filename
        line_no = tb.tb_lineno
        code_line = linecache.getline(filename, line_no).strip()
        st.write(f"Error on collect data for company for {cik} on file: {filename}, line #{line_no}, code line: {code_line}: {e}")
        return None
        
    return results

def rank_companies_by_growth_and_update_DB(cik_list):
    """
    Rank companies by growth consistency.
    Args: companies_dict: Dictionary with company names as keys and quarterly DataFrames as values
    Returns: results_df
    """
    
    # st.write(ss.quarterly_financials)  # debug
    results = []
    
    progress_bar = st.progress(0)
    total = len(cik_list)
    status = st.empty()
    # error=0
    
    # if ss.quarterly_financials is None or ss.quarterly_financials.empty:
    ss.quarterly_financials = load_quarterly_sec_data_from_db()
    st.toast('Read SEC Quarterly Financial Data from DB')

    # st.write(ss.quarterly_financials) # debug
    
    for i, cik in enumerate(cik_list): # :#companies_dict.items():
        status.write(f"Processing CIK {cik} ({i+1}/{total})")
        progress_bar.progress((i + 1) / total)
        try: 
            results_for_cik=collect_data_for_company(cik)
            if results_for_cik != None:
                temp_df = pd.DataFrame([results_for_cik])
                postgres_update(temp_df, 'stock_growth_analysis_results', ['cik'])  # Save results to PostgreSQL
                results.append(results_for_cik)
                            
        except Exception as e:
            st.write(f"Failed to append results data for {cik}: {e}")
            logging.error(f"Failed to append results data for {cik}: {e}")

    # results_df for all companies being processed
    if isinstance(results, list) and len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('Consolidated_Score', ascending=False)
    else:
        results_df=pd.DataFrame()
    # st.write(results) #debug
    # st.stop()
    
    return results_df

def get_column_specs_results_df():
    column_specs= {
        "cik": {"pg_name": "cik", "fmt": "{}"},
        "ticker": {"pg_name": "ticker", "fmt": "{}"},
        "company_and_ticker": {"pg_name": "company_and_ticker", "fmt": "{}"},
        "Consolidated_Score": {"pg_name": "consolidated_score", "fmt": "{:.1f}"},
        "Growth_Quality": {"pg_name": "growth_quality", "fmt": "{:.1f}"},
        "Recent_Momentum": {"pg_name": "recent_momentum", "fmt": "{:.1f}"},
        "Stability_Trend": {"pg_name": "stability_trend", "fmt": "{:.1f}"},
        "Value_Pressure": {"pg_name": "value_pressure", "fmt": "{:.1f}"},
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

def color_button(color_name):
    COLORS = {
        "blue":  "#070e79",
        "red":   "#881915",
        "green": "#094217",
        "gray":  "#3e4246",
        "orange": "#fd7e14",
        "purple": "#6f42c1",
        "dark gray": "#1E2024",
        "black": '#0E1117',
    }

    color = COLORS.get(color_name, '#0E1117')  # default black

    unique_key = f"btn_{color_name}_{uuid.uuid4().hex}"

    return stylable_container(
        key=unique_key,

        css_styles=f"""
            button {{
                background-color: {color} !important;
                color: white !important;

                border: 1px solid gray !important;
                border-radius: 6px;
                padding: 0.2em 1.2em;
                
                font-size: 0.8rem !important;      /* slightly smaller text */
                line-height: 1.0 !important;       /* tighter vertical spacing */
                
                margin-bottom: 10px !important;
            }}
            button:hover {{
                opacity: 0.85;
            }}
        """
    )

def write_sec_data_into_db(load_type):
    print('entering into write_sec_data_into_db function')
    
    # st.write(f"company_lookup_df = ",ss.company_lookup_df)  #debug
    
    #debug this only selects a few CIKs for testing
    # cik_list={'0000066740','0001368514'}#,'0001070494','0000318306','0001018724','0000789019','0000002488','0001045810','0000034782','0001652044','0000320193','0001018724','0001326801','0001730168','0001318605','0001067983','0000059478'} #debug
    
    ss.company_lookup_df=get_company_tickers_df()
    ss.company_lookup_df = (
        ss.company_lookup_df
        .drop_duplicates(subset="cik", keep="first")
        .sort_values("company_and_ticker")
    )
    
    if load_type == 'full':
        cik_list = ss.company_lookup_df['cik'].tolist()
        # cik_list=cik_list[6000:] #[:2] - limit to first 2 CIKs for testing
    
    if load_type == 'incremental':
        stock_growth_analysis_df = load_stock_growth_analysis_data_from_db()
        stock_growth_analysis_df['max_filing_date'] = pd.to_datetime(
                stock_growth_analysis_df['max_filing_date'],
                errors='coerce'
            )
        df = stock_growth_analysis_df.dropna(subset=['max_filing_date'])
        # st.write(df) #debug
        last_date_loaded=pd.to_datetime(df['max_filing_date'].max(), errors='coerce')
        # last_date_loaded=pd.to_datetime('2026-03-10') #debug
        # st.write(last_date_loaded) #debug
        filings_10q_10k_df = load_daily_SEC_submission_index(last_date_loaded, forms_filter_list=['10-Q','10-K']) # '8-K'
        merged_sec_filings=filings_10q_10k_df.merge(stock_growth_analysis_df[['cik','max_filing_date']],on='cik', how='left')
        # st.write('merged_sec_filings',merged_sec_filings) #debug
        filtered_sec_filings = merged_sec_filings[
            (merged_sec_filings['date'] > merged_sec_filings['max_filing_date'])
            ]
        st.write('SEC Filings to Load',filtered_sec_filings) #debug
        cik_list=filtered_sec_filings['cik'].to_list()
        # cik_list=['0001341439'] #debug
        print('Got list of cik to update for incremental load')
    
    # Now that you have cik_list, go get the data
    if len(cik_list)==0:
        st.write('No SEC filings to get')
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
            logging.error(f"Failed to load and write SEC data for {company_and_ticker}: {e}")
            ss.filings_df = pd.DataFrame()
            ss.quarterly_financials = pd.DataFrame()
            ss.annual_financials = pd.DataFrame()
            continue # Skip to next CIK    
    
    # Now go get the rest of the data for thesee companies and run regressions, and update the stock_growth_analysis_results
    ss.results_df = rank_companies_by_growth_and_update_DB(cik_list)
    st.write(ss.results_df) # debug?
    # st.stop() debug
    return

def load_quarterly_sec_data_from_db():
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
    # column_specs=get_column_specs_results_df()
    # pg_to_pretty = {v["pg_name"]: k for k, v in column_specs.items()}
    # df_pretty=df.rename(columns={pg_col: pg_to_pretty.get(pg_col, pg_col) for pg_col in df.columns})
    return df #, companies_data

def load_stock_transactions_from_db():
    status = st.empty()
    df = postgres_read('stock_transactions')
    df = df.sort_values('date',ascending=False)
    # st.dataframe(df)  # debug
    # column_specs=get_column_specs_results_df()
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

def postgres_delete_single_transaction(row):
    import psycopg2
    import pandas as pd

    connection_params = {
        "user": st.secrets["postgres_financial_data"]["user"],
        "host": st.secrets["postgres_financial_data"]["host"],
        "port": st.secrets["postgres_financial_data"]["port"],
        "database": st.secrets["postgres_financial_data"]["database"],
        "password": st.secrets["postgres_financial_data"]["password"]
    }

    conn = psycopg2.connect(**connection_params)
    cursor = conn.cursor()

    delete_sql = """
        DELETE FROM stock_transactions
        WHERE cik = %s
          AND date = %s
          AND action = %s;
    """

    cursor.execute(delete_sql, (
        row["cik"],
        pd.to_datetime(row["date"]).date(),
        row["action"]
    ))

    conn.commit()
    cursor.close()
    conn.close()

def enter_stock_transaction():
    # ss.show_transaction_form=True

    go_back_to_summary_btn = st.button('Go Back To Stock Analysis Summary')
    if go_back_to_summary_btn:
        ss.show_transaction_form=False
        ss.editable_stock_growth_analysis_df = ss.editable_stock_growth_analysis_df.iloc[0:0]
        st.rerun()

    if ss.company_lookup_df is None or ss.company_lookup_df.empty:
        ss.company_lookup_df = get_company_tickers_df()
        ss.company_lookup_df = (
            ss.company_lookup_df
            .drop_duplicates(subset="cik", keep="first")
            .sort_values("company_and_ticker")
        )

    if "transaction_df" not in ss:
        ss.transaction_df = load_stock_transactions_from_db()

    if "transaction_df_prev" not in ss:
        ss.transaction_df_prev = ss.transaction_df.copy()

    if "show_transaction_form" not in ss:
        ss.show_transaction_form = True

    st.subheader("Add a Stock Transaction")

    # -----------------------------
    # ENTRY FORM
    # -----------------------------
    if ss.show_transaction_form:
        with st.form("trade_entry_form"):
            col1, col2 = st.columns(2)

            with col1:
                trade_date = st.date_input("Trade Date", value=date.today())
                company_list = ss.company_lookup_df["company_and_ticker"].tolist()
                company_and_ticker = st.selectbox("Company", options=company_list)
                action = st.selectbox("Action", ["Buy", "Sell"])

            with col2:
                quantity = st.number_input("Quantity", min_value=1, step=1)
                price = st.number_input("Price per Share", min_value=0.00, step=0.01, format="%.2f")

            submit_add = st.form_submit_button("Add Transaction")

        if submit_add:
            cik = ss.company_lookup_df.loc[
                ss.company_lookup_df["company_and_ticker"] == company_and_ticker, "cik"
            ].iloc[0]

            new_row = {
                "cik": cik,
                "date": trade_date,
                "company_and_ticker": company_and_ticker,
                "action": action,
                "quantity": quantity,
                "price": price,
                "total": quantity * price,
            }

            postgres_update(pd.DataFrame([new_row]), "stock_transactions",
                            primary_key_columns=["cik", "date", "action"])

            ss.transaction_df = pd.concat([ss.transaction_df, pd.DataFrame([new_row])], ignore_index=True)
            ss.transaction_df_prev = ss.transaction_df.copy()

            st.toast("Transaction added successfully.")

    st.divider()

    # -----------------------------
    # EDITABLE TABLE
    # -----------------------------
    st.subheader("Edit or Delete Transactions")

    edited_df = st.data_editor(
        ss.transaction_df,
        num_rows="dynamic",
        column_config= {
            'quantity': st.column_config.NumberColumn(label="Quantity", step='int'),
            'price': st.column_config.NumberColumn(label="Price", format='dollar'),
            'total': st.column_config.NumberColumn(label="Total Amount", format='dollar')
            },
        hide_index=True,
    )

    # -----------------------------
    # DETECT DELETIONS
    # -----------------------------
    deleted_rows = ss.transaction_df_prev[
        ~ss.transaction_df_prev.apply(tuple, 1).isin(edited_df.apply(tuple, 1))
    ]

    for _, row in deleted_rows.iterrows():
        postgres_delete_single_transaction(row)
        st.toast("Transaction deleted.")

    # -----------------------------
    # DETECT ADDITIONS
    # -----------------------------
    new_rows = edited_df[
        ~edited_df.apply(tuple, 1).isin(ss.transaction_df_prev.apply(tuple, 1))
    ]

    for _, row in new_rows.iterrows():
        postgres_update(pd.DataFrame([row]), "stock_transactions",
                        primary_key_columns=["cik", "date", "action"])
        st.toast("New transaction added.")

    # -----------------------------
    # DETECT MODIFICATIONS
    # -----------------------------
    merged = edited_df.merge(ss.transaction_df_prev, indicator=True, how="outer")
    modified = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    for _, row in modified.iterrows():
        postgres_update(pd.DataFrame([row]), "stock_transactions",
                        primary_key_columns=["cik", "date", "action"])
        st.toast("Transaction updated.")

    # -----------------------------
    # UPDATE SESSION STATE
    # -----------------------------
    ss.transaction_df = edited_df
    ss.transaction_df_prev = edited_df.copy()

    if st.button("Exit Transaction Entry and Return to Stock Analysis"):
        ss.show_transaction_form = False
        st.rerun()
        
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

def show_regression_charts(cik):
    try: 
        quarterly_df=postgres_read('stock_quarterly_financials_sec',f"cik='{cik}'")
        company_and_ticker=quarterly_df['company_and_ticker'].iloc[0]
        # ticker=quarterly_df['ticker'].iloc[0]
        column_specs = get_column_specs_quarterly_df()
        pg_to_pretty = {spec["pg_name"]: pretty for pretty, spec in column_specs.items()}
        quarterly_df = quarterly_df.rename(columns=pg_to_pretty)
        # st.dataframe(quarterly_df) # debug
        analyze_yoy_growth(quarterly_df,company_and_ticker,plot_regression_bin=1)
        # st.write(ss.rankings_df) #debug
        company_description=ss.rankings_df.loc[ss.rankings_df['cik']==cik,'company_desc'].iloc[0]
        st.write('')
        st.write(f":blue[{company_description}]")
    except Exception as e:
        exc_type, exc_obj, tb = sys.exc_info()
        filename = tb.tb_frame.f_code.co_filename
        line_no = tb.tb_lineno
        code_line = linecache.getline(filename, line_no).strip()
        st.warning(f"Failed to append data due to {e}, file: {filename}, line #{line_no}, code line: {code_line}")
        # st.warning(f"Regression failed: {e}")
    return()

def reset_forms_ss_vars():
    ss.hide_menu=ss.view_stock_analysis_form=ss.show_transaction_form=ss.qtr_data_form=ss.investment_returns_form=ss.process_yahoo_and_statistics=False
    ss.load_sec_incremental_filings=ss.load_sec_full_filings=False
    ss.selected_company=None
    return

def update_primary_filter_session_value(key):
    temp_key = 'temp_'+key
    ss[key] = ss[temp_key]

def show_investment_returns():
    transactions_df=load_stock_transactions_from_db()
    ss.rankings_df=load_stock_growth_analysis_data_from_db()
    transactions_df = transactions_df.merge(
        ss.rankings_df[['cik', 'stock_price']], on='cik', how='left'
        ).rename(columns={'stock_price': 'current_price'})
    
    # --- Build investment_returns_df from transactions_df and rankings_df ---

    # Split buys and sells
    buys  = transactions_df[transactions_df['action'] == 'Buy'].copy()
    sells = transactions_df[transactions_df['action'] == 'Sell'].copy()

    buys['purchase_quantity'] = buys['quantity']
    buys['purchase_amount']   = buys['total']

    sells['sold_quantity'] = sells['quantity']
    sells['sold_amount']   = sells['total']

    # Aggregate buys
    buy_summary = buys.groupby(['cik', 'company_and_ticker']).agg(
        purchase_quantity=('purchase_quantity', 'sum'),
        purchase_amount=('purchase_amount', 'sum'),
        first_purchase_date=('date', 'min')   # needed for returns
    ).reset_index()

    # Aggregate sells
    sell_summary = sells.groupby(['cik', 'company_and_ticker']).agg(
        sold_quantity=('sold_quantity', 'sum'),
        sold_amount=('sold_amount', 'sum')
    ).reset_index()

    # Merge buy + sell summaries
    investment_returns_df = (
        buy_summary
        .merge(sell_summary, on=['cik', 'company_and_ticker'], how='left')
        .fillna({'sold_quantity': 0, 'sold_amount': 0})
    )

    # Average purchase price
    investment_returns_df['avg_purchase_price'] = (
        investment_returns_df['purchase_amount'] /
        investment_returns_df['purchase_quantity']
    )

    # Average sold price
    investment_returns_df['avg_sold_price'] = (
        investment_returns_df['sold_amount'] /
        investment_returns_df['sold_quantity'].replace(0, pd.NA)
    )

    # Current holdings
    investment_returns_df['current_holdings'] = (
        investment_returns_df['purchase_quantity'] -
        investment_returns_df['sold_quantity']
    )

    # Attach current price from rankings_df
    investment_returns_df['current_price'] = (
        investment_returns_df['cik'].map(
            ss.rankings_df.set_index('cik')['stock_price']
        )
    )

    # Current holdings value
    investment_returns_df['current_holdings_value'] = (
        investment_returns_df['current_holdings'] *
        investment_returns_df['current_price']
    )

    # Realized gains (simple average-cost method)
    investment_returns_df['realized_gains'] = investment_returns_df['sold_amount'] - (investment_returns_df['avg_purchase_price'] * investment_returns_df['sold_quantity'])

    # Unrealized gains
    investment_returns_df['unrealized_gains'] = (
        investment_returns_df['current_holdings_value'] -
        (investment_returns_df['avg_purchase_price'] * investment_returns_df['current_holdings'])
    )
    
    investment_returns_df['total_gains']=investment_returns_df['realized_gains']+investment_returns_df['unrealized_gains']

    # --- RETURN CALCULATIONS ---

    today = pd.Timestamp.today()

    # Ensure date is Timestamp
    investment_returns_df['first_purchase_date'] = pd.to_datetime(investment_returns_df['first_purchase_date'])

    investment_returns_df['months_held'] = (
        round((today - investment_returns_df['first_purchase_date']).dt.days / 30,1)
    )
    
    # Total return (current value vs cost basis)
    investment_returns_df['total_return_pct'] = (investment_returns_df['total_gains']/investment_returns_df['purchase_amount']) * 100
    
    # Clean up infinite or invalid values
    investment_returns_df = investment_returns_df.replace([np.inf, -np.inf], np.nan)

    # --- ADD TOTALS ROW ---

    # Compute totals for numeric columns
    totals = investment_returns_df.select_dtypes(include=[np.number]).sum()
    totals["company_and_ticker"] = "TOTAL"
    totals["cik"] = ""

    investment_returns_df = pd.concat(
        [investment_returns_df, totals.to_frame().T],
        ignore_index=True
    )

    # Compute portfolio-level total return
    total_purchase_amount = investment_returns_df['purchase_amount'].sum()
    total_sold_amount = investment_returns_df['sold_amount'].sum()
    total_current_value = investment_returns_df['current_holdings_value'].sum()

    portfolio_total_return = ((total_sold_amount + total_current_value) / total_purchase_amount - 1) * 100

    # Insert correct total return into totals row
    investment_returns_df.loc[
        investment_returns_df['company_and_ticker'] == "TOTAL",
        'total_return_pct'
    ] = portfolio_total_return
    
    # Columns to remove
    cols_to_remove = ["cik", "sold_quantity", "sold_amount", "avg_sold_price"]
    investment_returns_df = investment_returns_df.drop(columns=cols_to_remove, errors="ignore")
    
    # --- STYLING ---

    def color_gains(val):
        """Green for gains, red for losses."""
        try:
            if pd.isna(val):
                return ""
            if val > 0:
                return "color: green;"
            if val < 0:
                return "color: red;"
        except:
            pass
        return ""

    # Column groups
    quantity_cols = ["purchase_quantity", "current_holdings"]
    price_cols    = ["avg_purchase_price", "current_price"]
    amount_cols   = ["purchase_amount", "current_holdings_value","total_gains","realized_gains", "unrealized_gains"]
    percent_cols  = ["total_return_pct"]

    # Format dictionary
    fmt = {}

    # Quantities → comma integers
    for col in quantity_cols:
        fmt[col] = lambda v: f"{v:,.0f}" if pd.notna(v) and isinstance(v, (int, float)) else v

    # months_held to single decimal
    fmt['months_held'] = lambda v: f"{v:,.1f}" if pd.notna(v) and isinstance(v, (int, float)) else v
    
    # Prices → $ with 2 decimals
    for col in price_cols:
        fmt[col] = lambda v: f"${v:,.2f}" if pd.notna(v) and isinstance(v, (int, float)) else v

    # Amounts → $ with 0 decimals
    for col in amount_cols:
        fmt[col] = "${:,.0f}"

    # Percent returns → 1 decimal place
    for col in percent_cols:
        fmt[col] = lambda v: f"{v:.1f}%" if pd.notna(v) and isinstance(v, (int, float)) else v
            
    fmt["first_purchase_date"] = lambda d: d.strftime("%Y-%m-%d") if pd.notna(d) else ""
    
    # --- SORT TOTALS TO TOP, THEN BY PURCHASE AMOUNT DESC ---

    # Create a sort key: TOTAL row gets 0, others get 1
    investment_returns_df["sort_key"] = (
        investment_returns_df["company_and_ticker"].eq("TOTAL").map({True: 0, False: 1})
    )

    total_mask = investment_returns_df["company_and_ticker"] == "TOTAL"

    fields_to_blank_totals = ["months_held","avg_purchase_price","avg_sold_price","current_price","purchase_quantity"
                              ,"sold_quantity","current_holdings"]

    def highlight_total_row(row):
        if row["company_and_ticker"] == "TOTAL":
            return ["background-color: #f2f2f2"] * len(row)
        return [""] * len(row)

    styled_df = (
        investment_returns_df
            .style
            .apply(highlight_total_row, axis=1)
            # your existing formatting here...
    )

    for col in fields_to_blank_totals:
        if col in investment_returns_df.columns:
            investment_returns_df.loc[total_mask, col] = ""


    # Sort: TOTAL first, then by purchase_amount descending
    investment_returns_df = (
        investment_returns_df
            .sort_values(["sort_key", "purchase_amount"], ascending=[True, False])
            .reset_index(drop=True)
    )

    # Drop helper column
    investment_returns_df = investment_returns_df.drop(columns=["sort_key"])
    investment_returns_df = investment_returns_df[['company_and_ticker','purchase_amount','current_holdings_value','total_gains','total_return_pct','realized_gains','unrealized_gains','months_held','purchase_quantity','first_purchase_date','avg_purchase_price','current_holdings','current_price']]

    styled_df = (
        investment_returns_df.style
            .applymap(color_gains, subset=['total_gains',"realized_gains","unrealized_gains","total_return_pct"])
            .format(fmt)
    )

    st.write("These are the Investment Returns:")
    st.dataframe(styled_df,
                # column_order={'total_return'}, 
                column_config={
                     "company_and_ticker":st.column_config.Column("company_and_ticker",pinned=True)
                 },
                 )
    st.write("")



    st.write('These are the transactions to date:')
    st.dataframe(transactions_df)

    st.stop()

def display_stock_analysis_form(stock_growth_analysis_df):
    with color_button('green'):
        enter_transaction_btn = st.button('Enter Stock Buy/Sell Transaction')
    if enter_transaction_btn:
        ss.show_transaction_form=True
        st.rerun()

    if ss.rerun_the_application==True:
        ss.rerun_the_application=False
        st.rerun()
        
    editable_columns = ['category', 'notes']
    # stock_growth_analysis_df=stock_growth_analysis_df[stock_growth_analysis_df['revenue_growth_slope'] > 0] # Filter to only show companies with positive revenue growth slope

    columns = [ 'cik', 'ticker', 'company_and_ticker'] + editable_columns + ['industry','sector','stock_price', 'price_range_52wks', 'pct_chg_from_52wk_high', 'pct_chg_from_52wk_low', 
        'Consolidated_Score','Growth_Quality','Recent_Momentum','Stability_Trend','Value_Pressure', 'trailing_pe', 'forward_pe', 'trailing_ps',
        'Revenue_Growth_Slope','Revenue_R2','Revenue_Growth_PCT','Revenue_Avg_Residual_Last3','Revenue_Growth_N','Revenue_Growth_Outlier_PCT','Revenue_Growth_Median',
        'Income_Growth_Slope','Income_R2','Income_Growth_PCT','Income_Avg_Residual_Last3','Income_Growth_N','Income_Growth_Outlier_PCT',
        'Margin_Growth_Slope','Margin_R2','Margin_Avg_Residual_Last3','Margin_Growth_N','Margin_Growth_N_Outliers',
        'Last3Q_Revenue_Growth_PCT', 'Last3Q_Income_Growth_PCT', 'Last3Q_Margin_Growth_PCT', 'Last3Q_Median_Margin_PCT', 'Last3Q_Income_Positive',
        'max_filing_date'
        ]

    st.write("**Stock Growth Analysis Data:**")
    if ss.editable_stock_growth_analysis_df.empty or ss.rankings_df.empty:
        column_specs=get_column_specs_results_df()
        rename_map = {
            spec["pg_name"]: pretty
            for pretty, spec in column_specs.items()
        }
        ss.rankings_df = stock_growth_analysis_df.rename(columns=rename_map)
        
        # st.write(f"the len(ss.rankings_df) is {len(ss.rankings_df)}") #debug

        editable_stock_data=read_or_create_editable_table()
        ss.editable_stock_growth_analysis_df = ss.rankings_df.merge(editable_stock_data[['cik'] + editable_columns], on='cik', how='left')
    
        ss.editable_stock_growth_analysis_df = ss.editable_stock_growth_analysis_df.reindex(columns=columns)
        # st.dataframe(editable_stock_growth_analysis_df) # debug     
        ss.editable_stock_growth_analysis_df = ss.editable_stock_growth_analysis_df.sort_values(by='Consolidated_Score', ascending=False)
        # st.write(f"the len(ss.editable_stock_growth_analysis_df) is {len(ss.editable_stock_growth_analysis_df)}") #debug
            
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
                ,'Consolidated_Score','Growth_Quality','Recent_Momentum'
            ]
        cols_red_bottom_quintile = ['Revenue_Growth_N','Income_Growth_N','Margin_Growth_N']
        cols_for_color_dec = ['Value_Pressure','trailing_pe','trailing_ps','forward_pe','pct_chg_from_52wk_high','Revenue_Growth_Outlier_PCT','Income_Growth_Outlier_PCT']

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
    
    # categories = ss.editable_stock_growth_analysis_df['category'].unique().tolist()
    ss.categories_list = ['Buy Now','Owned','Strong Rev & Income Growth', 'Strong Rev, Neg Income','Inconsistent Growth', 'Declining Growth', 'Other', 'Uncategorized']
    
    company_options = [None] + ss.rankings_df["company_and_ticker"].tolist()
    
    counts = ss.rankings_df['industry'].value_counts(dropna=True)
    industry_list = [None] + counts.index.tolist()
    
    counts = ss.rankings_df['sector'].value_counts(dropna=True)
    sector_list = [None] + counts.index.tolist()
    
    with st.expander("Expand to show filters", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            category = st.multiselect('Which categories to include?', options=ss.categories_list,key='temp_filter_category', default=ss.filter_category, on_change=update_primary_filter_session_value, args=("filter_category",)) #default=ss.filters['category'],on_change=lambda: on_filter_change("category", ss.category)
            max_revenue_median = st.number_input("Max Revenue Median? (0 = No Filter)", value=int(ss.editable_stock_growth_analysis_df['Revenue_Growth_Median'].max() + 1), min_value=0, step=1000000, format="%d", on_change=update_primary_filter_session_value, args=("filter_min_revenue_growth",)) # max_value=int(ss.editable_stock_growth_analysis_df['Revenue_Growth_Median'].max() + 1, on_change=update_primary_filter_session_value, args=("filter_min_revenue_growth",))
            max_trailing_pe = st.number_input("Max trailing PE (0 = No Filter)?", key='filter_max_trailing_pe', min_value=0, max_value=300, step=10, format="%d", on_change=update_primary_filter_session_value, args=("filter_max_trailing_pe",))
            max_trailing_ps = st.number_input("Max trailing PS (0 = No Filter)?", key='filter_max_trailing_ps', min_value=0, max_value=300, step=10, format="%d", on_change=update_primary_filter_session_value, args=("filter_max_trailing_ps",))
            company_and_ticker=st.selectbox("Search for specific company (negates other filters, set to 'None' to clear):",options=company_options,index=0)
            
        with col2:
            min_last_filing_date = st.date_input("Min Last Filing Date (set to future date to ignore)?",key='temp_filter_min_last_filing_date', on_change=update_primary_filter_session_value, args=("filter_min_last_filing_date",))
            min_revenue_growth = st.number_input("Min Quarterly Revenue Growth %? (0 = No filter)",key="temp_filter_min_revenue_growth", min_value=0, max_value=100, step=1, format="%d", on_change=update_primary_filter_session_value, args=("filter_min_revenue_growth",))
            min_income_growth = st.number_input("Min Quarterly Income Growth % (0 = No filter)?", key='temp_filter_min_income_growth', min_value=0, max_value=10, step=1, format="%d", on_change=update_primary_filter_session_value, args=("filter_min_income_growth",))
            # min_revenue_r2 = st.number_input("Min Revenue R2 (0 = No Filter)?", key='temp_filter_min_revenue_r2', min_value=0.00, max_value=1.00, step=0.10, format="%.2f", on_change=update_primary_filter_session_value, args=("filter_min_revenue_r2",))
            industry = st.selectbox("Select industry to include ('None' to clear selection)", key='temp_filter_industry',options=industry_list,index=0, on_change=update_primary_filter_session_value, args=("filter_industry",))
            sector = st.selectbox("Select sector to include ('None' to clear selection)", key='temp_filter_sector',options=sector_list,index=0, on_change=update_primary_filter_session_value, args=("filter_sector",))
            # min_revenue_n_count = st.number_input("Min revenue N count?", key='temp_filter_min_revenue_n_count', min_value=0, max_value=ss.filter_min_revenue_n_count, step=10, format="%d", on_change=update_primary_filter_session_value, args=("filter_min_revenue_n_count",))
            # max_rev_outlier_pct = st.number_input("Max Revenue Outlier % (0 = No filter)?", min_value=0, max_value=100, step=5, format="%d", on_change=update_primary_filter_session_value, args=("filter_rev_outlier_pct",))

    if company_and_ticker != None:
        # st.write(company_and_ticker) #debug
        selected_cik = ss.rankings_df.loc[ss.rankings_df['company_and_ticker'] == company_and_ticker, "cik"].iloc[0]
        mask = ss.editable_stock_growth_analysis_df['cik'] == selected_cik
        ss.selected_company=selected_cik
    else:
        mask = ss.editable_stock_growth_analysis_df['Revenue_Growth_N'] >= ss.filter_min_revenue_n_count

        mask_categories = ["" if c == "Uncategorized" else c for c in ss.filter_category]
        mask &= (ss.editable_stock_growth_analysis_df['category'].isin(mask_categories))

        if ss.filter_max_revenue_median != 0:
            mask &= (ss.editable_stock_growth_analysis_df['Revenue_Growth_Median'] < ss.filter_max_revenue_median)
        
        if ss.filter_min_revenue_growth != 0:
            mask &= (ss.editable_stock_growth_analysis_df['Revenue_Growth_PCT'] >= ss.filter_min_revenue_growth)
        
        if ss.filter_min_income_growth != 0:
            mask &= (ss.editable_stock_growth_analysis_df['Income_Growth_PCT'] >= ss.filter_min_income_growth)

        if ss.filter_industry != None:
            values = [ss.filter_industry]
            mask &= (ss.editable_stock_growth_analysis_df['industry'].isin(values))

        if ss.filter_sector != None:
            values = [ss.filter_sector]
            mask &= (ss.editable_stock_growth_analysis_df['sector'].isin(values))

        # if ss.filter_min_revenue_r2 != 0:
        #     mask &= (ss.editable_stock_growth_analysis_df['Revenue_R2'] >= ss.filter_min_revenue_r2)

        # if max_rev_outlier_pct != 0:
        #     mask &= (ss.editable_stock_growth_analysis_df['Revenue_Growth_Outlier_PCT'] <=  max_rev_outlier_pct)
        
        if ss.filter_max_trailing_pe != 0:
            mask &= (ss.editable_stock_growth_analysis_df['trailing_pe'] <= ss.filter_max_trailing_pe)
        
        if ss.filter_max_trailing_ps != 0:
            mask &= (ss.editable_stock_growth_analysis_df['trailing_ps'] <= ss.filter_max_trailing_ps)
        
        today = datetime.now().date()
        if ss.filter_min_last_filing_date is not None and ss.filter_min_last_filing_date <= today:
            ss.editable_stock_growth_analysis_df['max_filing_date'] = pd.to_datetime(
                    ss.editable_stock_growth_analysis_df['max_filing_date'],
                    errors='coerce')    
            selected_date = ss.filter_min_last_filing_date
            if isinstance(selected_date, (list, tuple)):
                selected_date = selected_date[0] if len(selected_date) > 0 else None
            mask &= (ss.editable_stock_growth_analysis_df['max_filing_date'] >= pd.to_datetime(ss.filter_min_last_filing_date))
        
    ss.filtered_df = ss.editable_stock_growth_analysis_df[mask]
    st.write(f"Current filters selecting {len(ss.filtered_df)} companies")
        
    ss.filtered_df=ss.filtered_df.reset_index(drop=True)

    with color_button('red'):
        update_yahoo_and_stats_btn = st.button('Hit this button to update Yahoo Stock Data for CURRENTLY SELECTED Stocks (only when stock price needs updating or selection widens)')

    if update_yahoo_and_stats_btn:
        cik_list = ss.filtered_df['cik'].tolist()
        results_df = rank_companies_by_growth_and_update_DB(cik_list)
        with st.spinner(f"Saving Analysis Results to DB"):
            postgres_update(results_df, 'stock_growth_analysis_results', ['cik'])  # Save results to PostgreSQL
            ss.editable_stock_growth_analysis_df = ss.editable_stock_growth_analysis_df.iloc[0:0]
            st.toast('Updated stock_growth_analysis_results on DB')
        ss.view_stock_analysis_form=True
        st.rerun()

    # ss.df=filtered_df.copy()        
    styled = build_styler(ss.filtered_df)

    # st.write("styled df:",styled) #debug

    def on_change_handle():
        if "my_editor" not in ss:
            return
        
        editor_state = ss['my_editor']
        
        if "edited_rows" not in editor_state:
            return
        
        edited = editor_state["edited_rows"]
        df = ss.get("filtered_df")

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
        
        # Process checkbox clicks even if "data" is missing
        for row_index, changes in new_changes.items():
            cik = df.loc[row_index, "cik"]
            if changes.get("chart") is True:
                ss.selected_company = cik
                ss.rerun_the_application=True
                
            if "category" in changes or "notes" in changes:
                update_df = ss.filtered_df.loc[[row_index], ['cik','ticker','company_and_ticker','category','notes','stock_price','trailing_pe','trailing_ps']]
                new_category = changes.get("category", df.loc[row_index,"category"])
                new_notes = changes.get("notes", df.loc[row_index,"notes"])
                update_df['category']=new_category
                update_df['notes']=new_notes

                try:
                    postgres_update(update_df, 'editable_stock_data', ['cik']) 
                    st.toast('Change Committed to DB')
                except Exception as e:
                    st.warning(f'Change failed due to {e}')

    st.data_editor(styled, key="my_editor", on_change=on_change_handle, width='stretch', disabled=disabled_cols
                ,hide_index=True
                ,column_config= {
                'company_and_ticker': st.column_config.TextColumn(label='Company and Ticker',pinned=True),
                'chart':st.column_config.CheckboxColumn(label='Charts', width="small", pinned=True),
                # "Yahoo_Link": st.column_config.LinkColumn(label="Links",display_text="https://finance.yahoo.com",display_text="Open Chart ↗"),
                'category': st.column_config.SelectboxColumn(label="Category", pinned=True, options=ss.categories_list, width="small"),
                'stock_price': st.column_config.NumberColumn(label="Stock Price", format='dollar'),
                'price_range_52wks': st.column_config.TextColumn(),
                "notes": st.column_config.TextColumn(label="Notes", pinned=False, width="medium"),
                'pct_chg_from_52wk_high':st.column_config.NumberColumn(label="% from 52wk High²", format='%.1f', width="small"),
                'pct_chg_from_52wk_low':st.column_config.NumberColumn(label="% from 52wk Low", format='%.1f', width="small"),
                'trailing_pe':st.column_config.NumberColumn(label="P/E Trailing", format='%.1f', width="small"),
                'forward_pe':st.column_config.NumberColumn(label="P/E Fwd", format='%.1f', width="small"),
                'trailing_ps':st.column_config.NumberColumn(label="P/S Trailing", format='%.1f', width="small"),
                "Consolidated_Score": st.column_config.NumberColumn(label="Consol Value Score", format='%.1f', width="small"),
                'Growth_Quality': st.column_config.NumberColumn(label="Growth_Quality", format='%.1f', width="small"),
                'Recent_Momentum': st.column_config.NumberColumn(label="Recent_Momentum", format='%.1f', width="small"),
                'Stability_Trend':  st.column_config.NumberColumn(label="Stability_Trend", format='%.1f', width="small"),
                'Value_Pressure':  st.column_config.NumberColumn(label="Value_Pressure", format='%.1f', width="small"),
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

    # selection = ss.get("row_select", {})
    # selected_rows = selection.get("rows", [])
    # if selected_rows:
    #     row_idx = selected_rows[0]
    #     row = styled.iloc[row_idx]
    #     cik = row["cik"]
    #     ss.selected_company=cik
    #     # name = row["company_and_ticker"]
           
    if ss.selected_company is not None:
        cik = ss.selected_company
        # display_stock_analysis_form()
        show_regression_charts(cik)


    with color_button('gray'):
        return_menu2_btn=st.button('Return to Menu ')
    if return_menu2_btn:
        reset_forms_ss_vars()
        st.rerun()

# set page config and title
st.set_page_config( page_title="Stock Screener", layout="wide" )
st.markdown('<h2 style="color:#3894f0;">Stock Screener for Publically Traded Stocks</h2>', unsafe_allow_html=True)
st.write('Created by Rafael Avila leveraging Streamlit and Postgres, using SEC Filings data provided by SEC Edgar platform and Stock data from Yahoo Finance.')

def main():

# features to build:
# stock charts & some specific timeframes (a day ago, week ago, month ago, 6m ago, 1yr ago, 5yr ago)
# provide info on insider transactions
# provide info on analyst estimate revisions
# enable multiple users

# completed features:
# update of yahoo data for selected companies - completed 3/9/26
# updated charts to use st.write for title to avoid getting them cut off - completed 3/9/26
# Fixed bug causing stocks to be associated to non-primary ticker if multiple tickers exist - completed 3/9/26
# create & refine consolidated value score - completed 3/11/26
# enable charts with/without outliers - completed 3/11/26
# update bought/sold shares and provide summary of rate of return - completed 3/16/26
# look for new SEC filings, perform incremental updates  - completed 3/16/26

    load_full_sec_btn = load_incremental_sec_btn = investment_returns_btn = process_yahoo_and_stats_btn = view_stock_analysis_form_btn = qtr_data_btn = return_menu_btn = False

    if ss.hide_menu==False:
        st.write("Choose an action:")
        with color_button("blue"):
            view_stock_analysis_form_btn = st.button("View Stock Data, Update Categories, Enter Stock Transactions",key='view_data_btn')
        with color_button("blue"):
            load_incremental_sec_btn = st.button("Load Incremental Financial Data from SEC (All Companies)")
        with color_button("green"):
            investment_returns_btn = st.button("Show Investment Returns - TBD")
        with color_button("red"):
            load_full_sec_btn = st.button("Load Full Historical Financial Data from SEC (All Companies) - takes 2+ hours")
        with color_button("red"):
            process_yahoo_and_stats_btn = st.button("Get Yahoo Stock Data and Run Statistical Analysis (All Companies) - takes 30 mins")
        with color_button("red"):
            qtr_data_btn = st.button("Show Quarterly Financial Data")

    else:
        with color_button('gray'):
            return_menu_btn=st.button('Return to Menu')
        
    # For the menu button click, change the session state form and rerun
    if return_menu_btn:
        reset_forms_ss_vars()
        st.rerun()

    if view_stock_analysis_form_btn:
        reset_forms_ss_vars()
        ss.view_stock_analysis_form=True
        ss.hide_menu=True
        st.rerun()
        
    if load_incremental_sec_btn:
        reset_forms_ss_vars()
        ss.load_sec_incremental_filings = True
        write_sec_data_into_db('incremental')
        
    if ss.load_sec_incremental_filings == True:
        ss.rankings_df=load_stock_growth_analysis_data_from_db()
        st.write("Incremental Load Completed Successfully for New SEC Filings")
        reset_forms_ss_vars()
        ss.view_stock_analysis_form=True
        ss.hide_menu=True
        st.stop()

    if load_full_sec_btn:
        reset_forms_ss_vars()
        ss.load_sec_full_filings = True
        write_sec_data_into_db('full')

    if ss.load_sec_full_filings == True:
        ss.rankings_df=load_stock_growth_analysis_data_from_db()
        st.write("Full Load Completed Successfully for SEC Filings")
        reset_forms_ss_vars()
        ss.view_stock_analysis_form=True
        ss.hide_menu=True
        st.stop()
        
    if investment_returns_btn:
        reset_forms_ss_vars()
        ss.investment_returns_form=True
        ss.hide_menu=True
        st.rerun()
    
    if process_yahoo_and_stats_btn:
        reset_forms_ss_vars()
        ss.process_yahoo_and_statistics=True
        st.rerun()

    if qtr_data_btn:
        ss.qtr_data_form=True
        st.rerun()
    
    # This is the actual logic run by the menu buttons
    if ss.view_stock_analysis_form==True:
        if ss.show_transaction_form==False:
            stock_growth_analysis_df = load_stock_growth_analysis_data_from_db()
            display_stock_analysis_form(stock_growth_analysis_df)
        else:
            enter_stock_transaction()
        
    if ss.investment_returns_form==True:
        show_investment_returns()

    if ss.process_yahoo_and_statistics==True:
        # column_specs=get_column_specs()
        with st.spinner(f"Loadiing SEC Data"):
            ss.quarterly_financials = load_quarterly_sec_data_from_db()
        # st.write(companies_data['0000353184'])  # debug
        
        # Analyze and rank companies by growth consistency
        if ss.quarterly_financials is not None and not ss.quarterly_financials.empty:
            # st.markdown("---")
            st.subheader("📊 Growth Consistency Analysis")
            
            if ss.results_df.empty:
                # st.write('Test') #debug
                cik_list=ss.quarterly_financials['cik'].unique() #{'0000066740','0001368514'}#,'0001070494','0000318306','0001018724','0000789019','0000002488','0001045810','0000034782','0001652044','0000320193','0001018724','0001326801','0001730168','0001318605','0001067983','0000059478'} #debug
                # cik_list = cik_list[:20] #debug run just top 20 companies
                
                ss.results_df = rank_companies_by_growth_and_update_DB(cik_list)
                # Highlight top performer
                top_company = ss.results_df.iloc[0]
                # st.dataframe(top_company) # debug
                st.write(f"🏆 Top Performer: **{top_company['company_and_ticker']}** (Score: {top_company['Revenue_Consistency_Score']:.1f})")
            else:
                st.write('Collected data from Yahoo and Processed & Saved Regression Analysis')
                # with st.spinner("Loading Analysis Summary from DB"):
                #     stock_growth_analysis_df=load_stock_growth_analysis_data_from_db()
                # display_stock_analysis_form(stock_growth_analysis_df)
        
        with st.spinner("Loading Analysis Summary from DB"):
            stock_growth_analysis_df=load_stock_growth_analysis_data_from_db()
        reset_forms_ss_vars()
        ss.hide_menu=True
        ss.view_stock_analysis_form=True
        ss.process_yahoo_and_statistics=False
        st.rerun()

    if ss.qtr_data_form:
        quarterly_df = load_quarterly_sec_data_from_db()
        st.dataframe(quarterly_df)
                    
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
        st.dataframe(quarterly_df[cols],width='stretch')
                
main()
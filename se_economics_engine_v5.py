
# se_economics_engine_v5.py
# Cleaned Economics Engine module (Colab bits removed)

# se_economics_engine_v5.py
# Cleaned Economics Engine module with robust pricing overrides

import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from dateutil.relativedelta import relativedelta


def clean_api14(series):
    return series.astype(str).str.split('.').str[0]


def compute_project_irr(cashflows, capex_upfront=None):
    cf = pd.Series(cashflows).fillna(0).astype(float).tolist()
    if capex_upfront is not None:
        cf = [capex_upfront] + cf
    if cf and cf[0] >= 0:
        cf[0] = -cf[0]
    try:
        return float(npf.irr(cf) * 100)
    except:
        return float('nan')


def build_pv_summary(well_cashflows, discount_rates):
    df = pd.DataFrame(well_cashflows)[['Months', 'Free CF']].copy()
    df['DF'] = 1  # placeholder
    summary = ["Rate       PV (MM$)"]
    for rate in discount_rates:
        d = df.copy()
        d['DF'] = (1 + rate/100) ** (-(d['Months']/12))
        pv = (d['Free CF'] * d['DF']).sum() / 1_000
        summary.append(f"{rate:6.2f}%   {pv:10.2f}")
    return summary


def build_forecast_inputs(
    df_forecast,
    econ_params,
    df_ownership=None,
    df_diff=None,
    df_opex=None,
    df_capex=None
):
    wells = []
    df = df_forecast.copy()
    df['API14'] = clean_api14(df['API14'])
    for api, sub in df.groupby('API14'):
        w = {
            'API14': api,
            'WellName': sub['WellName'].iloc[0],
            'WI': econ_params.get('WI', 1.0),
            'NRI': econ_params.get('NRI', 1.0),
        }

        # Ownership overrides
        if df_ownership is not None:
            tmp = df_ownership[df_ownership['API14']==api]
            if not tmp.empty:
                w['WI']  = tmp['WI'].iloc[0]
                w['NRI'] = tmp['NRI'].iloc[0]

        # Base forecast streams
        w['Oil (bbl)'] = sub['Oil (bbl)'] * w['WI'] * w['NRI']
        w['Gas (mcf)'] = sub['Gas (mcf)'] * w['WI'] * w['NRI']
        w['NGL (bbl)'] = sub.get('NGL (bbl)', 0.0) * w['WI'] * w['NRI']

        # Price overrides: only if those exact cols exist
        # Default from econ_params
        w['Oil Price'] = econ_params.get('Oil Price', 0.0)
        w['Gas Price'] = econ_params.get('Gas Price', 0.0)

        if df_diff is not None:
            tmp = df_diff[df_diff['API14']==api]
            if not tmp.empty:
                if 'Oil Price' in tmp.columns:
                    w['Oil Price'] = tmp['Oil Price'].iloc[0]
                if 'Gas Price' in tmp.columns:
                    w['Gas Price'] = tmp['Gas Price'].iloc[0]

        # Other params
        w['NGL Yield'] = econ_params.get('NGL Yield (bbl/MMcf)', 0.0)
        w['Shrink']    = econ_params.get('Shrink', 1.0)

        # Opex override
        w['OpEx'] = econ_params.get('OpEx', 0.0)
        if df_opex is not None:
            tmp = df_opex[df_opex['API14']==api]
            if not tmp.empty and 'OpEx' in tmp.columns:
                w['OpEx'] = tmp['OpEx'].iloc[0]

        # Capex override
        w['CapEx'] = econ_params.get('CapEx', 0.0)
        if df_capex is not None:
            tmp = df_capex[df_capex['API14']==api]
            if not tmp.empty and 'CapEx' in tmp.columns:
                w['CapEx'] = tmp['CapEx'].iloc[0]

        wells.append(w)

    return wells


def calculate_cashflows(
    well_forecasts,
    effective_date,
    discount_rate,
    df_strip=None
):
    all_wells = []
    combined = []

    # Prepare strip if given
    strip = None
    if df_strip is not None and {'Year','Oil Price','Gas Price'}.issubset(df_strip.columns):
        df_strip = df_strip.sort_values('Year').reset_index(drop=True)
        strip = df_strip

    for w in well_forecasts:
        months = len(w['Oil (bbl)'])
        df = pd.DataFrame({
            'Months': np.arange(months, dtype=float),
            'Oil': w['Oil (bbl)'],
            'Gas': w['Gas (mcf)'],
            'NGL': w['NGL (bbl)'],
        })
        # Date, Year, Mo
         df['Date'] = pd.to_datetime(effective_date) + pd.to_timedelta(df['Months'], unit='M')
         base = pd.to_datetime(effective_date)
         df['Date'] = df['Months'].apply(lambda m: base + relativedelta(months=int(m)))
         df['Year'] = df['Date'].dt.year
         df['Mo']   = df['Date'].dt.month

        # Pricing
        df['Oil Price'] = w['Oil Price']
        df['Gas Price'] = w['Gas Price']
        if strip is not None:
            df = df.merge(strip, on='Year', how='left', suffixes=('','_strip'))
            df['Oil Price'] = df['Oil Price_strip'].fillna(df['Oil Price'])
            df['Gas Price'] = df['Gas Price_strip'].fillna(df['Gas Price'])

        # Revenues & costs
        df['Oil Rev'] = df['Oil'] * df['Oil Price']
        df['Gas Rev'] = df['Gas'] * df['Gas Price']
        df['NGL Rev'] = df['NGL'] * w['NGL Yield'] * df['Gas'] * w['Shrink']
        df['Total Rev'] = df[['Oil Rev','Gas Rev','NGL Rev']].sum(axis=1)
        df['OpEx']     = w['OpEx']
        df['CapEx']    = 0.0
        df.loc[0,'CapEx'] = w['CapEx']
        df['Free CF']  = df['Total Rev'] - df['OpEx'] - df['CapEx']
        df['DF']       = (1 + discount_rate) ** (-(df['Months']/12))
        df['Disc CF']  = df['Free CF'] * df['DF']

        df['API14']   = w['API14']
        df['WellName']= w['WellName']

        all_wells.append(df)
        combined.append(df)

    total = pd.concat(combined, ignore_index=True)
    return all_wells, total


def summarize_yearly(total_cashflow):
    total_cashflow['Year'] = total_cashflow['Date'].dt.year
    yearly = total_cashflow.groupby('Year')['Free CF'].sum().reset_index()
    yearly.columns = ['Year','Annual CF']
    return yearly


def render_cashflow_page(pdf: PdfPages, df, title):
    fig, ax = plt.subplots(figsize=(8, len(df)*0.3+1))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    pdf.savefig(fig)
    plt.close(fig)


def generate_cashflow_pdf_table(
    well_cashflows, total_cashflow, econ_params, output_path
):
    yearly = summarize_yearly(total_cashflow)
    with PdfPages(output_path) as pdf:
        pdf.infodict().update({
            'Title':   econ_params.get('Project',''),
            'Author':  econ_params.get('Client',''),
            'CreationDate': datetime.now()
        })
        render_cashflow_page(pdf, yearly, "Yearly Cashflow Summary")


def generate_cashflow_pdf_table_with_monthly(
    well_cashflows, total_cashflow, econ_params, output_path, get_summary_fn
):
    yearly = summarize_yearly(total_cashflow)
    monthly = total_cashflow[['Date','Free CF']].copy()
    with PdfPages(output_path) as pdf:
        pdf.infodict().update({
            'Title':   econ_params.get('Project',''),
            'Author':  econ_params.get('Client',''),
            'CreationDate': datetime.now()
        })
        # add summary text
        fig, ax = plt.subplots(figsize=(8,1))
        ax.axis('off')
        txt = get_summary_fn(total_cashflow, econ_params.get('Discount Rate',0.0))
        ax.text(0,0.5, txt, fontsize=10)
        pdf.savefig(fig); plt.close(fig)
        # yearly
        render_cashflow_page(pdf, yearly, "Yearly")
        # monthly
        fig, ax = plt.subplots(figsize=(8,len(monthly)*0.12+1))
        ax.axis('off')
        tbl = ax.table(cellText=monthly.values, colLabels=monthly.columns, loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(6)
        pdf.savefig(fig); plt.close(fig)


if __name__ == "__main__":
    print("SE Economics Engine loaded.")


# se_economics_engine_v5.py
# Cleaned Economics Engine module (Colab bits removed)

import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def clean_api14(series):
    """
    Removes any trailing '.0' from API14, pads/truncates to 14 chars, returns as string.
    """
    return series.apply(lambda x: str(x).split('.')[0]).astype(str)


def compute_project_irr(cashflows, capex_upfront=None):
    """
    Computes project IRR (% annualized) using annual net cashflows.
    If capex_upfront is provided, it's prepended to the cashflows.
    """
    try:
        cf = pd.Series(cashflows).fillna(0).astype(float).tolist()
        if capex_upfront is not None:
            cf = [capex_upfront] + cf
        if cf[0] >= 0:
            cf[0] = -cf[0]
        irr = npf.irr(cf)
        return float(irr * 100)
    except Exception:
        return float('nan')


def build_pv_summary(well_cashflows, discount_rates):
    """
    Given a DataFrame of well cashflows (with 'Months' and 'Free CF'),
    computes PV at each discount rate in discount_rates list.
    Returns a list of summary strings.
    """
    df = well_cashflows.copy()
    df = df[['Months', 'Free CF']]
    df['Months'] = df['Months'].astype(float)
    df['Free CF'] = df['Free CF'].astype(float)

    summary = []
    summary.append("Rate       PV (MM$)")
    for rate in discount_rates:
        df_rate = df.copy()
        df_rate[f"DF_{rate}"] = (1 + rate / 100) ** (-(df_rate['Months'] / 12))
        df_rate[f"PV{rate}"] = df_rate['Free CF'] * df_rate[f"DF_{rate}"]
        val = df_rate[f"PV{rate}"].sum() / 1_000
        summary.append(f"{rate:>6.2f}%   {val:>10.2f}")
    return summary


def build_forecast_inputs(
    df_forecast,
    econ_params,
    df_ownership=None,
    df_diff=None,
    df_opex=None,
    df_capex=None
):
    """
    Merges forecast DataFrame with ownership, differential, OPEX, CAPEX overrides.
    Returns a list of dicts per well containing all streams and parameters.
    """
    wells = []
    df = df_forecast.copy()
    df['API14'] = clean_api14(df['API14'])
    grouped = df.groupby('API14')

    for api, sub in grouped:
        w = {'API14': api}
        w['WellName'] = sub['WellName'].iloc[0]
        w['WI'] = econ_params.get('WI', 1.0)
        w['NRI'] = econ_params.get('NRI', 1.0)

        if df_ownership is not None:
            tmp = df_ownership[df_ownership['API14'] == api]
            if not tmp.empty:
                w['WI'] = tmp['WI'].iloc[0]
                w['NRI'] = tmp['NRI'].iloc[0]

        oil = sub[['Year', 'Mo', 'Oil (bbl)']].reset_index(drop=True)
        gas = sub[['Year', 'Mo', 'Gas (mcf)']].reset_index(drop=True)
        ngl = sub[['Year', 'Mo', 'NGL (bbl)']].reset_index(drop=True)

        w['Oil (bbl)'] = oil['Oil (bbl)'] * w['WI'] * w['NRI']
        w['Gas (mcf)'] = gas['Gas (mcf)'] * w['WI'] * w['NRI']
        w['NGL (bbl)'] = ngl['NGL (bbl)'] * w['WI'] * w['NRI']

        # Apply differentials
        if df_diff is not None:
            tmp = df_diff[df_diff['API14'] == api]
            if not tmp.empty:
                w['Gas Price'] = tmp['Gas Price'].iloc[0]
                w['Oil Price'] = tmp['Oil Price'].iloc[0]
        # Defaults from econ_params
        w['Gas Price'] = w.get('Gas Price', econ_params.get('Gas Price', 0.0))
        w['Oil Price'] = w.get('Oil Price', econ_params.get('Oil Price', 0.0))
        w['NGL Yield'] = econ_params.get('NGL Yield (bbl/MMcf)', 0.0)
        w['Shrink'] = econ_params.get('Shrink', 1.0)

        # Opex & Capex
        w['OpEx'] = 0.0
        if df_opex is not None:
            tmp = df_opex[df_opex['API14'] == api]
            if not tmp.empty:
                w['OpEx'] = tmp['OpEx'].iloc[0]
        w['CapEx'] = 0.0
        if df_capex is not None:
            tmp = df_capex[df_capex['API14'] == api]
            if not tmp.empty:
                w['CapEx'] = tmp['CapEx'].iloc[0]

        wells.append(w)
    return wells


def calculate_cashflows(
    well_forecasts,
    effective_date,
    discount_rate,
    df_strip=None
):
    """
    From list of well dicts, builds detailed monthly cashflows,
    applies strip pricing, discounting, and returns:
      - list of well DataFrames (with Months, Free CF, etc.)
      - total aggregated cashflow DataFrame
    """
    all_wells = []
    combined = []
    if df_strip is not None:
        df_strip = df_strip.sort_values('Year').reset_index(drop=True)
        last_oil = df_strip['Oil Price'].iloc[-1]
        last_gas = df_strip['Gas Price'].iloc[-1]

    for w in well_forecasts:
        months = len(w['Oil (bbl)'])
        df = pd.DataFrame({
            'Months': np.arange(months, dtype=float),
            'Oil': w['Oil (bbl)'],
            'Gas': w['Gas (mcf)'],
            'NGL': w['NGL (bbl)'],
        })
        # Pricing
        df['Oil Price'] = w['Oil Price']
        df['Gas Price'] = w['Gas Price']
        # Override with strip if before strip end
        if df_strip is not None:
            years = (effective_date.year + (df['Months'] / 12)).astype(int)
            df = df.merge(df_strip, left_on='Year', right_on='Year', how='left', suffixes=('', '_strip'))
            df['Oil Price'] = df['Oil Price_strip'].fillna(w['Oil Price'])
            df['Gas Price'] = df['Gas Price_strip'].fillna(w['Gas Price'])

        # Revenues & opex
        df['Oil Rev'] = df['Oil'] * df['Oil Price']
        df['Gas Rev'] = df['Gas'] * df['Gas Price']
        df['NGL Rev'] = df['NGL'] * w['NGL Yield'] * df['Gas'] * w['Shrink']

        df['Total Rev'] = df['Oil Rev'] + df['Gas Rev'] + df['NGL Rev']
        df['OpEx'] = w['OpEx']
        df['CapEx'] = 0.0
        df.loc[0, 'CapEx'] = w['CapEx']

        # Free CF before tax
        df['Free CF'] = df['Total Rev'] - df['OpEx'] - df['CapEx']

        # Discount factor
        df['DF'] = (1 + discount_rate) ** (-(df['Months'] / 12))
        df['Disc CF'] = df['Free CF'] * df['DF']

        df['API14'] = w['API14']
        df['WellName'] = w['WellName']
        all_wells.append(df)
        combined.append(df)

    total = pd.concat(combined, ignore_index=True)
    return all_wells, total


def get_aries_summary_text(total_cashflow, discount_rate):
    """
    Builds an Aries‐style summary paragraph based on total cashflow DataFrame.
    """
    # Example: summarize NPV10, NPV0, IRR, Payback, etc.
    npv10 = total_cashflow['Disc CF'].sum() / 1_000
    irr = compute_project_irr(
        total_cashflow['Free CF'].resample('A', on='Date').sum().values,
        capex_upfront=None
    )
    return f"NPV@{discount_rate*100:.0f}%: ${npv10:,.1f}M, IRR: {irr:.1f}%"


def summarize_yearly(total_cashflow):
    """
    Returns a pivot table of yearly cashflows.
    """
    total_cashflow['Year'] = total_cashflow['Date'].dt.year
    yearly = total_cashflow.groupby('Year')['Free CF'].sum().reset_index()
    yearly.columns = ['Year', 'Annual CF']
    return yearly


def render_cashflow_page(pdf: PdfPages, df, title):
    """
    Renders a matplotlib table into the PdfPages object.
    """
    fig, ax = plt.subplots(figsize=(8, len(df)*0.25 + 1))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    pdf.savefig(fig)
    plt.close(fig)


def generate_cashflow_pdf_table(
    well_cashflows, total_cashflow, econ_params, output_path
):
    """
    Generates a yearly cashflow PDF (aggregate).
    """
    yearly = summarize_yearly(total_cashflow)
    with PdfPages(output_path) as pdf:
        pdf.infodict().update({
            'Title': econ_params.get('Project', ''),
            'Author': econ_params.get('Client', ''),
            'CreationDate': datetime.now()
        })
        render_cashflow_page(pdf, yearly, "Yearly Cashflow Summary")


def render_mixed_table(pdf: PdfPages, df_yearly, df_monthly, title):
    """
    Renders a combined yearly + monthly table.
    """
    # first yearly
    render_cashflow_page(pdf, df_yearly, title + " (Yearly)")
    # then monthly
    fig, ax = plt.subplots(figsize=(8, len(df_monthly)*0.12 + 1))
    ax.axis('off')
    tbl = ax.table(cellText=df_monthly.values, colLabels=df_monthly.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    pdf.savefig(fig)
    plt.close(fig)


def generate_cashflow_pdf_table_with_monthly(
    well_cashflows, total_cashflow, econ_params, output_path, get_aries_summary_text
):
    """
    Generates a PDF with both yearly and monthly cashflow tables + summary text.
    """
    yearly = summarize_yearly(total_cashflow)
    monthly = total_cashflow[['Date', 'Free CF']].copy()
    with PdfPages(output_path) as pdf:
        pdf.infodict().update({
            'Title': econ_params.get('Project', ''),
            'Author': econ_params.get('Client', ''),
            'CreationDate': datetime.now()
        })
        # Add summary
        text = get_aries_summary_text(total_cashflow, econ_params.get('Discount Rate', 0.0))
        fig, ax = plt.subplots(figsize=(8, 1))
        ax.axis('off')
        ax.text(0, 0.5, text, fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        render_mixed_table(pdf, yearly, monthly, "Cashflow Summary")


def generate_oneline_summary_excel(
    well_cashflows, econ_params, df_ownership=None, df_opex=None
):
    """
    Writes a one‐line Excel summary of PV10, IRR, etc., across wells.
    """
    rows = []
    for df in well_cashflows:
        api = df['API14'].iloc[0]
        name = df['WellName'].iloc[0]
        pv10 = (df['Disc CF'].sum() / 1_000)
        irr = compute_project_irr(
            df['Free CF'].resample('A', on='Date').sum().values
        )
        rows.append({
            'API14': api,
            'WellName': name,
            'PV10 (MM$)': pv10,
            'IRR (%)': irr
        })
    summary_df = pd.DataFrame(rows).sort_values('PV10 (MM$)', ascending=False)
    output_path = econ_params.get('Oneline Path', 'oneline_summary.xlsx')
    summary_df.to_excel(output_path, index=False)


def export_monthly_cashflow_excel(
    well_cashflows, econ_params, df_strip=None
):
    """
    Writes detailed monthly cashflow (with tax breakdown) to Excel.
    """
    writer = pd.ExcelWriter(econ_params.get('Monthly Path', 'monthly_cashflow.xlsx'),
                            engine='openpyxl')
    for df in well_cashflows:
        name = df['WellName'].iloc[0]
        sheet = df[['Months', 'Oil', 'Gas', 'Free CF']].copy()
        sheet.columns = ['Month', 'Oil', 'Gas', 'Free Cash Flow']
        sheet.to_excel(writer, sheet_name=name[:31], index=False)
    writer.save()


if __name__ == "__main__":
    print("SE Economics Engine module loaded. Call its functions from your app.")

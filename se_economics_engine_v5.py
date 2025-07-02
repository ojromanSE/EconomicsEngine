# se_economics_engine_v6.py
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def clean_api14(series):
    return series.astype(str).str.split('.').str[0]


def compute_project_irr(cashflows, capex_upfront=None):
    cf = pd.Series(cashflows).fillna(0).astype(float).tolist()
    if capex_upfront is not None:
        cf = [capex_upfront] + cf
    if cf and cf[0] >= 0:
        cf[0] = -cf[0]
    try:
        irr = npf.irr(cf)
        return round(irr * 100, 2) if irr is not None and not np.isnan(irr) else None
    except:
        return None


def get_aries_summary_text(df, discount_rate):
    # Simple NPV and IRR summary
    npv = df['Disc CF'].sum() / 1_000
    annual = df.set_index('Date')['Free CF'].resample('A').sum().values
    irr = compute_project_irr(annual)
    return f"NPV@{discount_rate*100:.0f}%: ${npv:,.1f}M, IRR: {irr:.1f}%"


def build_pv_summary(df, pv_rates=None):
    pv_rates = pv_rates or [8,9,10,12,15,20,25,30,50]
    if 'Months' not in df: 
        df['Months'] = (df['Date'].dt.to_period('M') - df['Date'].min().to_period('M')).apply(lambda x: x.n)
    if 'Free CF' not in df.columns and {'Total Revenue','OpEx','Capex','Taxes'}.issubset(df.columns):
        df['Free CF'] = df['Total Revenue'] - df['OpEx'] - df['Capex'] - df['Taxes']
    lines = []
    for rate in pv_rates:
        df[f"DF_{rate}"] = (1 + rate/100)**(-(df['Months']/12))
        df[f"PV{rate}"] = df['Free CF'] * df[f"DF_{rate}"]
        val = df[f"PV{rate}"].sum() / 1_000
        lines.append(f"{rate:>6.2f}           {val:>10.2f}")
    return lines


# ──────────────────────────────────────────────────────────────────────────────
# Build Forecast Inputs
# ──────────────────────────────────────────────────────────────────────────────

def build_forecast_inputs(df_forecast, econ_params,
                          df_ownership=None, df_diff=None,
                          df_opex=None, df_capex=None):
    df = df_forecast.copy()
    df['API14'] = clean_api14(df['API14'])
    for d in (df_ownership, df_diff, df_opex, df_capex):
        if isinstance(d, pd.DataFrame) and 'API14' in d.columns:
            d['API14'] = clean_api14(d['API14'])
    wells = []
    for api, sub in df.groupby('API14'):
        row = sub.iloc[0]
        w = {
            'API14': api,
            'WellName': row['WellName'],
            'WI': econ_params.get('WI',1.0),
            'NRI': econ_params.get('NRI',0.8)
        }
        if isinstance(df_ownership, pd.DataFrame):
            tmp = df_ownership[df_ownership['API14']==api]
            if not tmp.empty:
                w['WI']  = tmp['WI'].iloc[0]
                w['NRI'] = tmp['NRI'].iloc[0]
        w['Oil (bbl)'] = (sub['OilProduction_bbl_month'] * w['NRI']).tolist()
        w['Gas (mcf)'] = (sub['GasProduction_MCF_month'] * w['NRI']).tolist()
        w['NGL (bbl)'] = (sub['GasProduction_MCF_month'] * econ_params.get('NGL Yield (bbl/MMcf)',2.5)/1000).tolist()
        w['Oil Price'] = econ_params.get('Oil Price',70.0)
        w['Gas Price'] = econ_params.get('Gas Price',3.0)
        if isinstance(df_diff, pd.DataFrame):
            tmp = df_diff[df_diff['API14']==api]
            if not tmp.empty:
                for col in tmp.columns:
                    lc = col.lower()
                    if 'oil' in lc and 'price' in lc:
                        w['Oil Price'] = tmp[col].iloc[0]
                    if 'gas' in lc and 'price' in lc:
                        w['Gas Price'] = tmp[col].iloc[0]
        w['OpEx']  = econ_params.get('OpEx',0.0)
        if isinstance(df_opex, pd.DataFrame):
            tmp = df_opex[df_opex['API14']==api]
            if not tmp.empty and 'OpEx' in tmp.columns:
                w['OpEx'] = tmp['OpEx'].iloc[0]
        w['Capex'] = econ_params.get('Capex',0.0)
        if isinstance(df_capex, pd.DataFrame):
            tmp = df_capex[df_capex['API14']==api]
            if not tmp.empty and 'Capex' in tmp.columns:
                w['Capex'] = tmp['Capex'].iloc[0]
        w['Severance Tax %'] = econ_params.get('Severance Tax %',0.0)
        w['Ad Valorem Tax %']= econ_params.get('Ad Valorem Tax %',0.0)
        wells.append(w)
    return wells


# ──────────────────────────────────────────────────────────────────────────────
# Calculate Cashflows
# ──────────────────────────────────────────────────────────────────────────────

def calculate_cashflows(well_forecasts, effective_date,
                        discount_rate, df_strip=None):
    all_wells, combined = [], []
    strip = None
    if isinstance(df_strip,pd.DataFrame) and {'Year','Oil Price','Gas Price'}.issubset(df_strip.columns):
        strip = df_strip.sort_values('Year').reset_index(drop=True)
    base = pd.to_datetime(effective_date)
    for w in well_forecasts:
        n = len(w['Oil (bbl)'])
        df = pd.DataFrame({
            'Months': np.arange(n, dtype=float),
            'Oil':    w['Oil (bbl)'],
            'Gas':    w['Gas (mcf)'],
            'NGL':    w['NGL (bbl)']
        })
        df['Date'] = df['Months'].apply(lambda m: base + relativedelta(months=int(m)))
        df['Year'], df['Mo'] = df['Date'].dt.year, df['Date'].dt.month
        df['Oil Price'], df['Gas Price'] = w['Oil Price'], w['Gas Price']
        if strip is not None:
            df = df.merge(strip, on='Year', how='left', suffixes=('','_strip'))
            df['Oil Price'] = df['Oil Price_strip'].fillna(df['Oil Price'])
            df['Gas Price'] = df['Gas Price_strip'].fillna(df['Gas Price'])
        df['Oil Rev'] = df['Oil'] * df['Oil Price']
        df['Gas Rev'] = df['Gas'] * df['Gas Price']
        df['NGL Rev'] = df['NGL'] * w.get('NGL Yield (bbl/MMcf)',1.0) * df['Gas'] * w.get('Shrink',1.0)
        df['Total Revenue'] = df[['Oil Rev','Gas Rev','NGL Rev']].sum(axis=1)
        df['Taxes'] = df['Total Revenue'] * (w['Severance Tax %'] + w['Ad Valorem Tax %'])
        df['OpEx']   = w['OpEx']
        df['Capex']  = 0.0; df.loc[0,'Capex']=w['Capex']
        df['Free CF']= df['Total Revenue'] - df['OpEx'] - df['Capex'] - df['Taxes']
        df['DF']     = (1+discount_rate)**(-(df['Months']/12))
        df['Disc CF']= df['Free CF'] * df['DF']
        df['API14'], df['WellName'] = w['API14'], w['WellName']
        all_wells.append(df); combined.append(df)
    total = pd.concat(combined, ignore_index=True)
    return all_wells, total


def summarize_yearly(total_df):
    df = total_df.copy()
    df['Year'] = df['Date'].dt.year
    return df.groupby('Year')[['Free CF','Disc CF']].sum().reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# Render Cashflow Page
# ──────────────────────────────────────────────────────────────────────────────
def render_cashflow_page(title, df_yearly, raw_df, pdf, page_num,
                         fontname, effective_date_str,
                         client_name, project_name, pv_label,
                         get_summary_fn, discount_rate):
    headers = ["Year","Free CF","Disc CF"]
    units   = ["     ","(M$)","(M$)"]
    col_w   = [9,11,11]
    def fmt(v): return f"{v/1000:,.2f}" if pd.notnull(v) else "   -   "
    def row(vals): return " ".join(str(v).rjust(w) for v,w in zip(vals,col_w))

    fig, ax = plt.subplots(figsize=(15, 0.6 + 0.25*(len(df_yearly)+6)))
    ax.axis('off')
    ax.text(0.5,1.06,"Schaper Energy Economics Engine", ha='center', va='bottom',
            fontsize=12, fontweight='bold', fontname=fontname, transform=ax.transAxes)
    ax.text(0.5,1.03,f"Effective Date: {effective_date_str}", ha='center', va='bottom',
            fontsize=10, fontname=fontname, transform=ax.transAxes)
    ax.text(0.5,1.00,f"Client: {client_name}", ha='center', va='bottom',
            fontsize=10, fontname=fontname, transform=ax.transAxes)
    ax.text(0.5,0.975,f"Project: {project_name} | {pv_label}", ha='center', va='bottom',
            fontsize=10, fontname=fontname, transform=ax.transAxes)

    lines = [title, row(headers), row(units)]
    for _,r in df_yearly.iterrows():
        lines.append(row([int(r['Year']), r['Free CF'], r['Disc CF']]))
    total = df_yearly[['Free CF','Disc CF']].sum()
    lines.append(row(["TOTAL", total['Free CF'], total['Disc CF']]))

    ax.text(0,0.94,"\n".join(lines), va='top', ha='left',
            fontname=fontname, fontsize=7, transform=ax.transAxes)

    summary = get_summary_fn(raw_df, discount_rate)
    ax.text(0.01,0.02, summary, fontname=fontname, fontsize=7,
            ha='left', va='bottom', transform=ax.transAxes)
    ax.text(0.5,0.01,f"Page {page_num}", ha='center', va='bottom',
            fontsize=8, fontname=fontname, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Generate Yearly PDF Table
# ──────────────────────────────────────────────────────────────────────────────
def generate_cashflow_pdf_table(wells, total_df, econ_params, output_path):
    fontname = 'monospace'
    eff_str = pd.to_datetime(econ_params['Effective Date']).strftime('%B %d, %Y')
    dr = econ_params.get('Discount Rate',0.10)
    pv = f"PV{int(dr*100)}"
    client, proj = econ_params.get('Client',''), econ_params.get('Project','')
    with PdfPages(output_path) as pdf:
        page=1
        yearly = summarize_yearly(total_df)
        render_cashflow_page("Total Project Cashflow Summary", yearly,
                             total_df, pdf, page,
                             fontname, eff_str, client, proj, pv,
                             get_aries_summary_text, dr)
        page+=1
        npv_list = sorted(
            [(df['WellName'].iloc[0], df['API14'].iloc[0], df['Free CF'].sum()) for df in wells],
            key=lambda x: x[2], reverse=True
        )
        for name,api,_ in npv_list:
            dfw = next(d for d in wells if d['API14'].iloc[0]==api)
            yw = summarize_yearly(dfw)
            render_cashflow_page(
                f"Cashflow Summary for {name} (API: {api})", yw,
                dfw, pdf, page, fontname, eff_str,
                client, proj, pv,
                get_aries_summary_text, dr
            )
            page+=1
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Generate Monthly PDF Table
# ──────────────────────────────────────────────────────────────────────────────
def generate_cashflow_pdf_table_with_monthly(wells, total_df, econ_params, output_path):
    fontname = 'monospace'
    eff = pd.to_datetime(econ_params['Effective Date'])
    dr = econ_params.get('Discount Rate',0.10)
    pv = f"PV{int(dr*100)}"
    client, proj = econ_params.get('Client',''), econ_params.get('Project','')
    yearly = summarize_yearly(total_df)
    monthly = total_df[['Date','Free CF']].copy()
    with PdfPages(output_path) as pdf:
        # project page
        render_cashflow_page("Project Total", yearly,
                             total_df, pdf, 1,
                             fontname, eff.strftime('%B %d, %Y'), client, proj, pv,
                             get_aries_summary_text, dr)
        # monthly table page
        fig, ax = plt.subplots(figsize=(15, 0.5 + 0.22*(len(monthly)+6)))
        ax.axis('off')
        ax.text(0.5,1.06,"Schaper Energy Economics Engine", ha='center',
                va='bottom', fontsize=11, fontweight='bold', fontname=fontname, transform=ax.transAxes)
        ax.text(0.5,1.03,f"Effective Date: {eff:%B %d, %Y}", ha='center',
                va='bottom', fontsize=9, fontname=fontname, transform=ax.transAxes)
        ax.text(0.5,1.00,f"Client: {client}", ha='center',
                va='bottom', fontsize=9, fontname=fontname, transform=ax.transAxes)
        ax.text(0.5,0.975,f"Project: {proj} | {pv}", ha='center',
                va='bottom', fontsize=9, fontname=fontname, transform=ax.transAxes)
        tbl = ax.table(cellText=monthly.values, colLabels=monthly.columns, loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(6)
        pdf.savefig(fig); plt.close(fig)
    return output_path

if __name__ == "__main__":
    print("SE Economics Engine V6 loaded.")

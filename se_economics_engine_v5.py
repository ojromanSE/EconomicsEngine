# se_economics_engine_v6.py
# Full‐column PDF engine with mixed monthly+annual pages implemented

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
        irr = npf.irr(cf)
        return round(irr * 100, 2) if irr and not np.isnan(irr) else None
    except:
        return None

def build_pv_summary(df, pv_rates=None):
    pv_rates = pv_rates or [8,9,10,12,15,20,25,30,50]
    if 'Months' not in df: 
        df['Months'] = (df['Date'].dt.to_period('M') - df['Date'].min().to_period('M')).apply(lambda x: x.n)
    if 'Free CF' not in df:
        df['Free CF'] = df['Total Revenue'] - df['OpEx'] - df['Capex'] - df['Taxes']
    lines = []
    for rate in pv_rates:
        df[f"DF_{rate}"] = (1+rate/100)**(-(df['Months']/12))
        df[f"PV{rate}"] = df['Free CF'] * df[f"DF_{rate}"]
        val = df[f"PV{rate}"].sum()/1_000
        lines.append(f"{rate:>6.2f}           {val:>10.2f}")
    return lines

def build_forecast_inputs(
    df_forecast, econ_params,
    df_ownership=None, df_diff=None,
    df_opex=None, df_capex=None
):
    """
    Merge forecast with ownership, differential, OPEX & CAPEX overrides.
    Returns list of per-well input dicts.
    """
    df = df_forecast.copy()
    df['API14'] = clean_api14(df['API14'])

    # normalize override API14 columns
    for d in (df_ownership, df_diff, df_opex, df_capex):
        if isinstance(d, pd.DataFrame) and 'API14' in d.columns:
            d['API14'] = clean_api14(d['API14'])

    wells = []
    for api, sub in df.groupby('API14'):
        # prepare series, filling missing streams with zeros
        oil_series   = sub['Oil (bbl)']
        gas_series   = sub['Gas (mcf)']
        ngl_series   = sub['NGL (bbl)'] if 'NGL (bbl)' in sub.columns else pd.Series(0.0, index=sub.index)
        water_series = sub['Water (bbl)'] if 'Water (bbl)' in sub.columns else pd.Series(0.0, index=sub.index)

        row = sub.iloc[0]
        w = {
            'API14':    api,
            'WellName': row['WellName'],
            'WI':       econ_params.get('WI', 1.0),
            'NRI':      econ_params.get('NRI', 1.0)
        }

        # Ownership override
        if isinstance(df_ownership, pd.DataFrame):
            tmp = df_ownership[df_ownership['API14'] == api]
            if not tmp.empty:
                w['WI'], w['NRI'] = tmp[['WI','NRI']].iloc[0]

        # Streams (apply WI & NRI)
        w['Oil (bbl)']   = (oil_series   * w['WI'] * w['NRI']).tolist()
        w['Gas (mcf)']   = (gas_series   * w['WI'] * w['NRI']).tolist()
        w['NGL (bbl)']   = (ngl_series   * w['WI'] * w['NRI']).tolist()
        w['Water (bbl)'] = (water_series * w['WI'] * w['NRI']).tolist()

        # Price defaults & overrides...
        w['Oil Price'] = econ_params.get('Oil Price', 0.0)
        w['Gas Price'] = econ_params.get('Gas Price', 0.0)
        if isinstance(df_diff, pd.DataFrame):
            tmp = df_diff[df_diff['API14'] == api]
            if not tmp.empty:
                for c in tmp.columns:
                    lc = c.lower()
                    if 'oil' in lc and 'price' in lc:
                        w['Oil Price'] = tmp[c].iloc[0]
                    if 'gas' in lc and 'price' in lc:
                        w['Gas Price'] = tmp[c].iloc[0]

        # Other parameters
        w['NGL Yield'] = econ_params.get('NGL Yield (bbl/MMcf)', 0.0)
        w['Shrink']    = econ_params.get('Shrink', 1.0)

        # OpEx & CapEx overrides
        w['OpEx']   = econ_params.get('OpEx', 0.0)
        if isinstance(df_opex, pd.DataFrame):
            tmp = df_opex[df_opex['API14'] == api]
            if not tmp.empty and 'OpEx' in tmp.columns:
                w['OpEx'] = tmp['OpEx'].iloc[0]

        w['Capex']  = econ_params.get('Capex', 0.0)
        if isinstance(df_capex, pd.DataFrame):
            tmp = df_capex[df_capex['API14'] == api]
            if not tmp.empty and 'CapEx' in tmp.columns:
                w['Capex'] = tmp['CapEx'].iloc[0]

        # Taxes
        w['Severance Tax %']  = econ_params.get('Severance Tax %', 0.0)
        w['Ad Valorem Tax %'] = econ_params.get('Ad Valorem Tax %', 0.0)

        wells.append(w)

    return wells


def calculate_cashflows(well_forecasts, effective_date, discount_rate, df_strip=None):
    all_wells, combined = [], []
    strip = None
    if isinstance(df_strip, pd.DataFrame) and {'Year','Oil Price','Gas Price'}.issubset(df_strip):
        strip = df_strip.sort_values('Year').reset_index(drop=True)
    base = pd.to_datetime(effective_date)
    for w in well_forecasts:
        n = len(w['Oil (bbl)'])
        df = pd.DataFrame({
            'Months':np.arange(n),
            'Oil':   w['Oil (bbl)'],
            'Gas':   w['Gas (mcf)'],
            'NGL':   w['NGL (bbl)'],
            'Water': w['Water (bbl)']
        })
        df['Date'] = df['Months'].apply(lambda m: base + relativedelta(months=int(m)))
        df['Year'], df['Mo'] = df['Date'].dt.year, df['Date'].dt.month
        df['Oil Price'], df['Gas Price'] = w['Oil Price'], w['Gas Price']
        if strip is not None:
            df = df.merge(strip,on='Year',how='left',suffixes=('','_strip'))
            if 'Oil Price_strip' in df:
                df['Oil Price'] = df['Oil Price_strip'].fillna(df['Oil Price'])
            if 'Gas Price_strip' in df:
                df['Gas Price'] = df['Gas Price_strip'].fillna(df['Gas Price'])
        df['Oil Rev']   = df['Oil'] * df['Oil Price']
        df['Gas Rev']   = df['Gas'] * df['Gas Price']
        df['NGL Rev']   = df['NGL'] * w['NGL Yield'] * df['Gas'] * w['Shrink']
        df['Total Revenue'] = df[['Oil Rev','Gas Rev','NGL Rev']].sum(axis=1)
        df['Taxes']       = df['Total Revenue'] * (w['Severance Tax %']+w['Ad Valorem Tax %'])
        df['OpEx']        = w['OpEx']
        df['Capex']       = 0.0; df.loc[0,'Capex']=w['Capex']
        df['Free CF']     = df['Total Revenue'] - df['OpEx'] - df['Capex'] - df['Taxes']
        df['DF']          = (1+discount_rate)**(-(df['Months']/12))
        df['Disc CF']     = df['Free CF'] * df['DF']
        df['API14']       = w['API14']
        df['WellName']    = w['WellName']
        all_wells.append(df)
        combined.append(df)
    total = pd.concat(combined,ignore_index=True)
    return all_wells, total

def summarize_yearly(total_cashflow):
    df = total_cashflow.copy()
    df['Year'] = df['Date'].dt.year
    return df.groupby('Year')[['Free CF','Disc CF']].sum().reset_index()

def render_cashflow_page(
    title, df_yearly, raw_df, pdf, page_num,
    fontname, eff_str, client, project, pv_label,
    get_summary_fn
):
    headers = ["Year","Oil Gross","Gas Gross","NGL Gross","Water Gross",
               "Oil Net","Gas Net","NGL Net","Oil $","Gas $","NGL $",
               "Total Rev","Taxes","OpEx","Capex","Free CF","Disc CF"]
    units   = ["(     )","(Mbbl)","(MMcf)","(Mbbl)","(Mbbl)",
               "(Mbbl)","(MMcf)","(Mbbl)","($/bbl)","($/mcf)","($/bbl)",
               "(M$)","(M$)","(M$)","(M$)","(M$)","(M$)"]
    col_w   = [9,10,10,10,10,10,10,10,8,8,8,11,8,8,8,11,11]
    fmt     = lambda v,scale=1000: f"{v/scale:,.2f}" if pd.notnull(v) else "   -   "
    fmtp    = lambda v: f"{v:,.2f}" if pd.notnull(v) else "   -   "
    row     = lambda vals: " ".join(str(v).rjust(w) for v,w in zip(vals,col_w))

    fig, ax = plt.subplots(figsize=(15,0.6+0.25*(len(df_yearly)+6)))
    ax.axis('off')
    ax.text(0.5,1.06,"Schaper Energy Economics Engine",ha='center',va='bottom',
            fontsize=12,fontweight='bold',fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,1.03,f"Effective Date: {eff_str}",ha='center',va='bottom',
            fontsize=10,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,1.00,f"Client: {client}",ha='center',va='bottom',
            fontsize=10,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,0.975,f"Project: {project} | {pv_label}",ha='center',va='bottom',
            fontsize=10,fontname=fontname,transform=ax.transAxes)

    lines = [title, row(headers), row(units)]
    for _,r in df_yearly.iterrows():
        year = int(r['Year'])
        vals = [
            year,
            fmt(raw_df[raw_df['Date'].dt.year==year]['Oil'].sum()),
            fmt(raw_df[raw_df['Date'].dt.year==year]['Gas'].sum()),
            fmt(raw_df[raw_df['Date'].dt.year==year]['NGL'].sum()),
            fmt(raw_df[raw_df['Date'].dt.year==year]['Water'].sum()),
            fmt(r['Free CF']), fmt(r['Disc CF'])
        ] + ['']*(len(headers)-7)
        lines.append(row(vals))
    tot = df_yearly[['Free CF','Disc CF']].sum()
    total_vals = ["TOTAL"] + ['']*(len(headers)-3) + [fmt(tot['Free CF']),fmt(tot['Disc CF'])]
    lines.append(row(total_vals))

    ax.text(0,0.94,"\n".join(lines),va='top',ha='left',
            fontname=fontname,fontsize=7,transform=ax.transAxes)
    summary = get_summary_fn(raw_df, econ_params={'Discount Rate':None})
    ax.text(0.01,0.02,summary, fontname=fontname,fontsize=7,
            ha='left',va='bottom',transform=ax.transAxes)
    ax.text(0.5,0.01,f"Page {page_num}",ha='center',va='bottom',
            fontsize=8,fontname=fontname,transform=ax.transAxes)

    pdf.savefig(fig); plt.close(fig)

def generate_cashflow_pdf_table(
    well_cashflows, total_cashflow, econ_params, output_path="Cashflow_Report.pdf"
):
    fontname = 'monospace'
    eff_str  = pd.to_datetime(econ_params['Effective Date']).strftime('%B %d, %Y')
    dr       = econ_params.get('Discount Rate',0.0)
    pv_label = f"PV{int(dr*100)}"
    client   = econ_params.get('Client','')
    project  = econ_params.get('Project','')

    with PdfPages(output_path) as pdf:
        yearly = summarize_yearly(total_cashflow)
        render_cashflow_page(
            "Total Project Cashflow Summary",
            yearly, total_cashflow,
            pdf, 1, fontname, eff_str, client, project, pv_label,
            compute_project_irr
        )
        for i,df in enumerate(well_cashflows,start=2):
            yearly_w = summarize_yearly(df)
            render_cashflow_page(
                f"Cashflow for {df['WellName'].iloc[0]} (API: {df['API14'].iloc[0]})",
                yearly_w, df,
                pdf, i, fontname, eff_str, client, project, pv_label,
                compute_project_irr
            )
    return output_path

def render_mixed_table(
    pdf, df_monthly, df_yearly, df_full,
    title, page_num, fontname,
    eff_str, client, project, pv_label,
    get_aries_summary_text
):
    # Combine monthly & annual in one page
    fmt   = lambda v: f"{v/1_000:,.2f}" if pd.notnull(v) else "   -   "
    label = lambda dt: dt.strftime("%b %Y")
    fig, ax = plt.subplots(figsize=(15,0.5+0.22*(len(df_monthly)+len(df_yearly)+6)))
    ax.axis('off')
    # header
    ax.text(0.5,1.06,"Schaper Energy Economics Engine",ha='center',va='bottom',
            fontsize=12,fontweight='bold',fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,1.03,f"Effective Date: {eff_str}",ha='center',va='bottom',
            fontsize=9,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,1.00,f"Client: {client}",ha='center',va='bottom',
            fontsize=9,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,0.975,f"Project: {project} | {pv_label}",ha='center',va='bottom',
            fontsize=9,fontname=fontname,transform=ax.transAxes)
    # title
    ax.text(0,0.94,title,ha='left',va='top',fontsize=10,
            fontname=fontname,transform=ax.transAxes)
    # build lines
    lines = []
    lines.append("   Month     Free CF (M$)    Disc CF (M$)")
    for _,r in df_monthly.iterrows():
        lines.append(f"{label(r['Date']):>9}     {fmt(r['Free CF']):>10}    {fmt(r['Disc CF']):>10}")
    lines.append("")
    lines.append("   Year      Free CF (M$)    Disc CF (M$)")
    for _,r in df_yearly.iterrows():
        lines.append(f"{int(r['Year']):>9}     {fmt(r['Free CF']):>10}    {fmt(r['Disc CF']):>10}")
    # summary
    summary = get_aries_summary_text(df_full, econ_params.get('Discount Rate',0.0))
    # render
    ax.text(0,0.50,"\n".join(lines),va='top',ha='left',
            fontname=fontname,fontsize=7,transform=ax.transAxes)
    ax.text(0.01,0.02,summary, fontname=fontname,fontsize=7,
            ha='left',va='bottom',transform=ax.transAxes)
    ax.text(0.5,0.01,f"Page {page_num}",ha='center',va='bottom',
            fontsize=8,fontname=fontname,transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)

def generate_cashflow_pdf_table_with_monthly(
    well_cashflows, total_cashflow, econ_params, output_path, get_aries_summary_text
):
    fontname = 'monospace'
    eff_str  = pd.to_datetime(econ_params['Effective Date']).strftime('%B %d, %Y')
    dr       = econ_params.get('Discount Rate',0.0)
    pv_label = f"PV{int(dr*100)}"
    client   = econ_params.get('Client','')
    project  = econ_params.get('Project','')

    with PdfPages(output_path) as pdf:
        all_data = pd.concat(well_cashflows, ignore_index=True)
        # monthly (first 12 months)
        df_mon = all_data[all_data['Months'] < 12].groupby('Date', as_index=False)[['Free CF','Disc CF']].sum()
        df_yr  = summarize_yearly(all_data)
        # project mixed page
        render_mixed_table(
            pdf, df_mon, df_yr, all_data,
            "PROJECT TOTAL", 1, fontname,
            eff_str, client, project, pv_label,
            get_aries_summary_text
        )
        # per‐well mixed pages
        npv_list = [
            (df['WellName'].iloc[0], df['API14'].iloc[0], df['Disc CF'].sum())
            for df in well_cashflows
        ]
        for i, (name, api, _) in enumerate(sorted(npv_list, key=lambda x: x[2], reverse=True), start=2):
            df = next(d for d in well_cashflows if d['API14'].iloc[0]==api)
            df_mon = df[df['Months'] < 12].groupby('Date', as_index=False)[['Free CF','Disc CF']].sum()
            df_yr  = summarize_yearly(df)
            render_mixed_table(
                pdf, df_mon, df_yr, df,
                f"{name} (API: {api})", i, fontname,
                eff_str, client, project, pv_label,
                get_aries_summary_text
            )
    return output_path

if __name__ == "__main__":
    print("SE Economics Engine v6 loaded.")

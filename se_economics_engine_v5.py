# se_economics_engine_v6.py
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ──────────────────────────────────────────────────────────────────────────────
# 1. Helpers and Summaries
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
    npv = df['Disc CF'].sum() / 1_000
    annual_cf = df.set_index('Date')['Free CF'].resample('A').sum().values
    irr = compute_project_irr(annual_cf)
    return f"NPV@{discount_rate*100:.0f}%: ${npv:,.1f}M, IRR: {irr:.1f}%"


def build_pv_summary(df, pv_rates=None):
    pv_rates = pv_rates or [8,9,10,12,15,20,25,30,50]
    if 'Months' not in df.columns:
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
# 2. Build Forecast Inputs
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
                w['WI'], w['NRI'] = tmp['WI'].iloc[0], tmp['NRI'].iloc[0]
        w['Oil Gross (Mbbl)'] = (sub['OilProduction_bbl_month'] / 1000).tolist()
        w['Gas Gross (MMcf)']  = (sub['GasProduction_MCF_month'] / 1000).tolist()
        w['NGL Gross (Mbbl)']  = ((sub['GasProduction_MCF_month'] * econ_params.get('NGL Yield (bbl/MMcf)',2.5) / 1000)).tolist()
        w['Water Gross (Mbbl)'] = (sub.get('WaterProduction_bbl_month',0) / 1000).tolist()
        w['Oil Net (Mbbl)']    = [v * w['NRI'] for v in w['Oil Gross (Mbbl)']]
        w['Gas Net (MMcf)']    = [v * w['NRI'] for v in w['Gas Gross (MMcf)']]
        w['NGL Net (Mbbl)']    = [v * w['NRI'] for v in w['NGL Gross (Mbbl)']]
        w['Water Net (Mbbl)']  = [v * w['NRI'] for v in w['Water Gross (Mbbl)']]
        w['Oil Price'] = econ_params.get('Oil Price',70.0)
        w['Gas Price'] = econ_params.get('Gas Price',3.0)
        if isinstance(df_diff, pd.DataFrame):
            tmp = df_diff[df_diff['API14']==api]
            if not tmp.empty:
                for col in tmp.columns:
                    lc = col.lower()
                    if 'oil' in lc and 'price' in lc: w['Oil Price'] = tmp[col].iloc[0]
                    if 'gas' in lc and 'price' in lc: w['Gas Price'] = tmp[col].iloc[0]
        w['OpEx']  = econ_params.get('OpEx',0.0)
        w['CapEx'] = econ_params.get('CapEx',0.0)
        if isinstance(df_opex,pd.DataFrame):
            tmp = df_opex[df_opex['API14']==api]
            if not tmp.empty and 'OpEx' in tmp.columns: w['OpEx'] = tmp['OpEx'].iloc[0]
        if isinstance(df_capex,pd.DataFrame):
            tmp = df_capex[df_capex['API14']==api]
            if not tmp.empty and 'CapEx' in tmp.columns: w['CapEx'] = tmp['CapEx'].iloc[0]
        w['Severance Tax %']  = econ_params.get('Severance Tax %',0.0)
        w['Ad Valorem Tax %'] = econ_params.get('Ad Valorem Tax %',0.0)
        wells.append(w)
    return wells

# ──────────────────────────────────────────────────────────────────────────────
# 3. Calculate Cashflows
# ──────────────────────────────────────────────────────────────────────────────

def calculate_cashflows(wells, effective_date, discount_rate, df_strip=None):
    all_wells, combined = [], []
    strip = None
    if isinstance(df_strip,pd.DataFrame) and {'Year','Oil Price','Gas Price'}.issubset(df_strip.columns):
        strip = df_strip.sort_values('Year').reset_index(drop=True)
    base = pd.to_datetime(effective_date)
    for w in wells:
        n = len(w['Oil Gross (Mbbl)'])
        df = pd.DataFrame({
            'Months': np.arange(n),
            'Oil Gross (Mbbl)': w['Oil Gross (Mbbl)'],
            'Gas Gross (MMcf)':  w['Gas Gross (MMcf)'],
            'NGL Gross (Mbbl)':  w['NGL Gross (Mbbl)'],
            'Water Gross (Mbbl)':w['Water Gross (Mbbl)']
        })
        df['Date'] = df['Months'].apply(lambda m: base + relativedelta(months=int(m)))
        df['Year'], df['Mo'] = df['Date'].dt.year, df['Date'].dt.month
        if strip is not None:
            df = df.merge(strip, on='Year', how='left', suffixes=('','_strip'))
            df['Oil Price'] = df['Oil Price_strip'].fillna(w['Oil Price'])
            df['Gas Price'] = df['Gas Price_strip'].fillna(w['Gas Price'])
        else:
            df['Oil Price'], df['Gas Price'] = w['Oil Price'], w['Gas Price']
        df['NGL Price']   = df['Oil Price'] * (econ_params.get('NGL Yield (bbl/MMcf)',2.5)/1000)
        df['Oil Net (Mbbl)']= df['Oil Gross (Mbbl)'] * w['NRI']
        df['Gas Net (MMcf)']= df['Gas Gross (MMcf)'] * w['NRI']
        df['NGL Net (Mbbl)']= df['NGL Gross (Mbbl)'] * w['NRI']
        df['Water Net (Mbbl)']= df['Water Gross (Mbbl)'] * w['NRI']
        df['Oil Rev'] = df['Oil Net (Mbbl)'] * df['Oil Price']
        df['Gas Rev'] = df['Gas Net (MMcf)'] * df['Gas Price']
        df['NGL Rev'] = df['NGL Net (Mbbl)']* df['NGL Price']
        df['Total Revenue']= df[['Oil Rev','Gas Rev','NGL Rev']].sum(axis=1)
        df['Taxes'] = df['Total Revenue'] * (w['Severance Tax %'] + w['Ad Valorem Tax %'])
        df['OpEx']  = w['OpEx']
        df['CapEx']=0.0; df.loc[0,'CapEx']=w['CapEx']
        df['Free CF']= df['Total Revenue'] - df['OpEx'] - df['CapEx'] - df['Taxes']
        df['Disc CF']= df['Free CF'] * ((1+discount_rate)**(-(df['Months']/12)))
        df['API14'], df['WellName'] = w['API14'], w['WellName']
        all_wells.append(df); combined.append(df)
    total = pd.concat(combined, ignore_index=True)
    return all_wells, total

# ──────────────────────────────────────────────────────────────────────────────
# 4. Summarize Yearly for All Columns
# ──────────────────────────────────────────────────────────────────────────────

def summarize_yearly(total):
    df = total.copy()
    df['Year'] = df['Date'].dt.year
    agg = {
        'Oil Gross (Mbbl)': 'sum', 'Gas Gross (MMcf)': 'sum',
        'NGL Gross (Mbbl)': 'sum','Water Gross (Mbbl)':'sum',
        'Oil Net (Mbbl)': 'sum','Gas Net (MMcf)': 'sum',
        'NGL Net (Mbbl)':'sum','Water Net (Mbbl)':'sum',
        'Oil Price':'mean','Gas Price':'mean','NGL Price':'mean',
        'Total Revenue':'sum','Taxes':'sum','OpEx':'sum','CapEx':'sum',
        'Free CF':'sum','Disc CF':'sum'
    }
    out = df.groupby('Year', as_index=False).agg(agg)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 5. Render Cashflow Page with Full Columns
# ──────────────────────────────────────────────────────────────────────────────

def render_cashflow_page(title, df_yearly, raw_df, pdf, page_num,
                         fontname, effective_date_str,
                         client_name, project_name, pv_label,
                         get_summary_fn, discount_rate):
    headers = [
        "Year","Oil Gross","Gas Gross","NGL Gross","Water Gross",
        "Oil Net","Gas Net","NGL Net","Water Net",
        "Oil $","Gas $","NGL $",
        "Total Rev","Taxes","OpEx","CapEx","Free CF","Disc CF"
    ]
    units = [
        "     ","(M bbl)","(MMcf)","(M bbl)","(M bbl)",
        "(M bbl)","(MMcf)","(M bbl)","(M bbl)",
        "($/bbl)","($/mcf)","($/bbl)",
        "(M$)","(M$)","(M$)","(M$)","(M$)","(M$)"
    ]
    col_w = [6,10,10,10,10,10,10,10,10,8,8,8,11,8,8,8,11,11]

    def fmt(v,scale=1): return f"{v/scale:,.2f}" if pd.notnull(v) else "   -   "
    def fmtp(v): return f"{v:,.2f}" if pd.notnull(v) else "   -   "
    def row(vals): return " ".join(str(val).rjust(w) for val,w in zip(vals, col_w))

    fig, ax = plt.subplots(figsize=(15, 0.6+0.25*(len(df_yearly)+6)))
    ax.axis('off')
    ax.text(0.5,1.06,"Schaper Energy Economics Engine",ha='center',va='bottom',fontsize=12,fontweight='bold',fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,1.03,f"Effective Date: {effective_date_str}",ha='center',va='bottom',fontsize=10,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,1.00,f"Client: {client_name}",ha='center',va='bottom',fontsize=10,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,0.975,f"Project: {project_name} | {pv_label}",ha='center',va='bottom',fontsize=10,fontname=fontname,transform=ax.transAxes)

    lines = [title, row(headers), row(units)]
    for _, r in df_yearly.iterrows():
        vals = [r['Year'], fmt(r['Oil Gross (Mbbl)']), fmt(r['Gas Gross (MMcf)']), fmt(r['NGL Gross (Mbbl)']), fmt(r['Water Gross (Mbbl)']),
                fmt(r['Oil Net (Mbbl)']), fmt(r['Gas Net (MMcf)']), fmt(r['NGL Net (Mbbl)']), fmt(r['Water Net (Mbbl)']),
                fmtp(r['Oil Price']), fmtp(r['Gas Price']), fmtp(r['NGL Price']),
                fmt(r['Total Revenue']), fmt(r['Taxes']), fmt(r['OpEx']), fmt(r['CapEx']), fmt(r['Free CF']), fmt(r['Disc CF'])]
        lines.append(row(vals))

    total = df_yearly[['Oil Gross (Mbbl)','Gas Gross (MMcf)','NGL Gross (Mbbl)','Water Gross (Mbbl)',
                       'Oil Net (Mbbl)','Gas Net (MMcf)','NGL Net (Mbbl)','Water Net (Mbbl)',
                       'Total Revenue','Taxes','OpEx','CapEx','Free CF','Disc CF']].sum()
    total_vals=["TOTAL", fmt(total['Oil Gross (Mbbl)']), fmt(total['Gas Gross (MMcf)']), fmt(total['NGL Gross (Mbbl)']), fmt(total['Water Gross (Mbbl)']),
                fmt(total['Oil Net (Mbbl)']), fmt(total['Gas Net (MMcf)']), fmt(total['NGL Net (Mbbl)']), fmt(total['Water Net (Mbbl)']),
                "","","", fmt(total['Total Revenue']), fmt(total['Taxes']), fmt(total['OpEx']), fmt(total['CapEx']), fmt(total['Free CF']), fmt(total['Disc CF'])]
    lines.append(row(total_vals))

    ax.text(0,0.94,"\n".join(lines),va='top',ha='left',fontsize=7,fontname=fontname,transform=ax.transAxes)
    summary=get_summary_fn(raw_df, discount_rate)
    ax.text(0.01,0.02,summary,fontsize=7,fontname=fontname,ha='left',va='bottom',transform=ax.transAxes)
    ax.text(0.5,0.01,f"Page {page_num}",ha='center',va='bottom',fontsize=8,fontname=fontname,transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Generate Yearly PDF
# ──────────────────────────────────────────────────────────────────────────────

def generate_cashflow_pdf_table(wells, total_df, econ_params, output_path):
    fontname='monospace'
    eff_str=pd.to_datetime(econ_params['Effective Date']).strftime('%B %d, %Y')
    dr=econ_params.get('Discount Rate',0.10)
    pv_label=f"PV{int(dr*100)}"
    client,proj=econ_params.get('Client',''),econ_params.get('Project','')
    with PdfPages(output_path) as pdf:
        page=1
        yearly=summarize_yearly(total_df)
        render_cashflow_page("Total Project Cashflow Summary",yearly,total_df,pdf,page,fontname,eff_str,client,proj,pv_label,get_aries_summary_text,dr)
        page+=1
        npv_list=sorted([(df['WellName'].iloc[0],df['API14'].iloc[0],df['Free CF'].sum()) for df in wells],key=lambda x:x[2],reverse=True)
        for name,api,_ in npv_list:
            dfw=next(d for d in wells if d['API14'].iloc[0]==api)
            yr_w=summarize_yearly(dfw)
            render_cashflow_page(f"Cashflow Summary for {name} (API: {api})",yr_w,dfw,pdf,page,fontname,eff_str,client,proj,pv_label,get_aries_summary_text,dr)
            page+=1
    return output_path

# ──────────────────────────────────────────────────────────────────────────────
# 7. Generate Monthly PDF
# ──────────────────────────────────────────────────────────────────────────────

def generate_cashflow_pdf_table_with_monthly(wells, total_df, econ_params, output_path):
    fontname='monospace'
    eff=pd.to_datetime(econ_params['Effective Date'])
    dr=econ_params.get('Discount Rate',0.10)
    pv_label=f"PV{int(dr*100)}"
    client,proj=econ_params.get('Client',''),econ_params.get('Project','')
    yearly=summarize_yearly(total_df)
    monthly=total_df[['Date','Free CF']].copy()
    with PdfPages(output_path) as pdf:
        render_cashflow_page("Project Total",yearly,total_df,pdf,1,fontname,eff.strftime('%B %d, %Y'),client,proj,pv_label,get_aries_summary_text,dr)
        fig,ax=plt.subplots(figsize=(15, 0.5+0.22*(len(monthly)+6)))
        ax.axis('off')
        ax.text(0.5,1.06,"Schaper Energy Economics Engine",ha='center',va='bottom',fontsize=11,fontweight='bold',fontname=fontname,transform=ax.transAxes)
        ax.text(0.5,1.03,f"Effective Date: {eff:%B %d, %Y}",ha='center',va='bottom',fontsize=9,fontname=fontname,transform=ax.transAxes)
        ax.text(0.5,1.00,f"Client: {client}",ha='center',va='bottom',fontsize=9,fontname=fontname,transform=ax.transAxes)
        ax.text(0.5,0.975,f"Project: {proj} | {pv_label}",ha='center',va='bottom',fontsize=9,fontname=fontname,transform=ax.transAxes)
        tbl=ax.table(cellText=monthly.values,colLabels=monthly.columns,loc='center')
        tbl.auto_set_font_size(False);tbl.set_fontsize(6)
        pdf.savefig(fig);plt.close(fig)
    return output_path

if __name__ == "__main__":
    print("SE Economics Engine V6 loaded.")

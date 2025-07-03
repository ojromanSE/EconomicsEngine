import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import io
from datetime import datetime

# ------------------------------------------------------------------------------
# 0. Page config
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Schaper Energy Economics Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# 1. App header
# ------------------------------------------------------------------------------
st.title("🔋 Schaper Energy Economics Engine")

# ------------------------------------------------------------------------------
# 2. Step 1: Upload Production Forecast
# ------------------------------------------------------------------------------
st.header("1. Upload Production Forecast (CSV)")
uploaded_forecast = st.file_uploader("Choose forecast CSV", type="csv")
if uploaded_forecast:
    df_forecast = pd.read_csv(uploaded_forecast)
    @st.cache_data
    def clean_api14(series: pd.Series) -> pd.Series:
        return series.apply(lambda x: str(x).split('.')[0]).astype(str).str.zfill(14)
    df_forecast["API14"] = clean_api14(df_forecast["API14"])
    st.success("Forecast loaded!")
    st.dataframe(df_forecast.head())
else:
    st.info("Upload a CSV to proceed.")

# ------------------------------------------------------------------------------
# 3. Step 2: Upload Economic Inputs
# ------------------------------------------------------------------------------
st.header("2. Upload Economic Inputs (Excel)")
effective_date = st.date_input(
    "Effective Date",
    value=datetime.strptime("05.01.2025", "%m.%d.%Y"),
)
excel_file = st.file_uploader("Upload multi‐sheet Excel", type=["xls", "xlsx"])
if excel_file:
    @st.cache_data
    def load_econ_sheets(uploaded) -> dict:
        xl = pd.ExcelFile(uploaded)
        dfs = {}
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            if "API14" in df.columns:
                df["API14"] = clean_api14(df["API14"])
            if sheet == "Ownership":
                df[["WI", "NRI"]] = df[["WI", "NRI"]].apply(pd.to_numeric, errors="coerce")
            elif sheet == "Strip":
                df[["Year", "Oil Price", "Gas Price"]] = df[["Year", "Oil Price", "Gas Price"]].apply(pd.to_numeric, errors="coerce")
                df = df.sort_values("Year").reset_index(drop=True)
            elif sheet == "Differentials":
                df[["Oil Diff", "Gas Diff", "NGL Diff"]] = df[["Oil Diff", "Gas Diff", "NGL Diff"]].apply(pd.to_numeric, errors="coerce")
            elif sheet == "Expenses":
                df[["OpEx","Oil OpEx","Gas OpEx","Water OpEx"]] = df[["OpEx","Oil OpEx","Gas OpEx","Water OpEx"]].apply(pd.to_numeric, errors="coerce")
            elif sheet == "Capital":
                df[["Capex","Abandonment Cost"]] = df[["Capex","Abandonment Cost"]].apply(pd.to_numeric, errors="coerce")
            dfs[sheet] = df
        return dfs

    dfs = load_econ_sheets(excel_file)
    st.success(f"Loaded sheets: {', '.join(dfs.keys())}")
    for name, df in dfs.items():
        st.subheader(name)
        st.dataframe(df.head())

    econ_params = {
        "Effective Date": effective_date,
        "Discount Rate": 0.09,
        "Severance Tax %": 0.06,
        "Ad Valorem Tax %": 0.00,
        "Client": "Schaper Energy Consulting LLC",
        "Project": "Mineral Evaluation PV Tool",
        "NGL Yield (bbl/MMcf)": 72,
        "Shrink": 0.48,
    }
    st.write("### Economic Parameters", econ_params)
else:
    st.info("Upload your economic‐inputs Excel to proceed.")

# ------------------------------------------------------------------------------
# 4. Core Engine Functions
# ------------------------------------------------------------------------------
def compute_project_irr(cashflows, capex_upfront=None):
    try:
        cf = pd.Series(cashflows).fillna(0).astype(float).tolist()
        if capex_upfront is not None:
            cf = [capex_upfront] + cf
        if cf[0] >= 0:
            return None
        irr = npf.irr(cf)
        return round(irr * 100, 2) if irr is not None and not np.isnan(irr) else None
    except:
        return None

def build_pv_summary(df, pv_rates=None):
    pv_rates = pv_rates or [8,9,10,12,15,20,25,30,50]
    lines = []
    df = df.copy()
    if 'Months' not in df.columns:
        df['Months'] = (df['Date'].dt.to_period('M') - df['Date'].min().to_period('M')).apply(lambda x: x.n)
    if 'Free CF' not in df.columns:
        df['Free CF'] = df['Total Revenue'] - df['OpEx'] - df['Capex'] - df['Taxes']
    for rate in pv_rates:
        df[f"DF_{rate}"] = (1 + rate/100) ** (-(df['Months']/12))
        df[f"PV{rate}"] = df['Free CF'] * df[f"DF_{rate}"]
        val = df[f"PV{rate}"].sum() / 1_000
        lines.append(f"{rate:>6.2f}           {val:>10.2f}")
    return lines

def build_forecast_inputs(df_forecast, econ_params, df_ownership=None, df_diff=None, df_opex=None, df_capex=None):
    ngl_yield = econ_params.get('NGL Yield (bbl/MMcf)', 2.5)
    for d in [df_ownership, df_diff, df_opex, df_capex]:
        if d is not None:
            d['API14'] = d['API14'].astype(str).str.strip()
    df = df_forecast.copy()
    df['API14'] = df['API14'].astype(str).str.strip()
    if df_ownership is not None:
        df = df.merge(df_ownership, on="API14", how="left")
    if df_diff is not None:
        df = df.merge(df_diff, on="API14", how="left")
    if df_opex is not None:
        df = df.merge(df_opex, on="API14", how="left")
    if df_capex is not None:
        df = df.merge(df_capex, on="API14", how="left")
    df['WI']      = df['WI'].fillna(econ_params.get('WI',1.0))
    df['NRI']     = df['NRI'].fillna(econ_params.get('NRI',0.8))
    df['Oil Diff']= df['Oil Diff'].fillna(econ_params.get('Oil Diff',-3.0))
    df['Gas Diff']= df['Gas Diff'].fillna(econ_params.get('Gas Diff',0.0))
    df['NGL Diff']= df['NGL Diff'].fillna(econ_params.get('NGL Diff',0.3))
    df['OpEx']    = df['OpEx'].fillna(0.0)
    df['Oil OpEx']= df['Oil OpEx'].fillna(0.0)
    df['Gas OpEx']= df['Gas OpEx'].fillna(0.0)
    df['Water OpEx']= df['Water OpEx'].fillna(0.0)
    df['Capex']   = df['Capex'].fillna(0.0)
    df['Abandonment Cost']=df['Abandonment Cost'].fillna(0.0)
    wells = []
    for api, g in df.groupby("API14"):
        dates = pd.to_datetime(g['Date'])
        oil   = g['OilProduction_bbl_month']
        gas   = g['GasProduction_MCF_month']
        ngl   = gas * ngl_yield / 1000
        water = g['WaterProduction_bbl_month']
        row0  = g.iloc[0]
        wells.append({
            'API14': api,
            'WellName': row0['WellName'],
            'Dates': dates.tolist(),
            'Oil (bbl)': oil.tolist(),
            'Gas (mcf)': gas.tolist(),
            'NGL (bbl)': ngl.tolist(),
            'Water (bbl)': water.tolist(),
            'WI': row0['WI'],
            'NRI':row0['NRI'],
            'Oil Diff':row0['Oil Diff'],
            'Gas Diff':row0['Gas Diff'],
            'NGL Diff':row0['NGL Diff'],
            'OpEx':row0['OpEx'],
            'Oil OpEx':row0['Oil OpEx'],
            'Gas OpEx':row0['Gas OpEx'],
            'Water OpEx':row0['Water OpEx'],
            'Capex':row0['Capex'],
            'Abandonment Cost':row0['Abandonment Cost'],
            **econ_params
        })
    return wells

def calculate_cashflows(wells, effective_date, discount_rate, df_strip=None):
    effective_date = pd.to_datetime(effective_date)
    total_df = None
    all_cf = []
    strip_oil_last = strip_gas_last = None
    if df_strip is not None:
        df_strip = df_strip.sort_values('Year')
        strip_oil_last = df_strip['Oil Price'].iloc[-1]
        strip_gas_last = df_strip['Gas Price'].iloc[-1]
    for w in wells:
        wi, nri = w['WI'], w['NRI']
        oil_g = np.array(w['Oil (bbl)'])
        gas_g = np.array(w['Gas (mcf)'])
        ngl_g = np.array(w.get('NGL (bbl)', [0]*len(oil_g)))
        water = np.array(w.get('Water (bbl)', [0]*len(oil_g)))
        df = pd.DataFrame({
            'Date': pd.to_datetime(w['Dates']),
            'Oil Gross (bbl)': oil_g,
            'Gas Gross (mcf)': gas_g,
            'NGL Gross (bbl)': ngl_g,
            'Water Gross (bbl)': water,
            'Oil Net (bbl)': oil_g*nri,
            'Gas Net (mcf)': gas_g*nri,
            'NGL Net (bbl)': ngl_g*nri
        })
        df['Months'] = (df['Date'].dt.to_period('M')-effective_date.to_period('M')).apply(lambda x: x.n)
        df = df[df['Months']>=0]
        df['Year'] = df['Date'].dt.year
        if df_strip is not None:
            df = df.merge(df_strip, on='Year', how='left')
            df['Oil Price'] = df['Oil Price'].fillna(strip_oil_last) + w['Oil Diff']
            df['Gas Price'] = df['Gas Price'].fillna(strip_gas_last) + w['Gas Diff']
        else:
            df['Oil Price'] = w.get('Oil Price',70.0) + w['Oil Diff']
            df['Gas Price'] = w.get('Gas Price',3.0) + w['Gas Diff']
        df['NGL Price']     = df['Oil Price'] * w['NGL Diff']
        df['Oil Revenue']   = df['Oil Net (bbl)']*df['Oil Price']
        df['Gas Revenue']   = df['Gas Net (mcf)']*df['Gas Price']
        df['NGL Revenue']   = df['NGL Net (bbl)']*df['NGL Price']
        df['Total Revenue'] = df[['Oil Revenue','Gas Revenue','NGL Revenue']].sum(axis=1)
        df['Taxes']         = df['Total Revenue']*(w['Severance Tax %']+w['Ad Valorem Tax %'])
        df['OpEx']          = (
            w['OpEx']*wi +
            df['Oil Gross (bbl)']*w['Oil OpEx']*wi +
            df['Gas Gross (mcf)']*w['Gas OpEx']*wi +
            df['Water Gross (bbl)']*w['Water OpEx']*wi
        )
        df['Capex'] = 0.0
        if not df.empty:
            df.loc[df['Months']==0,'Capex'] = w['Capex']*wi
            df.iloc[-1,df.columns.get_loc('Capex')] += w['Abandonment Cost']*wi
        df['Free CF']       = df['Total Revenue']-df['OpEx']-df['Capex']-df['Taxes']
        df['Discount Factor']= (1+discount_rate)**(-(df['Months']/12))
        df['Discounted CF'] = df['Free CF']*df['Discount Factor']
        df['API14']   = w['API14']
        df['WellName']= w['WellName']
        df['WI']      = wi
        df['NRI']     = nri
        all_cf.append(df)
        total_df = pd.concat([total_df,df]) if total_df is not None else df.copy()
    total_cashflow = total_df.groupby('Date',as_index=False).agg({
        'Free CF':'sum','Discounted CF':'sum'
    })
    return all_cf, total_cashflow

def summarize_yearly(df):
    df = df.copy()
    df['Year'] = df['Date'].dt.year
    df['Total Revenue'] = df[['Oil Revenue','Gas Revenue','NGL Revenue']].sum(axis=1)
    df['Free CF']       = df['Total Revenue']-df['Taxes']-df['OpEx']-df['Capex']
    df['is_dec']        = df['Date'].dt.month==12
    return df.groupby('Year',as_index=False).agg({
        'Oil Gross (bbl)':'sum','Gas Gross (mcf)':'sum','NGL Gross (bbl)':'sum','Water Gross (bbl)':'sum',
        'Oil Net (bbl)':'sum','Gas Net (mcf)':'sum','NGL Net (bbl)':'sum','Total Revenue':'sum',
        'Taxes':'sum','OpEx':'sum','Capex':'sum','Free CF':'sum','Discounted CF':'sum','is_dec':'max'
    })

def get_aries_summary_text(df, meta):
    df = df.copy()
    df['Net CF'] = df['Free CF']
    initial_capex = df.loc[df['Months']==0,'Capex'].sum() if (df['Months']==0).any() else 0.0
    df.loc[df['Months']==0,'Net CF'] -= initial_capex
    df['YearIdx'] = (df['Months']/12).apply(np.floor).astype(int)
    cf_list = [-initial_capex] + [df[df['YearIdx']==y]['Net CF'].sum() for y in range(1,21)]
    irr = round(npf.irr(cf_list)*100,2) if cf_list[0]<0 else None
    irr_line = f"RATE-OF-RETURN %   {irr:>8.2f}" if irr is not None else "RATE-OF-RETURN %       N/A"
    df['Cum Net CF'] = df['Net CF'].cumsum()
    pm = df[df['Cum Net CF']>=0]['Months'].min()
    payout = round(pm/12,2) if pd.notnull(pm) else None
    payout_line = f"PAYOUT TIME, YRS.   {payout:>8.2f}" if payout is not None else "PAYOUT TIME, YRS.       N/A"
    pv_lines = build_pv_summary(df)
    header = f"{'':<20}        OIL              GAS             WATER"
    left = [f"GROSS WELLS         {1.0:>10.2f}",
            f"GROSS RES., MB      {df['Oil Gross (bbl)'].sum()/1000:>10.2f}",
            f"NET RES., MB        {df['Oil Net (bbl)'].sum()/1000:>10.2f}",
            f"INITIAL N.R.I., %   {meta.get('NRI',0.0):>10.2f}"]
    right= [f"GROSS WELLS         {0.0:>10.2f}",
            f"GROSS RES., MMcf    {df['Gas Gross (mcf)'].sum()/1000:>10.2f}",
            f"NET RES., MMcf      {df['Gas Net (mcf)'].sum()/1000:>10.2f}",
            f"INITIAL W.I., %     {meta.get('WI',0.0):>10.2f}"]
    water= ["GROSS WELLS         0.00",
            f"GROSS WATER, MB     {df['Water Gross (bbl)'].sum()/1000:>10.2f}",
            "NET RES., MB        0.00",
            ""]
    lines = [header] + [f"{l}    {r}    {w}" for l,r,w in zip(left,right,water)]
    duration_months = ((df['Date'].max().to_period("M")-df['Date'].min().to_period("M")).n+1)
    dur = round(duration_months/12,2)
    lines.append(f"LIFE YRS          {dur:>12.2f}")
    lines.append(irr_line)
    lines.append(payout_line)
    lines.append("")
    lines.append(" P.W. %            P.W., M$")
    lines.extend(pv_lines)
    lines.append("")
    return "\n".join(lines)

def render_cashflow_page(title, df_yearly, raw_df, pdf, page_num,
                         fontname, effective_date_str,
                         client_name, project_name,
                         pv_label,
                         get_aries_summary_text):
    headers = [
        "Year", "Oil Gross", "Gas Gross", "NGL Gross", "Water Gross",
        "Oil Net", "Gas Net", "NGL Net",
        "Oil $", "Gas $", "NGL $",
        "Total Rev", "Taxes", "OpEx", "Capex", "Free CF", "Disc CF"
    ]
    units = [
        "     ", "(Mbbl)", "(MMcf)", "(Mbbl)", "(Mbbl)",
        "(Mbbl)", "(MMcf)", "(Mbbl)",
        "($/bbl)", "($/mcf)", "($/bbl)",
        "(M$)", "(M$)", "(M$)", "(M$)", "(M$)", "(M$)"
    ]
    col_widths = [9,10,10,10,10,10,10,10,8,8,8,11,8,8,8,11,11]
    def fmt(val,scale=1000):
        return f"{val/scale:,.2f}" if pd.notnull(val) else "   -   "
    def fmt_p(val):
        return f"{val:,.2f}" if pd.notnull(val) else "   -   "
    def format_row(vals,widths):
        return " ".join(str(v).rjust(w) for v,w in zip(vals,widths))

    df_detail = df_yearly[(df_yearly['is_dec'])&(df_yearly['Year']<=pd.to_datetime(effective_date_str).year+19)]
    total_row = df_yearly.sum(numeric_only=True)

    fig, ax = plt.subplots(figsize=(15,0.6+0.25*(len(df_detail)+6)))
    ax.axis("off")
    ax.text(0.5,1.06,"Schaper Energy Economics Engine",ha='center',va='bottom',
            fontsize=12,fontweight='bold',fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,1.03,f"Effective Date: {effective_date_str}",ha='center',va='bottom',
            fontsize=10,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,1.00,f"Client: {client_name}",ha='center',va='bottom',
            fontsize=10,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,0.975,f"Project: {project_name} | {pv_label}",ha='center',va='bottom',
            fontsize=10,fontname=fontname,transform=ax.transAxes)

    lines = [title, format_row(headers,col_widths), format_row(units,col_widths)]
    for _,r in df_detail.iterrows():
        row = [
            int(r['Year']),
            fmt(r['Oil Gross (bbl)']), fmt(r['Gas Gross (mcf)']), fmt(r['NGL Gross (bbl)']),
            fmt(r.get('Water Gross (bbl)',0.0)),
            fmt(r['Oil Net (bbl)']), fmt(r['Gas Net (mcf)']), fmt(r['NGL Net (bbl)']),
            fmt_p(r['Eff Oil Price']), fmt_p(r['Eff Gas Price']), fmt_p(r['Eff NGL Price']),
            fmt(r['Total Revenue']), fmt(r['Taxes']), fmt(r['OpEx']), fmt(r['Capex']),
            fmt(r['Free CF']), fmt(r['Discounted CF'])
        ]
        lines.append(format_row(row,col_widths))

    total_fmt = [
        "TOTAL",
        fmt(total_row['Oil Gross (bbl)']), fmt(total_row['Gas Gross (mcf)']), fmt(total_row['NGL Gross (bbl)']),
        fmt(total_row.get('Water Gross (bbl)',0.0)),
        fmt(total_row['Oil Net (bbl)']), fmt(total_row['Gas Net (mcf)']), fmt(total_row['NGL Net (bbl)']),
        "", "", "",
        fmt(total_row['Total Revenue']), fmt(total_row['Taxes']), fmt(total_row['OpEx']), fmt(total_row['Capex']),
        fmt(total_row['Free CF']), fmt(total_row['Discounted CF'])
    ]
    lines.append(format_row(total_fmt,col_widths))

    wi = raw_df['WI'].iloc[0] if 'WI' in raw_df.columns else 1.0
    nri = raw_df['NRI'].iloc[0] if 'NRI' in raw_df.columns else 0.75
    summary_text = get_aries_summary_text(raw_df, {'WI':wi,'NRI':nri})

    ax.text(0,0.94,"\n".join(lines),va='top',ha='left',
            fontname=fontname,fontsize=7,transform=ax.transAxes)
    ax.text(0.01,0.02,summary_text,
            fontname=fontname,fontsize=7,transform=ax.transAxes,ha='left',va='bottom')
    ax.text(0.5,0.01,f"Page {page_num}",ha='center',va='bottom',
            fontsize=8,fontname=fontname,transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)

def render_mixed_table(df_mon, df_ann, df_full, df_yr, title, pdf, pg,
                       wi=1.0, nri=0.75,
                       client_name="TBD", project_name="TBD", pv_label="PVXX"):
    df_mon = df_mon.copy(); df_ann = df_ann.copy(); df_full = df_full.copy(); df_yr=df_yr.copy()
    df_mon['Label'] = df_mon['Date'].dt.strftime("%b %Y").str.rjust(9)
    df_ann['Label'] = df_ann['Year'].astype(str).str.rjust(9)
    combined = pd.concat([df_mon,df_ann],ignore_index=True)
    total = df_full.drop(columns=['Date','Year','Month','is_dec','Label'],errors='ignore').sum(numeric_only=True)
    total['Label'] = 'TOTAL'.rjust(9)

    headers = [
        "Year", "Oil Gross", "Gas Gross", "NGL Gross", "Water Gross",
        "Oil Net", "Gas Net", "NGL Net",
        "Oil $", "Gas $", "NGL $",
        "Total Rev", "Taxes", "OpEx", "Capex", "Free CF", "Disc CF"
    ]
    units = [
        "     ", "(Mbbl)", "(MMcf)", "(Mbbl)", "(Mbbl)",
        "(Mbbl)", "(MMcf)", "(Mbbl)",
        "($/bbl)", "($/mcf)", "($/bbl)",
        "(M$)", "(M$)", "(M$)", "(M$)", "(M$)", "(M$)"
    ]
    col_w = [9,10,10,10,10,10,10,10,8,8,8,11,8,8,8,11,11]

    def fmt(v,s=1000): return f"{v/s:,.2f}" if pd.notnull(v) else "   -   "
    def fmt_p(v): return f"{v:,.2f}" if pd.notnull(v) else "   -   "
    def fr(vals,widths): return " ".join(str(v).rjust(w) for v,w in zip(vals,widths))

    fig, ax = plt.subplots(figsize=(15,0.5+0.22*(len(combined)+6)))
    ax.axis("off")
    ax.text(0.5,1.06,"Schaper Energy Economics Engine",ha='center',va='bottom',
            fontsize=11,fontweight='bold',fontname='monospace',transform=ax.transAxes)
    ax.text(0.5,1.03,f"Effective Date: {df_full['Date'].min():%B %d, %Y}",ha='center',va='bottom',
            fontsize=9,fontname='monospace',transform=ax.transAxes)
    ax.text(0.5,1.00,f"Client: {client_name}",ha='center',va='bottom',
            fontsize=9,fontname='monospace',transform=ax.transAxes)
    ax.text(0.5,0.975,f"Project: {project_name} | {pv_label}",ha='center',va='bottom',
            fontsize=9,fontname='monospace',transform=ax.transAxes)

    lines = [f"Cashflow Summary for {title}", fr(headers,col_w), fr(units,col_w)]
    for _,r in combined.iterrows():
        row = [
            r['Label'],
            fmt(r['Oil Gross (bbl)']), fmt(r['Gas Gross (mcf)']), fmt(r['NGL Gross (bbl)']),
            fmt(r.get('Water Gross (bbl)',0.0)),
            fmt(r['Oil Net (bbl)']), fmt(r['Gas Net (mcf)']), fmt(r['NGL Net (bbl)']),
            fmt_p(r['Eff Oil Price']), fmt_p(r['Eff Gas Price']), fmt_p(r['Eff NGL Price']),
            fmt(r['Total Revenue']), fmt(r['Taxes']), fmt(r['OpEx']), fmt(r['Capex']),
            fmt(r['Free CF']), fmt(r['Discounted CF'])
        ]
        lines.append(fr(row,col_w))
    total_row = [
        total['Label'],
        fmt(total['Oil Gross (bbl)']), fmt(total['Gas Gross (mcf)']), fmt(total['NGL Gross (bbl)']),
        fmt(total.get('Water Gross (bbl)',0.0)),
        fmt(total['Oil Net (bbl)']), fmt(total['Gas Net (mcf)']), fmt(total['NGL Net (bbl)']),
        "", "", "",
        fmt(total['Total Revenue']), fmt(total['Taxes']), fmt(total['OpEx']), fmt(total['Capex']),
        fmt(total['Free CF']), fmt(total['Discounted CF'])
    ]
    lines.append(fr(total_row,col_w))

    summary = get_aries_summary_text(df_full, {'WI':wi,'NRI':nri})
    ax.text(0,0.94,"\n".join(lines),va='top',ha='left',fontname='monospace',
            fontsize=7,transform=ax.transAxes)
    ax.text(0.01,0.02,summary,fontname='monospace',fontsize=7,
            transform=ax.transAxes,ha='left',va='bottom')
    ax.text(0.5,0.01,f"Page {pg}",ha='center',va='bottom',fontsize=8,
            fontname='monospace',transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)

def generate_cashflow_pdf_table(well_cashflows, total_cashflow_df, econ_params, output_path="Cashflow_Report.pdf"):
    fontname = 'monospace'
    eff_date = pd.to_datetime(econ_params['Effective Date'])
    dr = econ_params.get("Discount Rate",0.1)
    pv_label = f"PV{int(dr*100)}"
    client = econ_params.get("Client","")
    project = econ_params.get("Project","")
    final_agg_year = eff_date.year + 49

    npv_list = []
    for df in well_cashflows:
        yearly = summarize_yearly(df)
        total_npv = yearly[yearly['Year']<=final_agg_year]['Discounted CF'].sum()
        npv_list.append((df['WellName'].iloc[0], df['API14'].iloc[0], total_npv))
    sorted_wells = sorted(npv_list, key=lambda x: x[2], reverse=True)

    with PdfPages(output_path) as pdf:
        page=1
        proj_yearly = summarize_yearly(total_cashflow_df)
        render_cashflow_page("Total Project Cashflow Summary", proj_yearly, total_cashflow_df,
                             pdf, page, fontname, eff_date.strftime('%B %d, %Y'),
                             client, project, pv_label, get_aries_summary_text)
        page+=1
        for name,api,_ in sorted_wells:
            df = next(d for d in well_cashflows if d['API14'].iloc[0]==api)
            yr = summarize_yearly(df)
            render_cashflow_page(f"Cashflow Summary for {name} (API: {api})",
                                 yr, df, pdf, page, fontname,
                                 eff_date.strftime('%B %d, %Y'), client, project,
                                 pv_label, get_aries_summary_text)
            page+=1
    return output_path

def generate_cashflow_pdf_table_with_monthly(
    well_cashflows, total_cashflow_df, econ_params,
    output_path="Cashflow_Report_Monthly.pdf", get_aries_summary_text=None
):
    fontname = 'monospace'
    eff_date = pd.to_datetime(econ_params['Effective Date'])
    dr = econ_params.get("Discount Rate",0.1)
    pv_label = f"PV{int(dr*100)}"
    client = econ_params.get("Client","")
    project = econ_params.get("Project","")
    summary_end = eff_date.year+19

    with PdfPages(output_path) as pdf:
        pg=1
        all_data = pd.concat(well_cashflows, ignore_index=True)
        all_data['Months'] = (all_data['Date'].dt.to_period('M')-eff_date.to_period('M')).apply(lambda x: x.n)
        all_data['Total Revenue'] = all_data[['Oil Revenue','Gas Revenue','NGL Revenue']].sum(axis=1)
        all_data['Free CF'] = all_data['Total Revenue']-all_data['OpEx']-all_data['Capex']-all_data['Taxes']
        all_data['Discount Factor'] = (1+dr)**(-(all_data['Months']/12))
        all_data['Discounted CF'] = all_data['Free CF']*all_data['Discount Factor']

        monthly = all_data[all_data['Months']<12].groupby('Date',as_index=False).sum(numeric_only=True)
        yearly = summarize_yearly(all_data)
        annual = yearly[(yearly['Year']>eff_date.year)&(yearly['Year']<=summary_end)&(yearly['is_dec'])]

        render_mixed_table(monthly, annual, all_data, yearly,
                           "PROJECT TOTAL", pdf, pg, client_name=client,
                           project_name=project, pv_label=pv_label)
        pg+=1

        npv_list = [(d['WellName'].iloc[0],d['API14'].iloc[0], d['Discounted CF'].sum()) for d in well_cashflows]
        for name,api,_ in sorted(npv_list, key=lambda x: x[2], reverse=True):
            df = next(d for d in well_cashflows if d['API14'].iloc[0]==api).copy()
            df['Months'] = (df['Date'].dt.to_period('M')-eff_date.to_period('M')).apply(lambda x: x.n)
            df['Total Revenue'] = df[['Oil Revenue','Gas Revenue','NGL Revenue']].sum(axis=1)
            df['Free CF'] = df['Total Revenue']-df['OpEx']-df['Capex']-df['Taxes']
            df['Discount Factor'] = (1+dr)**(-(df['Months']/12))
            df['Discounted CF'] = df['Free CF']*df['Discount Factor']

            mon = df[df['Months']<12].groupby('Date',as_index=False).sum(numeric_only=True)
            yrly = summarize_yearly(df)
            ann = yrly[(yrly['Year']>eff_date.year)&(yrly['Year']<=summary_end)&(yrly['is_dec'])]

            render_mixed_table(mon, ann, df, yrly,
                               f"{name} (API: {api})", pdf, pg,
                               wi=df['WI'].iloc[0], nri=df['NRI'].iloc[0],
                               client_name=client, project_name=project,
                               pv_label=pv_label)
            pg+=1
    return output_path

def generate_oneline_summary_excel(
    well_cashflows, econ_params, df_ownership=None, df_opex=None, output_path=None
):
    today_str = datetime.today().strftime('%m.%d.%Y')
    client_safe = econ_params.get('Client','UnknownClient').replace(" ","")
    if output_path is None:
        output_path = f"SE_Economics_{client_safe}_Oneline_Report_{today_str}.xlsx"
    summary_rows = []
    for df in well_cashflows:
        api  = df['API14'].iloc[0]; name=df['WellName'].iloc[0]
        tmp = df.copy()
        tmp['Free CF'] = tmp['Total Revenue']-tmp['OpEx']-tmp['Capex']-tmp['Taxes']
        wi = nri = None
        if df_ownership is not None:
            m = df_ownership[df_ownership['API14']==api]
            if not m.empty:
                wi=float(m['WI'].iloc[0]); nri=float(m['NRI'].iloc[0])
        gross_oil = tmp['Oil Gross (bbl)'].sum()
        gross_gas = tmp['Gas Gross (mcf)'].sum()
        gross_ngl = tmp['NGL Gross (bbl)'].sum()
        net_oil   = tmp['Oil Net (bbl)'].sum()
        net_gas   = tmp['Gas Net (mcf)'].sum()
        net_ngl   = tmp['NGL Net (bbl)'].sum()
        total_capex=tmp['Capex'].sum(); total_opex=tmp['OpEx'].sum(); total_cf=tmp['Free CF'].sum()
        pv_values={}; total_dcf=0.0
        for rate in [0,9]+list(range(10,101,10)):
            tmp['Discount Factor']=(1+rate/100)**(-(tmp['Months']/12))
            tmp['Discounted CF']=tmp['Free CF']*tmp['Discount Factor']
            pv=tmp['Discounted CF'].sum()
            pv_values[f'PV{rate}']=round(pv/1000,2)
            if rate==10: total_dcf=pv
        row = {
            'API14':api,'WellName':name,
            'WI (%)':f"{wi*100:.2f}%" if wi is not None else None,
            'NRI (%)':f"{nri*100:.2f}%" if nri is not None else None,
            'Gross Oil (Mbbl)':round(gross_oil/1000,2),
            'Gross Gas (Mmcf)':round(gross_gas/1000,2),
            'Gross NGL (Mbbl)':round(gross_ngl/1000,2),
            'Net Oil (Mbbl)':round(net_oil/1000,2),
            'Net Gas (Mmcf)':round(net_gas/1000,2),
            'Net NGL (Mbbl)':round(net_ngl/1000,2),
            'Total Capex (M$)':round(total_capex/1000,2),
            'Total OpEx (M$)':round(total_opex/1000,2),
            'Net CF (M$)':round(total_cf/1000,2),
            'Discounted CF (M$)':round(total_dcf/1000,2),
            **pv_values
        }
        summary_rows.append(row)
    df_summary = pd.DataFrame(summary_rows).sort_values('PV10',ascending=False).reset_index(drop=True)
    df_summary.to_excel(output_path,index=False)
    return output_path

def export_monthly_cashflow_excel(well_cashflows, econ_params, output_path=None):
    today_str = datetime.today().strftime('%m.%d.%Y')
    client_safe = econ_params.get('Client','UnknownClient').replace(" ","")
    if output_path is None:
        output_path = f"SE_Economics_{client_safe}_Monthly_Report_{today_str}.xlsx"
    all_rows=[]
    for df in well_cashflows:
        dfm = df.copy()
        api=dfm['API14'].iloc[0]; name=dfm['WellName'].iloc[0]
        wi = dfm.get('WI',pd.Series([1.0])).iloc[0]
        fixed = dfm.get('OpEx',pd.Series([0.0])).iloc[0]
        oil_opex = dfm.get('Oil OpEx',pd.Series([0.0])).iloc[0]
        gas_opex = dfm.get('Gas OpEx',pd.Series([0.0])).iloc[0]
        sev = dfm.get('Severance Tax %',pd.Series([0.0])).iloc[0]
        adv = dfm.get('Ad Valorem Tax %',pd.Series([0.0])).iloc[0]
        dfm['Severance Tax ($)']=dfm['Total Revenue']*sev
        dfm['Ad Valorem Tax ($)']=dfm['Total Revenue']*adv
        dfm['Fixed OpEx ($)']=fixed
        dfm['Oil Var OpEx ($)']=dfm['Oil Gross (bbl)']*oil_opex*wi
        dfm['Gas Var OpEx ($)']=dfm['Gas Gross (mcf)']*gas_opex*wi
        if 'Net CF' not in dfm.columns:
            dfm['Net CF']=dfm['Total Revenue']-dfm['OpEx']-dfm['Capex']-dfm['Taxes']
        dfm['API14']=api; dfm['WellName']=name
        all_rows.append(dfm)
    df_all=pd.concat(all_rows,ignore_index=True)
    ordered=[
        'API14','WellName','Date','Months','Year',
        'Oil Gross (bbl)','Gas Gross (mcf)','NGL Gross (bbl)',
        'Oil Net (bbl)','Gas Net (mcf)','NGL Net (bbl)',
        'Oil Price','Gas Price','NGL Price',
        'Oil Revenue','Gas Revenue','NGL Revenue','Total Revenue',
        'Severance Tax ($)','Ad Valorem Tax ($)',
        'Fixed OpEx ($)','Oil Var OpEx ($)','Gas Var OpEx ($)',
        'OpEx','Capex','Net CF'
    ]
    cols=[c for c in ordered if c in df_all.columns]
    df_all=df_all[cols]
    df_all.to_excel(output_path,index=False,sheet_name="Monthly Cashflow")
    return output_path

# ------------------------------------------------------------------------------
# 5. Cached data prep
# ------------------------------------------------------------------------------
@st.cache_data
def prep_wells(df_forecast, econ_params, dfs_overrides):
    return build_forecast_inputs(
        df_forecast,
        econ_params,
        dfs_overrides.get("Ownership"),
        dfs_overrides.get("Differentials"),
        dfs_overrides.get("Expenses"),
        dfs_overrides.get("Capital")
    )

@st.cache_data
def run_cashflows(well_inputs, effective_date, discount_rate, df_strip):
    return calculate_cashflows(well_inputs, effective_date, discount_rate, df_strip)

# ------------------------------------------------------------------------------
# 6. Export Buttons
# ------------------------------------------------------------------------------
st.header("3. Exports")
if uploaded_forecast and excel_file:
    if st.button("▶️ Run All & Download Reports"):
        wells = prep_wells(df_forecast, econ_params, dfs)
        well_cfs, total_cfs = run_cashflows(
            wells, econ_params["Effective Date"], econ_params["Discount Rate"], dfs.get("Strip")
        )

        # Monthly PDF
        buf_mon_pdf = io.BytesIO()
        generate_cashflow_pdf_table_with_monthly(
            well_cfs, total_cfs, econ_params,
            output_path=buf_mon_pdf, get_aries_summary_text=get_aries_summary_text
        )
        buf_mon_pdf.seek(0)
        st.download_button("Download Monthly PDF", buf_mon_pdf, "Cashflow_Monthly.pdf", "application/pdf")

        # Yearly PDF
        buf_yr_pdf = io.BytesIO()
        generate_cashflow_pdf_table(well_cfs, total_cfs, econ_params, output_path=buf_yr_pdf)
        buf_yr_pdf.seek(0)
        st.download_button("Download Yearly PDF", buf_yr_pdf, "Cashflow_Yearly.pdf", "application/pdf")

        # Oneline XLSX
        oneline_path = generate_oneline_summary_excel(well_cfs, econ_params, dfs.get("Ownership"), dfs.get("Expenses"))
        with open(oneline_path, "rb") as f: data = f.read()
        st.download_button("Download Oneline XLSX", data, oneline_path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Detailed Monthly XLSX
        monthly_path = export_monthly_cashflow_excel(well_cfs, econ_params)
        with open(monthly_path, "rb") as f: data2 = f.read()
        st.download_button("Download Monthly XLSX", data2, monthly_path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.success("All reports generated!")
else:
    st.info("Please complete Steps 1 & 2 before exporting.")

st.markdown("---")
st.write("© 2025 Schaper Energy Consulting LLC")


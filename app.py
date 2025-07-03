import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import io
import tempfile, os
from datetime import datetime

# ------------------------------------------------------------------------------
# 0. Page config
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Schaper Energy Economics Engine",
    layout="wide"
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
excel_file = st.file_uploader("Upload multi-sheet Excel", type=["xls", "xlsx"])
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
                df[["Year","Oil Price","Gas Price"]] = df[["Year","Oil Price","Gas Price"]].apply(pd.to_numeric, errors="coerce")
                df = df.sort_values("Year").reset_index(drop=True)
            elif sheet == "Differentials":
                df[["Oil Diff","Gas Diff","NGL Diff"]] = df[["Oil Diff","Gas Diff","NGL Diff"]].apply(pd.to_numeric, errors="coerce")
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
    st.info("Upload your economic-inputs Excel to proceed.")

# ------------------------------------------------------------------------------
# 4. Core Engine Functions (full implementations)
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
    df = df.copy()
    if 'Months' not in df:
        df['Months'] = (df['Date'].dt.to_period('M') - df['Date'].min().to_period('M')).apply(lambda x: x.n)
    if 'Free CF' not in df:
        df['Free CF'] = df['Total Revenue'] - df['OpEx'] - df['Capex'] - df['Taxes']
    lines = []
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
    defaults = {
        'WI': 1.0, 'NRI': 0.8, 'Oil Diff': -3.0,
        'Gas Diff': 0.0, 'NGL Diff': 0.3,
        'OpEx': 0.0, 'Oil OpEx': 0.0,
        'Gas OpEx': 0.0, 'Water OpEx': 0.0,
        'Capex': 0.0, 'Abandonment Cost': 0.0
    }
    for k,v in defaults.items():
        df[k] = df[k].fillna(econ_params.get(k, v))
    wells = []
    for api, g in df.groupby("API14"):
        dates = pd.to_datetime(g['Date'])
        oil = g['OilProduction_bbl_month']
        gas = g['GasProduction_MCF_month']
        ngl = gas * ngl_yield / 1000
        water = g['WaterProduction_bbl_month']
        r = g.iloc[0]
        wells.append({
            'API14': api,
            'WellName': r['WellName'],
            'Dates': dates.tolist(),
            'Oil (bbl)': oil.tolist(),
            'Gas (mcf)': gas.tolist(),
            'NGL (bbl)': ngl.tolist(),
            'Water (bbl)': water.tolist(),
            'WI': r['WI'], 'NRI': r['NRI'],
            'Oil Diff': r['Oil Diff'], 'Gas Diff': r['Gas Diff'],
            'NGL Diff': r['NGL Diff'],
            'OpEx': r['OpEx'], 'Oil OpEx': r['Oil OpEx'],
            'Gas OpEx': r['Gas OpEx'], 'Water OpEx': r['Water OpEx'],
            'Capex': r['Capex'], 'Abandonment Cost': r['Abandonment Cost'],
            **econ_params
        })
    return wells

def calculate_cashflows(wells, effective_date, discount_rate, df_strip=None):
    eff = pd.to_datetime(effective_date)
    total_df = None
    all_cf = []
    strip_oil = strip_gas = None
    if df_strip is not None:
        df_strip = df_strip.sort_values('Year')
        strip_oil = df_strip['Oil Price'].iloc[-1]
        strip_gas = df_strip['Gas Price'].iloc[-1]
    for w in wells:
        wi, nri = w['WI'], w['NRI']
        oil_g = np.array(w['Oil (bbl)'])
        gas_g = np.array(w['Gas (mcf)'])
        ngl_g = np.array(w.get('NGL (bbl)', [0]*len(oil_g)))
        water = np.array(w.get('Water (bbl)', [0] * len(oil_g)))

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
        df['Months'] = (df['Date'].dt.to_period('M')-eff.to_period('M')).apply(lambda x: x.n)
        df = df[df['Months']>=0]
        df['Year'] = df['Date'].dt.year
        if df_strip is not None:
            df = df.merge(df_strip, on='Year', how='left')
            df['Oil Price'] = df['Oil Price'].fillna(strip_oil) + w['Oil Diff']
            df['Gas Price'] = df['Gas Price'].fillna(strip_gas) + w['Gas Diff']
        else:
            df['Oil Price'] = w.get('Oil Price',70.0) + w['Oil Diff']
            df['Gas Price'] = w.get('Gas Price',3.0) + w['Gas Diff']
        df['NGL Price'] = df['Oil Price'] * w['NGL Diff']
        df['Oil Revenue'] = df['Oil Net (bbl)'] * df['Oil Price']
        df['Gas Revenue'] = df['Gas Net (mcf)'] * df['Gas Price']
        df['NGL Revenue'] = df['NGL Net (bbl)'] * df['NGL Price']
        df['Total Revenue'] = df[['Oil Revenue','Gas Revenue','NGL Revenue']].sum(axis=1)
        df['Taxes'] = df['Total Revenue'] * (w['Severance Tax %'] + w['Ad Valorem Tax %'])
        df['OpEx'] = (
            w['OpEx']*wi +
            df['Oil Gross (bbl)']*w['Oil OpEx']*wi +
            df['Gas Gross (mcf)']*w['Gas OpEx']*wi +
            df['Water Gross (bbl)']*w['Water OpEx']*wi
        )
        df['Capex'] = 0.0
        if not df.empty:
            df.loc[df['Months']==0,'Capex'] = w['Capex']*wi
            df.iloc[-1, df.columns.get_loc('Capex')] += w['Abandonment Cost']*wi
        df['Free CF'] = df['Total Revenue'] - df['OpEx'] - df['Capex'] - df['Taxes']
        df['Discount Factor'] = (1+discount_rate)**(-(df['Months']/12))
        df['Discounted CF'] = df['Free CF'] * df['Discount Factor']
        df['API14'] = w['API14']
        df['WellName'] = w['WellName']
        df['WI'] = wi
        df['NRI'] = nri
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
    df['Free CF'] = df['Total Revenue'] - df['Taxes'] - df['OpEx'] - df['Capex']
    df['is_dec'] = df['Date'].dt.month == 12
    return df.groupby('Year',as_index=False).agg({
        'Oil Gross (bbl)':'sum','Gas Gross (mcf)':'sum','NGL Gross (bbl)':'sum','Water Gross (bbl)':'sum',
        'Oil Net (bbl)':'sum','Gas Net (mcf)':'sum','NGL Net (bbl)':'sum',
        'Total Revenue':'sum','Taxes':'sum','OpEx':'sum','Capex':'sum','Free CF':'sum','Discounted CF':'sum','is_dec':'max'
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
    left = [
        f"GROSS WELLS         {1.0:>10.2f}",
        f"GROSS RES., MB      {df['Oil Gross (bbl)'].sum()/1000:>10.2f}",
        f"NET RES., MB        {df['Oil Net (bbl)'].sum()/1000:>10.2f}",
        f"INITIAL N.R.I., %   {meta.get('NRI',0.0):>10.2f}"
    ]
    right = [
        f"GROSS WELLS         0.00",
        f"GROSS RES., MMcf    {df['Gas Gross (mcf)'].sum()/1000:>10.2f}",
        f"NET RES., MMcf      {df['Gas Net (mcf)'].sum()/1000:>10.2f}",
        f"INITIAL W.I., %     {meta.get('WI',0.0):>10.2f}"
    ]
    water = [
        f"GROSS WELLS         0.00",
        f"GROSS WATER, MB     {df['Water Gross (bbl)'].sum()/1000:>10.2f}",
        "NET RES., MB        0.00",""
    ]
    lines = [header] + [f"{l}    {r}    {w}" for l,r,w in zip(left,right,water)]
    months = (df['Date'].max().to_period("M")-df['Date'].min().to_period("M")).n+1
    dur = round(months/12,2)
    lines += [f"LIFE YRS          {dur:>12.2f}", irr_line, payout_line, "", " P.W. %            P.W., M$"] + pv_lines + [""]
    return "\n".join(lines)

def render_cashflow_page(title, df_yearly, raw_df, pdf, page_num,
                         fontname, effective_date_str,
                         client_name, project_name,
                         pv_label,
                         get_aries_summary_text):
    headers = ["Year","Oil Gross","Gas Gross","NGL Gross","Water Gross",
               "Oil Net","Gas Net","NGL Net","Oil $","Gas $","NGL $",
               "Total Rev","Taxes","OpEx","Capex","Free CF","Disc CF"]
    units   = ["","(Mbbl)","(MMcf)","(Mbbl)","(Mbbl)","(Mbbl)","(MMcf)","(Mbbl)",
               "($/bbl)","($/mcf)","($/bbl)","(M$)","(M$)","(M$)","(M$)","(M$)","(M$)"]
    widths  = [9,10,10,10,10,10,10,10,8,8,8,11,8,8,8,11,11]
    def fmt(v): return f"{v/1000:,.2f}" if pd.notnull(v) else " - "
    def fmt_p(v): return f"{v:,.2f}" if pd.notnull(v) else " - "
    def row(vals): return " ".join(str(x).rjust(w) for x,w in zip(vals,widths))

    df_detail = df_yearly[(df_yearly['is_dec'])&(df_yearly['Year']<=pd.to_datetime(effective_date_str).year+19)]
    total = df_yearly.sum(numeric_only=True)

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

    lines = [title, row(headers), row(units)]
    for _,r in df_detail.iterrows():
        vals = [
            int(r['Year']),
            fmt(r['Oil Gross (bbl)']), fmt(r['Gas Gross (mcf)']),
            fmt(r['NGL Gross (bbl)']), fmt(r.get('Water Gross (bbl)',0)),
            fmt(r['Oil Net (bbl)']), fmt(r['Gas Net (mcf)']),
            fmt(r['NGL Net (bbl)']), fmt_p(r['Eff Oil Price']),
            fmt_p(r['Eff Gas Price']), fmt_p(r['Eff NGL Price']),
            fmt(r['Total Revenue']), fmt(r['Taxes']), fmt(r['OpEx']),
            fmt(r['Capex']), fmt(r['Free CF']), fmt(r['Discounted CF'])
        ]
        lines.append(row(vals))
    total_vals = ["TOTAL"] + [
        fmt(total[c]) if c in total else "" for c in [
            'Oil Gross (bbl)','Gas Gross (mcf)','NGL Gross (bbl)','Water Gross (bbl)',
            'Oil Net (bbl)','Gas Net (mcf)','NGL Net (bbl)'
        ]
    ] + ["","",""] + [
        fmt(total['Total Revenue']), fmt(total['Taxes']), fmt(total['OpEx']),
        fmt(total['Capex']), fmt(total['Free CF']), fmt(total['Discounted CF'])
    ]
    lines.append(row(total_vals))

    wi = raw_df['WI'].iloc[0] if 'WI' in raw_df else 1.0
    nri= raw_df['NRI'].iloc[0] if 'NRI' in raw_df else 0.75
    summary = get_aries_summary_text(raw_df, {'WI':wi,'NRI':nri})

    ax.text(0,0.94,"\n".join(lines),ha='left',va='top',
            fontsize=7,fontname=fontname,transform=ax.transAxes)
    ax.text(0.01,0.02,summary,ha='left',va='bottom',
            fontsize=7,fontname=fontname,transform=ax.transAxes)
    ax.text(0.5,0.01,f"Page {page_num}",ha='center',va='bottom',
            fontsize=8,fontname=fontname,transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)

def render_mixed_table(df_mon, df_ann, df_full, df_yr, title, pdf, pg,
                       wi=1.0, nri=0.75,
                       client_name="TBD", project_name="TBD", pv_label="PVXX"):
    df_mon = df_mon.copy()
    df_ann = df_ann.copy()
    df_full = df_full.copy()
    df_yr = df_yr.copy()

    # Label columns
    df_mon['Label'] = df_mon['Date'].dt.strftime("%b %Y").str.rjust(9)
    df_ann['Label'] = df_ann['Year'].astype(str).str.rjust(9)

    combined = pd.concat([df_mon, df_ann], ignore_index=True)

    total = (
        df_full
        .drop(columns=['Date', 'Year', 'Month', 'is_dec', 'Label'], errors='ignore')
        .sum(numeric_only=True)
    )
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
    col_w = [9, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 11, 8, 8, 8, 11, 11]

    def fmt(v, scale=1_000):
        return f"{v/scale:,.2f}" if pd.notnull(v) else "   -   "

    def fmt_price(v):
        return f"{v:,.2f}" if pd.notnull(v) else "   -   "

    def format_row(vals, widths):
        return " ".join(str(v).rjust(w) for v, w in zip(vals, widths))

    fig, ax = plt.subplots(figsize=(15, 0.5 + 0.22 * (len(combined) + 6)))
    ax.axis("off")

    # Header text
    ax.text(0.5, 1.06, "Schaper Energy Economics Engine", ha='center', va='bottom',
            fontsize=11, fontweight='bold', fontname='monospace', transform=ax.transAxes)
    ax.text(0.5, 1.03, f"Effective Date: {df_full['Date'].min():%B %d, %Y}", ha='center', va='bottom',
            fontsize=9, fontname='monospace', transform=ax.transAxes)
    ax.text(0.5, 1.00, f"Client: {client_name}", ha='center', va='bottom',
            fontsize=9, fontname='monospace', transform=ax.transAxes)
    ax.text(0.5, 0.975, f"Project: {project_name} | {pv_label}", ha='center', va='bottom',
            fontsize=9, fontname='monospace', transform=ax.transAxes)

    # Table lines
    lines = [
        f"Cashflow Summary for {title}",
        format_row(headers, col_w),
        format_row(units, col_w)
    ]

    for _, r in combined.iterrows():
        row = [
            r['Label'],
            fmt(r['Oil Gross (bbl)']), fmt(r['Gas Gross (mcf)']), fmt(r['NGL Gross (bbl)']), fmt(r.get('Water Gross (bbl)', 0.0)),
            fmt(r['Oil Net (bbl)']), fmt(r['Gas Net (mcf)']), fmt(r['NGL Net (bbl)']),
            fmt_price(r['Eff Oil Price']), fmt_price(r['Eff Gas Price']), fmt_price(r['Eff NGL Price']),
            fmt(r['Total Revenue']), fmt(r['Taxes']), fmt(r['OpEx']), fmt(r['Capex']),
            fmt(r['Free CF']), fmt(r['Discounted CF'])
        ]
        lines.append(format_row(row, col_w))

    # Total row
    total_row = [
        total['Label'],
        fmt(total['Oil Gross (bbl)']), fmt(total['Gas Gross (mcf)']), fmt(total['NGL Gross (bbl)']), fmt(total.get('Water Gross (bbl)', 0.0)),
        fmt(total['Oil Net (bbl)']), fmt(total['Gas Net (mcf)']), fmt(total['NGL Net (bbl)']),
        "", "", "",
        fmt(total['Total Revenue']), fmt(total['Taxes']), fmt(total['OpEx']), fmt(total['Capex']),
        fmt(total['Free CF']), fmt(total['Discounted CF'])
    ]
    lines.append(format_row(total_row, col_w))

    # Aries summary
    summary = get_aries_summary_text(df_full, {'WI': wi, 'NRI': nri})

    ax.text(0, 0.94, "\n".join(lines), va='top', ha='left',
            fontname='monospace', fontsize=7, transform=ax.transAxes)
    ax.text(0.01, 0.02, summary, va='bottom', ha='left',
            fontname='monospace', fontsize=7, transform=ax.transAxes)
    ax.text(0.5, 0.01, f"Page {pg}", ha='center', va='bottom',
            fontsize=8, fontname='monospace', transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close(fig)


def generate_cashflow_pdf_table(well_cashflows, total_cashflow_df, econ_params, output_path="Cashflow_Report.pdf"):
    fontname = 'monospace'
    eff_date = pd.to_datetime(econ_params['Effective Date'])
    dr = econ_params.get("Discount Rate", 0.10)
    pv_label = f"PV{int(dr * 100)}"
    client = econ_params.get("Client", "")
    project = econ_params.get("Project", "")
    final_year = eff_date.year + 49

    # Sort wells by NPV descending
    npv_list = []
    for df in well_cashflows:
        yr = summarize_yearly(df)
        total_npv = yr[yr['Year'] <= final_year]['Discounted CF'].sum()
        npv_list.append((df['WellName'].iloc[0], df['API14'].iloc[0], total_npv))
    sorted_wells = sorted(npv_list, key=lambda x: x[2], reverse=True)

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(output_path) as pdf:
        # Project summary page
        proj_yr = summarize_yearly(total_cashflow_df)
        render_cashflow_page(
            "Total Project Cashflow Summary",
            proj_yr,
            total_cashflow_df,
            pdf,
            page_num=1,
            fontname=fontname,
            effective_date_str=eff_date.strftime('%B %d, %Y'),
            client_name=client,
            project_name=project,
            pv_label=pv_label,
            get_aries_summary_text=get_aries_summary_text
        )

        # One page per well
        page = 2
        for name, api, _ in sorted_wells:
            df = next(d for d in well_cashflows if d['API14'].iloc[0] == api)
            yr = summarize_yearly(df)
            render_cashflow_page(
                f"Cashflow Summary for {name} (API: {api})",
                yr,
                df,
                pdf,
                page_num=page,
                fontname=fontname,
                effective_date_str=eff_date.strftime('%B %d, %Y'),
                client_name=client,
                project_name=project,
                pv_label=pv_label,
                get_aries_summary_text=get_aries_summary_text
            )
            page += 1

    return output_path


from matplotlib.backends.backend_pdf import PdfPages

def generate_cashflow_pdf_table_with_monthly(
    well_cashflows,
    total_cashflow_df,
    econ_params,
    output_path="Cashflow_Report_Monthly.pdf",
    get_aries_summary_text=None
):
    # Formatting & parameters
    fontname = 'monospace'
    eff_date = pd.to_datetime(econ_params['Effective Date'])
    dr = econ_params.get("Discount Rate", 0.10)
    pv_label = f"PV{int(dr * 100)}"
    client = econ_params.get("Client", "")
    project = econ_params.get("Project", "")
    summary_end = eff_date.year + 19

    # Write to PDFPages
    with PdfPages(output_path) as pdf:
        # --- Page 1: Project total mixed table ---
        # Prepare combined data
        all_data = pd.concat(well_cashflows, ignore_index=True)
        all_data['Months'] = (
            all_data['Date'].dt.to_period('M')
            - eff_date.to_period('M')
        ).apply(lambda x: x.n)
        all_data['Total Revenue'] = all_data[['Oil Revenue','Gas Revenue','NGL Revenue']].sum(axis=1)
        all_data['Free CF'] = all_data['Total Revenue'] - all_data['OpEx'] - all_data['Capex'] - all_data['Taxes']
        all_data['Discount Factor'] = (1 + dr) ** (-(all_data['Months']/12))
        all_data['Discounted CF'] = all_data['Free CF'] * all_data['Discount Factor']

        # Build monthly & annual summaries
        monthly = all_data[all_data['Months'] < 12].groupby('Date', as_index=False).sum(numeric_only=True)
        yearly  = summarize_yearly(all_data)
        annual  = yearly[
            (yearly['Year'] > eff_date.year) &
            (yearly['Year'] <= summary_end) &
            (yearly['is_dec'])
        ]

        # Render page 1
        render_mixed_table(
            df_mon=monthly,
            df_ann=annual,
            df_full=all_data,
            df_yr=yearly,
            title="PROJECT TOTAL",
            pdf=pdf,
            pg=1,
            wi=1.0,
            nri=1.0,
            client_name=client,
            project_name=project,
            pv_label=pv_label
        )

        # --- Pages 2+ : one per well ---
        # Precompute ordering by NPV
        npv_list = [
            (d['WellName'].iloc[0], d['API14'].iloc[0], d['Discounted CF'].sum())
            for d in well_cashflows
        ]
        sorted_wells = sorted(npv_list, key=lambda x: x[2], reverse=True)

        page = 2
        for name, api, _ in sorted_wells:
            df = next(d for d in well_cashflows if d['API14'].iloc[0] == api).copy()
            # Recompute metrics for this well
            df['Months'] = (
                df['Date'].dt.to_period('M')
                - eff_date.to_period('M')
            ).apply(lambda x: x.n)
            df['Total Revenue'] = df[['Oil Revenue','Gas Revenue','NGL Revenue']].sum(axis=1)
            df['Free CF'] = df['Total Revenue'] - df['OpEx'] - df['Capex'] - df['Taxes']
            df['Discount Factor'] = (1 + dr) ** (-(df['Months']/12))
            df['Discounted CF'] = df['Free CF'] * df['Discount Factor']

            mon = df[df['Months'] < 12].groupby('Date', as_index=False).sum(numeric_only=True)
            yrly = summarize_yearly(df)
            ann  = yrly[
                (yrly['Year'] > eff_date.year) &
                (yrly['Year'] <= summary_end) &
                (yrly['is_dec'])
            ]

            render_mixed_table(
                df_mon=mon,
                df_ann=ann,
                df_full=df,
                df_yr=yrly,
                title=f"{name} (API: {api})",
                pdf=pdf,
                pg=page,
                wi=df['WI'].iloc[0],
                nri=df['NRI'].iloc[0],
                client_name=client,
                project_name=project,
                pv_label=pv_label
            )
            page += 1

    return output_path



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
# 5. Export Buttons
# ------------------------------------------------------------------------------
st.header("3. Exports")
if uploaded_forecast and excel_file:
    if st.button("▶️ Run All & Download Reports"):
        wells = prep_wells(df_forecast, econ_params, dfs)
        well_cfs, total_cfs = run_cashflows(
            wells, econ_params["Effective Date"], econ_params["Discount Rate"], dfs.get("Strip")
        )

        # Monthly PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        generate_cashflow_pdf_table_with_monthly(
            well_cfs, total_cfs, econ_params,
            output_path=tmp_path, get_aries_summary_text=get_aries_summary_text
        )
        buf_mon_pdf = io.BytesIO()
        with open(tmp_path,"rb") as f: buf_mon_pdf.write(f.read())
        buf_mon_pdf.seek(0)
        os.remove(tmp_path)
        st.download_button("Download Monthly PDF", buf_mon_pdf, "Cashflow_Monthly.pdf", "application/pdf")

        # Yearly PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
            yearly_path = tmp2.name
        generate_cashflow_pdf_table(well_cfs, total_cfs, econ_params, output_path=yearly_path)
        buf_yr_pdf = io.BytesIO()
        with open(yearly_path,"rb") as f: buf_yr_pdf.write(f.read())
        buf_yr_pdf.seek(0)
        os.remove(yearly_path)
        st.download_button("Download Yearly PDF", buf_yr_pdf, "Cashflow_Yearly.pdf", "application/pdf")

        # Oneline XLSX
        oneline_path = generate_oneline_summary_excel(well_cfs, econ_params, dfs.get("Ownership"), dfs.get("Expenses"))
        with open(oneline_path, "rb") as f: data = f.read()
        st.download_button("Download Oneline XLSX", data, oneline_path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # Detailed Monthly XLSX
        monthly_path = export_monthly_cashflow_excel(well_cfs, econ_params)
        with open(monthly_path, "rb") as f: data2 = f.read()
        st.download_button("Download Monthly XLSX", data2, monthly_path, "application/vnd.openxmlformats-officedocument-spreadsheetml.sheet")

        st.success("All reports generated!")
else:
    st.info("Please complete Steps 1 & 2 before exporting.")

st.markdown("---")
st.write("© 2025 Schaper Energy Consulting LLC")



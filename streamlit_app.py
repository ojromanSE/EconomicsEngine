import streamlit as st
import pandas as pd
import tempfile
from datetime import datetime
import se_economics_engine_v5 as engine

st.set_page_config(page_title="Schaper Econ Engine", layout="wide")
st.title("📊 Schaper Energy Consulting Economics Engine")

# Sidebar inputs
st.sidebar.header("1. Upload Inputs")
forecast_file = st.sidebar.file_uploader("Forecast CSV", type=["csv"])
excel_file    = st.sidebar.file_uploader("Economic Inputs Excel", type=["xlsx", "xls"])

st.sidebar.header("2. Econ Parameters")
effective_date = st.sidebar.date_input("Effective Date", value=datetime.today())
discount_rate  = st.sidebar.number_input("Discount Rate", value=0.09, step=0.005)
severance_tax  = st.sidebar.number_input("Severance Tax %", value=0.06, step=0.005)
ad_valorem_tax = st.sidebar.number_input("Ad Valorem Tax %", value=0.00, step=0.005)
ngl_yield      = st.sidebar.number_input("NGL Yield (bbl/MMcf)", value=72.0)
shrink         = st.sidebar.number_input("Shrink", value=0.48)

client_name  = st.sidebar.text_input("Client", value="Schaper Energy Consulting LLC")
project_name = st.sidebar.text_input("Project", value="Mineral Evaluation PV Tool")

run = st.sidebar.button("▶️ Run Reports")

if run:
    if not forecast_file or not excel_file:
        st.sidebar.error("Please upload both CSV and Excel files.")
    else:
        # 1) Load & normalize forecast
        df = pd.read_csv(forecast_file)
        df.columns = df.columns.str.strip()  # trim whitespace
        lower_map = {c.lower(): c for c in df.columns}

        # map common variants → engine’s expected names
        col_map = {
            'API14'     : ['api14', 'api', 'well api'],
            'Year'      : ['year', 'yr'],
            'Mo'        : ['mo', 'month'],
            'Oil (bbl)' : ['oil (bbl)', 'oil_bbl', 'oil'],
            'Gas (mcf)' : ['gas (mcf)', 'gas_mcf', 'gas'],
            'NGL (bbl)' : ['ngl (bbl)', 'ngl_bbl', 'ngl']
        }
        rename_dict = {}
        for std_name, variants in col_map.items():
            for v in variants:
                if v in lower_map:
                    rename_dict[lower_map[v]] = std_name
                    break

        df = df.rename(columns=rename_dict)

        # sanity-check
        st.write("🧐 Forecast columns:", df.columns.tolist())

        # now safe to clean API14 & hand off to engine
        df['API14'] = engine.clean_api14(df['API14'])

        # 2) Load Excel overrides
        xl = pd.ExcelFile(excel_file)
        df_ownership = xl.parse("Ownership")     if "Ownership"     in xl.sheet_names else None
        df_strip     = xl.parse("Strip")         if "Strip"         in xl.sheet_names else None
        df_diff      = xl.parse("Differentials") if "Differentials" in xl.sheet_names else None
        df_opex      = xl.parse("Expenses")      if "Expenses"      in xl.sheet_names else None
        df_capex     = xl.parse("Capital")       if "Capital"       in xl.sheet_names else None

        for df_ in (df_ownership, df_diff, df_opex, df_capex):
            if df_ is not None and 'API14' in df_.columns:
                df_['API14'] = engine.clean_api14(df_['API14'])

        # 3) Build params & run
        econ_params = {
            "Effective Date": effective_date,
            "Discount Rate":  discount_rate,
            "Severance Tax %": severance_tax,
            "Ad Valorem Tax %": ad_valorem_tax,
            "Client": client_name,
            "Project": project_name,
            "NGL Yield (bbl/MMcf)": ngl_yield,
            "Shrink": shrink
        }

        wells, total = engine.calculate_cashflows(
            engine.build_forecast_inputs(
                df, econ_params,
                df_ownership=df_ownership,
                df_diff=df_diff,
                df_opex=df_opex,
                df_capex=df_capex
            ),
            effective_date=econ_params["Effective Date"],
            discount_rate=econ_params["Discount Rate"],
            df_strip=df_strip
        )

        # 4) Generate & offer downloads
        tmp1 = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
        tmp2 = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name

        engine.generate_cashflow_pdf_table(wells, total, econ_params, tmp1)
        engine.generate_cashflow_pdf_table_with_monthly(
            wells, total, econ_params, tmp2, engine.get_aries_summary_text
        )

        with open(tmp1, "rb") as f1:
            st.download_button("📥 Download Yearly Report", f1.read(),
                               file_name=f"Yearly_{datetime.today():%Y%m%d}.pdf",
                               mime="application/pdf")
        with open(tmp2, "rb") as f2:
            st.download_button("📥 Download Monthly Report", f2.read(),
                               file_name=f"Monthly_{datetime.today():%Y%m%d}.pdf",
                               mime="application/pdf")

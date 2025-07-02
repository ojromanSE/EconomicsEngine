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
        st.stop()

    # 1) Load forecast
    df = pd.read_csv(forecast_file)
    df.columns = df.columns.str.strip()

    # Check required columns
    missing = []
    for col in ["API14", "Date", "OilProduction_bbl_month", "GasProduction_MCF_month"]:
        if col not in df.columns:
            missing.append(col)
    if missing:
        st.error(f"Your CSV is missing these columns: {missing}")
        st.stop()

    # Parse dates into Year & Mo
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Mo"]   = df["Date"].dt.month

    # Map your production columns into the engine’s expected names
    df["Oil (bbl)"] = df["OilProduction_bbl_month"]
    df["Gas (mcf)"] = df["GasProduction_MCF_month"]
    # You don’t have an NGL stream → default to zero
    df["NGL (bbl)"] = 0.0

    # Clean up API14
    df["API14"] = engine.clean_api14(df["API14"])

    st.write("🔍 Sample of parsed forecast data:")
    st.dataframe(df[["API14", "WellName", "Date", "Year", "Mo", "Oil (bbl)", "Gas (mcf)", "NGL (bbl)"]].head())

    # 2) Load Excel overrides
    xl = pd.ExcelFile(excel_file)
    df_ownership = xl.parse("Ownership")     if "Ownership"     in xl.sheet_names else None
    df_strip     = xl.parse("Strip")         if "Strip"         in xl.sheet_names else None
    df_diff      = xl.parse("Differentials") if "Differentials" in xl.sheet_names else None
    df_opex      = xl.parse("Expenses")      if "Expenses"      in xl.sheet_names else None
    df_capex     = xl.parse("Capital")       if "Capital"       in xl.sheet_names else None

    for df_ in (df_ownership, df_diff, df_opex, df_capex):
        if df_ is not None and "API14" in df_.columns:
            df_["API14"] = engine.clean_api14(df_["API14"])

    # 3) Build econ_params & run engine
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

    # Build forecasts and cashflows
    well_inputs = engine.build_forecast_inputs(
        df, econ_params,
        df_ownership=df_ownership,
        df_diff=df_diff,
        df_opex=df_opex,
        df_capex=df_capex
    )
    well_cfs, total_cf = engine.calculate_cashflows(
        well_inputs,
        effective_date=econ_params["Effective Date"],
        discount_rate=econ_params["Discount Rate"],
        df_strip=df_strip
    )

    # 4) Generate PDF outputs
    tmp_yearly  = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    tmp_monthly = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name

    engine.generate_cashflow_pdf_table(well_cfs, total_cf, econ_params, tmp_yearly)
    engine.generate_cashflow_pdf_table_with_monthly(
        well_cfs, total_cf, econ_params, tmp_monthly, engine.get_aries_summary_text
    )

    # Download buttons
    with open(tmp_yearly, "rb") as fy:
        st.download_button("📥 Download Yearly Report", fy.read(),
                           file_name=f"Yearly_Cashflow_{datetime.today():%Y%m%d}.pdf",
                           mime="application/pdf")
    with open(tmp_monthly, "rb") as fm:
        st.download_button("📥 Download Monthly Report", fm.read(),
                           file_name=f"Monthly_Cashflow_{datetime.today():%Y%m%d}.pdf",
                           mime="application/pdf")

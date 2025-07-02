# app.py
import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
import os
import io

# Optional: let user upload a logo
LOGO_PATH = None
uploaded_logo = st.file_uploader("Upload logo image (optional)", type=["png","jpg","jpeg"])
if uploaded_logo:
    LOGO_PATH = "logo_upload.png"
    with open(LOGO_PATH, "wb") as f:
        f.write(uploaded_logo.getbuffer())

# PDF class with logo and page number
class PDFWithPageNumbers(FPDF):
    def __init__(self, logo_path=None):
        super().__init__()
        self.logo_path = logo_path

    def footer(self):
        self.set_y(-15)
        self.set_font("Times", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
        if self.logo_path and os.path.exists(self.logo_path):
            logo_width = 12
            x_position = self.w - self.r_margin - logo_width
            y_position = self.h - 15
            self.image(self.logo_path, x=x_position, y=y_position, w=logo_width)


# all your existing helper functions:
def load_oneline(file_buffer):
    df = pd.read_excel(file_buffer, sheet_name="Oneline")
    df.columns = df.columns.str.strip()
    return df

def calculate_variances(begin_df, final_df, npv_column):
    key_columns = [
        "Net Total Revenue ($)", "Net Operating Expense ($)", "Inital Approx WI", "Initial Approx NRI",
        "Net Res Oil (Mbbl)", "Net Res Gas (MMcf)", "Net Capex ($)", "Net Res NGL (Mbbl)", npv_column
    ]
    merged = begin_df.merge(final_df, on=["PROPNUM","LEASE_NAME"], suffixes=("_begin","_final"), how="left")
    for col in key_columns:
        if f"{col}_begin" in merged and f"{col}_final" in merged:
            merged[f"{col} Variance"] = merged[f"{col}_final"] - merged[f"{col}_begin"]
    merged['Reserve Category Begin'] = merged['SE_RSV_CAT_begin']
    merged['Reserve Category Final'] = merged['SE_RSV_CAT_final']
    return merged

def generate_explanations(variance_df, npv_column):
    explanations = []
    thresholds = {
        "Net Total Revenue ($)": 0.05,
        "Net Operating Expense ($)": 0.05,
        "Inital Approx WI": 0.05,
        "Net Res Oil (Mbbl)": 0.05,
        "Net Res Gas (MMcf)": 0.05,
        "Net Capex ($)": 0.05,
        "Net Res NGL (Mbbl)": 0.05,
        npv_column: 0.05
    }
    for _, row in variance_df.iterrows():
        max_var = 0; max_col = None
        for col, thresh in thresholds.items():
            vb = row.get(f"{col}_begin", 0)
            vf = row.get(f"{col}_final", 0)
            if vb and abs(vf - vb)/abs(vb) > thresh and abs(vf - vb) > max_var:
                max_var = abs(vf - vb); max_col = col
        expl = ""
        if max_col:
            var = row[f"{max_col} Variance"]
            expl = f"{max_col} changed by {var:,.2f}."
        explanations.append({
            "PROPNUM": row["PROPNUM"],
            "LEASE_NAME": row["LEASE_NAME"],
            "Key Metric": max_col or "",
            "Variance Value": max_var,
            "Explanation": expl
        })
    return pd.DataFrame(explanations)

def identify_negative_npv(variance_df, npv_column):
    return variance_df[
        (variance_df[f"{npv_column}_begin"]>0) & (variance_df[f"{npv_column}_final"]<=0)
    ][["PROPNUM","LEASE_NAME"]]

def calculate_nri_wi_ratio(begin_df, final_df):
    def comp(df):
        df = df[df['Inital Approx WI']!=0]
        df['NRI/WI Ratio'] = df['Initial Approx NRI'] / df['Inital Approx WI']
        return df[['PROPNUM','LEASE_NAME','NRI/WI Ratio']]
    b = comp(begin_df).rename(columns={'NRI/WI Ratio':'Begin Ratio'})
    f = comp(final_df).rename(columns={'NRI/WI Ratio':'Final Ratio'})
    merged = b.merge(f, on=['PROPNUM','LEASE_NAME'], how='outer')
    merged['Outlier Source'] = merged.apply(
        lambda r: 'Begin' if not 0.70<r['Begin Ratio']<0.85 else
                  ('Final' if not 0.70<r['Final Ratio']<0.85 else None),
        axis=1
    )
    return merged.dropna(subset=['Outlier Source'])


st.title("Schaper Energy Oneline Comparison")

# File upload
begin_file = st.file_uploader("Upload BEGIN Excel", type=["xlsx"])
final_file = st.file_uploader("Upload FINAL Excel", type=["xlsx"])
npv_column = st.text_input("NPV column name (e.g. NPV at 9%)")

if begin_file and final_file and npv_column:
    begin_df = load_oneline(begin_file)
    final_df = load_oneline(final_file)

    # compute
    var_df = calculate_variances(begin_df, final_df, npv_column)
    expl_df = generate_explanations(var_df, npv_column)
    neg_df  = identify_negative_npv(var_df, npv_column)
    nri_df  = calculate_nri_wi_ratio(begin_df, final_df)

    st.subheader("Variance Summary")
    st.dataframe(var_df)

    st.subheader("Explanations")
    st.dataframe(expl_df)

    st.subheader("Wells with Negative or Zero NPV")
    st.dataframe(neg_df)

    st.subheader("NRI/WI Outliers")
    st.dataframe(nri_df)

    # Prepare Excel in-memory
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
        var_df.to_excel(writer, sheet_name="Variance", index=False)
        expl_df.to_excel(writer, sheet_name="Explanations", index=False)
        neg_df.to_excel(writer, sheet_name="Neg NPV Wells", index=False)
        nri_df.to_excel(writer, sheet_name="NRI/WI Outliers", index=False)
    towrite.seek(0)

    st.download_button(
        label="Download Excel Report",
        data=towrite,
        file_name="variance_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Prepare PDF in-memory
    pdf_buffer = io.BytesIO()
    pdf = PDFWithPageNumbers(logo_path=LOGO_PATH)
    pdf.add_page()
    pdf.set_font("Times", size=12)
    pdf.cell(0,10, f"Overall Variance Report", ln=True)
    # (You can expand here: loop categories/pages as in your original function.)
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)

    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name="variance_report.pdf",
        mime="application/pdf"
    )

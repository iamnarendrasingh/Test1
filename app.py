import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# ============================================================
# Session Visits Dashboard - V1
# ============================================================

st.set_page_config(page_title="Session Visits Dashboard-V1", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type")

def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def bucket_0_to_7_plus(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    bins = [-0.1, 0, 1, 2, 3, 4, 5, 6, np.inf]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7+"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)

def add_export_footer_rows(df_out, notes):
    footer = pd.DataFrame({df_out.columns[0]: [""] + notes})
    for c in df_out.columns[1:]:
        footer[c] = ""
    return pd.concat([df_out, footer], ignore_index=True)

# -----------------------------
# CENTERED TITLE
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>Session Visits Dashboard-V1</h1>",
    unsafe_allow_html=True
)

# -----------------------------
# Upload
# -----------------------------
uploaded = st.file_uploader("Upload Excel / CSV", type=["xlsx", "xls", "csv"])
if not uploaded:
    st.stop()

df = read_uploaded_file(uploaded)
df.columns = [c.strip() for c in df.columns]

# -----------------------------
# Column detection
# -----------------------------
col_state = pick_col(df, ["StateName", "State"])
col_gender = pick_col(df, ["Gender"])
col_child = pick_col(df, ["ChildId", "ChildID", "Child", "Child_ID"])
col_progsub = pick_col(df, ["ProgramSubType", "Program Sub Type", "ProgramSubTypeName"])

metric_cols = [
    "NumeracySessions",
    "LiteracySessions",
    "WorksheetNumeracy",
    "WorksheetLiteracy",
    "LearningEnggagement",
    "LearningEnggagementSessions",
]

df = safe_numeric(df, ["SessionAttended", "Total"] + metric_cols)

# Normalize gender values (important for color consistency)
if col_gender:
    df[col_gender] = (
        df[col_gender]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({"M": "Male", "F": "Female"})
        .fillna("Unknown")
    )

# -----------------------------
# Upload Summary
# -----------------------------
st.subheader("Upload Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Total Submissions (Rows)", f"{len(df):,}")
c2.metric("Total Children (Unique)", f"{df[col_child].nunique():,}" if col_child else "NA")
c3.metric("Unique States", df[col_state].nunique() if col_state else "NA")

# -----------------------------
# StateName-wise Submissions + Gender (COLOR CODED)
# -----------------------------
if col_state and col_gender:
    st.markdown("### StateName-wise Submissions by Gender")

    sg = df[[col_state, col_gender]].copy()
    sg[col_state] = sg[col_state].fillna("Unknown")
    sg[col_gender] = sg[col_gender].fillna("Unknown")

    sg_counts = (
        sg.groupby([col_state, col_gender])
        .size()
        .reset_index(name="Rows")
    )

    # Keep chart readable: Top 10 states + Others
    totals = sg_counts.groupby(col_state)["Rows"].sum().sort_values(ascending=False)
    top_states = set(totals.head(10).index)

    sg_counts[col_state] = sg_counts[col_state].apply(
        lambda x: x if x in top_states else "Others"
    )

    sg_counts = (
        sg_counts.groupby([col_state, col_gender], as_index=False)["Rows"].sum()
    )

    gender_colors = alt.Scale(
        domain=["Male", "Female", "Unknown"],
        range=["#1f77b4", "#e377c2", "#7f7f7f"]
    )

    stacked = alt.Chart(sg_counts).mark_bar().encode(
        x=alt.X(f"{col_state}:N", title="StateName", sort="-y", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Rows:Q", title="Submissions (Rows)"),
        color=alt.Color(
            f"{col_gender}:N",
            title="Gender",
            scale=gender_colors,
            legend=alt.Legend(orient="right")
        ),
        tooltip=[
            alt.Tooltip(f"{col_state}:N", title="StateName"),
            alt.Tooltip(f"{col_gender}:N", title="Gender"),
            alt.Tooltip("Rows:Q", title="Rows", format=",.0f"),
        ],
    ).properties(height=420)

    st.altair_chart(stacked, use_container_width=True)

    # Table view
    pivot = sg_counts.pivot_table(
        index=col_state,
        columns=col_gender,
        values="Rows",
        fill_value=0,
        aggfunc="sum"
    ).reset_index()

    pivot["Total"] = pivot.drop(columns=[col_state]).sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False)

    st.dataframe(pivot, use_container_width=True)

else:
    st.info("StateName or Gender column not available for this view.")

st.divider()

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
f = df.copy()

sel_state = "All"
sel_progsub = "All"
sel_gender = "All"

if col_state:
    states = sorted(f[col_state].dropna().unique())
    sel_state = st.sidebar.selectbox("StateName", ["All"] + states)
    if sel_state != "All":
        f = f[f[col_state] == sel_state]

if col_progsub:
    progsubs = sorted(f[col_progsub].dropna().unique())
    sel_progsub = st.sidebar.selectbox("ProgramSubType", ["All"] + progsubs)
    if sel_progsub != "All":
        f = f[f[col_progsub] == sel_progsub]

if col_gender:
    genders = sorted(f[col_gender].dropna().unique())
    sel_gender = st.sidebar.selectbox("Gender", ["All"] + genders)
    if sel_gender != "All":
        f = f[f[col_gender] == sel_gender]

# -----------------------------
# Export with footer
# -----------------------------
st.subheader("Export")

export_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
footer_notes = [
    f"Export Timestamp: {export_ts}",
    f"Filters Applied - StateName: {sel_state}",
    f"Filters Applied - ProgramSubType: {sel_progsub}",
    f"Filters Applied - Gender: {sel_gender}",
]

f_export = add_export_footer_rows(f.copy(), footer_notes)

st.download_button(
    "Download filtered data (CSV)",
    data=f_export.to_csv(index=False).encode("utf-8"),
    file_name="session_visits_dashboard_v1_filtered.csv",
    mime="text/csv",
)

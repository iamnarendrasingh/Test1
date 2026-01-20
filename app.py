import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

def bar_with_labels(data, x_col, y_col, x_title, y_title):
    base = alt.Chart(data)
    bar = base.mark_bar().encode(
        x=alt.X(f"{x_col}:N", title=x_title),
        y=alt.Y(f"{y_col}:Q", title=y_title),
        tooltip=[x_col, y_col],
    )
    text = base.mark_text(dy=-6).encode(
        x=x_col,
        y=y_col,
        text=alt.Text(y_col, format=".0f"),
    )
    return bar + text

# -----------------------------
# FIXED BUCKET FUNCTION (0–7+)
# -----------------------------
def bucket_0_to_7_plus(series):
    """
    Buckets: 0,1,2,3,4,5,6,7+
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    bins = [-0.1, 0, 1, 2, 3, 4, 5, 6, np.inf]   # 9 edges
    labels = ["0", "1", "2", "3", "4", "5", "6", "7+"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)

BUCKET_0_7_ORDER = ["0", "1", "2", "3", "4", "5", "6", "7+"]

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
col_child = pick_col(df, ["ChildId", "ChildID"])

metric_cols = [
    "NumeracySessions",
    "LiteracySessions",
    "WorksheetNumeracy",
    "WorksheetLiteracy",
    "LearningEnggagement",
    "LearningEnggagementSessions",
]

df = safe_numeric(df, ["SessionAttended", "Total"] + metric_cols)

# -----------------------------
# Upload summaries
# -----------------------------
st.subheader("Upload Summary")

c1, c2 = st.columns(2)
c1.metric("Total Submissions (Rows)", f"{len(df):,}")
c2.metric("Unique States", df[col_state].nunique() if col_state else "NA")

if col_state:
    st.markdown("### StateName-wise Submissions")
    st.dataframe(
        df[col_state].value_counts().reset_index(name="Rows"),
        use_container_width=True
    )

if col_gender:
    st.markdown("### Gender-wise Submissions")
    st.dataframe(
        df[col_gender].value_counts().reset_index(name="Rows"),
        use_container_width=True
    )

st.divider()

# -----------------------------
# Create buckets for metrics
# -----------------------------
for mc in metric_cols:
    if mc in df.columns:
        df[f"{mc}_Bucket"] = bucket_0_to_7_plus(df[mc]).astype(str)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
f = df.copy()

if col_state:
    states = sorted(f[col_state].dropna().unique())
    sel_state = st.sidebar.selectbox("StateName", ["All"] + states)
    if sel_state != "All":
        f = f[f[col_state] == sel_state]

if col_gender:
    genders = sorted(f[col_gender].dropna().unique())
    sel_gender = st.sidebar.selectbox("Gender", ["All"] + genders)
    if sel_gender != "All":
        f = f[f[col_gender] == sel_gender]

show_raw = st.sidebar.checkbox("Show raw data")

# -----------------------------
# KPIs
# -----------------------------
st.subheader("Key Metrics (Filtered)")

k1, k2, k3 = st.columns(3)

k1.metric("Records", f"{len(f):,}")
k2.metric("Unique Children", f[col_child].nunique() if col_child else "NA")
k3.metric("Avg SessionAttended", f["SessionAttended"].mean().round(2) if "SessionAttended" in f else "NA")

st.divider()

# -----------------------------
# Metric Distributions (0–7+)
# -----------------------------
st.subheader("Session Category Distributions (0–7+)")

for mc in metric_cols:
    bucket_col = f"{mc}_Bucket"
    if bucket_col not in f.columns:
        continue

    st.markdown(f"### {mc}")

    dist = (
        f[bucket_col]
        .value_counts()
        .reindex(BUCKET_0_7_ORDER, fill_value=0)
        .reset_index()
    )
    dist.columns = ["Bucket", "Rows"]

    st.dataframe(dist, use_container_width=True)

    chart = bar_with_labels(
        dist,
        x_col="Bucket",
        y_col="Rows",
        x_title="Bucket",
        y_title="Rows",
    )
    st.altair_chart(chart, use_container_width=True)

    st.divider()

# -----------------------------
# Export
# -----------------------------
st.subheader("Export")

st.download_button(
    "Download filtered data (CSV)",
    data=f.to_csv(index=False),
    file_name="session_visits_dashboard_v1.csv",
    mime="text/csv",
)

# -----------------------------
# Raw Data
# -----------------------------
if show_raw:
    st.subheader("Raw Data")
    st.dataframe(f, use_container_width=True)

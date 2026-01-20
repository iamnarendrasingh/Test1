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
    """
    Buckets: 0,1,2,3,4,5,6,7+
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0).clip(lower=0)
    bins = [-0.1, 0, 1, 2, 3, 4, 5, 6, np.inf]  # 9 edges -> 8 buckets
    labels = ["0", "1", "2", "3", "4", "5", "6", "7+"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True)

BUCKET_0_7_ORDER = ["0", "1", "2", "3", "4", "5", "6", "7+"]

def add_export_footer_rows(df_out, notes):
    """
    Append export timestamp + filter selections as footer rows in the exported CSV.
    """
    footer = pd.DataFrame({df_out.columns[0]: [""] + notes})
    for c in df_out.columns[1:]:
        footer[c] = ""
    return pd.concat([df_out, footer], ignore_index=True)

def bar_with_labels(data, x_col, y_col, x_title, y_title):
    """
    Bar chart with data labels (counts shown on bars).
    """
    base = alt.Chart(data)
    bar = base.mark_bar().encode(
        x=alt.X(f"{x_col}:N", title=x_title, sort=None),
        y=alt.Y(f"{y_col}:Q", title=y_title),
        tooltip=[
            alt.Tooltip(f"{x_col}:N", title=x_title),
            alt.Tooltip(f"{y_col}:Q", title=y_title, format=",.0f"),
        ],
    )
    text = base.mark_text(dy=-6, fontSize=12).encode(
        x=alt.X(f"{x_col}:N", sort=None),
        y=alt.Y(f"{y_col}:Q"),
        text=alt.Text(f"{y_col}:Q", format=",.0f"),
    )
    return bar + text

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
    st.info("Upload a file to begin.")
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

# Ensure numeric columns are numeric
df = safe_numeric(df, ["SessionAttended", "Total"] + metric_cols)

# Normalize gender values (for consistent color coding)
if col_gender:
    g = df[col_gender].astype(str).str.strip().str.title()
    g = g.replace({"M": "Male", "F": "Female"})
    # If your data includes "Male ", "female", etc., normalize further:
    g = g.replace({"Man": "Male", "Woman": "Female"})
    df[col_gender] = g.fillna("Unknown")

# Create 0–7+ bucket columns for metric columns
for mc in metric_cols:
    if mc in df.columns:
        df[f"{mc}_Bucket"] = bucket_0_to_7_plus(df[mc]).astype(str)

# ============================================================
# Upload Summary (UNFILTERED view)
# ============================================================
st.subheader("Upload Summary (Full File)")

total_rows = len(df)
total_children = df[col_child].nunique() if col_child else np.nan
unique_states = df[col_state].nunique() if col_state else np.nan

c1, c2, c3 = st.columns(3)
c1.metric("Total Submissions (Rows)", f"{total_rows:,}")
c2.metric("Total Children (Unique)", f"{int(total_children):,}" if not np.isnan(total_children) else "NA")
c3.metric("Unique States", f"{int(unique_states):,}" if not np.isnan(unique_states) else "NA")

# StateName-wise Submissions by Gender (COLOR CODED)
if col_state and col_gender:
    st.markdown("### StateName-wise Submissions by Gender (Top 10 + Others)")

    sg = df[[col_state, col_gender]].copy()
    sg[col_state] = sg[col_state].fillna("Unknown").astype(str)
    sg[col_gender] = sg[col_gender].fillna("Unknown").astype(str)

    sg_counts = sg.groupby([col_state, col_gender]).size().reset_index(name="Rows")

    totals = sg_counts.groupby(col_state)["Rows"].sum().sort_values(ascending=False)
    top_states = set(totals.head(10).index)

    sg_counts[col_state] = sg_counts[col_state].apply(lambda x: x if x in top_states else "Others")
    sg_counts = sg_counts.groupby([col_state, col_gender], as_index=False)["Rows"].sum()

    gender_colors = alt.Scale(
        domain=["Male", "Female", "Unknown"],
        range=["#1f77b4", "#e377c2", "#7f7f7f"]
    )

    stacked = alt.Chart(sg_counts).mark_bar().encode(
        x=alt.X(f"{col_state}:N", title="StateName", sort="-y", axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("Rows:Q", title="Submissions (Rows)"),
        color=alt.Color(f"{col_gender}:N", title="Gender", scale=gender_colors),
        tooltip=[
            alt.Tooltip(f"{col_state}:N", title="StateName"),
            alt.Tooltip(f"{col_gender}:N", title="Gender"),
            alt.Tooltip("Rows:Q", title="Rows", format=",.0f"),
        ],
    ).properties(height=420)

    st.altair_chart(stacked, use_container_width=True)

    # Pivot table
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
    st.info("StateName or Gender column not available for StateName-wise Submissions by Gender view.")

st.divider()

# ============================================================
# Filters (FILTERED view)
# ============================================================
st.sidebar.header("Filters")
f = df.copy()

sel_state = "All"
sel_progsub = "All"
sel_gender = "All"

if col_state:
    states = sorted([x for x in f[col_state].dropna().unique()])
    sel_state = st.sidebar.selectbox("StateName", ["All"] + states)
    if sel_state != "All":
        f = f[f[col_state] == sel_state]

if col_progsub:
    progsubs = sorted([x for x in f[col_progsub].dropna().unique()])
    sel_progsub = st.sidebar.selectbox("ProgramSubType", ["All"] + progsubs)
    if sel_progsub != "All":
        f = f[f[col_progsub] == sel_progsub]

if col_gender:
    genders = sorted([x for x in f[col_gender].dropna().unique()])
    sel_gender = st.sidebar.selectbox("Gender", ["All"] + genders)
    if sel_gender != "All":
        f = f[f[col_gender] == sel_gender]

show_raw = st.sidebar.checkbox("Show raw data (filtered)", value=False)

# ============================================================
# KPIs (FILTERED)
# ============================================================
st.subheader("Key Metrics (Filtered View)")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Submissions (Rows)", f"{len(f):,}")
k2.metric("Total Children (Unique)", f"{f[col_child].nunique():,}" if col_child else "NA")
k3.metric("Unique States", f"{f[col_state].nunique():,}" if col_state else "NA")
k4.metric("Avg SessionAttended", f"{f['SessionAttended'].mean():.2f}" if "SessionAttended" in f.columns and len(f) else "NA")

st.divider()

# ============================================================
# Metric Distributions (0–7+) (FILTERED)
# ============================================================
st.subheader("Session Category Distributions (0–7+) — Filtered")

present_metrics = [mc for mc in metric_cols if mc in f.columns and f"{mc}_Bucket" in f.columns]
if not present_metrics:
    st.info("None of the requested metric columns were found in the uploaded file.")
else:
    for mc in present_metrics:
        st.markdown(f"### {mc}")

        bucket_col = f"{mc}_Bucket"

        dist = (
            f[bucket_col]
            .fillna("0")
            .astype(str)
            .value_counts()
            .reindex(BUCKET_0_7_ORDER, fill_value=0)
            .reset_index()
        )
        dist.columns = ["Bucket", "Rows"]

        # Table
        st.dataframe(dist, use_container_width=True)

        # Chart with data labels
        chart = bar_with_labels(
            dist,
            x_col="Bucket",
            y_col="Rows",
            x_title="Bucket",
            y_title="Rows",
        )
        st.altair_chart(chart, use_container_width=True)

        st.divider()

# ============================================================
# Export (FILTERED) with timestamp + filters at bottom
# ============================================================
st.subheader("Export (Filtered Data)")

export_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
footer_notes = [
    f"Export Timestamp: {export_ts}",
    f"Filters Applied - StateName: {sel_state}",
    f"Filters Applied - ProgramSubType: {sel_progsub}",
    f"Filters Applied - Gender: {sel_gender}",
]

f_export = add_export_footer_rows(f.copy(), footer_notes)

st.download_button(
    "Download filtered data (CSV) with export details at bottom",
    data=f_export.to_csv(index=False).encode("utf-8"),
    file_name="session_visits_dashboard_v1_filtered.csv",
    mime="text/csv",
)

# ============================================================
# Raw data (FILTERED)
# ============================================================
if show_raw:
    st.subheader("Raw Data (Filtered)")
    st.dataframe(f, use_container_width=True)

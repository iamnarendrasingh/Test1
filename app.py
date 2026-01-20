import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Session Visits Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def read_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)  # requires openpyxl for .xlsx
    raise ValueError("Unsupported file type. Upload CSV or Excel (.xlsx/.xls).")

def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def bar_with_labels(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
    value_format: str = ".1f",
):
    """Altair bar chart with data labels and tooltips."""
    base = alt.Chart(data)

    bar = base.mark_bar().encode(
        x=alt.X(f"{x_col}:N", title=x_title, sort=None),
        y=alt.Y(f"{y_col}:Q", title=y_title),
        tooltip=[
            alt.Tooltip(f"{x_col}:N", title=x_title),
            alt.Tooltip(f"{y_col}:Q", title=y_title, format=value_format),
        ],
    )

    text = base.mark_text(dy=-6, fontSize=12).encode(
        x=alt.X(f"{x_col}:N", sort=None),
        y=alt.Y(f"{y_col}:Q"),
        text=alt.Text(f"{y_col}:Q", format=value_format),
    )

    return bar + text

# --- Bucketing logic depends on ProgramSubType ---
def bucket_sessions_by_program_subtype(program_subtype: str, session_attended: float) -> str:
    """
    Bucketing rules:
      1) life skill -> buckets of size 5: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30+
      2) COMMUNITY LEARNING -> 0, 1-5, 6-10, 11-20, 21-30, 31+
      Else -> default buckets: 0, 1-3, 4-5, 6-10, 11-15, 16-20, 21-30, 31+
    """
    ps = "" if program_subtype is None else str(program_subtype).strip().lower()
    s = pd.to_numeric(session_attended, errors="coerce")
    if pd.isna(s) or s < 0:
        s = 0.0
    s = float(s)

    if ps == "life skill" or "life skill" in ps:
        if s >= 30:
            return "30+"
        lo = int(s // 5) * 5
        hi = lo + 4
        return f"{lo}-{hi}"

    if ps == "community learning" or "community" in ps:
        if s == 0:
            return "0"
        if 1 <= s <= 5:
            return "1-5"
        if 6 <= s <= 10:
            return "6-10"
        if 11 <= s <= 20:
            return "11-20"
        if 21 <= s <= 30:
            return "21-30"
        return "31+"

    if s == 0:
        return "0"
    if 1 <= s <= 3:
        return "1-3"
    if 4 <= s <= 5:
        return "4-5"
    if 6 <= s <= 10:
        return "6-10"
    if 11 <= s <= 15:
        return "11-15"
    if 16 <= s <= 20:
        return "16-20"
    if 21 <= s <= 30:
        return "21-30"
    return "31+"

def ordered_buckets_for_program_subtype(program_subtype: str) -> list[str]:
    ps = "" if program_subtype is None else str(program_subtype).strip().lower()

    if ps == "life skill" or "life skill" in ps:
        return ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30+"]

    if ps == "community learning" or "community" in ps:
        return ["0", "1-5", "6-10", "11-20", "21-30", "31+"]

    return ["0", "1-3", "4-5", "6-10", "11-15", "16-20", "21-30", "31+"]

# -----------------------------
# UI - Header
# -----------------------------
st.title("Session Visits Dashboard")

uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])
if not uploaded:
    st.info("Upload a file to begin.")
    st.stop()

# -----------------------------
# Load Data
# -----------------------------
try:
    df = read_uploaded_file(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]

# Identify columns
col_region = pick_col(df, ["RegionName", "Region", "region"])
col_state = pick_col(df, ["StateName", "State", "state"])
col_proj = pick_col(df, ["ProjectID", "ProjectId", "Project", "project"])
col_child = pick_col(df, ["ChildId", "ChildID", "child_id", "childid"])
col_gender = pick_col(df, ["Gender", "gender"])
col_progsub = pick_col(df, ["ProgramSubType", "ProgramSubTypeName", "Program Sub Type", "programsubtype"])

# Convert numerics
df = safe_numeric(df, [
    "SessionAttended", "Total",
    "InPersonSessions", "VirtualSessions",
    "WorksheetSessions", "WorkshopSessions",
    "SummerCampSessions", "RegularSessions",
    "NumeracySessions", "LiteracySessions"
])

# -----------------------------
# Data quality checks & cleaning for attendance math
# -----------------------------
if "SessionAttended" in df.columns and "Total" in df.columns:
    df["DQ_TotalNonPositive"] = df["Total"] <= 0
    df["DQ_AttendedGTTotal"] = df["SessionAttended"] > df["Total"]

    df["Total_safe"] = df["Total"].where(df["Total"] > 0, np.nan)
    df["SessionAttended_safe"] = np.where(
        df["Total_safe"].notna(),
        np.minimum(df["SessionAttended"], df["Total"]),
        np.nan
    )

    df["AttendancePct_safe"] = np.where(
        df["Total_safe"] > 0,
        (df["SessionAttended_safe"] / df["Total_safe"]) * 100,
        np.nan
    )
else:
    df["DQ_TotalNonPositive"] = False
    df["DQ_AttendedGTTotal"] = False
    df["Total_safe"] = np.nan
    df["SessionAttended_safe"] = np.nan
    df["AttendancePct_safe"] = np.nan

# Ensure ProgramSubType exists
if col_progsub:
    df[col_progsub] = df[col_progsub].astype(str).replace({"nan": np.nan, "None": np.nan})
else:
    col_progsub = "ProgramSubType"
    df[col_progsub] = np.nan

# Create bucket column (uses original SessionAttended for visit buckets)
if "SessionAttended" in df.columns:
    df["SessionAttendedBucket"] = df.apply(
        lambda r: bucket_sessions_by_program_subtype(r.get(col_progsub), r.get("SessionAttended")),
        axis=1
    )
else:
    df["SessionAttendedBucket"] = "NA"

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
f = df.copy()

if col_region:
    regions = sorted([x for x in f[col_region].dropna().unique()])
    selected_region = st.sidebar.selectbox("Region", ["All"] + regions)
    if selected_region != "All":
        f = f[f[col_region] == selected_region]

if col_state:
    states = sorted([x for x in f[col_state].dropna().unique()])
    selected_state = st.sidebar.selectbox("State", ["All"] + states)
    if selected_state != "All":
        f = f[f[col_state] == selected_state]

if col_proj:
    projs = sorted([x for x in f[col_proj].dropna().unique()])
    selected_proj = st.sidebar.selectbox("Project", ["All"] + projs)
    if selected_proj != "All":
        f = f[f[col_proj] == selected_proj]

progsubs = sorted([x for x in f[col_progsub].dropna().unique()])
selected_progsub = st.sidebar.selectbox("ProgramSubType", ["All"] + progsubs)
if selected_progsub != "All":
    f = f[f[col_progsub] == selected_progsub]

# Gender filter
if col_gender:
    genders = sorted([x for x in f[col_gender].dropna().unique()])
    selected_gender = st.sidebar.selectbox("Gender", ["All"] + genders)
    if selected_gender != "All":
        f = f[f[col_gender] == selected_gender]

st.sidebar.divider()
show_raw = st.sidebar.checkbox("Show raw data table", value=False)

bucket_order = ordered_buckets_for_program_subtype(selected_progsub if selected_progsub != "All" else "")

# -----------------------------
# Data Quality Summary (Filtered)
# -----------------------------
bad_gt = int(f["DQ_AttendedGTTotal"].sum()) if "DQ_AttendedGTTotal" in f.columns else 0
bad_tot = int(f["DQ_TotalNonPositive"].sum()) if "DQ_TotalNonPositive" in f.columns else 0
if bad_gt > 0 or bad_tot > 0:
    st.warning(
        f"Data quality detected in filtered data: "
        f"{bad_gt:,} rows have SessionAttended > Total; "
        f"{bad_tot:,} rows have Total <= 0. "
        "Attendance % is computed using cleaned values (attended clipped to total; non-positive totals excluded)."
    )

# -----------------------------
# KPIs (use SAFE sums)
# -----------------------------
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

records = len(f)
unique_children = f[col_child].nunique() if col_child else np.nan

attended_safe_sum = f["SessionAttended_safe"].sum(skipna=True) if "SessionAttended_safe" in f.columns else np.nan
total_safe_sum = f["Total_safe"].sum(skipna=True) if "Total_safe" in f.columns else np.nan

overall_pct = (attended_safe_sum / total_safe_sum * 100) if (isinstance(total_safe_sum, (int, float, np.number)) and total_safe_sum and total_safe_sum > 0) else np.nan

# Extra KPIs derived from existing data
zero_visit = int((f["SessionAttended"] == 0).sum()) if "SessionAttended" in f.columns else 0
zero_visit_pct = (zero_visit / records * 100) if records > 0 else np.nan

avg_visits = f["SessionAttended"].mean() if "SessionAttended" in f.columns and records > 0 else np.nan

hf10 = int((f["SessionAttended"] >= 10).sum()) if "SessionAttended" in f.columns else 0
hf10_pct = (hf10 / records * 100) if records > 0 else np.nan

k1.metric("Records", f"{records:,}")
k2.metric("Unique Children", f"{int(unique_children):,}" if not np.isnan(unique_children) else "NA")
k3.metric("Sessions Attended (Safe)", f"{attended_safe_sum:,.0f}" if not np.isnan(attended_safe_sum) else "NA")
k4.metric("Total Sessions (Safe)", f"{total_safe_sum:,.0f}" if not np.isnan(total_safe_sum) else "NA")
k5.metric("Attendance % (Safe)", f"{overall_pct:,.1f}%" if not np.isnan(overall_pct) else "NA")
k6.metric("Zero-Visit Records", f"{zero_visit:,}" + (f" ({zero_visit_pct:.1f}%)" if not np.isnan(zero_visit_pct) else ""))
k7.metric("Avg Visits / Record", f"{avg_visits:.2f}" if not np.isnan(avg_visits) else "NA")

st.divider()

# -----------------------------
# Session Attended Buckets (Counts) - TABLE FORMAT
# -----------------------------
st.subheader("Session Attended Buckets (Counts) - Table")

if "SessionAttended" in f.columns:
    bdf = (
        f["SessionAttendedBucket"]
        .value_counts()
        .reindex(bucket_order, fill_value=0)
        .reset_index()
    )
    bdf.columns = ["Bucket", "Count"]
    bdf["Bucket"] = bdf["Bucket"].astype(str)
    bdf["Count"] = bdf["Count"].astype(int)

    total_records = int(bdf["Count"].sum())
    bdf["Percent"] = np.where(total_records > 0, (bdf["Count"] / total_records) * 100, 0.0).round(1)

    st.dataframe(bdf, use_container_width=True)
else:
    st.info("SessionAttended column not available; cannot compute buckets.")

st.divider()

# -----------------------------
# ProgramSubType Effectiveness (extra from same data)
# -----------------------------
st.subheader("ProgramSubType Effectiveness (derived)")

if col_progsub and "SessionAttended" in f.columns:
    prog = (
        f.groupby(col_progsub)
        .agg(
            Records=("SessionAttended", "size"),
            AvgVisits=("SessionAttended", "mean"),
            ZeroVisitPct=("SessionAttended", lambda x: (x.eq(0).mean() * 100) if len(x) else np.nan),
            HighFreq10Pct=("SessionAttended", lambda x: (x.ge(10).mean() * 100) if len(x) else np.nan),
        )
        .reset_index()
    )
    prog["AvgVisits"] = prog["AvgVisits"].round(2)
    prog["ZeroVisitPct"] = prog["ZeroVisitPct"].round(1)
    prog["HighFreq10Pct"] = prog["HighFreq10Pct"].round(1)

    st.dataframe(prog.sort_values("AvgVisits", ascending=False), use_container_width=True)
else:
    st.info("ProgramSubType or SessionAttended not available.")

st.divider()

# -----------------------------
# Top Projects by Attendance % (table + labeled chart) - uses SAFE values
# -----------------------------
st.subheader("Top Projects by Attendance % (table + labeled chart)")

if col_proj and "SessionAttended_safe" in f.columns and "Total_safe" in f.columns:
    p = f.groupby(col_proj)[["SessionAttended_safe", "Total_safe"]].sum().reset_index()
    p = p.rename(columns={"SessionAttended_safe": "SessionAttended", "Total_safe": "Total"})
    p["AttendancePct"] = np.where(p["Total"] > 0, (p["SessionAttended"] / p["Total"]) * 100, np.nan)
    p = p.sort_values("AttendancePct", ascending=False)

    st.dataframe(p.head(20), use_container_width=True)

    top10 = p.head(10).copy()
    top10[col_proj] = top10[col_proj].astype(str)

    proj_chart = bar_with_labels(
        top10,
        x_col=col_proj,
        y_col="AttendancePct",
        x_title="Project",
        y_title="Attendance %",
        value_format=".1f",
    )
    st.altair_chart(proj_chart, use_container_width=True)
else:
    st.info("Project or required numeric columns not available.")

st.divider()

# -----------------------------
# State-wise Attendance % (Bottom) - Bubble chart (name-only)
# -----------------------------
st.subheader("State-wise Attendance % (Bottom View)")

if col_state and "SessionAttended_safe" in f.columns and "Total_safe" in f.columns:
    state_agg = f.groupby(col_state)[["SessionAttended_safe", "Total_safe"]].sum().reset_index()
    state_agg = state_agg.rename(columns={"SessionAttended_safe": "SessionAttended", "Total_safe": "Total"})
    state_agg["AttendancePct"] = np.where(state_agg["Total"] > 0, (state_agg["SessionAttended"] / state_agg["Total"]) * 100, np.nan)
    state_agg[col_state] = state_agg[col_state].astype(str)

    bubble = alt.Chart(state_agg).mark_circle().encode(
        x=alt.X("AttendancePct:Q", title="Attendance % (Safe)"),
        y=alt.Y(f"{col_state}:N", title="State", sort="-x"),
        size=alt.Size("Total:Q", title="Total Sessions (Safe)"),
        tooltip=[
            alt.Tooltip(f"{col_state}:N", title="State"),
            alt.Tooltip("SessionAttended:Q", title="Sessions Attended (Safe)", format=",.0f"),
            alt.Tooltip("Total:Q", title="Total Sessions (Safe)", format=",.0f"),
            alt.Tooltip("AttendancePct:Q", title="Attendance % (Safe)", format=".1f"),
        ],
    )
    st.altair_chart(bubble, use_container_width=True)
else:
    st.info("State or required columns not available to create the view.")

st.divider()

# -----------------------------
# Download filtered data (extra quick win)
# -----------------------------
st.subheader("Export")
st.download_button(
    "Download filtered data (CSV)",
    data=f.to_csv(index=False).encode("utf-8"),
    file_name="session_visits_filtered.csv",
    mime="text/csv",
)

# -----------------------------
# Raw data
# -----------------------------
if show_raw:
    st.subheader("Raw Data (Filtered)")
    st.dataframe(f, use_container_width=True)

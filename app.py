import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="YM Attendance Dashboard", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def read_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
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

# -----------------------------
# UI - Header
# -----------------------------
st.title("YM Attendance Dashboard (Upload → Filter → View)")

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

# Identify core columns based on your dummy data
col_region = pick_col(df, ["RegionName", "Region", "region"])
col_state  = pick_col(df, ["StateName", "State", "state"])
col_proj   = pick_col(df, ["ProjectID", "ProjectId", "Project", "project"])
col_child  = pick_col(df, ["ChildId", "ChildID", "child_id", "childid"])
col_gender = pick_col(df, ["Gender", "gender"])
col_month  = pick_col(df, ["DoJ_YM_Str", "YM", "Month", "month", "YearMonth"])

# Numeric columns we expect
df = safe_numeric(df, [
    "SessionAttended", "Total",
    "InPersonSessions", "VirtualSessions",
    "WorksheetSessions", "WorkshopSessions",
    "SummerCampSessions", "RegularSessions",
    "NumeracySessions", "LiteracySessions"
])

# Attendance rate
if "SessionAttended" in df.columns and "Total" in df.columns:
    df["AttendancePct"] = np.where(df["Total"] > 0, (df["SessionAttended"] / df["Total"]) * 100, np.nan)

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

st.sidebar.divider()
show_raw = st.sidebar.checkbox("Show raw data table", value=False)

# -----------------------------
# KPIs
# -----------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

records = len(f)
unique_children = f[col_child].nunique() if col_child else np.nan
attended_sum = f["SessionAttended"].sum() if "SessionAttended" in f.columns else np.nan
total_sum = f["Total"].sum() if "Total" in f.columns else np.nan
overall_pct = (attended_sum / total_sum * 100) if (isinstance(total_sum, (int, float, np.number)) and total_sum and total_sum > 0) else np.nan

kpi1.metric("Records", f"{records:,}")
kpi2.metric("Unique Children", f"{int(unique_children):,}" if not np.isnan(unique_children) else "NA")
kpi3.metric("Sessions Attended", f"{attended_sum:,.0f}" if not np.isnan(attended_sum) else "NA")
kpi4.metric("Total Sessions", f"{total_sum:,.0f}" if not np.isnan(total_sum) else "NA")
kpi5.metric("Attendance %", f"{overall_pct:,.1f}%" if not np.isnan(overall_pct) else "NA")

st.divider()

# -----------------------------
# Charts Row 1
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Attendance by Month")
    if col_month and "SessionAttended" in f.columns and "Total" in f.columns:
        tmp = f.groupby(col_month, dropna=False)[["SessionAttended", "Total"]].sum().reset_index()
        tmp["AttendancePct"] = np.where(tmp["Total"] > 0, (tmp["SessionAttended"] / tmp["Total"]) * 100, np.nan)
        # Try to sort if month looks like YYYY-MM
        try:
            tmp["_dt"] = pd.to_datetime(tmp[col_month].astype(str), errors="coerce")
            tmp = tmp.sort_values("_dt")
        except Exception:
            pass
        st.line_chart(tmp.set_index(col_month)["AttendancePct"])
        st.caption("Shows % attendance (SessionAttended / Total) by month-like field.")
    else:
        st.warning("Month column or required numeric columns not found. Expected: DoJ_YM_Str (or similar), SessionAttended, Total.")

with c2:
    st.subheader("Attendance Distribution (Record Level)")
    if "AttendancePct" in f.columns:
        hist_data = f["AttendancePct"].dropna()
        if len(hist_data) > 0:
            st.bar_chart(pd.cut(hist_data, bins=[0, 20, 40, 60, 80, 100], include_lowest=True).value_counts().sort_index())
        else:
            st.info("No non-null attendance % values after filtering.")
    else:
        st.warning("AttendancePct could not be computed (need SessionAttended and Total).")

# -----------------------------
# Charts Row 2
# -----------------------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("Attendance by Gender")
    if col_gender and "SessionAttended" in f.columns and "Total" in f.columns:
        g = f.groupby(col_gender)[["SessionAttended", "Total"]].sum()
        g["AttendancePct"] = np.where(g["Total"] > 0, (g["SessionAttended"] / g["Total"]) * 100, np.nan)
        st.bar_chart(g["AttendancePct"])
    else:
        st.info("Gender or required columns not available.")

with c4:
    st.subheader("Session Mix (In-Person vs Virtual)")
    if "InPersonSessions" in f.columns or "VirtualSessions" in f.columns:
        mix = {}
        if "InPersonSessions" in f.columns:
            mix["InPersonSessions"] = float(f["InPersonSessions"].sum())
        if "VirtualSessions" in f.columns:
            mix["VirtualSessions"] = float(f["VirtualSessions"].sum())
        mix_df = pd.DataFrame({"Type": list(mix.keys()), "Count": list(mix.values())}).set_index("Type")
        st.bar_chart(mix_df["Count"])
    else:
        st.info("InPersonSessions / VirtualSessions columns not available.")

st.divider()

# -----------------------------
# Top Projects table
# -----------------------------
st.subheader("Top Projects by Attendance %")
if col_proj and "SessionAttended" in f.columns and "Total" in f.columns:
    p = f.groupby(col_proj)[["SessionAttended", "Total"]].sum().reset_index()
    p["AttendancePct"] = np.where(p["Total"] > 0, (p["SessionAttended"] / p["Total"]) * 100, np.nan)
    p = p.sort_values("AttendancePct", ascending=False)
    st.dataframe(p.head(20), use_container_width=True)
else:
    st.info("Project or required numeric columns not available.")

# -----------------------------
# Raw data
# -----------------------------
if show_raw:
    st.subheader("Raw Data (Filtered)")
    st.dataframe(f, use_container_width=True)

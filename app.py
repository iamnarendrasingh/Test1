import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="YM Attendance Dashboard", layout="wide")

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

def line_with_point_labels(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_title: str,
    y_title: str,
    value_format: str = ".1f",
):
    """Altair line chart with points + labels + tooltips."""
    base = alt.Chart(data)

    line = base.mark_line().encode(
        x=alt.X(f"{x_col}:N", title=x_title, sort=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(f"{y_col}:Q", title=y_title),
        tooltip=[
            alt.Tooltip(f"{x_col}:N", title=x_title),
            alt.Tooltip(f"{y_col}:Q", title=y_title, format=value_format),
        ],
    )

    points = base.mark_point(size=70).encode(
        x=alt.X(f"{x_col}:N", sort=None),
        y=alt.Y(f"{y_col}:Q"),
        tooltip=[
            alt.Tooltip(f"{x_col}:N", title=x_title),
            alt.Tooltip(f"{y_col}:Q", title=y_title, format=value_format),
        ],
    )

    labels = base.mark_text(dy=-10, fontSize=11).encode(
        x=alt.X(f"{x_col}:N", sort=None),
        y=alt.Y(f"{y_col}:Q"),
        text=alt.Text(f"{y_col}:Q", format=value_format),
    )

    return line + points + labels

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

# Identify columns (based on your dummy data)
col_region = pick_col(df, ["RegionName", "Region", "region"])
col_state  = pick_col(df, ["StateName", "State", "state"])
col_proj   = pick_col(df, ["ProjectID", "ProjectId", "Project", "project"])
col_child  = pick_col(df, ["ChildId", "ChildID", "child_id", "childid"])
col_gender = pick_col(df, ["Gender", "gender"])
col_month  = pick_col(df, ["DoJ_YM_Str", "YM", "Month", "month", "YearMonth"])

# Convert numerics
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

# Hardening: ensure AttendancePct in range
if "AttendancePct" in df.columns:
    df["AttendancePct"] = pd.to_numeric(df["AttendancePct"], errors="coerce")
    df.loc[(df["AttendancePct"] < 0) | (df["AttendancePct"] > 100), "AttendancePct"] = np.nan

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
    st.subheader("Attendance by Month (with labels)")

    if col_month and "SessionAttended" in f.columns and "Total" in f.columns:
        tmp = f.groupby(col_month, dropna=False)[["SessionAttended", "Total"]].sum().reset_index()
        tmp["AttendancePct"] = np.where(tmp["Total"] > 0, (tmp["SessionAttended"] / tmp["Total"]) * 100, np.nan)

        # Sort if month is like YYYY-MM
        try:
            tmp["_dt"] = pd.to_datetime(tmp[col_month].astype(str), errors="coerce")
            tmp = tmp.sort_values("_dt")
        except Exception:
            pass

        tmp[col_month] = tmp[col_month].astype(str)
        chart = line_with_point_labels(
            tmp,
            x_col=col_month,
            y_col="AttendancePct",
            x_title="Month",
            y_title="Attendance %",
            value_format=".1f",
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Month column or required numeric columns not found. Expected: DoJ_YM_Str (or similar), SessionAttended, Total.")

with c2:
    st.subheader("Attendance Distribution (Record Level) with labels")

    if "AttendancePct" in f.columns:
        hist_data = f["AttendancePct"].dropna()
        if len(hist_data) > 0:
            bins = [0, 20, 40, 60, 80, 100]
            labels = ["0–20", "20–40", "40–60", "60–80", "80–100"]

            binned = pd.cut(hist_data, bins=bins, labels=labels, include_lowest=True)
            counts = binned.value_counts().reindex(labels, fill_value=0)

            hist_df = pd.DataFrame({"Attendance Band": labels, "Count": counts.values})
            chart = bar_with_labels(
                hist_df,
                x_col="Attendance Band",
                y_col="Count",
                x_title="Attendance Band",
                y_title="Record Count",
                value_format=".0f",
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No non-null attendance % values after filtering.")
    else:
        st.warning("AttendancePct could not be computed (need SessionAttended and Total).")

st.divider()

# -----------------------------
# Charts Row 2
# -----------------------------
c3, c4 = st.columns(2)

with c3:
    st.subheader("Attendance by Gender (with labels)")

    if col_gender and "SessionAttended" in f.columns and "Total" in f.columns:
        g = f.groupby(col_gender)[["SessionAttended", "Total"]].sum().reset_index()
        g["AttendancePct"] = np.where(g["Total"] > 0, (g["SessionAttended"] / g["Total"]) * 100, np.nan)

        g[col_gender] = g[col_gender].astype(str)
        chart = bar_with_labels(
            g,
            x_col=col_gender,
            y_col="AttendancePct",
            x_title="Gender",
            y_title="Attendance %",
            value_format=".1f",
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Gender or required columns not available.")

with c4:
    st.subheader("Session Mix (In-Person vs Virtual) with labels")

    if "InPersonSessions" in f.columns or "VirtualSessions" in f.columns:
        mix = []
        if "InPersonSessions" in f.columns:
            mix.append({"Type": "InPersonSessions", "Count": float(f["InPersonSessions"].sum())})
        if "VirtualSessions" in f.columns:
            mix.append({"Type": "VirtualSessions", "Count": float(f["VirtualSessions"].sum())})

        mix_df = pd.DataFrame(mix)
        if len(mix_df) > 0:
            chart = bar_with_labels(
                mix_df,
                x_col="Type",
                y_col="Count",
                x_title="Type",
                y_title="Sessions",
                value_format=".0f",
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No session mix values found after filtering.")
    else:
        st.info("InPersonSessions / VirtualSessions columns not available.")

st.divider()

# -----------------------------
# Optional: Additional labelled chart (Top 10 Projects)
# -----------------------------
st.subheader("Top Projects by Attendance % (table + labeled chart)")

if col_proj and "SessionAttended" in f.columns and "Total" in f.columns:
    p = f.groupby(col_proj)[["SessionAttended", "Total"]].sum().reset_index()
    p["AttendancePct"] = np.where(p["Total"] > 0, (p["SessionAttended"] / p["Total"]) * 100, np.nan)
    p = p.sort_values("AttendancePct", ascending=False)

    # Table
    st.dataframe(p.head(20), use_container_width=True)

    # Labeled bar chart (top 10)
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

# -----------------------------
# Raw data
# -----------------------------
if show_raw:
    st.subheader("Raw Data (Filtered)")
    st.dataframe(f, use_container_width=True)

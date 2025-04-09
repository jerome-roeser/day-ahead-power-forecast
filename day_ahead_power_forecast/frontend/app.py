import datetime as dt

import matplotlib.dates as dates
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(layout="wide")


### API call ==================================================================
# base_url = "http://127.0.0.1:8000"
base_url = "https://power-v2-pdymu2v2na-ew.a.run.app"
# ------------------------------------------------------------------------------


def call_visu(today_date):
    params_visu = {
        "input_date": today_date,  # '2000-05-15' (dt.date())
        "power_source": "pv",
        "capacity": "true",
    }

    endpoint_visu = "/visualisation"
    url_visu = f"{base_url}{endpoint_visu}"
    response_visu = requests.get(url_visu, params_visu).json()

    plot_df = pd.DataFrame.from_dict(response_visu)
    plot_df.utc_time = pd.to_datetime(plot_df.utc_time, utc=True)

    return plot_df


# Session states and Callbacks =================================================
# (see:https://docs.streamlit.io/library/advanced-features/button-behavior-and-examples)

# initialize session states
if "today" not in st.session_state:
    st.session_state["today"] = dt.date(2020, 6, 2)  # default date

if "plot_df" not in st.session_state:
    st.session_state["plot_df"] = call_visu(st.session_state["today"])


# define callbacks
def add_day():
    st.session_state["today"] += dt.timedelta(days=1)
    st.session_state["plot_df"] = call_visu(st.session_state["today"])


def sub_day():
    st.session_state["today"] -= dt.timedelta(days=1)
    st.session_state["plot_df"] = call_visu(st.session_state["today"])


# ==============================================================================
# ====================== Streamlit Interface ===================================

### Sidebar ====================================================================
st.sidebar.markdown("""
   # User Input
   """)

# Calender select
calender_today = st.sidebar.date_input(
    label="Simulated today",
    value=st.session_state["today"],
    min_value=dt.date(2020, 1, 1),
    max_value=dt.date(2022, 12, 30),
)

if st.session_state["today"] != calender_today:
    st.session_state["today"] = calender_today
    st.session_state["plot_df"] = call_visu(st.session_state["today"])


# Move a day forth and back
columns = st.sidebar.columns(2)
columns[0].button("Day before", on_click=sub_day)
columns[1].button("Day after", on_click=add_day)

# Show values
show_true = st.sidebar.radio("Show true values", ("No", "Yes"))


### Main window ====================================================================
# <p style="text-align: center;">Text_content</p>

"""
# Day-Ahead Power Forecast

#

"""

# #### Today: **{st.session_state['today']}**   Day-Ahead: \
#      **{st.session_state['today'] + pd.Timedelta(days=1)}**

daybehind = st.session_state["today"] - pd.Timedelta(days=1)
today = st.session_state["today"]
dayahead = st.session_state["today"] + pd.Timedelta(days=1)


columns = st.columns(7)
columns[1].write(f" ##### Day-Behind: **{daybehind.strftime('%m/%d/%Y')}**")
columns[3].write(f" ##### Today: **{today.strftime('%m/%d/%Y')}**")
columns[5].write(f" ###### Day-Ahead: **{dayahead.strftime('%m/%d/%Y')}**")

### Show plots =================================================================
# used in the plots
today_date = st.session_state["today"]
plot_df = st.session_state["plot_df"]
# ------------------------------------------------------------------------------

### capacity

# time variables
today_dt = pd.Timestamp(today_date, tz="UTC")
time = plot_df.utc_time.values

sep_future = today_dt + pd.Timedelta(days=1)
sep_past = today_dt
sep_order = today_dt + pd.Timedelta(hours=12)

# plot
fig, ax = plt.subplots(figsize=(15, 5))

ax.axvline(sep_past, color="k", linewidth=0.7)
ax.axvline(sep_future, color="k", linewidth=0.7)
ax.vlines(sep_order, ymin=0, ymax=100, color="k", linewidth=0.7, linestyle="--")

# stats
alpha_stats = 0.2
ax.step(
    time,
    plot_df["min"].values,
    where="pre",
    color="k",
    linestyle=":",
    alpha=alpha_stats,
    label="min",
)
ax.step(
    time,
    plot_df["max"].values,
    where="pre",
    color="k",
    linestyle=":",
    alpha=alpha_stats,
    label="max",
)
ax.step(
    time,
    plot_df["mean"].values,
    where="pre",
    color="k",
    linestyle="-",
    alpha=alpha_stats,
    label="mean",
)

lower_bound = plot_df["mean"].values - 1 * plot_df["std"].values
upper_bound = plot_df["mean"].values + 1 * plot_df["std"].values
ax.fill_between(
    time,
    lower_bound,
    upper_bound,
    step="pre",
    color="gray",
    alpha=alpha_stats,
    label="std",
)

# true current production data
current = 37  # current production data
ax.step(
    time[:current],
    plot_df.cap_fac.values[:current],
    where="pre",
    color="orange",
    linewidth=4,
    label="true",
)

# prediction day ahead data
hori = -24
ax.step(
    time[hori:],
    plot_df.pred.values[hori:],
    where="pre",
    color="orange",
    linewidth=4,
    linestyle=":",
    label="pred",
)

###
if show_true == "Yes":
    ax.step(
        time[-36:],
        plot_df.cap_fac.values[-36:],
        where="pre",
        color="orange",
        linewidth=4,
        linestyle="-",
        alpha=0.4,
    )
    st.sidebar.write("")
else:
    st.sidebar.write("")

# date ticks
ax.xaxis.set_major_locator(dates.HourLocator(byhour=range(24), interval=12, tz="UTC"))
ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M %d/%m/%Y"))

ax.set_xlim(today_dt - pd.Timedelta(days=1), today_dt + pd.Timedelta(days=2))
ax.set_ylim(0, 120.0)
ax.set_xlabel("Time")
ax.set_ylabel("Capacity factor in %")

ax.annotate("Day-Ahead", (0.77, 0.9), xycoords="subfigure fraction")
ax.annotate("Today", (0.48, 0.9), xycoords="subfigure fraction")
ax.annotate("Day-Behind", (0.15, 0.9), xycoords="subfigure fraction")
ax.annotate("Order book closed", (0.51, 0.77), xycoords="subfigure fraction")
# ax.set_title(f"Day Ahead prediction for { sep_future.strftime('%d/%m/%Y') }")

ax.legend()
##
st.pyplot(fig)

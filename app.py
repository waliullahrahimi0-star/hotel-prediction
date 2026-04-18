"""
Hotel Reservation Cancellation Predictor
A professional machine learning application built with Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Roboto', 'Source Sans Pro', sans-serif; }
section[data-testid="stSidebar"] { background-color: #f8f9fb; border-right: 1px solid #e8eaf0; }
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }
.main-title { font-size: 2rem; font-weight: 700; color: #111827; letter-spacing: -0.5px; margin-bottom: 0.2rem; }
.main-subtitle { font-size: 1rem; color: #6b7280; font-weight: 400; margin-bottom: 2rem; }
.result-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1.6rem 2rem; margin-bottom: 1rem; }
.result-card-success { border-left: 5px solid #16a34a; background: #f0fdf4; }
.result-card-fail { border-left: 5px solid #dc2626; background: #fff5f5; }
.result-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #9ca3af; margin-bottom: 0.5rem; }
.result-value { font-size: 1.6rem; font-weight: 700; color: #111827; margin-bottom: 0.3rem; }
.result-message { font-size: 0.9rem; color: #374151; line-height: 1.6; }
.sidebar-section { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #9ca3af; padding: 0.8rem 0 0.3rem 0; border-top: 1px solid #e5e7eb; margin-top: 0.5rem; }
.sidebar-section:first-of-type { border-top: none; padding-top: 0; }
.prob-bar-container { background: #f3f4f6; border-radius: 6px; height: 8px; margin-top: 0.6rem; overflow: hidden; }
.prob-bar-fill-success { background: #16a34a; height: 8px; border-radius: 6px; }
.prob-bar-fill-fail { background: #dc2626; height: 8px; border-radius: 6px; }
.divider { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }
.summary-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 0.5rem; }
.summary-table thead tr { border-bottom: 1px solid #d1d5db; }
.summary-table thead th { text-align: left; padding: 0.55rem 0.75rem; font-size: 0.78rem; font-weight: 500; color: #9ca3af; letter-spacing: 0.02em; }
.summary-table tbody tr { border-bottom: 1px solid #f3f4f6; }
.summary-table tbody tr:last-child { border-bottom: none; }
.summary-table tbody td { padding: 0.6rem 0.75rem; color: #111827; font-size: 0.875rem; line-height: 1.5; }
.summary-table tbody td:first-child { color: #374151; font-weight: 400; width: 48%; }
.summary-table tbody td:last-child { color: #111827; font-weight: 500; }
.summary-table tbody tr:nth-child(even) { background: #f9fafb; }
div[data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; }
div[data-testid="stButton"] button { background: #111827; color: #ffffff; border: none; border-radius: 7px; padding: 0.55rem 1.5rem; font-weight: 500; font-size: 0.9rem; width: 100%; transition: background 0.2s; }
div[data-testid="stButton"] button:hover { background: #374151; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

CATEGORICAL_COLS = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
NUMERICAL_COLS = [
    "no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights",
    "required_car_parking_space", "lead_time", "arrival_year", "arrival_month",
    "arrival_date", "repeated_guest", "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled", "avg_price_per_room", "no_of_special_requests",
]
MEAL_OPTIONS    = ["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"]
ROOM_OPTIONS    = ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4",
                   "Room_Type 5", "Room_Type 6", "Room_Type 7"]
SEGMENT_OPTIONS = ["Online", "Offline", "Corporate", "Aviation", "Complementary"]
MONTH_NAMES     = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
                   7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = pd.read_csv("Hotel_Reservations.csv")
    df["target"] = (df["booking_status"] == "Canceled").astype(int)
    return df

@st.cache_resource(show_spinner=False)
def train_model():
    df = load_and_prepare_data()
    X = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y = df["target"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    num_t = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_t = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                      ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre   = ColumnTransformer([("num", num_t, NUMERICAL_COLS), ("cat", cat_t, CATEGORICAL_COLS)])
    pipe  = Pipeline([("preprocessor", pre),
                      ("classifier", RandomForestClassifier(
                          n_estimators=150, max_depth=12, min_samples_split=10,
                          min_samples_leaf=5, random_state=42, class_weight="balanced", n_jobs=-1))])
    pipe.fit(X_train, y_train)
    return pipe

with st.spinner("Preparing model — this may take a moment on first load…"):
    model = train_model()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Input Details")
    st.markdown("Enter the booking details below to generate a prediction.")
    st.markdown("---")

    st.markdown('<div class="sidebar-section">Guest Details</div>', unsafe_allow_html=True)
    no_adults        = st.number_input("Adults staying", 0, 4, 2,
                                       help="Number of adults included in the booking")
    no_children      = st.number_input("Children staying", 0, 10, 0,
                                       help="Number of children included in the booking")
    repeated_guest   = st.selectbox("Returning guest", ["No","Yes"],
                                    help="Whether the guest has stayed at this hotel before")
    prev_cancels     = st.number_input("Previous cancellations", 0, 13, 0,
                                       help="Number of times this guest has cancelled previously")
    prev_ok          = st.number_input("Successful previous bookings", 0, 58, 0,
                                       help="Previous bookings completed without cancellation")
    special_req      = st.slider("Special requests count", 0, 5, 0,
                                 help="Total number of special requests made by the guest")

    st.markdown('<div class="sidebar-section">Stay Details</div>', unsafe_allow_html=True)
    weekend_nights   = st.number_input("Weekend nights booked", 0, 7, 1,
                                       help="Number of weekend nights (Saturday or Sunday)")
    week_nights      = st.number_input("Weekday nights booked", 0, 17, 2,
                                       help="Number of weekday nights (Monday to Friday)")
    meal_plan        = st.selectbox("Meal plan selected", MEAL_OPTIONS,
                                    help="Type of meal arrangement included in the reservation")
    room_type        = st.selectbox("Room type booked", ROOM_OPTIONS,
                                    help="Category of room reserved by the guest")
    car_parking      = st.selectbox("Car parking required", ["No","Yes"],
                                    help="Whether the guest has requested a parking space")
    avg_price        = st.number_input("Average room price per night (EUR)", 0.0, 600.0, 100.0,
                                       step=5.0, help="Average price charged per room per night")

    st.markdown('<div class="sidebar-section">Booking and Arrival</div>', unsafe_allow_html=True)
    lead_time        = st.number_input("Days before arrival booking was made", 0, 500, 30,
                                       help="Days between booking date and arrival date")
    booking_type     = st.selectbox("Booking type", SEGMENT_OPTIONS,
                                    help="Channel through which the reservation was made")
    arrival_year     = st.selectbox("Arrival year", [2017,2018], index=1,
                                    help="Year in which the guest is expected to arrive")
    arrival_month    = st.selectbox("Arrival month", list(MONTH_NAMES.keys()),
                                    format_func=lambda x: MONTH_NAMES[x], index=5,
                                    help="Month in which the guest is expected to arrive")
    arrival_date     = st.number_input("Arrival day of month", 1, 31, 15,
                                       help="Day of the month the guest is expected to arrive")
    st.markdown("---")
    predict_button   = st.button("Generate Prediction")

# ── Main ─────────────────────────────────────────────────────────────────────
col_header, _ = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">Hotel Cancellation Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Predicts whether a hotel reservation is likely to be '
                'cancelled based on booking details and guest behaviour.</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

if predict_button:
    rg_val  = 1 if repeated_guest == "Yes" else 0
    cp_val  = 1 if car_parking    == "Yes" else 0

    input_df = pd.DataFrame([{
        "type_of_meal_plan": meal_plan, "room_type_reserved": room_type,
        "market_segment_type": booking_type,
        "no_of_adults": float(no_adults), "no_of_children": float(no_children),
        "no_of_weekend_nights": float(weekend_nights), "no_of_week_nights": float(week_nights),
        "required_car_parking_space": float(cp_val), "lead_time": float(lead_time),
        "arrival_year": float(arrival_year), "arrival_month": float(arrival_month),
        "arrival_date": float(arrival_date), "repeated_guest": float(rg_val),
        "no_of_previous_cancellations": float(prev_cancels),
        "no_of_previous_bookings_not_canceled": float(prev_ok),
        "avg_price_per_room": float(avg_price),
        "no_of_special_requests": float(special_req),
    }])

    pred   = model.predict(input_df)[0]
    proba  = model.predict_proba(input_df)[0]
    cancel_p     = round(proba[1] * 100, 1)
    not_cancel_p = round(proba[0] * 100, 1)

    col1, col2 = st.columns([3, 2])
    with col1:
        if pred == 1:
            st.markdown(f"""
            <div class="result-card result-card-fail">
                <div class="result-label">Predicted Outcome</div>
                <div class="result-value">Likely Cancelled</div>
                <div class="result-message">
                    Based on the booking details provided, this reservation profile is consistent
                    with bookings that have been cancelled. The model assigns a
                    <strong>{cancel_p}%</strong> probability of cancellation.
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-fail" style="width:{cancel_p}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-card-success">
                <div class="result-label">Predicted Outcome</div>
                <div class="result-value">Booking Likely to Proceed</div>
                <div class="result-message">
                    Based on the booking details provided, this reservation profile is consistent
                    with bookings that are completed. The model assigns a
                    <strong>{not_cancel_p}%</strong> probability that the booking will proceed.
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-success" style="width:{not_cancel_p}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card" style="height:100%;">
            <div class="result-label">Confidence Breakdown</div>
            <div style="margin-top:0.8rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
                    <span style="font-size:0.85rem;color:#374151;">Booking Proceeds</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#16a34a;">{not_cancel_p}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-success" style="width:{not_cancel_p}%"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.8rem;margin-bottom:0.4rem;">
                    <span style="font-size:0.85rem;color:#374151;">Cancelled</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#dc2626;">{cancel_p}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-fail" style="width:{cancel_p}%"></div>
                </div>
            </div>
            <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid #e5e7eb;">
                <div class="result-label">Room Type</div>
                <div style="font-size:0.95rem;color:#111827;font-weight:500;">{room_type}</div>
                <div class="result-label" style="margin-top:0.6rem;">Booking Type</div>
                <div style="font-size:0.95rem;color:#111827;font-weight:500;">{booking_type}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Summary of Inputs Used")
    summary_rows = [
        ("Adults staying",                       str(int(no_adults))),
        ("Children staying",                     str(int(no_children))),
        ("Weekend nights booked",                str(int(weekend_nights))),
        ("Weekday nights booked",                str(int(week_nights))),
        ("Meal plan selected",                   meal_plan),
        ("Car parking required",                 car_parking),
        ("Room type booked",                     room_type),
        ("Days before arrival booking was made", str(int(lead_time))),
        ("Arrival year",                         str(int(arrival_year))),
        ("Arrival month",                        MONTH_NAMES[arrival_month]),
        ("Arrival day of month",                 str(int(arrival_date))),
        ("Booking type",                         booking_type),
        ("Returning guest",                      repeated_guest),
        ("Previous cancellations",               str(int(prev_cancels))),
        ("Successful previous bookings",         str(int(prev_ok))),
        ("Average room price per night (EUR)",   f"€{avg_price:,.2f}"),
        ("Special requests count",               str(int(special_req))),
    ]
    rows_html = "".join(f"<tr><td>{f}</td><td>{v}</td></tr>" for f,v in summary_rows)
    st.markdown(f"""
    <table class="summary-table">
        <thead><tr><th>Feature</th><th>Value Entered</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="result-card" style="text-align:center;padding:3rem 2rem;">
        <div style="font-size:1.6rem;margin-bottom:0.5rem;color:#d1d5db;">◈</div>
        <div style="font-size:0.95rem;color:#9ca3af;">
            Complete the fields in the sidebar and click <strong>Generate Prediction</strong>.
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
with st.expander("How this works"):
    st.markdown("""
**Model**

This tool uses a tuned Random Forest classifier trained on historical hotel reservation data. The model was selected after comparing it against Logistic Regression and Decision Tree baselines, consistently achieving the strongest performance across all evaluation metrics.

**Training data**

The model was trained on a dataset of over 36,000 hotel reservations, each with a confirmed outcome of either cancelled or not cancelled. All records have known outcomes, making the dataset well-suited for supervised binary classification.

**Features used**

The model draws on guest profile information, stay characteristics, booking timing, pricing, room type, meal plan, and the guest's prior cancellation history. Lead time and the number of special requests carry particularly strong predictive signal.

**Limitations**

No model can predict reservation cancellations with complete certainty. This tool is intended as a decision-support aid for hotel operations and revenue management teams, not as a definitive classification system. Results may reflect patterns specific to the hotels and time period represented in the training data, and should be interpreted alongside other operational context.
""")

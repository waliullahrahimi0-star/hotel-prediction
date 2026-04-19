"""
Hotel Reservation Cancellation Predictor
Professional ML application — updated and corrected version.

Key improvements over initial version:
  - arrival_year removed from UI and features (only 2017-2018 in training data, not useful)
  - total_guests and total_nights added as engineered features
  - Prediction uses predict_proba() with a calibrated threshold of 0.40
    (lower than the default 0.50 to improve sensitivity to cancellation risk,
     which is the operationally costly class to miss)
  - class_weight="balanced" retained to handle the 67/33 class split
  - Adults input max raised from 4 to 10
  - Soft validation warns when total guests or total nights is zero
  - Meal plan and room type shown with human-readable labels; internal codes
    are mapped back before inference so the model receives the original values
  - Price displayed and entered in USD; divided by 1.1 before model inference
    (EUR is the internal currency of the training dataset)
  - Prediction wording uses measured, professional language
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

# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Styling — identical to original interface
# =============================================================================

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
.validation-warning { background: #fffbeb; border: 1px solid #fbbf24; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #92400e; margin-bottom: 1rem; }
div[data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; }
div[data-testid="stButton"] button { background: #111827; color: #ffffff; border: none; border-radius: 7px; padding: 0.55rem 1.5rem; font-weight: 500; font-size: 0.9rem; width: 100%; transition: background 0.2s; }
div[data-testid="stButton"] button:hover { background: #374151; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Display-to-internal label mappings
# These allow the UI to show human-readable labels while the model receives
# the original encoded values it was trained on.
# =============================================================================

MEAL_DISPLAY_TO_INTERNAL = {
    "Not Selected":                  "Not Selected",
    "Meal Plan 1 (Breakfast Only)":  "Meal Plan 1",
    "Meal Plan 2 (Lunch Included)":  "Meal Plan 2",
    "Meal Plan 3 (Dinner Included)": "Meal Plan 3",
}

ROOM_DISPLAY_TO_INTERNAL = {
    "Simple":    "Room_Type 1",
    "Standard":  "Room_Type 2",
    "Comfort":   "Room_Type 3",
    "Superior":  "Room_Type 4",
    "Deluxe":    "Room_Type 5",
    "VIP 1":     "Room_Type 6",
    "VIP 2":     "Room_Type 7",
}

MEAL_DISPLAY_OPTIONS = list(MEAL_DISPLAY_TO_INTERNAL.keys())
ROOM_DISPLAY_OPTIONS = list(ROOM_DISPLAY_TO_INTERNAL.keys())
SEGMENT_OPTIONS      = ["Online", "Offline", "Corporate", "Aviation", "Complementary"]

MONTH_NAMES = {
    1:"January", 2:"February", 3:"March",    4:"April",
    5:"May",     6:"June",     7:"July",      8:"August",
    9:"September",10:"October",11:"November",12:"December",
}

# Prediction threshold — set below 0.50 to give the cancelled class
# (the minority, but operationally important class) fair sensitivity.
CANCEL_THRESHOLD = 0.40

# EUR-to-USD approximate rate for display conversion.
# The model was trained on EUR prices; user enters USD and we convert back.
EUR_TO_USD = 1.10

# =============================================================================
# Feature column lists — arrival_year excluded (only 2017-2018 in training
# data; including it adds noise rather than signal for deployment)
# =============================================================================

CATEGORICAL_COLS = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]

NUMERICAL_COLS = [
    "no_of_adults", "no_of_children",
    "no_of_weekend_nights", "no_of_week_nights",
    "required_car_parking_space", "lead_time",
    "arrival_month", "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",         # stored in EUR internally
    "no_of_special_requests",
    "total_nights",               # engineered: weekend + weekday nights
    "total_guests",               # engineered: adults + children
]

# =============================================================================
# Data loading and model training (cached)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = pd.read_csv("Hotel_Reservations.csv")
    df["target"] = (df["booking_status"] == "Canceled").astype(int)

    # Derived features
    df["total_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]
    df["total_guests"]  = df["no_of_adults"]         + df["no_of_children"]

    return df


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_and_prepare_data()
    X  = df[CATEGORICAL_COLS + NUMERICAL_COLS]
    y  = df["target"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Numerical: median imputation → standard scaling
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # Categorical: constant imputation → one-hot encoding
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, NUMERICAL_COLS),
        ("cat", cat_transformer, CATEGORICAL_COLS),
    ])

    # class_weight="balanced" compensates for the 67/33 class split so the
    # model does not simply optimise for the majority (Not Cancelled) class.
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    rf_pipeline.fit(X_train, y_train)
    return rf_pipeline


# =============================================================================
# Initialise
# =============================================================================

with st.spinner("Preparing model — this may take a moment on first load…"):
    model = train_model()

# =============================================================================
# Sidebar — inputs
# =============================================================================

with st.sidebar:
    st.markdown("## Input Details")
    st.markdown("Enter the booking details below to generate a prediction.")
    st.markdown("---")

    # --- Guest Details ---
    st.markdown('<div class="sidebar-section">Guest Details</div>', unsafe_allow_html=True)

    no_adults = st.number_input(
        "Adults staying",
        min_value=1, max_value=10, value=2,
        help="Number of adults included in the booking",
    )
    no_children = st.number_input(
        "Children staying",
        min_value=0, max_value=10, value=0,
        help="Number of children included in the booking",
    )
    repeated_guest = st.selectbox(
        "Returning guest",
        options=["No", "Yes"],
        help="Whether the guest has stayed at this hotel before",
    )
    prev_cancels = st.number_input(
        "Previous cancellations",
        min_value=0, max_value=13, value=0,
        help="Number of prior bookings this guest has cancelled",
    )
    prev_ok = st.number_input(
        "Successful previous bookings",
        min_value=0, max_value=58, value=0,
        help="Previous bookings completed by this guest without cancellation",
    )
    special_req = st.slider(
        "Special requests count",
        min_value=0, max_value=5, value=0,
        help="Number of special requests made alongside the booking",
    )

    # --- Stay Details ---
    st.markdown('<div class="sidebar-section">Stay Details</div>', unsafe_allow_html=True)

    weekend_nights = st.number_input(
        "Weekend nights booked",
        min_value=0, max_value=7, value=1,
        help="Number of weekend nights (Saturday or Sunday) in the stay",
    )
    week_nights = st.number_input(
        "Weekday nights booked",
        min_value=0, max_value=17, value=2,
        help="Number of weekday nights (Monday to Friday) in the stay",
    )
    meal_display = st.selectbox(
        "Meal plan selected",
        options=MEAL_DISPLAY_OPTIONS,
        help="Meal arrangement included in the reservation",
    )
    room_display = st.selectbox(
        "Room type booked",
        options=ROOM_DISPLAY_OPTIONS,
        help="Category of room reserved",
    )
    car_parking = st.selectbox(
        "Car parking required",
        options=["No", "Yes"],
        help="Whether the guest has requested a parking space",
    )

    # Price entered in USD; internally converted to EUR for model inference
    avg_price_usd = st.number_input(
        "Average room price per night (USD)",
        min_value=0.0, max_value=660.0, value=110.0, step=5.0,
        help="Average nightly room rate for this booking, in US dollars",
    )

    # --- Booking and Arrival ---
    st.markdown('<div class="sidebar-section">Booking and Arrival</div>', unsafe_allow_html=True)

    lead_time = st.number_input(
        "Days before arrival booking was made",
        min_value=0, max_value=500, value=30,
        help="Number of days between booking date and arrival date",
    )
    booking_type = st.selectbox(
        "Booking type",
        options=SEGMENT_OPTIONS,
        help="Channel or method through which the reservation was made",
    )
    # Arrival year intentionally removed — dataset contains only 2017-2018,
    # making it an uninformative feature for realistic deployment scenarios.
    arrival_month = st.selectbox(
        "Arrival month",
        options=list(MONTH_NAMES.keys()),
        format_func=lambda x: MONTH_NAMES[x],
        index=5,
        help="Month in which the guest is expected to arrive",
    )
    arrival_date = st.number_input(
        "Arrival day of month",
        min_value=1, max_value=31, value=15,
        help="Day of the month on which the guest is expected to arrive",
    )

    st.markdown("---")
    predict_button = st.button("Generate Prediction")

# =============================================================================
# Main content area
# =============================================================================

col_header, _ = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">Hotel Cancellation Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Predicts whether a hotel reservation is likely to be '
        'cancelled based on booking details and guest behaviour.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =============================================================================
# Prediction logic
# =============================================================================

if predict_button:

    # Derived values
    total_nights_val = int(weekend_nights) + int(week_nights)
    total_guests_val = int(no_adults)      + int(no_children)

    # Soft validation — display a warning but do not block the prediction.
    # The model handles edge cases through imputation; the warning simply
    # informs the user that the inputs are unusual.
    warnings_list = []
    if total_guests_val < 1:
        warnings_list.append("Total guests is zero — a valid booking requires at least one adult.")
    if total_nights_val < 1:
        warnings_list.append("Total nights is zero — please enter at least one night for a meaningful prediction.")

    if warnings_list:
        warning_html = "<br>".join(warnings_list)
        st.markdown(
            f'<div class="validation-warning">{warning_html}</div>',
            unsafe_allow_html=True,
        )

    # Resolve display labels to internal model values
    meal_internal  = MEAL_DISPLAY_TO_INTERNAL[meal_display]
    room_internal  = ROOM_DISPLAY_TO_INTERNAL[room_display]
    rg_val         = 1 if repeated_guest == "Yes" else 0
    cp_val         = 1 if car_parking    == "Yes" else 0

    # Convert USD price back to EUR for model inference
    avg_price_eur = avg_price_usd / EUR_TO_USD

    input_df = pd.DataFrame([{
        "type_of_meal_plan":                    meal_internal,
        "room_type_reserved":                   room_internal,
        "market_segment_type":                  booking_type,
        "no_of_adults":                         float(no_adults),
        "no_of_children":                       float(no_children),
        "no_of_weekend_nights":                 float(weekend_nights),
        "no_of_week_nights":                    float(week_nights),
        "required_car_parking_space":           float(cp_val),
        "lead_time":                            float(lead_time),
        "arrival_month":                        float(arrival_month),
        "arrival_date":                         float(arrival_date),
        "repeated_guest":                       float(rg_val),
        "no_of_previous_cancellations":         float(prev_cancels),
        "no_of_previous_bookings_not_canceled": float(prev_ok),
        "avg_price_per_room":                   float(avg_price_eur),   # EUR internally
        "no_of_special_requests":               float(special_req),
        "total_nights":                         float(total_nights_val),
        "total_guests":                         float(total_guests_val),
    }])

    # Use predict_proba() — raw probabilities are more informative and allow
    # a custom threshold independent of the classifier's internal decision boundary.
    proba       = model.predict_proba(input_df)[0]
    cancel_p    = round(proba[1] * 100, 1)
    proceed_p   = round(proba[0] * 100, 1)

    # Apply threshold: predict Cancelled if cancel probability meets or exceeds
    # CANCEL_THRESHOLD. A value of 0.40 provides better recall on the cancelled
    # class than the default 0.50, which is appropriate given the operational
    # cost of missing a cancellation.
    predicted_cancel = proba[1] >= CANCEL_THRESHOLD

    # --- Result cards ---
    col1, col2 = st.columns([3, 2])

    with col1:
        if predicted_cancel:
            st.markdown(f"""
            <div class="result-card result-card-fail">
                <div class="result-label">Predicted Outcome</div>
                <div class="result-value">Reservation Shows Elevated Cancellation Risk</div>
                <div class="result-message">
                    Based on the booking details provided, this reservation profile is consistent
                    with bookings that have been cancelled. The model estimates a
                    <strong>{cancel_p}%</strong> probability of cancellation.
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-fail" style="width:{cancel_p}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-card-success">
                <div class="result-label">Predicted Outcome</div>
                <div class="result-value">Booking Shows Low Risk of Cancellation</div>
                <div class="result-message">
                    Based on the booking details provided, this reservation profile is consistent
                    with bookings that proceed to check-in. The model estimates a
                    <strong>{proceed_p}%</strong> probability that the booking will be honoured.
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-success" style="width:{proceed_p}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="result-card" style="height:100%;">
            <div class="result-label">Confidence Breakdown</div>
            <div style="margin-top:0.8rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem;">
                    <span style="font-size:0.85rem;color:#374151;">Booking Proceeds</span>
                    <span style="font-size:0.85rem;font-weight:600;color:#16a34a;">{proceed_p}%</span>
                </div>
                <div class="prob-bar-container">
                    <div class="prob-bar-fill-success" style="width:{proceed_p}%"></div>
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
                <div style="font-size:0.95rem;color:#111827;font-weight:500;">{room_display}</div>
                <div class="result-label" style="margin-top:0.6rem;">Booking Type</div>
                <div style="font-size:0.95rem;color:#111827;font-weight:500;">{booking_type}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # Summary of Inputs Used
    # =========================================================================

    st.markdown("---")
    st.markdown("### Summary of Inputs Used")

    summary_rows = [
        ("Adults staying",                       str(int(no_adults))),
        ("Children staying",                     str(int(no_children))),
        ("Weekend nights booked",                str(int(weekend_nights))),
        ("Weekday nights booked",                str(int(week_nights))),
        ("Meal plan selected",                   meal_display),
        ("Car parking required",                 car_parking),
        ("Room type booked",                     room_display),
        ("Days before arrival booking was made", str(int(lead_time))),
        ("Arrival month",                        MONTH_NAMES[arrival_month]),
        ("Arrival day of month",                 str(int(arrival_date))),
        ("Booking type",                         booking_type),
        ("Returning guest",                      repeated_guest),
        ("Previous cancellations",               str(int(prev_cancels))),
        ("Successful previous bookings",         str(int(prev_ok))),
        ("Average room price per night (USD)",   f"${avg_price_usd:,.2f}"),
        ("Special requests count",               str(int(special_req))),
    ]

    rows_html = "".join(
        f"<tr><td>{f}</td><td>{v}</td></tr>" for f, v in summary_rows
    )

    st.markdown(f"""
    <table class="summary-table">
        <thead><tr><th>Feature</th><th>Value Entered</th></tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="result-card" style="text-align:center;padding:3rem 2rem;">
        <div style="font-size:1.6rem;margin-bottom:0.5rem;color:#d1d5db;">◈</div>
        <div style="font-size:0.95rem;color:#9ca3af;">
            Complete the fields in the sidebar and click <strong>Generate Prediction</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# How This Works (Expander)
# =============================================================================

st.markdown("---")
with st.expander("How this works"):
    st.markdown("""
**Model**

This tool uses a tuned Random Forest classifier trained on historical hotel reservation data. The model was selected after comparing it against Logistic Regression and Decision Tree baselines. The training dataset contains approximately 36,000 reservations with confirmed outcomes.

**Class imbalance and threshold**

Approximately 33% of reservations in the training data were cancelled. To prevent the model from simply favouring the majority class, class weighting is applied during training and the prediction threshold is set to 0.40 rather than the default 0.50. This means a booking is flagged as at risk of cancellation when the model assigns a cancellation probability of 40% or greater — improving sensitivity to the class that carries the greatest operational cost if missed.

**Training data**

Only reservations with a confirmed outcome — cancelled or completed — were used for training. The arrival year field was excluded, as the dataset covers only 2017 and 2018, making it an uninformative feature for deployment beyond that window.

**Features used**

The model draws on guest composition, stay duration, room type, meal plan, booking channel, pricing, lead time, and the guest's prior cancellation history. Lead time and prior cancellation behaviour carry the strongest predictive signal.

**Limitations**

No model can predict reservation cancellations with complete certainty. Probability estimates are based on patterns in historical data and may not reflect current booking conditions. This tool is intended as a decision-support aid for revenue management teams and should be used alongside other operational context.
""")

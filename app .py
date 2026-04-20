"""
Hotel Reservation Cancellation Predictor — v3
Enhancements:
  - Feature importance chart: "Key Factors Influencing This Prediction"
  - Dynamic natural-language explanation of prediction drivers
  - Right panel updated: Key Drivers replaces Room Type / Booking Type
  - Summary table now includes "Impact on Prediction" column
  - Sidebar and all model / training logic are completely unchanged
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use("Agg")   # non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
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
# Styling — identical to v2 plus additions for feature importance section
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
/* ── Feature importance section ─────────────────────────────────────────── */
.fi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1.4rem 1.8rem 1rem 1.8rem;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}
.fi-title {
    font-size: 1rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.2rem;
}
.fi-subtitle {
    font-size: 0.82rem;
    color: #6b7280;
    margin-bottom: 0.8rem;
}
.explanation-box {
    background: #f0f7ff;
    border: 1px solid #dbeafe;
    border-left: 4px solid #2563EB;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.875rem;
    color: #1e3a5f;
    line-height: 1.7;
    margin-top: 1rem;
}
/* ── Key drivers bullets ─────────────────────────────────────────────────── */
.driver-list { margin-top: 0.5rem; }
.driver-item {
    display: flex;
    align-items: flex-start;
    gap: 0.45rem;
    margin-bottom: 0.45rem;
    font-size: 0.82rem;
    color: #374151;
    line-height: 1.55;
}
.driver-icon-up   { color: #dc2626; font-weight: 700; flex-shrink: 0; }
.driver-icon-down { color: #16a34a; font-weight: 700; flex-shrink: 0; }
.driver-icon-neut { color: #9ca3af; font-weight: 700; flex-shrink: 0; }
/* ── Summary table (3 columns) ───────────────────────────────────────────── */
.summary-table { width: 100%; border-collapse: collapse; font-size: 0.875rem; margin-top: 0.5rem; }
.summary-table thead tr { border-bottom: 1px solid #d1d5db; }
.summary-table thead th { text-align: left; padding: 0.55rem 0.75rem; font-size: 0.78rem; font-weight: 500; color: #9ca3af; letter-spacing: 0.02em; }
.summary-table tbody tr { border-bottom: 1px solid #f3f4f6; }
.summary-table tbody tr:last-child { border-bottom: none; }
.summary-table tbody td { padding: 0.6rem 0.75rem; color: #111827; font-size: 0.875rem; line-height: 1.5; }
.summary-table tbody td:first-child { color: #374151; font-weight: 400; width: 40%; }
.summary-table tbody td:nth-child(2) { color: #111827; font-weight: 500; width: 25%; }
.summary-table tbody td:nth-child(3) { width: 35%; }
.summary-table tbody tr:nth-child(even) { background: #f9fafb; }
.impact-up   { color: #dc2626; font-size: 0.82rem; }
.impact-down { color: #16a34a; font-size: 0.82rem; }
.impact-neut { color: #9ca3af; font-size: 0.82rem; }
/* ── Validation warning ──────────────────────────────────────────────────── */
.validation-warning { background: #fffbeb; border: 1px solid #fbbf24; border-radius: 8px; padding: 0.75rem 1rem; font-size: 0.85rem; color: #92400e; margin-bottom: 1rem; }
/* ── Expander ────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] { border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; }
/* ── Button ──────────────────────────────────────────────────────────────── */
div[data-testid="stButton"] button { background: #111827; color: #ffffff; border: none; border-radius: 7px; padding: 0.55rem 1.5rem; font-weight: 500; font-size: 0.9rem; width: 100%; transition: background 0.2s; }
div[data-testid="stButton"] button:hover { background: #374151; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Display-to-internal label mappings (unchanged)
# =============================================================================

MEAL_DISPLAY_TO_INTERNAL = {
    "Not Selected":                  "Not Selected",
    "Meal Plan 1 (Breakfast Only)":  "Meal Plan 1",
    "Meal Plan 2 (Lunch Included)":  "Meal Plan 2",
    "Meal Plan 3 (Dinner Included)": "Meal Plan 3",
}
ROOM_DISPLAY_TO_INTERNAL = {
    "Simple":   "Room_Type 1",
    "Standard": "Room_Type 2",
    "Comfort":  "Room_Type 3",
    "Superior": "Room_Type 4",
    "Deluxe":   "Room_Type 5",
    "VIP 1":    "Room_Type 6",
    "VIP 2":    "Room_Type 7",
}
MEAL_DISPLAY_OPTIONS = list(MEAL_DISPLAY_TO_INTERNAL.keys())
ROOM_DISPLAY_OPTIONS = list(ROOM_DISPLAY_TO_INTERNAL.keys())
SEGMENT_OPTIONS      = ["Online", "Offline", "Corporate", "Aviation", "Complementary"]
MONTH_NAMES = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

CANCEL_THRESHOLD = 0.40
EUR_TO_USD       = 1.10

# =============================================================================
# Feature column lists (unchanged)
# =============================================================================

CATEGORICAL_COLS = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
NUMERICAL_COLS   = [
    "no_of_adults", "no_of_children",
    "no_of_weekend_nights", "no_of_week_nights",
    "required_car_parking_space", "lead_time",
    "arrival_month", "arrival_date",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "total_nights",
    "total_guests",
]

# =============================================================================
# Feature display name mappings (for chart and table)
# =============================================================================

_NUM_DISPLAY = {
    "no_of_adults":                         "Adults in booking",
    "no_of_children":                       "Children in booking",
    "no_of_weekend_nights":                 "Weekend nights",
    "no_of_week_nights":                    "Weekday nights",
    "required_car_parking_space":           "Car parking",
    "lead_time":                            "Lead time (days)",
    "arrival_month":                        "Arrival month",
    "arrival_date":                         "Arrival day",
    "repeated_guest":                       "Returning guest",
    "no_of_previous_cancellations":         "Previous cancellations",
    "no_of_previous_bookings_not_canceled": "Prior successful bookings",
    "avg_price_per_room":                   "Room price per night",
    "no_of_special_requests":              "Special requests",
    "total_nights":                         "Total nights",
    "total_guests":                         "Total guests",
}
_CAT_DISPLAY = {
    "type_of_meal_plan":    "Meal plan type",
    "room_type_reserved":   "Room type",
    "market_segment_type":  "Booking channel",
}

# =============================================================================
# Data loading and model training (cached — unchanged)
# =============================================================================

@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    df = pd.read_csv("Hotel_Reservations.csv")
    df["target"] = (df["booking_status"] == "Canceled").astype(int)
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
    num_t = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())])
    cat_t = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                      ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre   = ColumnTransformer([("num", num_t, NUMERICAL_COLS),
                               ("cat", cat_t, CATEGORICAL_COLS)])
    pipe  = Pipeline([
        ("preprocessor", pre),
        ("classifier", RandomForestClassifier(
            n_estimators=150, max_depth=12,
            min_samples_split=10, min_samples_leaf=5,
            random_state=42, class_weight="balanced", n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# =============================================================================
# Feature importance helpers (NEW in v3)
# =============================================================================

@st.cache_data(show_spinner=False)
def get_top_importances(_mdl, n=7):
    """
    Aggregate one-hot encoded feature importances back to parent-level groups,
    then return the top-n as a pandas Series sorted descending.
    Uses a leading underscore so Streamlit skips hashing the model object.
    """
    rf_clf     = _mdl.named_steps["classifier"]
    ohe_names  = (
        _mdl.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["enc"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    all_names  = NUMERICAL_COLS + ohe_names
    importances = rf_clf.feature_importances_

    agg = {}
    for feat, imp in zip(all_names, importances):
        if feat in NUMERICAL_COLS:
            key = _NUM_DISPLAY.get(feat, feat)
        else:
            key = next(
                (_CAT_DISPLAY[c] for c in CATEGORICAL_COLS if feat.startswith(c + "_")),
                feat,
            )
        agg[key] = agg.get(key, 0.0) + imp

    series = pd.Series(agg).sort_values(ascending=False)
    return series.head(n)


def build_explanation(lead_t, special_req, avg_usd, prev_cancel,
                      bk_type, ret_guest, total_n, cancel_prob):
    """
    Construct a single natural-language sentence explaining the prediction
    in terms of the user's actual inputs.
    """
    avg_eur = avg_usd / EUR_TO_USD
    drivers = []

    # Lead time — model's strongest predictor (importance 0.36)
    if lead_t > 150:
        drivers.append("a very long lead time")
    elif lead_t > 60:
        drivers.append("a moderately long lead time")
    elif lead_t < 14:
        drivers.append("a short lead time close to the arrival date")

    # Special requests — second strongest (importance 0.18)
    if special_req == 0:
        drivers.append("no special requests indicating lower booking commitment")
    elif special_req >= 2:
        drivers.append("multiple special requests indicating strong commitment to the stay")

    # Room price
    if avg_eur > 200:
        drivers.append("a high nightly room rate")
    elif avg_eur < 70:
        drivers.append("a low nightly room rate, which is associated with flexible bookings")

    # Booking channel
    if bk_type == "Online":
        drivers.append("an online booking channel, which historically carries elevated cancellation rates")
    elif bk_type in ["Corporate", "Offline"]:
        drivers.append(f"a {bk_type.lower()} booking channel, associated with more committed reservations")

    # Prior cancellations
    if prev_cancel >= 2:
        drivers.append("a history of multiple prior cancellations")
    elif prev_cancel == 1:
        drivers.append("one prior cancellation on record")

    # Returning guest
    if ret_guest == "Yes":
        drivers.append("returning guest status, which reduces the likelihood of cancellation")

    # Zero nights edge case
    if total_n == 0:
        drivers.append("a booking with no confirmed nights")

    # Compose the sentence
    if not drivers:
        if cancel_prob >= CANCEL_THRESHOLD:
            return ("The overall combination of booking characteristics is consistent with "
                    "reservations that carry an elevated cancellation risk, based on "
                    "patterns observed in historical hotel booking data.")
        else:
            return ("The overall combination of booking characteristics is consistent with "
                    "reservations that typically proceed to check-in, based on patterns "
                    "in historical hotel booking data.")

    if len(drivers) == 1:
        body_str = drivers[0]
    elif len(drivers) == 2:
        body_str = f"{drivers[0]} and {drivers[1]}"
    else:
        body_str = ", ".join(drivers[:-1]) + ", and " + drivers[-1]

    if cancel_prob >= CANCEL_THRESHOLD:
        return (f"This prediction is primarily influenced by {body_str}. "
                "These characteristics are commonly associated with a higher "
                "tendency for reservations to be cancelled.")
    else:
        return (f"This prediction benefits from {body_str}. "
                "Together, these factors are associated with reservations that "
                "are more likely to proceed to check-in.")


def get_key_drivers(lead_t, special_req, avg_usd, prev_cancel,
                    bk_type, ret_guest, total_n, cancel_prob):
    """
    Return a list of (icon_class, text) tuples for the Key Drivers panel.
    icon_class is one of: 'driver-icon-up', 'driver-icon-down', 'driver-icon-neut'.
    Returns at most 4 items.
    """
    avg_eur = avg_usd / EUR_TO_USD
    drivers = []

    if lead_t > 100:
        drivers.append(("driver-icon-up", "Long lead time increased cancellation risk"))
    elif lead_t < 14:
        drivers.append(("driver-icon-down", "Short lead time reduced cancellation risk"))
    else:
        drivers.append(("driver-icon-neut", "Moderate lead time had a neutral effect"))

    if special_req == 0:
        drivers.append(("driver-icon-up", "No special requests suggests lower booking commitment"))
    elif special_req >= 2:
        drivers.append(("driver-icon-down",
                         f"{special_req} special requests indicated stronger commitment"))
    else:
        drivers.append(("driver-icon-neut", "One special request had a slight positive effect"))

    if prev_cancel >= 1:
        drivers.append(("driver-icon-up",
                         f"{prev_cancel} prior cancellation(s) increased risk"))
    elif ret_guest == "Yes":
        drivers.append(("driver-icon-down", "Returning guest status reduced cancellation risk"))

    if bk_type == "Online":
        drivers.append(("driver-icon-up",
                         "Online channel is associated with higher cancellation rates"))
    elif bk_type in ["Corporate", "Offline"]:
        drivers.append(("driver-icon-down",
                         f"{bk_type} channel is associated with lower cancellation rates"))

    if avg_eur > 200:
        drivers.append(("driver-icon-up", "High room price is associated with elevated risk"))
    elif avg_eur < 70:
        drivers.append(("driver-icon-neut",
                         "Low room price indicates a flexible, budget-oriented booking"))

    return drivers[:4]


def get_impact_html(feature_label, raw_value):
    """
    Return an HTML string for the Impact column of the summary table.
    Only the most interpretable features receive directional labels;
    others display a neutral dash.
    """
    try:
        if feature_label == "Days before arrival booking was made":
            v = int(float(raw_value))
            if v > 100:
                return '<span class="impact-up">↑ Increases cancellation risk</span>'
            elif v <= 14:
                return '<span class="impact-down">↓ Reduces cancellation risk</span>'
            else:
                return '<span class="impact-neut">→ Moderate effect</span>'

        elif feature_label == "Special requests count":
            v = int(float(raw_value))
            if v == 0:
                return '<span class="impact-up">↑ Increases cancellation risk</span>'
            elif v >= 2:
                return '<span class="impact-down">↓ Reduces cancellation risk</span>'
            else:
                return '<span class="impact-neut">→ Slight positive effect</span>'

        elif feature_label == "Previous cancellations":
            v = int(float(raw_value))
            if v >= 2:
                return '<span class="impact-up">↑ Significantly increases risk</span>'
            elif v == 1:
                return '<span class="impact-up">↑ Slightly increases risk</span>'
            else:
                return '<span class="impact-neut">→ No prior cancellation history</span>'

        elif feature_label == "Returning guest":
            if raw_value == "Yes":
                return '<span class="impact-down">↓ Reduces cancellation risk</span>'
            else:
                return '<span class="impact-neut">→ First-time guest</span>'

        elif feature_label == "Booking type":
            if raw_value == "Online":
                return '<span class="impact-up">↑ Slight increase in risk</span>'
            elif raw_value in ["Corporate", "Offline"]:
                return '<span class="impact-down">↓ Slight reduction in risk</span>'
            else:
                return '<span class="impact-neut">→ Neutral</span>'

        elif feature_label == "Average room price per night (USD)":
            v = float(raw_value.replace("$", "").replace(",", ""))
            eur = v / EUR_TO_USD
            if eur > 200:
                return '<span class="impact-up">↑ High price increases risk</span>'
            elif eur < 70:
                return '<span class="impact-neut">→ Low price, minor effect</span>'
            else:
                return '<span class="impact-neut">→ Moderate pricing, neutral</span>'

        elif feature_label == "Successful previous bookings":
            v = int(float(raw_value))
            if v >= 3:
                return '<span class="impact-down">↓ Strong history reduces risk</span>'
            elif v >= 1:
                return '<span class="impact-neut">→ Some positive history</span>'
            else:
                return '<span class="impact-neut">→ No prior booking history</span>'

        elif feature_label in ("Weekend nights booked", "Weekday nights booked"):
            v = int(float(raw_value))
            if v == 0:
                return '<span class="impact-up">↑ Zero nights unusual</span>'
            else:
                return '<span class="impact-neut">→ Normal stay duration</span>'

        elif feature_label == "Adults staying":
            v = int(float(raw_value))
            if v == 0:
                return '<span class="impact-up">↑ No adults — unusual booking</span>'
            else:
                return '<span class="impact-neut">→ Standard guest count</span>'

        else:
            return '<span class="impact-neut">—</span>'

    except Exception:
        return '<span class="impact-neut">—</span>'


def render_feature_chart(top_feats):
    """
    Render a horizontal bar chart of feature importances using matplotlib.
    Returns the figure for use with st.pyplot().
    """
    labels = top_feats.index.tolist()[::-1]
    values = top_feats.values.tolist()[::-1]
    n      = len(labels)

    fig, ax = plt.subplots(figsize=(7, max(2.5, n * 0.42)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bar_colors = ["#2563EB" if i == n - 1 else "#93c5fd" for i in range(n)]
    ax.barh(labels, values, color=bar_colors, height=0.58, edgecolor="none")

    ax.set_xlabel("Relative Importance", fontsize=8.5, color="#6b7280", labelpad=6)
    ax.tick_params(axis="y", labelsize=8.5, colors="#374151", pad=4)
    ax.tick_params(axis="x", labelsize=7.5, colors="#9ca3af")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#e5e7eb")
    ax.grid(axis="x", color="#f3f4f6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Value labels on bars
    for i, (val, label) in enumerate(zip(values, labels)):
        ax.text(val + 0.002, i, f"{val:.3f}", va="center",
                fontsize=7.5, color="#6b7280")

    plt.tight_layout(pad=0.6)
    return fig


# =============================================================================
# Initialise model
# =============================================================================

with st.spinner("Preparing model — this may take a moment on first load…"):
    model = train_model()

# Pre-compute importances once (cached)
top_importances = get_top_importances(model)

# =============================================================================
# Sidebar — UNCHANGED from v2
# =============================================================================

with st.sidebar:
    st.markdown("## Input Details")
    st.markdown("Enter the booking details below to generate a prediction.")
    st.markdown("---")

    st.markdown('<div class="sidebar-section">Guest Details</div>', unsafe_allow_html=True)
    no_adults    = st.number_input("Adults staying", 1, 10, 2,
                                   help="Number of adults included in the booking")
    no_children  = st.number_input("Children staying", 0, 10, 0,
                                   help="Number of children included in the booking")
    repeated_guest = st.selectbox("Returning guest", ["No","Yes"],
                                  help="Whether the guest has stayed at this hotel before")
    prev_cancels = st.number_input("Previous cancellations", 0, 13, 0,
                                   help="Number of prior bookings this guest has cancelled")
    prev_ok      = st.number_input("Successful previous bookings", 0, 58, 0,
                                   help="Previous bookings completed without cancellation")
    special_req  = st.slider("Special requests count", 0, 5, 0,
                             help="Number of special requests made alongside the booking")

    st.markdown('<div class="sidebar-section">Stay Details</div>', unsafe_allow_html=True)
    weekend_nights = st.number_input("Weekend nights booked", 0, 7, 1,
                                     help="Number of weekend nights (Saturday or Sunday)")
    week_nights    = st.number_input("Weekday nights booked", 0, 17, 2,
                                     help="Number of weekday nights (Monday to Friday)")
    meal_display   = st.selectbox("Meal plan selected", MEAL_DISPLAY_OPTIONS,
                                  help="Meal arrangement included in the reservation")
    room_display   = st.selectbox("Room type booked", ROOM_DISPLAY_OPTIONS,
                                  help="Category of room reserved")
    car_parking    = st.selectbox("Car parking required", ["No","Yes"],
                                  help="Whether the guest has requested a parking space")
    avg_price_usd  = st.number_input("Average room price per night (USD)",
                                     0.0, 660.0, 110.0, step=5.0,
                                     help="Average nightly room rate in US dollars")

    st.markdown('<div class="sidebar-section">Booking and Arrival</div>', unsafe_allow_html=True)
    lead_time      = st.number_input("Days before arrival booking was made", 0, 500, 30,
                                     help="Days between booking date and arrival date")
    booking_type   = st.selectbox("Booking type", SEGMENT_OPTIONS,
                                  help="Channel through which the reservation was made")
    arrival_month  = st.selectbox("Arrival month", list(MONTH_NAMES.keys()),
                                  format_func=lambda x: MONTH_NAMES[x], index=5,
                                  help="Month the guest is expected to arrive")
    arrival_date   = st.number_input("Arrival day of month", 1, 31, 15,
                                     help="Day of the month the guest is expected to arrive")
    st.markdown("---")
    predict_button = st.button("Generate Prediction")

# =============================================================================
# Main content area — title and subtitle (unchanged)
# =============================================================================

col_header, _ = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">Hotel Cancellation Predictor</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Predicts whether a hotel reservation is likely to be '
        "cancelled based on booking details and guest behaviour.</div>",
        unsafe_allow_html=True,
    )

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# =============================================================================
# Prediction logic
# =============================================================================

if predict_button:
    total_nights_val = int(weekend_nights) + int(week_nights)
    total_guests_val = int(no_adults)      + int(no_children)

    # Soft validation
    warnings_msgs = []
    if total_guests_val < 1:
        warnings_msgs.append(
            "Total guests is zero — a valid booking requires at least one adult."
        )
    if total_nights_val < 1:
        warnings_msgs.append(
            "Total nights is zero — please enter at least one night for a meaningful prediction."
        )
    if warnings_msgs:
        st.markdown(
            '<div class="validation-warning">' + "<br>".join(warnings_msgs) + "</div>",
            unsafe_allow_html=True,
        )

    meal_internal  = MEAL_DISPLAY_TO_INTERNAL[meal_display]
    room_internal  = ROOM_DISPLAY_TO_INTERNAL[room_display]
    rg_val         = 1 if repeated_guest == "Yes" else 0
    cp_val         = 1 if car_parking    == "Yes" else 0
    avg_price_eur  = avg_price_usd / EUR_TO_USD

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
        "avg_price_per_room":                   float(avg_price_eur),
        "no_of_special_requests":               float(special_req),
        "total_nights":                         float(total_nights_val),
        "total_guests":                         float(total_guests_val),
    }])

    proba            = model.predict_proba(input_df)[0]
    cancel_p         = round(proba[1] * 100, 1)
    proceed_p        = round(proba[0] * 100, 1)
    predicted_cancel = proba[1] >= CANCEL_THRESHOLD

    # ── Build key drivers for right panel ────────────────────────────────────
    key_drivers = get_key_drivers(
        int(lead_time), int(special_req), avg_price_usd,
        int(prev_cancels), booking_type, repeated_guest,
        total_nights_val, proba[1],
    )
    drivers_html = '<div class="driver-list">'
    for icon_cls, text in key_drivers:
        arrow = "↑" if "up" in icon_cls else ("↓" if "down" in icon_cls else "→")
        drivers_html += (
            f'<div class="driver-item">'
            f'<span class="{icon_cls}">{arrow}</span>'
            f'<span>{text}</span>'
            f"</div>"
        )
    drivers_html += "</div>"

    # ── Result cards ──────────────────────────────────────────────────────────
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
            </div>""", unsafe_allow_html=True)
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
            </div>""", unsafe_allow_html=True)

    # ── Right panel: Confidence + Key Drivers (UPDATED from v2) ──────────────
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
                <div class="result-label">Key Drivers of This Prediction</div>
                {drivers_html}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Feature importance chart (NEW in v3) ──────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="fi-card">
        <div class="fi-title">Key Factors Influencing This Prediction</div>
        <div class="fi-subtitle">
            The chart below shows the features the model relies on most heavily
            when assessing cancellation risk — ranked by relative importance.
        </div>
    </div>""", unsafe_allow_html=True)

    chart_col, _ = st.columns([3, 2])
    with chart_col:
        fig = render_feature_chart(top_importances)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Dynamic explanation
    explanation = build_explanation(
        int(lead_time), int(special_req), avg_price_usd,
        int(prev_cancels), booking_type, repeated_guest,
        total_nights_val, proba[1],
    )
    st.markdown(
        f'<div class="explanation-box">{explanation}</div>',
        unsafe_allow_html=True,
    )

    # ── Summary of Inputs Used (UPDATED — three columns) ──────────────────────
    st.markdown("---")
    st.markdown("### Summary of Inputs Used")

    summary_rows = [
        ("Adults staying",                       str(int(no_adults)),       get_impact_html("Adults staying", str(no_adults))),
        ("Children staying",                     str(int(no_children)),     get_impact_html("Children staying", str(no_children))),
        ("Weekend nights booked",                str(int(weekend_nights)),  get_impact_html("Weekend nights booked", str(weekend_nights))),
        ("Weekday nights booked",                str(int(week_nights)),     get_impact_html("Weekday nights booked", str(week_nights))),
        ("Meal plan selected",                   meal_display,              '<span class="impact-neut">—</span>'),
        ("Car parking required",                 car_parking,               '<span class="impact-neut">—</span>'),
        ("Room type booked",                     room_display,              '<span class="impact-neut">—</span>'),
        ("Days before arrival booking was made", str(int(lead_time)),       get_impact_html("Days before arrival booking was made", str(lead_time))),
        ("Arrival month",                        MONTH_NAMES[arrival_month],'<span class="impact-neut">—</span>'),
        ("Arrival day of month",                 str(int(arrival_date)),    '<span class="impact-neut">—</span>'),
        ("Booking type",                         booking_type,              get_impact_html("Booking type", booking_type)),
        ("Returning guest",                      repeated_guest,            get_impact_html("Returning guest", repeated_guest)),
        ("Previous cancellations",               str(int(prev_cancels)),    get_impact_html("Previous cancellations", str(prev_cancels))),
        ("Successful previous bookings",         str(int(prev_ok)),         get_impact_html("Successful previous bookings", str(prev_ok))),
        ("Average room price per night (USD)",   f"${avg_price_usd:,.2f}",  get_impact_html("Average room price per night (USD)", f"${avg_price_usd:,.2f}")),
        ("Special requests count",               str(int(special_req)),     get_impact_html("Special requests count", str(special_req))),
    ]

    rows_html = "".join(
        f"<tr><td>{feat}</td><td>{val}</td><td>{impact}</td></tr>"
        for feat, val, impact in summary_rows
    )

    st.markdown(f"""
    <table class="summary-table">
        <thead>
            <tr>
                <th>Feature</th>
                <th>Value Entered</th>
                <th>Impact on Prediction</th>
            </tr>
        </thead>
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

# =============================================================================
# How This Works (unchanged)
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

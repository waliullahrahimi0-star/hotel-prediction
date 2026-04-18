# Hotel Cancellation Predictor

A machine learning application that predicts whether a hotel reservation is likely to be cancelled, based on booking details and guest behaviour.

---

## Project Overview

This project implements a complete end-to-end machine learning pipeline covering data preparation, feature engineering, model training, evaluation, hyperparameter tuning, and deployment via a Streamlit web application.

Three classification models are compared: Logistic Regression (baseline), Decision Tree, and Random Forest. The tuned Random Forest is selected as the final model.

---

## Repository Contents

| File | Description |
|------|-------------|
| `app.py` | Streamlit application (trains model on launch, serves predictions) |
| `full_model_code.py` | Standalone end-to-end pipeline script |
| `notebook_steps.ipynb` | Step-by-step Google Colab notebook (21 steps) |
| `requirements.txt` | Python dependencies |
| `Hotel_Reservations.csv` | Source dataset (place in same directory as app.py) |

---

## Getting Started

### 1. Install dependencies

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

### 2. Ensure the dataset is present

Place `Hotel_Reservations.csv` in the same directory as `app.py`.

### 3. Run the application

```bash
streamlit run app.py
```

The application opens in your browser at `http://localhost:8501`.

---

## How It Works

On first launch, the dataset is loaded and the Random Forest model is trained. This typically takes 20–40 seconds. Streamlit's caching mechanism prevents retraining on subsequent interactions within the same session.

Fill in the booking details in the left sidebar and click **Generate Prediction**. The main panel displays the predicted outcome, probability score, confidence breakdown, and a summary of all values entered.

---

## Target Variable

| Label | Value | Meaning |
|-------|-------|---------|
| 1 — Cancelled | `Canceled` | Reservation was cancelled before arrival |
| 0 — Not Cancelled | `Not_Canceled` | Reservation was completed |

---

## Feature Glossary

| Display Label | Dataset Column |
|---------------|----------------|
| Adults staying | `no_of_adults` |
| Children staying | `no_of_children` |
| Weekend nights booked | `no_of_weekend_nights` |
| Weekday nights booked | `no_of_week_nights` |
| Meal plan selected | `type_of_meal_plan` |
| Car parking required | `required_car_parking_space` |
| Room type booked | `room_type_reserved` |
| Days before arrival booking was made | `lead_time` |
| Arrival year | `arrival_year` |
| Arrival month | `arrival_month` |
| Arrival day of month | `arrival_date` |
| Booking type | `market_segment_type` |
| Returning guest | `repeated_guest` |
| Previous cancellations | `no_of_previous_cancellations` |
| Successful previous bookings | `no_of_previous_bookings_not_canceled` |
| Average room price per night | `avg_price_per_room` |
| Special requests count | `no_of_special_requests` |

---

## Running the Standalone Script

```bash
python full_model_code.py
```

Trains all three models, prints evaluation metrics, performs GridSearchCV tuning, and outputs feature importances to the console.

---

## Limitations

- Dataset covers two years (2017–2018) from a specific hotel property or group.
- Predictions may not generalise to different hotel types or markets without retraining.
- The model is intended as a decision-support aid, not a definitive classification system.

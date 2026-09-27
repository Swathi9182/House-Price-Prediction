import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = joblib.load("house_price_model.pkl")


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🏠 House Price Prediction")

st.write(
    "Enter the housing details below to predict the estimated "
    "house price using a trained Random Forest model."
)

st.divider()


# ==========================================================
# INPUT SECTION
# ==========================================================

st.subheader("🏡 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=5.0
    )

    house_age = st.number_input(
        "House Age",
        min_value=0.0,
        value=20.0
    )

    ave_rooms = st.number_input(
        "Average Rooms",
        min_value=0.0,
        value=5.0
    )

    ave_bedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.0,
        value=1.0
    )


with col2:
    population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000.0
    )

    ave_occup = st.number_input(
        "Average Occupancy",
        min_value=0.0,
        value=3.0
    )

    latitude = st.number_input(
        "Latitude",
        value=35.0
    )

    longitude = st.number_input(
        "Longitude",
        value=-119.0
    )


st.divider()


# ==========================================================
# PREDICTION
# ==========================================================

if st.button("🔮 Predict House Price", use_container_width=True):

    user_data = pd.DataFrame([[
        med_inc,
        house_age,
        ave_rooms,
        ave_bedrms,
        population,
        ave_occup,
        latitude,
        longitude
    ]], columns=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude"
    ])

    prediction = model.predict(user_data)

    price = prediction[0] * 100000

    st.success(
        f"🏠 Estimated House Price: ${price:,.2f}"
    )

    st.info(
        "The prediction is generated using the trained "
        "Random Forest regression model."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "House Price Prediction | Machine Learning Project"
)
# ==========================================================
# MODEL PERFORMANCE
# ==========================================================

st.divider()

st.subheader("📊 Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "SVR"
    ],
    "MAE": [
        0.533200,
        0.454171,
        0.327731,
        0.398599
    ],
    "RMSE": [
        0.745581,
        0.703033,
        0.505986,
        0.597498
    ],
    "R²": [
        0.575788,
        0.622823,
        0.804624,
        0.727563
    ]
})

st.dataframe(
    performance,
    use_container_width=True,
    hide_index=True
)

st.write(
    "**Selected model:** Random Forest "
    "(R² = 0.8046)"
)
# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.divider()

st.subheader("🔍 Feature Importance")

st.write(
    "The chart below shows the relative importance of each "
    "feature used by the Random Forest model."
)

try:
    st.image(
        "feature_importance.png",
        caption="Random Forest Feature Importance",
        use_container_width=True
    )
except FileNotFoundError:
    st.warning("Feature importance image not found.")
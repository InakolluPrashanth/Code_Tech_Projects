import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.write("A beginner-friendly machine learning project using Linear Regression.")

df = pd.read_csv("data/house_prices.csv")
features = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score", "parking"]
X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

st.subheader("Model Performance")
c1, c2 = st.columns(2)
c1.metric("Mean Absolute Error", f"₹{mae:,.0f}")
c2.metric("R² Score", f"{r2:.3f}")

st.subheader("Enter House Details")
area = st.number_input("Area (sq ft)", 500, 5000, 1500, step=100)
bedrooms = st.slider("Bedrooms", 1, 5, 2)
bathrooms = st.slider("Bathrooms", 1, 6, 2)
age = st.slider("House Age (years)", 0, 30, 5)
location_score = st.slider("Location Score", 1, 10, 7)
parking = st.slider("Parking Spaces", 0, 2, 1)

if st.button("Predict Price"):
    sample = pd.DataFrame([{
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "location_score": location_score,
        "parking": parking
    }])
    prediction = model.predict(sample)[0]
    st.success(f"Estimated House Price: ₹{prediction:,.0f}")

with st.expander("How it works"):
    st.write("""
    1. Load the house-price dataset.
    2. Split it into training and testing data.
    3. Train a Linear Regression model.
    4. Evaluate the model using MAE and R².
    5. Use the trained model to predict a new house price.
    """)

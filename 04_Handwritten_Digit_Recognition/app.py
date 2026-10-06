import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")
st.title("✍️ Handwritten Digit Recognition")
st.write("Recognize handwritten digits 0–9 using the built-in scikit-learn digits dataset.")

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=3000)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.metric("Test Accuracy", f"{accuracy:.2%}")

st.subheader("Try a Test Image")
sample_index = st.slider("Select a sample from the test set", 0, len(X_test)-1, 0)

sample = X_test[sample_index]
actual = y_test[sample_index]
prediction = model.predict([sample])[0]
confidence = model.predict_proba([sample]).max()

fig, ax = plt.subplots()
ax.imshow(sample.reshape(8, 8), cmap="gray")
ax.axis("off")
st.pyplot(fig)

st.write(f"**Actual digit:** {actual}")
st.write(f"**Predicted digit:** {prediction}")
st.write(f"**Confidence:** {confidence:.1%}")

if prediction == actual:
    st.success("Correct prediction!")
else:
    st.warning("The model made an incorrect prediction for this sample.")

with st.expander("How it works"):
    st.write("""
    1. Load the built-in handwritten digits dataset.
    2. Split the images into training and testing sets.
    3. Train Logistic Regression.
    4. Predict the digit represented by an image.
    5. Compare prediction with the actual label.
    """)

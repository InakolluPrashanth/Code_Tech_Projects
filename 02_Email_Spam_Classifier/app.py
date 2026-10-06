import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")
st.title("📧 Email Spam Classifier")
st.write("Classify a message as Spam or Not Spam using NLP and Naive Bayes.")

df = pd.read_csv("data/spam_messages.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
    ("classifier", MultinomialNB())
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.metric("Test Accuracy", f"{accuracy:.2%}")

st.subheader("Test a Message")
message = st.text_area(
    "Enter an email/SMS:",
    "Congratulations! You won a free prize. Claim now!"
)

if st.button("Classify"):
    result = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = probabilities.max()

    if result == "spam":
        st.error(f"🚨 SPAM — confidence {confidence:.1%}")
    else:
        st.success(f"✅ NOT SPAM — confidence {confidence:.1%}")

with st.expander("How it works"):
    st.write("""
    1. Convert text into TF-IDF numerical features.
    2. Train a Multinomial Naive Bayes classifier.
    3. Predict whether a new message is spam or not spam.
    """)

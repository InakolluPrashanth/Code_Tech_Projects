import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommendation Engine", page_icon="🎬")
st.title("🎬 Movie Recommendation Engine")
st.write("A content-based recommendation system using TF-IDF and cosine similarity.")

df = pd.read_csv("data/movies.csv")
df["content"] = df["genres"].fillna("") + " " + df["description"].fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["content"])
similarity = cosine_similarity(matrix)

indices = pd.Series(df.index, index=df["title"])

movie = st.selectbox("Choose a movie", df["title"].tolist())
count = st.slider("Number of recommendations", 1, 8, 5)

if st.button("Recommend"):
    idx = indices[movie]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommendations = []

    for i, score in scores[1:count+1]:
        recommendations.append({
            "Movie": df.iloc[i]["title"],
            "Genres": df.iloc[i]["genres"],
            "Similarity": f"{score:.2f}"
        })

    st.dataframe(pd.DataFrame(recommendations), use_container_width=True)

with st.expander("How it works"):
    st.write("""
    1. Combine each movie's genres and description.
    2. Convert the text into TF-IDF vectors.
    3. Calculate cosine similarity between movies.
    4. Return the most similar movies.
    """)

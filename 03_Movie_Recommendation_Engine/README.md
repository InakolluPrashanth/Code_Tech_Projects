# Movie Recommendation Engine

## CODTECH AI Internship - Task 1

A content-based movie recommendation system that suggests movies similar to a selected movie.

### Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit

### AI/ML Concepts
- TF-IDF Vectorization
- Cosine Similarity
- Content-based Recommendation

### How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Workflow
Movie Metadata → TF-IDF Vectors → Cosine Similarity → Recommendations

### Note
The included dataset is a small educational dataset. A production recommender would use a much larger, cleaned dataset and more sophisticated ranking methods.

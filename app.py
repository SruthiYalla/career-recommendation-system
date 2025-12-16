import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="Career Recommendation System",
    page_icon="🎓",
    layout="centered"
)

# Title & description
st.markdown(
    "<h1 style='text-align: center;'>🎓 Career Recommendation System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Enter your skills and get top career recommendations</p>",
    unsafe_allow_html=True
)

st.divider()

# Load dataset
df = pd.read_csv("career_skills_dataset.csv")

# Input section
st.subheader("🧠 Enter Your Skills")
user_skills = st.text_input(
    "Example: python sql machine-learning",
    placeholder="Type your skills here..."
)

st.divider()

# Button
if st.button("🚀 Recommend Careers"):
    if user_skills.strip() == "":
        st.warning("⚠️ Please enter at least one skill")
    else:
        # Combine dataset skills with user input
        all_skills = df['Skills'].tolist()
        all_skills.append(user_skills)

        # TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_skills)

        # Cosine similarity
        similarity_scores = cosine_similarity(
            tfidf_matrix[-1], tfidf_matrix[:-1]
        ).flatten()

        # Top 3 matches
        top_indices = similarity_scores.argsort()[-3:][::-1]

        st.success("✅ Top Career Recommendations")

        # Display results nicely
        for rank, i in enumerate(top_indices, start=1):
            st.markdown(
                f"""
                <div style="padding:10px; margin-bottom:10px; 
                            border-radius:10px; 
                            background-color:#f0f2f6;">
                <h4>#{rank} 🎯 {df.iloc[i]['Career']}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

# Footer
st.divider()
st.markdown(
    "<p style='text-align:center; font-size:12px;'>Mini Project | Career Recommendation System</p>",
    unsafe_allow_html=True
)

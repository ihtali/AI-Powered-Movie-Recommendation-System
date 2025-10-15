import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -----------------------------
# Custom CSS for Modern Look
# -----------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
body {
    background-color: #f9f9f9;
    font-family: 'Arial', sans-serif;
}
h1 {
    color: #0d47a1;
    text-align: center;
}
.stButton>button {
    background-color: #0d47a1;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("🎬 Movie Recommender System")
st.sidebar.write("""
Select a movie you like and get personalized recommendations.
Built with **Python, pandas, scikit-learn, and Streamlit**.
""")

# -----------------------------
# Load Data (cached for speed)
# -----------------------------
@st.cache_data
def load_data():
    ratings = pd.read_csv(
        'u.data',
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    movies = pd.read_csv(
        'u.item',
        sep='|',
        encoding='latin-1',
        usecols=[0, 1],
        names=['movie_id', 'title']
    )
    data = pd.merge(ratings, movies, on='movie_id')
    user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating')
    user_movie_matrix_filled = user_movie_matrix.fillna(0)
    movie_similarity = cosine_similarity(user_movie_matrix_filled.T)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
    return movie_similarity_df

movie_similarity_df = load_data()

# -----------------------------
# Main Interface
# -----------------------------
st.title("🎬 Movie Recommendation System")

st.markdown(
    "Select a movie from the dropdown below and get **5 recommended movies** based on user ratings!"
)

# Movie selection
movie_list = movie_similarity_df.columns.tolist()
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Recommendation function
def recommend_movie(movie_name, num_recommendations=5):
    similar_movies = movie_similarity_df[movie_name].sort_values(ascending=False)
    return similar_movies.iloc[1:num_recommendations+1].index.tolist()

# Show recommendations in a stylish way
if st.button("Get Recommendations"):
    recommendations = recommend_movie(selected_movie)
    
    st.success(f"Because you liked **{selected_movie}**, you might also enjoy:")
    
    cols = st.columns(5)  # Show recommendations in 5 columns
    for col, movie in zip(cols, recommendations):
        col.info(f"🎬 {movie}")

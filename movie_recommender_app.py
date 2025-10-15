import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="CineMatch - Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Enhanced Styling
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .subheader {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .recommendation-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load and Prepare Data
# -----------------------------
@st.cache_data
def load_data():
    try:
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
        return movie_similarity_df, data
    except FileNotFoundError:
        st.error("❌ Data files not found. Please ensure 'u.data' and 'u.item' are in the correct directory.")
        return None, None

movie_similarity_df, movie_data = load_data()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🎯 Settings")
    
    if movie_similarity_df is not None:
        num_recommendations = st.slider(
            "Number of Recommendations",
            min_value=3,
            max_value=10,
            value=5,
            help="Choose how many movie recommendations you'd like to see"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        if movie_data is not None:
            st.write(f"**Total Ratings:** {len(movie_data):,}")
            st.write(f"**Unique Movies:** {movie_data['title'].nunique():,}")
            st.write(f"**Unique Users:** {movie_data['user_id'].nunique():,}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.write("This recommender uses collaborative filtering and cosine similarity to find movies similar to your selection based on user ratings patterns.")

# -----------------------------
# Main Interface
# -----------------------------
if movie_similarity_df is None:
    st.stop()

# Header Section
st.markdown('<h1 class="main-header">CineMatch</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Discover your next favorite movie with AI-powered recommendations</p>', unsafe_allow_html=True)

# Main Content
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🎭 Choose Your Movie")
    
    # Enhanced movie selection with search
    movie_list = movie_similarity_df.columns.tolist()
    selected_movie = st.selectbox(
        "Search or select a movie:",
        movie_list,
        index=movie_list.index("Toy Story (1995)") if "Toy Story (1995)" in movie_list else 0,
        help="Start typing to search through our movie database"
    )
    
    # Movie stats (if available)
    if movie_data is not None and selected_movie:
        movie_stats = movie_data[movie_data['title'] == selected_movie]
        if not movie_stats.empty:
            avg_rating = movie_stats['rating'].mean()
            rating_count = len(movie_stats)
            st.metric("Average Rating", f"{avg_rating:.1f} ⭐")
            st.metric("Number of Ratings", f"{rating_count:,}")

with col2:
    st.markdown("### 💫 Your Recommendations")
    
    # Recommendation function
    def recommend_movie(movie_name, num_recommendations=5):
        if movie_name not in movie_similarity_df.columns:
            return []
        similar_movies = movie_similarity_df[movie_name].sort_values(ascending=False)
        return similar_movies.iloc[1:num_recommendations+1]

    # Show recommendations with enhanced UI
    if st.button("🎬 Get Recommendations", use_container_width=True):
        if selected_movie:
            with st.spinner('🔍 Finding the perfect matches for you...'):
                recommendations = recommend_movie(selected_movie, num_recommendations)
                
                if not recommendations.empty:
                    st.balloons()
                    
                    # Selected movie card
                    st.markdown("#### 🎯 You Selected:")
                    st.markdown(f'<div class="movie-card"><h4>{selected_movie}</h4></div>', unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown(f"#### 🎉 Recommended for You:")
                    
                    # Display recommendations in a beautiful grid
                    cols = st.columns(2)
                    for i, (movie, score) in enumerate(recommendations.items()):
                        with cols[i % 2]:
                            similarity_percentage = min(score * 100, 99)
                            st.markdown(f"""
                            <div class="movie-card">
                                <div class="recommendation-badge">#{i+1}</div>
                                <h4>{movie}</h4>
                                <p><strong>Match:</strong> {similarity_percentage:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("❌ Sorry, we couldn't find recommendations for that movie. Please try another one.")
        else:
            st.warning("⚠️ Please select a movie first!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Streamlit & Collaborative Filtering</p>
    </div>
    """, 
    unsafe_allow_html=True
)
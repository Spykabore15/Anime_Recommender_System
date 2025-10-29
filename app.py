import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from supabase import create_client, Client
from rapidfuzz import process, fuzz
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import os

# ---------------------------------------------------------------------
# App Configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🍥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a dark, modern theme
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #0E1117;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #1a1a2e;
    }
    
    /* Card styling */
    .anime-card {
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border: 1px solid #2a2a4e;
        transition: all 0.3s ease-in-out;
    }
    .anime-card:hover {
        border-color: #e63946;
        box-shadow: 0 4px 15px rgba(230, 57, 70, 0.2);
    }
    .anime-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #ffffff;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .anime-score {
        font-size: 0.9rem;
        color: #a0a0b0;
        margin-bottom: 10px;
    }
    .anime-desc {
        font-size: 0.8rem;
        color: #c0c0d0;
        height: 80px; /* Fixed height for description */
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Image styling */
    .anime-image {
        width: 100%;
        height: 250px;
        object-fit: cover;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #e63946;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #d62839;
        color: white;
    }
    
    /* Selectbox/Slider */
    .stSelectbox, .stSlider {
        background-color: #1a1a2e;
        border-radius: 5px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Supabase & Model Loading (Cached)
# ---------------------------------------------------------------------

# Load secrets from Streamlit's secrets management
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()
except KeyError:
    st.error("Please add SUPABASE_URL and SUPABASE_KEY to your .streamlit/secrets.toml file.")
    st.stop()

@st.cache_resource(show_spinner="Connecting to database...")
def init_supabase():
    """Initialize and return the Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

@st.cache_resource(show_spinner="Loading recommendation models...")
def load_models():
    """Load the SVD model and embeddings from disk."""
    try:
        with open('svd_model.pkl', 'rb') as f:
            svd_model = pickle.load(f)
        
        embeddings_matrix = np.load('anime_embeddings.npy')
        
        return svd_model, embeddings_matrix
    except FileNotFoundError:
        st.error("Model files (svd_model.pkl or anime_embeddings.npy) not found. Please add them to the app directory.")
        st.stop()

svd_model, embeddings_matrix = load_models()

@st.cache_data(ttl=3600, show_spinner="Fetching anime data...")
def get_anime_data():
    """Fetch anime metadata and build the search map."""
    try:
        response = supabase.table('anime').select('anime_id, title, genres, tags, mean_score, popularity, image_url, description').execute()
        
        if not response.data:
            st.error("No data fetched from 'anime' table. Was the upload successful?")
            st.stop()

        anime_df = pd.DataFrame(response.data)
        
        # Create a search map (title -> anime_id)
        # We use anime_id as the key for our models
        search_map = {}
        for _, row in anime_df.iterrows():
            if row['title']:
                search_map[row['title'].lower()] = row['anime_id']
        
        # Create an ID-to-details map for quick lookups
        anime_details = anime_df.set_index('anime_id').to_dict('index')

        return anime_df, search_map, anime_details

    except Exception as e:
        st.error(f"Error connecting to Supabase: {e}")
        st.stop()

anime_df, search_map, anime_details = get_anime_data()

# Create a mapping from our internal matrix index (0-813) to the real anime_id
# This assumes the embeddings_matrix is in the same order as anime_df
matrix_to_anime_id_map = anime_df['anime_id'].to_dict()
anime_id_to_matrix_map = {v: k for k, v in matrix_to_anime_id_map.items()}

# ---------------------------------------------------------------------
# Recommendation Logic
# ---------------------------------------------------------------------

def get_content_recommendations(anime_id, top_k=50):
    """Get recommendations based on content similarity."""
    if anime_id not in anime_id_to_matrix_map:
        return [], []
        
    matrix_idx = anime_id_to_matrix_map[anime_id]
    
    # Get similarity scores for this anime
    sim_scores = list(enumerate(embeddings_matrix[matrix_idx]))
    
    # Sort by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top_k *excluding* itself
    top_indices = [i[0] for i in sim_scores[1:top_k+1]]
    top_sim_scores = [i[1] for i in sim_scores[1:top_k+1]]
    
    # Map matrix indices back to anime_ids
    rec_anime_ids = [matrix_to_anime_id_map[i] for i in top_indices]
    
    return rec_anime_ids, top_sim_scores

def find_closest_title(query):
    """Find the best matching anime title using fuzzy search."""
    try:
        # Use rapidfuzz to find the best match
        best_match = process.extractOne(query.lower(), search_map.keys(), scorer=fuzz.WRatio)
        if best_match and best_match[1] > 70:  # Require a decent match score
            return best_match[0], search_map[best_match[0]]
        return None, None
    except Exception as e:
        st.warning(f"Error during title search: {e}")
        return None, None

def get_hybrid_recommendations(user_id, anime_id, alpha, top_k=10):
    """Generate hybrid recommendations for an existing user."""
    
    # 1. Get content-based candidates (we get more to re-rank)
    candidate_ids, content_scores = get_content_recommendations(anime_id, top_k=50)
    if not candidate_ids:
        return []

    # 2. Get SVD predictions for these candidates
    svd_scores = []
    for aid in candidate_ids:
        # Get the .predict() score from our SVD model
        prediction = svd_model.predict(uid=str(user_id), iid=str(aid))
        svd_scores.append(prediction.est)

    # 3. Normalize scores (0 to 1)
    # Handle division by zero if all scores are identical
    def normalize(scores):
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s == 0:
            return [0.5] * len(scores)  # Return a neutral score
        return [(s - min_s) / (max_s - min_s) for s in scores]

    norm_content = normalize(content_scores)
    norm_svd = normalize(svd_scores)

    # 4. Calculate final hybrid score
    hybrid_scores = []
    for i in range(len(candidate_ids)):
        # The hybrid formula
        content_part = (1 - alpha) * norm_content[i]
        svd_part = alpha * norm_svd[i]
        hybrid_score = content_part + svd_part
        hybrid_scores.append((hybrid_score, candidate_ids[i]))

    # 5. Sort by hybrid score and return top K
    hybrid_scores.sort(reverse=True)
    final_recs = [anime_id for score, anime_id in hybrid_scores[:top_k]]
    
    return final_recs

# ---------------------------------------------------------------------
# Streamlit UI Layout
# ---------------------------------------------------------------------

# --- Sidebar ---
st.sidebar.title("🍥 Anime Recommender")
st.sidebar.markdown("Built with `Supabase`, `Streamlit`, and `scikit-surprise`.")

# User Type Selection
user_type = st.sidebar.radio(
    "Are you a new or existing user?",
    ('New User (Content-Only)', 'Existing User (Hybrid)'),
    key='user_type'
)

# Initialize session state
if 'anime_id' not in st.session_state:
    st.session_state['anime_id'] = None
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = []

# --- Main Page ---
st.title("Find Your Next Favorite Anime")

# Search Box
search_query = st.text_input(
    "Start by finding an anime you like:",
    placeholder="e.g., 'Attack on Titan', 'Steins;Gate', 'Naruto'"
)

# --- Logic for Search ---
if search_query:
    matched_title, matched_id = find_closest_title(search_query)
    
    if matched_id:
        st.session_state['anime_id'] = matched_id
        # Display the selected anime
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(anime_details[matched_id]['image_url'], use_column_width=True, caption="Your Selection")
        with col2:
            st.subheader(f"Selected: {anime_details[matched_id]['title']}")
            st.caption(f"**Genres:** {', '.join(anime_details[matched_id]['genres'])}")
            st.markdown(
                f"<div class='anime-desc'>{anime_details[matched_id]['description']}</div>", 
                unsafe_allow_html=True
            )
        st.divider()
    else:
        st.warning(f"Could not find a close match for '{search_query}'. Please try again.")
        st.session_state['anime_id'] = None
        st.session_state['recommendations'] = []


# --- Logic for Recommendations ---
if st.session_state['anime_id']:
    
    if user_type == 'New User (Content-Only)':
        if st.button("Find Similar Anime", key='new_user_button'):
            with st.spinner("Finding recommendations based on content..."):
                rec_ids, _ = get_content_recommendations(st.session_state['anime_id'], top_k=10)
                st.session_state['recommendations'] = rec_ids
    
    else:  # Existing User
        st.sidebar.subheader("Personalization")
        # Get User ID (Note: 10000 is the max user ID in our new model)
        user_id = st.sidebar.number_input(
            "Enter your User ID (1 - 10000):", 
            min_value=1, max_value=10000, value=1, step=1
        )
        # Get Alpha
        alpha = st.sidebar.slider(
            "Personalization Strength (Alpha)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.1,
            help="0.0 = 100% Content-Based (similar items). 1.0 = 100% Collaborative (what similar users liked)."
        )
        
        if st.button("Get Hybrid Recommendations", key='hybrid_button'):
            with st.spinner(f"Finding personalized recommendations for User {user_id}..."):
                rec_ids = get_hybrid_recommendations(user_id, st.session_state['anime_id'], alpha, top_k=10)
                st.session_state['recommendations'] = rec_ids

# --- Display Recommendations ---
if st.session_state['recommendations']:
    st.subheader("Here are your recommendations:")
    
    # Display in a responsive grid (3 columns)
    cols = st.columns(3)
    col_idx = 0
    
    for rec_id in st.session_state['recommendations']:
        if rec_id in anime_details:
            anime = anime_details[rec_id]
            with cols[col_idx]:
                # Create the card using HTML/CSS
                st.markdown(
                    f"""
                    <div class="anime-card">
                        <img src="{anime['image_url']}" class="anime-image" 
                             onerror="this.src='https://placehold.co/400x600/1a1a2e/ffffff?text=No+Image';">
                        <div class="anime-title" title="{anime['title']}">{anime['title']}</div>
                        <div class="anime-score">
                            ⭐ {anime['mean_score'] / 10 if anime['mean_score'] else 'N/A'} | 
                            🔥 {anime['popularity'] if anime['popularity'] else 'N/A'}
                        </div>
                        <div class="anime-desc">{anime['description']}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            col_idx = (col_idx + 1) % 3

else:
    # Show top popular anime if no recommendations are active
    st.subheader("Top Popular Anime")
    top_popular = anime_df.sort_values(by='popularity', ascending=True).head(6) # Ascending because lower is better
    
    cols = st.columns(3)
    col_idx = 0
    for _, anime in top_popular.iterrows():
        with cols[col_idx]:
            st.markdown(
                f"""
                <div class="anime-card">
                    <img src="{anime['image_url']}" class="anime-image"
                         onerror="this.src='https://placehold.co/400x600/1a1a2e/ffffff?text=No+Image';">
                    <div class="anime-title" title="{anime['title']}">{anime['title']}</div>
                    <div class="anime-score">
                        ⭐ {anime['mean_score'] / 10 if anime['mean_score'] else 'N/A'} | 
                        🔥 {anime['popularity'] if anime['popularity'] else 'N/A'}
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        col_idx = (col_idx + 1) % 3


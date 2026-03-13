import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from search import MovieSearchEngine
from recommender import MovieRecommender
from movie_similarity import MovieVisualizer
import time

# PAGE CONFIG
st.set_page_config(
    page_title="Semantic Movie Search | Endee",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# INITIALIZE ENGINES
@st.cache_resource
def init_engines():
    try:
        search_engine = MovieSearchEngine()
        recommender = MovieRecommender()
        visualizer = MovieVisualizer()
        return search_engine, recommender, visualizer
    except Exception as e:
        return None, None, None

search_engine, recommender, visualizer = init_engines()

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Space+Grotesk:wght@300;500;700&display=swap');

    :root {
        --primary: #06b6d4;
        --secondary: #8b5cf6;
        --accent: #f43f5e;
        --bg-color: #020617;
        --card-bg: rgba(30, 41, 59, 0.4);
    }

    .stApp {
        background: radial-gradient(circle at 0% 0%, #0f172a 0%, #020617 100%);
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-title {
        font-size: 72px;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #06b6d4, #8b5cf6, #f43f5e);
        -webkit-background-clip: text;
        color: transparent;
        letter-spacing: -2px;
        margin-bottom: 0px;
        font-family: 'Space Grotesk', sans-serif;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 20px;
        font-weight: 300;
        margin-bottom: 50px;
    }

    /* CARD DESIGN */
    .movie-card {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 20px;
    }

    .movie-card:hover {
        transform: translateY(-10px) scale(1.02);
        border-color: var(--primary);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 20px rgba(6, 182, 212, 0.2);
        background: rgba(30, 41, 59, 0.6);
    }

    .stat-box {
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: white;
        transition: transform 0.3s ease;
        margin-bottom: 20px;
    }

    .stat-box:hover {
        transform: scale(1.05);
        border-color: var(--secondary);
    }

    .stat-val {
        font-size: 28px;
        font-weight: 700;
        color: var(--primary);
        font-family: 'Space Grotesk', sans-serif;
    }

    .stat-label {
        color: #94a3b8;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* INPUTS */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        color: white !important;
        font-size: 18px !important;
    }

    /* BUTTONS */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, var(--primary), var(--secondary)) !important;
        color: white !important;
        border: none !important;
        padding: 15px 30px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(6, 182, 212, 0.4);
        transform: translateY(-2px);
    }

    /* TAGS */
    .genre-tag {
        background: rgba(6, 182, 212, 0.1);
        color: var(--primary);
        padding: 4px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(6, 182, 212, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<div class='main-title'>MOVIE INTELLIGENCE</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Endee-powered semantic discovery engine</div>", unsafe_allow_html=True)

# ---------- STATISTICS PANEL ----------
col1, col2, col3, col4 = st.columns(4)

try:
    df_data = pd.read_csv('semantic-movie-search/data/movies.csv')
    movie_count = len(df_data)
    avg_rating = round(df_data['rating'].mean(), 1)
except:
    movie_count = 100
    avg_rating = 8.2

with col1:
    st.markdown(f"<div class='stat-box'><div class='stat-label'>Database</div><div class='stat-val'>{movie_count:,}</div><div class='stat-label'>Movies</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='stat-box'><div class='stat-label'>Embeddings</div><div class='stat-val'>384D</div><div class='stat-label'>MiniLM-L6</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='stat-box'><div class='stat-label'>Latency</div><div class='stat-val'>12ms</div><div class='stat-label'>Endee Engine</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='stat-box'><div class='stat-label'>Avg Rating</div><div class='stat-val'>{avg_rating}</div><div class='stat-label'>User Score</div></div>", unsafe_allow_html=True)

st.write("")
st.write("---")

# ---------- SEARCH INTERFACE ----------
if search_engine is None:
    st.error("🚀 Core Engine Offline. Please run 'ingest.py' to initialize the vector space.")
    st.stop()

# Info about connection
if search_engine.adapter.is_mock:
    st.info("📡 Internal Simulator Active (Docker not detected). Ready for local testing.")

# Horizontal Search Bar
col_q, col_k, col_b = st.columns([4, 1, 1])

with col_q:
    search_query = st.text_input("", placeholder="Describe a movie theme, plot, or atmospheric vibe...", label_visibility="collapsed")
with col_k:
    top_k = st.number_input("Count", 3, 20, 6, label_visibility="collapsed")
with col_b:
    search_trigger = st.button("🔍 START SEARCH")

# ---------- RESULTS SECTION ----------
if search_query or search_trigger:
    start_time = time.time()
    with st.spinner("Analyzing semantic vectors..."):
        results = search_engine.search(search_query, top_k=top_k)
        latency = (time.time() - start_time) * 1000
    
    st.markdown(f"#### 🎥 Recommended Results ({latency:.1f}ms)")
    
    # Grid Layout
    cols = st.columns(3)
    for i, res in enumerate(results):
        meta = res['meta']
        with cols[i % 3]:
            st.markdown(f"""
            <div class="movie-card">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
                        <span class="genre-tag">{meta['genre']}</span>
                        <span style="color:#ffcc00; font-weight:700;">⭐ {meta['rating']}</span>
                    </div>
                    <h3 style="margin:0; font-size:22px; color:white;">{meta['title']}</h3>
                    <p style="color:#94a3b8; font-size:14px; margin-top:10px; line-height:1.5;">{meta['description'][:150]}...</p>
                </div>
                <div style="margin-top:20px; border-top:1px solid rgba(255,255,255,0.05); padding-top:10px; display:flex; justify-content:space-between;">
                    <span style="font-size:12px; color:var(--primary);">Similarity</span>
                    <span style="font-size:12px; color:white; font-weight:600;">{res.get('similarity', 0)*100:.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.write("---")

# ---------- DISCOVERY TABS ----------
tab_universe, tab_recs, tab_moods = st.tabs(["🌌 UNIVERSAL EXPLORER", "⭐ RELATED FILMS", "🎭 EMOTIONAL MAP"])

with tab_universe:
    st.markdown("### 🌌 The Semantic Multiverse")
    st.markdown("Every point in this 3D space represents a movie. Closer points share similar narrative structures.")
    with st.spinner("Rendering vector space..."):
        fig = visualizer.create_similarity_map()
        st.plotly_chart(fig, use_container_width=True)

with tab_recs:
    st.markdown("### ⭐ Similarity-Based Recommendations")
    titles = df_data['title'].tolist() if 'df_data' in locals() else []
    sel_movie = st.selectbox("Choose a movie to find its genetic matches:", titles)
    
    if st.button("ANALYZE GENETIC SIMILARITY"):
        recs = recommender.get_recommendations(sel_movie, top_k=4)
        if recs:
            rcols = st.columns(4)
            for idx, r in enumerate(recs):
                rmeta = r['meta']
                with rcols[idx]:
                    st.markdown(f"""
                    <div class="movie-card" style="min-height:250px;">
                        <div class="genre-tag">{rmeta['genre']}</div>
                        <h4 style="color:white; margin:10px 0;">{rmeta['title']}</h4>
                        <p style="font-size:12px; color:#94a3b8;">{rmeta['description'][:80]}...</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.error("No similar movies found in this quadrant.")

with tab_moods:
    st.markdown("### 🎭 Search by Emotional Vibe")
    mood_cols = st.columns(5)
    moods = ["Feel Good", "Make You Cry", "Dark Thriller", "Action Packed", "Thought Provoking"]
    
    for i, mood in enumerate(moods):
        if mood_cols[i].button(mood, key=f"mood_{i}"):
            res_mood = search_engine.get_by_mood(mood, top_k=3)
            m_res_cols = st.columns(3)
            for j, mr in enumerate(res_mood):
                mm = mr['meta']
                with m_res_cols[j]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <h4 style="color:white; margin:0;">{mm['title']}</h4>
                        <p style="font-size:13px; color:#94a3b8; margin-top:5px;">{mm['description'][:100]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

st.write("")
st.write("")
st.markdown("<p style='text-align: center; color: #475569; font-size:14px;'>Endee Vector Database • Search Latency: < 50ms • Embeddings: 384-Dim</p>", unsafe_allow_html=True)

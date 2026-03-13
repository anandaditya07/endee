from utils.data_loader import load_movie_data
from utils.embedding_utils import EmbeddingEngine
from utils.endee_adapter import EndeeAdapter
import os

def run_ingestion():
    # 1. Load Data
    print("Loading movie dataset...")
    df = load_movie_data("semantic-movie-search/data/movies.csv")
    
    # 2. Initialize Engines
    adapter = EndeeAdapter()
    engine = EmbeddingEngine()
    
    # 3. Create or Get Index
    index_name = "movie_search"
    print(f"Initializing Endee index: {index_name}...")
    
    # Create index (if not exists)
    try:
        index = adapter.create_index(name=index_name, dimension=384)
    except:
        index = adapter.get_index(name=index_name)

    # 4. Generate Embeddings & Prep Metadata
    print(f"Generating embeddings for {len(df)} movies...")
    descriptions = df['description'].tolist()
    vectors = engine.generate_embeddings(descriptions)
    
    items = []
    for i, row in df.iterrows():
        items.append({
            "id": i + 1,
            "vector": vectors[i],
            "meta": {
                "title": row['title'],
                "description": row['description'],
                "genre": row['genre'],
                "rating": float(row['rating'])
            }
        })
    
    # 5. Upsert to Endee
    print("Upserting to Endee vector database...")
    index.upsert(items)
    
    print("✅ Ingestion complete! Use streamlit_app.py to explore.")

if __name__ == "__main__":
    run_ingestion()

from utils.endee_adapter import EndeeAdapter
from utils.embedding_utils import EmbeddingEngine

class MovieSearchEngine:
    def __init__(self, index_name="movie_search"):
        self.adapter = EndeeAdapter()
        self.index = self.adapter.get_index(name=index_name)
        self.engine = EmbeddingEngine()

    def search(self, query, top_k=5, genre_filter=None):
        """Perform semantic search using query embeddings"""
        query_vector = self.engine.generate_embeddings([query])[0]
        
        filter_payload = []
        if genre_filter:
            filter_payload.append({"genre": {"$eq": genre_filter}})
            
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_payload if filter_payload else None,
            include_vectors=True
        )
        return results

    def get_by_mood(self, mood, top_k=5):
        """Semantic search based on 'mood' keywords mapped to movie descriptions"""
        mood_prompts = {
            "Feel Good": "Uplifting, happy, inspirational, heartwarming story",
            "Make You Cry": "Sad, tragic, emotional, heartbreaking drama",
            "Dark Thriller": "Dark, intense, psychological, mysterious, grit",
            "Action Packed": "High energy, explosive, fast-paced, excitement",
            "Thought Provoking": "Philosophical, complex, deeply intellectual, perspective-shifting"
        }
        
        query = mood_prompts.get(mood, mood)
        return self.search(query, top_k=top_k)

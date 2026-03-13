from utils.endee_adapter import EndeeAdapter
from utils.embedding_utils import EmbeddingEngine

class MovieRecommender:
    def __init__(self, index_name="movie_search"):
        self.adapter = EndeeAdapter()
        self.index = self.adapter.get_index(name=index_name)
        self.engine = EmbeddingEngine()

    def get_recommendations(self, movie_title, top_k=6):
        """Find movies similar to a given title using vector similarity"""
        # Find the vector for the given movie title first
        all_movies = self.index.query(vector=[0]*384, top_k=200) # Simple way to find target in small set
        target_movie = next((m for m in all_movies if m['meta']['title'].lower() == movie_title.lower()), None)
        
        if not target_movie:
            return []
            
        # Query for similar vectors
        results = self.index.query(
            vector=target_movie['vector'],
            top_k=top_k + 1, # +1 to exclude itself
            include_vectors=False
        )
        
        # Exclude the original movie
        recommendations = [r for r in results if r['meta']['title'].lower() != movie_title.lower()]
        return recommendations[:top_k]

    def explain_recommendation(self, target, recommended):
        """Generates a simple text explanation for why a movie was recommended"""
        t_meta = target['meta']
        r_meta = recommended['meta']
        
        reasons = []
        if t_meta['genre'] == r_meta['genre']:
            reasons.append(f"both are {t_meta['genre']} films")
        
        # Simple text similarity check (placeholder for more complex analysis)
        t_desc = t_meta['description'].lower()
        r_desc = r_meta['description'].lower()
        
        keywords = ['space', 'future', 'war', 'love', 'psycological', 'crime', 'family']
        for k in keywords:
            if k in t_desc and k in r_desc:
                reasons.append(f"they share themes of {k}")
        
        if not reasons:
            return "They share deep semantic similarities and narrative tones."
        
        return f"Recommended because {' and '.join(reasons)}."

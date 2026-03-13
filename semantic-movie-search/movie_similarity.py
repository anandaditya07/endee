import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
from utils.endee_adapter import EndeeAdapter
from utils.embedding_utils import EmbeddingEngine

class MovieVisualizer:
    def __init__(self, index_name="movie_search"):
        self.adapter = EndeeAdapter()
        self.index = self.adapter.get_index(name=index_name)
        self.engine = EmbeddingEngine()

    def get_all_vectors(self, limit=100):
        # Using a dummy zero vector to trigger a global search (adapter handles this)
        results = self.index.query(vector=[0]*384, top_k=limit, include_vectors=True)
        return results

    def create_similarity_map(self):
        """Creates a Plotly 3D scatter plot of movie embeddings"""
        raw_data = self.get_all_vectors(limit=100)
        
        if not raw_data:
            return None
            
        vectors = np.array([res['vector'] for res in raw_data])
        metas = [res['meta'] for res in raw_data]
        
        # Reduce to 3D for visualization
        pca = PCA(n_components=3)
        coords = pca.fit_transform(vectors)
        
        # Clustering for color
        kmeans = KMeans(n_clusters=int(np.sqrt(len(vectors))), random_state=42)
        clusters = kmeans.fit_predict(vectors)
        
        df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
        df['title'] = [m['title'] for m in metas]
        df['genre'] = [m['genre'] for m in metas]
        df['rating'] = [m['rating'] for m in metas]
        df['cluster'] = clusters.astype(str)
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='cluster',
            hover_name='title',
            hover_data=['genre', 'rating'],
            title="Semantic Movie Universe (3D Projection)",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title="Semantic X",
                yaxis_title="Semantic Y",
                zaxis_title="Semantic Z"
            )
        )
        
        return fig

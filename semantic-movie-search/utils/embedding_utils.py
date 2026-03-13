from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initializes the embedding model using Sentence Transformers"""
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, text_list):
        """Converts a list of strings into a list of vectors"""
        if isinstance(text_list, str):
            text_list = [text_list]
        
        embeddings = self.model.encode(text_list)
        return [emb.tolist() for emb in embeddings]

import pandas as pd
import os

def load_movie_data(file_path):
    """Loads and basic cleans the movie dataset"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Movie data not found at {file_path}")
    
    df = pd.read_csv(file_path)
    # Basic cleaning
    df['title'] = df['title'].fillna('Unknown Title')
    df['description'] = df['description'].fillna('')
    df['genre'] = df['genre'].fillna('Unknown')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)
    
    return df

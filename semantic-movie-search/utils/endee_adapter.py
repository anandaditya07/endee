from endee import Endee, Precision
import numpy as np
import os
import json

class MockIndex:
    def __init__(self, name):
        self.name = name
        self.storage_file = f"semantic-movie-search/data/mock_{name}.json"
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save(self):
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        with open(self.storage_file, 'w') as f:
            json.dump(self.data, f)

    def upsert(self, items):
        ids = [item['id'] for item in items]
        self.data = [d for d in self.data if d['id'] not in ids]
        self.data.extend(items)
        self._save()
        return True

    def query(self, vector, top_k=5, filter=None, include_vectors=True):
        if not self.data:
            return []
        
        results = []
        vec_a = np.array(vector)
        norm_a = np.linalg.norm(vec_a)
        
        for item in self.data:
            if filter:
                match = True
                for f in filter:
                    key = list(f.keys())[0]
                    val = f[key]
                    if isinstance(val, dict):
                        op = list(val.keys())[0]
                        target = val[op]
                        if op == "$eq" and item['meta'].get(key) != target: match = False
                        if op == "$gt" and float(item['meta'].get(key, 0)) <= float(target): match = False
                    else:
                        if item['meta'].get(key) != val: match = False
                if not match: continue

            vec_b = np.array(item['vector'])
            norm_b = np.linalg.norm(vec_b)
            
            if norm_a == 0 or norm_b == 0:
                score = 0
            else:
                score = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            
            res = {
                "id": item['id'],
                "similarity": float(score),
                "meta": item['meta']
            }
            if include_vectors:
                res['vector'] = item['vector']
            results.append(res)
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

class EndeeAdapter:
    def __init__(self, use_mock=False):
        self.is_mock = use_mock
        self.client = None
        
        if not use_mock:
            try:
                import requests
                requests.get("http://localhost:8080/api/v1/health", timeout=0.1)
                self.client = Endee()
            except:
                self.is_mock = True
        
        if self.is_mock:
            self.client = "MOCK_CLIENT"

    def create_index(self, name, dimension, precision=None):
        if self.is_mock:
            return MockIndex(name)
        return self.client.create_index(name, dimension, precision)

    def get_index(self, name):
        if self.is_mock:
            return MockIndex(name)
        return self.client.get_index(name)

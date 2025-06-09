import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class SimpleRAG:
    def __init__(self, data_dir="data", embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v1"):
        self.texts = []
        self.model = SentenceTransformer(embedding_model)
        self.index = None
        self._load_and_embed(data_dir)

    def _load_and_embed(self, data_dir):
        embeddings = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    self.texts.append(text)
                    emb = self.model.encode(text)
                    embeddings.append(emb)
        if embeddings:
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(np.array(embeddings))

    def retrieve(self, query, k=2):
        if not self.index:
            return []
        q_vec = self.model.encode([query])
        D, I = self.index.search(np.array(q_vec), k)
        return [self.texts[i] for i in I[0]]

import faiss
import numpy as np
import os
import pickle
import json
from .config import Config

class MemoryManager:
    def __init__(self, resources):
        self.res = resources
        self.index = None
        self.history = [] # Stores actual text: {"query": q, "answer": a}
        
        if not os.path.exists(Config.MEMORY_DB_DIR):
            os.makedirs(Config.MEMORY_DB_DIR)
            
        self._init_index()

    def _init_index(self):
        # Initialize flat IP index for memory
        self.index = faiss.IndexFlatIP(384) 

    def add_interaction(self, query, answer):
        # 1. Create a "Semantic Key" for quick embedding
        answer_summary = answer[:200].replace("\n", " ") + "..." if len(answer) > 200 else answer
        semantic_key = f"Q: {query} | A: {answer_summary}"
        
        # Store full content in history list for retrieval display
        full_fact = f"User: {query}\nAI: {answer}"
        
        # 2. Maintain strict FIFO (Last 10 conversations)
        self.history.append({"key": semantic_key, "content": full_fact})
        
        if len(self.history) > 10:
            self.history.pop(0)
            
        # 3. Reset and Rebuild DB every time
        self._rebuild_index()

    def _rebuild_index(self):
        self.index.reset()
        if not self.history:
            self._save_to_disk()
            return
            
        # Bulk embed only the short semantic keys
        keys = [item["key"] for item in self.history]
        embeddings = self.res["text_embedder"].encode(keys).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Save to disk for inspection
        self._save_to_disk()

    def _save_to_disk(self):
        """Saves history to JSON and FAISS index to binary file in memory_db folder"""
        history_path = os.path.join(Config.MEMORY_DB_DIR, "history.json")
        index_path = os.path.join(Config.MEMORY_DB_DIR, "index.bin")
        
        # Save History JSON
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=4)
                
            # Save FAISS Index
            if self.index:
                faiss.write_index(self.index, index_path)
        except Exception as e:
            print(f" Failed to save memory to disk: {e}")

    def search_memory(self, query, top_k=2):
        if not self.history:
            return ""
            
        q_embed = self.res["text_embedder"].encode([query]).astype('float32')
        faiss.normalize_L2(q_embed)
        
        search_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_embed, search_k)
        
        relevant_indices = set()
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and score > Config.MEMORY_SCORE_THRESHOLD:
                relevant_indices.add(int(idx))
                
        last_idx = len(self.history) - 1
        relevant_indices.add(last_idx)
        
        final_indices = sorted(list(relevant_indices))
        relevant_context = [self.history[i]["content"] for i in final_indices]
        
        return "\n---\n".join(relevant_context)
    
    def clear(self):
        self.history = []
        self.index.reset()
        self._save_to_disk()

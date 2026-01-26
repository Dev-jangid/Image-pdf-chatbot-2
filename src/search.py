import numpy as np
import faiss
from .config import Config

class MultimodalSearch:
    def __init__(self, resources, data):
        self.res = resources
        self.data = data

    def query(self, prompt):
        results = {"text_hits": [], "image_hits": []}
        
        # 1. Detect Visual Intent (Strictly Visual/Grapgical terms only)
        query_lower = prompt.lower()
        is_visual = any(word in query_lower for word in Config.VISUAL_TRIGGERS)
        
        # Boost visual intent if query is very short and contains a visual word
        if is_visual and len(prompt.split()) <= 3:
            # High confidence visual request
            pass 
        
        # 1. Text Search (Hybrid: FAISS + BM25)
        if self.data.get("text_index") is not None:
            # A. Dense Search (FAISS)
            q_embed = self.res["text_embedder"].encode([prompt]).astype('float32')
            faiss.normalize_L2(q_embed)
            
            top_k_retrieval = min(Config.TEXT_TOP_K * 2, self.data["text_index"].ntotal)
            d_scores, d_indices = self.data["text_index"].search(q_embed, top_k_retrieval)
            
            # Map indices to scores
            dense_scores = {idx: score for idx, score in zip(d_indices[0], d_scores[0])}
            
            # B. Sparse Search (BM25)
            bm25_scores = {}
            if self.data.get("bm25") is not None:
                tokenized_query = prompt.lower().split()
                # Get raw scores for all documents
                raw_bm25 = self.data["bm25"].get_scores(tokenized_query)
                # Normalize BM25 scores to 0-1 range
                max_bm25 = max(raw_bm25) if len(raw_bm25) > 0 and max(raw_bm25) > 0 else 1.0
                bm25_scores = {i: s/max_bm25 for i, s in enumerate(raw_bm25) if s > 0}
            
            # C. Hybrid Fusion (Weighted Sum)
            # Combine candidates from both (Union of Top K from Dense and Top K from Sparse)
            # Since BM25 is efficient, we computed it for all, but let's focus on union of candidates
            
            all_candidates = set(dense_scores.keys()) | set(sorted(bm25_scores.keys(), key=bm25_scores.get, reverse=True)[:top_k_retrieval])
            
            hybrid_results = []
            for idx in all_candidates:
                if idx == -1: continue # FAISS padding
                
                s_dense = dense_scores.get(idx, 0.0)
                s_sparse = bm25_scores.get(idx, 0.0)
                
                # Hybrid Score = Alpha * Dense + (1 - Alpha) * Sparse
                final_score = (Config.HYBRID_ALPHA * s_dense) + ((1 - Config.HYBRID_ALPHA) * s_sparse)
                hybrid_results.append((idx, final_score))
            
            # Sort by final score
            hybrid_results.sort(key=lambda x: x[1], reverse=True)
            
            # Select Top K
            top_indices = [idx for idx, score in hybrid_results[:Config.TEXT_TOP_K] if score > Config.TEXT_SCORE_THRESHOLD]
            results["text_hits"] = [self.data["texts"][i] for i in top_indices]
            
            # Identify relevant pages from text hits for image re-ranking
            relevant_pages = {hit["page"] for hit in results["text_hits"]}

        # 2. Image Search (FAISS Semantic + Page Coherence)
        if self.data.get("image_index") is not None:
            # Fix: Ensure inputs are moved to the same device as the model (e.g., CUDA)
            inputs = self.res["clip_processor"](text=[prompt], return_tensors="pt", padding=True).to(Config.DEVICE)
            t_features = self.res["clip_model"].get_text_features(**inputs).detach().cpu().numpy().astype('float32')
            faiss.normalize_L2(t_features)
            
            # Adjust threshold based on intent
            base_threshold = Config.IMAGE_VISUAL_THRESHOLD if is_visual else Config.IMAGE_BASE_THRESHOLD
            top_k = Config.IMAGE_TOP_K if not is_visual else max(Config.IMAGE_TOP_K, 3)
            top_k_search = min(top_k * 3, self.data["image_index"].ntotal) # Fetch more candidates
            
            scores, indices = self.data["image_index"].search(t_features, top_k_search)
            
            # Rerank images: Boost score if image is on a relevant text page
            reranked_images = []
            for idx, score in zip(indices[0], scores[0]):
                if idx == -1: continue
                img_data = self.data["images"][idx]
                
                # Boost logic: 15% boost if page matches text context
                final_score = score
                if img_data["page"] in relevant_pages:
                    final_score *= 1.15
                
                if final_score > base_threshold:
                    reranked_images.append((img_data, final_score))
            
            # Sort by boosted score
            reranked_images.sort(key=lambda x: x[1], reverse=True)
            results["image_hits"] = [{**img, "score": float(s)} for img, s in reranked_images[:top_k]]
            
        return results, is_visual

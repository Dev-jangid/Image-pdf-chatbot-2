import fitz
import io
import os
import hashlib
import pickle
import numpy as np
import streamlit as st
import faiss
from PIL import Image
import torch
from rank_bm25 import BM25Okapi
from .config import Config

class MultimodalProcessor:
    def __init__(self, resources):
        self.res = resources
        if not os.path.exists(Config.VECTOR_DB_DIR):
            os.makedirs(Config.VECTOR_DB_DIR)
        if not os.path.exists(Config.IMAGE_STORAGE_DIR):
            os.makedirs(Config.IMAGE_STORAGE_DIR)

    def _get_file_hash(self, file_bytes):
        return hashlib.md5(file_bytes).hexdigest()

    def process_file(self, uploaded_file):
        """Processes PDF or Image files with persistence"""
        file_bytes = uploaded_file.read()
        file_hash = self._get_file_hash(file_bytes)
        storage_path = os.path.join(Config.VECTOR_DB_DIR, f"{file_hash}.pkl")

        if os.path.exists(storage_path):
            with st.status(" Loading cached FAISS indices...", expanded=False):
                with open(storage_path, "rb") as f:
                    return pickle.load(f)

        if uploaded_file.type == "application/pdf":
            data = self._process_pdf(file_bytes, file_hash)
        elif uploaded_file.type.startswith("image/"):
            data = self._process_image_standalone(file_bytes, file_hash)
        else:
            return None

        if data:
            with open(storage_path, "wb") as f:
                pickle.dump(data, f)
        
        return data

    def _process_pdf(self, file_bytes, file_hash):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        images = []
        
        with st.status(" Analyzing PDF Structure...", expanded=True) as status:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text").strip()
                if page_text:
                    chunks = self.res["text_splitter"].split_text(page_text)
                    for chunk in chunks:
                        texts.append({"content": chunk, "page": page_num + 1})
                
                for img_idx, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    pix = doc.extract_image(xref)
                    pil_img = Image.open(io.BytesIO(pix["image"])).convert("RGB")
                    
                    if (pil_img.width < Config.MIN_IMAGE_WIDTH or 
                        pil_img.height < Config.MIN_IMAGE_HEIGHT):
                        continue
                    
                    pil_img = self._pad_image(pil_img)
                    image_filename = f"{file_hash}_p{page_num+1}_img{img_idx}.jpg"
                    image_save_path = os.path.join(Config.IMAGE_STORAGE_DIR, image_filename)
                    pil_img.save(image_save_path)

                    try:
                        img_text = self.res["ocr_manager"].extract_text(image_save_path)
                        if img_text:
                            texts.append({"content": f"[Image Context]: {img_text}", "page": page_num + 1})
                    except Exception as e:
                        print(f"⚠️ OCR Error: {e}")
                        
                    images.append({
                        "image": pil_img,
                        "path": image_save_path,
                        "page": page_num + 1,
                        "id": f"p{page_num+1}_i{img_idx}"
                    })
            
            status.update(label=" Building FAISS & BM25 Indices...", state="running")
            text_index, bm25 = self._build_text_index(texts)
            image_index = self._build_image_index(images)
            status.update(label="Ready for Questions!", state="complete", expanded=False)

        return {
            "texts": texts,
            "images": images,
            "text_index": text_index,
            "image_index": image_index,
            "bm25": bm25,
            "metadata": {"type": "pdf", "pages": len(doc)}
        }

    def _process_image_standalone(self, file_bytes, file_hash):
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        pil_img = self._pad_image(pil_img)

        image_filename = f"{file_hash}_standalone.jpg"
        image_save_path = os.path.join(Config.IMAGE_STORAGE_DIR, image_filename)
        pil_img.save(image_save_path)
        
        texts = []
        try:
            img_text = self.res["ocr_manager"].extract_text(image_save_path)
            if img_text:
                texts.append({"content": img_text, "page": 1})
        except Exception as e:
            print(f" Standalone OCR Error: {e}")

        images = [{"image": pil_img, "page": 1, "id": "standalone"}]
        image_index = self._build_image_index(images)
        text_index, bm25 = self._build_text_index(texts)

        return {
            "texts": texts,
            "images": images,
            "text_index": text_index,
            "image_index": image_index,
            "bm25": bm25,
            "metadata": {"type": "image", "pages": 1}
        }

    def _build_text_index(self, texts):
        if not texts:
            return None, None
        
        text_embeddings = self.res["text_embedder"].encode([t["content"] for t in texts])
        text_embeddings = np.array(text_embeddings).astype('float32')
        faiss.normalize_L2(text_embeddings)
        index = faiss.IndexFlatIP(text_embeddings.shape[1])
        index.add(text_embeddings)
        
        tokenized_corpus = [doc["content"].lower().split() for doc in texts]
        bm25 = BM25Okapi(tokenized_corpus)
        return index, bm25

    def _build_image_index(self, images):
        if not images:
            return None
            
        embeddings = []
        for img_data in images:
            with torch.no_grad():
                inputs = self.res["clip_processor"](images=img_data["image"], return_tensors="pt").to(Config.DEVICE)
                features = self.res["clip_model"].get_image_features(**inputs)
                
                # Robust tensor-to-numpy conversion
                if hasattr(features, "cpu"):
                    features = features.cpu().detach().numpy()
                else:
                    features = np.array(features)
                    
                # Ensure it's a 2D array (1, Dim) for concatenation
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                elif features.ndim == 2 and features.shape[0] != 1:
                    # If multiple images were processed in one pass (rare here)
                    pass 
                
                embeddings.append(features)
        
        if not embeddings:
            return None

        embeddings = np.concatenate(embeddings, axis=0).astype('float32')
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def _pad_image(self, pil_img):
        """Pads an image to square to prevent CLIP distortion"""
        width, height = pil_img.size
        if width == height:
            return pil_img
        
        max_dim = max(width, height)
        new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        new_img.paste(pil_img, ((max_dim - width) // 2, (max_dim - height) // 2))
        return new_img

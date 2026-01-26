import os
import torch
import streamlit as st
import dotenv

dotenv.load_dotenv()

class Config:
    try:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
    except Exception:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    TEXT_EMBEDDER_MODEL = "all-MiniLM-L6-v2"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
    GROQ_MODEL = "llama-3.1-8b-instant"
    IMAGE_TOP_K = 3
    TEXT_TOP_K = 3
    MAX_TOKENS = 1000
    
    # Image Filtering
    MIN_IMAGE_WIDTH = 250
    MIN_IMAGE_HEIGHT = 250
    VECTOR_DB_DIR = "vector_db"
    HYBRID_ALPHA = 0.5 # Weight for Dense vector search (0.0 to 1.0)
    MEMORY_WINDOW = 3 # Number of past exchanges to remember
    MEMORY_DB_DIR = "memory_db"
    IMAGE_STORAGE_DIR = "extracted_images"
    
    # Thresholds
    TEXT_SCORE_THRESHOLD = 0.25
    IMAGE_BASE_THRESHOLD = 0.18
    IMAGE_VISUAL_THRESHOLD = 0.12
    MEMORY_SCORE_THRESHOLD = 0.3
    
    # Device Detection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_GPU = torch.cuda.is_available()

    # Centralized Intent Triggers
    VISUAL_TRIGGERS = ["image", "photo", "pic", "picture", "show", "see", "diagram", "chart", "figure", "plot", "drawing", "sketch", "look", "graph", "visual", "fig", "illustration", "map", "table", "graphic"]
    EXAMPLE_TRIGGERS = ["example", "instance", "sample", "demonstration"]
    INFO_TRIGGERS = ["tell", "explain", "about", "details", "brief", "summarize", "what", "how", "why", "meaning", "define"]
    JUNK_TRIGGERS = ["clear", "wrong", "reset", "clean", "bad", "incorrect", "false", "good", "hello", "hi", "test", "nonsense", "asdf", "...", "?", "."]

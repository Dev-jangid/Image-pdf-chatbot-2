import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Config
from src.styles import apply_custom_css
from src.processor import MultimodalProcessor
from src.search import MultimodalSearch
from src.prompts import Prompts
from src.memory import MemoryManager
from src.ocr import OCRManager
from src.chat import generate_chat_response

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Image-PDF Multimodal AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ENVIRONMENT ---
load_dotenv()

# --- RESOURCE LOADING ---
@st.cache_resource
def load_resources():
    return {
        "text_embedder": SentenceTransformer(Config.TEXT_EMBEDDER_MODEL, device=Config.DEVICE),
        "clip_model": CLIPModel.from_pretrained(Config.CLIP_MODEL_NAME).to(Config.DEVICE),
        "clip_processor": CLIPProcessor.from_pretrained(Config.CLIP_MODEL_NAME),
        "text_splitter": RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        ),
        "groq_client": Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None,
        "ocr_manager": OCRManager()
    }

def reset_storage():
    """Total reset: Wipes all physical DBs, images, and Streamlit caches."""
    import shutil
    import gc
    import time
    
    # 1. Clear session state immediately
    for key in list(st.session_state.keys()):
        if key not in ["uploader_key", "initialized"]:
            del st.session_state[key]
    
    # 2. Clear Streamlit Data Cache (Parsed documents only)
    st.cache_data.clear()
    
    # 3. Force release of all file locks
    gc.collect()
    time.sleep(0.5)
    
    # 4. Wipe physical folders
    folders = [Config.VECTOR_DB_DIR, Config.MEMORY_DB_DIR, Config.IMAGE_STORAGE_DIR]
    for folder in folders:
        if os.path.exists(folder):
            for i in range(5):
                try:
                    shutil.rmtree(folder, ignore_errors=True)
                    break
                except Exception:
                    time.sleep(0.3)
        os.makedirs(folder, exist_ok=True)
    

def main():
    apply_custom_css()
    
    # --- AUTO-CLEAR ON BROWSER REFRESH ---
    if "initialized" not in st.session_state:
        reset_storage()
        st.session_state.initialized = True
        st.rerun()

    resources = load_resources()
    
    # Session State Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.vault = None
        st.session_state.last_file = None
            
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0
    
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = MemoryManager(resources)

    # Sidebar
    with st.sidebar:
        st.title("Image-PDF Multimodal AI")
        st.markdown("---")
        # Dynamic key allows us to force-reset the uploader widget
        uploader_key = st.session_state.get("uploader_key", 0)
        uploaded_file = st.file_uploader(
            "Drop PDF or Image", 
            type=["pdf", "png", "jpg", "jpeg"],
            key=f"uploader_{uploader_key}"
        )
        
        if uploaded_file:
            if st.session_state.vault is None or uploaded_file.name != st.session_state.get("last_file"):
                processor = MultimodalProcessor(resources)
                st.session_state.vault = processor.process_file(uploaded_file)
                st.session_state.last_file = uploaded_file.name
                st.success("Analysis Complete!")
        
        st.markdown("---")
        if st.button("Clear Workspace"):
            # 1. Prepare for hard reset
            new_key = st.session_state.get("uploader_key", 0) + 1
            
            # 2. Total Wipe (Physical & Session)
            reset_storage()
            
            # 3. Re-initialize minimal state
            st.session_state.uploader_key = new_key
            st.rerun()

        st.markdown("---")
        with st.expander("🛠️ System Health", expanded=False):
            st.write(f"**Device:** `{Config.DEVICE.upper()}`")
            st.write(f"**RAM Mode:** `{'Optimized' if not Config.USE_GPU else 'Performance'}`")
            
            if st.session_state.vault:
                n_texts = len(st.session_state.vault.get("texts", []))
                n_imgs = len(st.session_state.vault.get("images", []))
                st.write(f"**Indexed Text Chunks:** {n_texts}")
                st.write(f"**Indexed Images:** {n_imgs}")
                
                # Verify physical files
                files = os.listdir(Config.IMAGE_STORAGE_DIR) if os.path.exists(Config.IMAGE_STORAGE_DIR) else []
                st.write(f"**Physical Images:** {len(files)}")
            else:
                st.info("No document loaded.")

    # Main Interface
    # Hero Section Removed as per request
    
    # Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "images" in msg and msg["images"]:
                cols = st.columns(len(msg["images"]))
                for i, img_data in enumerate(msg["images"]):
                    score_val = img_data.get("score", 0.0)
                    caption = f"Page {img_data['page']} | Confidence: {score_val:.2f}"
                    cols[i].image(img_data["image"], caption=caption)

    # Chat Input
    if prompt := st.chat_input("   "):
        if not st.session_state.vault:
            st.warning("Please upload a document first.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG Pipeline with Contextual Refinement
        search_query = prompt
        
        # Check for Vague or Context-Dependent Intent
        if len(st.session_state.messages) >= 3:
            # We want to rewrite for both visual follow-ups AND example follow-ups
            pronoun_triggers = ["it", "that", "this", "them", "those", "here", "there"]
            
            is_vague = len(prompt.split()) < 10
            has_trigger = any(t in prompt.lower() for t in Config.VISUAL_TRIGGERS + Config.EXAMPLE_TRIGGERS)
            has_pronoun = any(f" {t} " in f" {prompt.lower()} " for t in pronoun_triggers)
            
            if is_vague and (has_trigger or has_pronoun):
                prev_user_q = st.session_state.messages[-3]["content"]
                search_query = f"{prev_user_q} {prompt}"
                print(f"Contextual Search Rewrite: '{prompt}' -> '{search_query}'")

        # 1. Junk/Empty Query Filter
        is_empty = not prompt.strip()
        is_too_short = len(prompt.strip()) < 2
        is_junk = prompt.lower().strip() in Config.JUNK_TRIGGERS
        
        if is_empty or is_too_short or is_junk:
            response = Prompts.JUNK_RESPONSE if is_junk else Prompts.EMPTY_RESPONSE
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            return

        # 2. Contextual RAG Pipeline
        search = MultimodalSearch(resources, st.session_state.vault)
        hits, is_visual = search.query(search_query)
        
        # Memory Retrieval (for Vector Context)
        memory_context = st.session_state.memory_manager.search_memory(prompt)
        
        # Construct Context
        if hits["text_hits"]:
            final_context = "\n".join([f"[Page {h['page']}]: {h['content']}" for h in hits["text_hits"]])
        else:
            final_context = "CRITICAL: NO RELEVANT CONTENT FOUND IN THE DOCUMENT FOR THIS QUERY."
        
        # LLM Call
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            image_placeholder = st.empty()
            
            with st.spinner("Synthesizing answer..."):
                try:
                    # 1. Prepare history for the chat function (pairs of user/bot)
                    chat_history = []
                    raw_msgs = st.session_state.messages[:-1] # exclude current message
                    for i in range(0, len(raw_msgs), 2):
                        if i + 1 < len(raw_msgs):
                            chat_history.append({
                                "user": raw_msgs[i]["content"],
                                "bot": raw_msgs[i+1]["content"]
                            })

                    # 2. Call the centralized response generator
                    response = generate_chat_response(
                        client=resources["groq_client"],
                        user_input=prompt,
                        context=final_context,
                        history=chat_history,
                        img_count=len(hits["image_hits"]),
                        is_visual=is_visual
                    )
                    
                    response_placeholder.markdown(response)
                    
                    # Update Memory Vector DB
                    st.session_state.memory_manager.add_interaction(prompt, response)
                    
                    # --- STRICT IMAGE DISPLAY LOGIC ---
                    # ONLY display images if the system detected EXPLICIT visual intent (is_visual from search)
                    display_images = []
                    if hits["image_hits"] and is_visual:
                        display_images = hits["image_hits"]
                        with image_placeholder.container():
                            st.markdown("---")
                            st.markdown("#### Visual Context")
                            n_cols = min(3, len(display_images))
                            img_cols = st.columns(n_cols)
                            for idx, img in enumerate(display_images):
                                with img_cols[idx % n_cols]:
                                    score_val = img.get("score", 0.0)
                                    caption = f"Page {img['page']} | Match: {score_val:.2f}"
                                    st.image(img["image"], use_container_width=True, caption=caption)
                    
                    # --- Assistant Response Stored ---
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "images": display_images
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

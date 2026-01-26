import streamlit as st


class OCRManager:
    def __init__(self, languages=['en']):
        self.languages = languages
        self.reader = None
        self._initialize_reader()

    def _initialize_reader(self):
        """Initializes the EasyOCR reader."""
        try:
            import easyocr
            from .config import Config
            self.reader = easyocr.Reader(self.languages, gpu=Config.USE_GPU)
        except Exception as e:
            st.error(f"Failed to initialize EasyOCR: {e}")

    def extract_text(self, image_path):
        """Extracts text from an image at the given path."""
        if not self.reader:
            return ""
        
        try:
            results = self.reader.readtext(image_path, detail=0)
            return " ".join(results) if results else ""
        except Exception as e:
            print(f"OCR Extraction Error: {e}")
            return ""

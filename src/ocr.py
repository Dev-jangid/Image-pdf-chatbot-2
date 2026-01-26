import streamlit as st


class OCRManager:
    def __init__(self):
        self.use_tesseract = False
        self.reader = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes the OCR engine (Tesseract preferred, EasyOCR fallback)."""
        # 1. Try Tesseract first (Lighter, preferred for Cloud with packages.txt)
        try:
            import pytesseract
            # Check if tesseract is actually callable
            pytesseract.get_tesseract_version()
            self.use_tesseract = True
            print("OCR: Using Tesseract Engine")
            return
        except Exception:
            # If Tesseract fails (e.g. not installed locally), fall back silently
            pass

        # 2. Fallback to EasyOCR
        try:
            import easyocr
            from .config import Config
            # Suppress verbose loading
            self.reader = easyocr.Reader(['en'], gpu=Config.USE_GPU, verbose=False)
            print("OCR: Using EasyOCR Engine")
        except Exception as e:
            st.error(f"OCR Error: Could not initialize text extraction engines. {e}")

    def extract_text(self, image_path):
        """Extracts text from an image using the active engine."""
        if self.use_tesseract:
            try:
                import pytesseract
                from PIL import Image
                return pytesseract.image_to_string(Image.open(image_path)).strip()
            except Exception as e:
                print(f"Tesseract Extraction Error: {e}")
                return ""
        
        if self.reader:
            try:
                results = self.reader.readtext(image_path, detail=0)
                return " ".join(results)
            except Exception as e:
                print(f"EasyOCR Extraction Error: {e}")
                return ""
        
        return ""

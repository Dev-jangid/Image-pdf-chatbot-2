# Professional Image Confidence Enhancement Plan

To elevate the "Image-PDF Multimodal AI" to a professional standard with <1% false positive rate, we will implement a **"Triple-Lock Verification Protocol"**. This goes beyond simple vector matching to ensure that when an image is shown, it is inextricably linked to the user's query.

---

## 🚀 Phase 1: The "Dual-Factor" Authentication (Immediate Impact)
**Concept**: An image should only be shown if it looks right (CLIP) AND reads right (OCR).

1.  **OCR Semantic Verification**:
    *   **Current State**: We rely 100% on CLIP vector similarity.
    *   **Upgrade**: utilize the `src/ocr.py` module during the *search phase*.
    *   **Logic**: If the user asks for "Interest Rate Chart", we scan the OCR text of candidate images. If the specific words "Interest Rate" appear *inside* the image, we apply a massive **40% confidence boost**.
    *   **Result**: Eliminates generic "chart-looking" images that don't contain the relevant data.

2.  **Dynamic Z-Score Thresholding**:
    *   **Current State**: Fixed threshold (0.25).
    *   **Upgrade**: Calculate the *average* score of all images in the document. An image is only "High Confidence" if its score is **2x higher** than the average background noise. This adapts strictly to "messy" vs "clean" documents automatically.

---

## 🧠 Phase 2: Model Super-Resolution (High Precision)
**Concept**: Better eyes see better details.

3.  **Upgrade to `clip-vit-large-patch14`**:
    *   **Current State**: `base-patch32` (Good speed, medium detail).
    *   **Upgrade**: Switch to `openai/clip-vit-large-patch14`.
    *   **Impact**: This model sees minimal details (like small text labels in diagrams) that the base model misses. It requires more RAM but provides "Human-Level" recognition.

4.  **High-DPI Re-Indexing**:
    *   **Current Action**: Increase extraction DPI from standard to 300 DPI.
    *   **Benefit**: Gives the CLIP model a "4k" view of diagrams instead of "720p", dramatically improving feature detection in complex blueprints.

---

## 🔗 Phase 3: Contextual Anchoring (The "Caption" Lock)
**Concept**: Images don't exist in a vacuum; they have captions.

5.  **Caption-to-Image Bonding**:
    *   **Logic**: During PDF processing, scan the text immediately *below* or *above* an image.
    *   **Binding**: Index this "Caption Text" essentially as a second search tag for the image.
    *   **Result**: If the user asks for "Figure 2.1", and the text below an image says "Figure 2.1: Workflow", it is a 100% guaranteed match, bypassing vectors entirely.

---

## 📋 Execution Roadmap (Next Steps)

1.  **Step 1**: Implement **Model Upgrade** in `config.py` (Switch to `large-patch14`).
2.  **Step 2**: Modify `search.py` to add **OCR Verification** boost logic.
3.  **Step 3**: Implementation of **Dynamic Thresholding** to remove the manual `0.25` guess.

*Shall we proceed with Step 1 (Model Upgrade) and Step 2 (OCR Boost)?*

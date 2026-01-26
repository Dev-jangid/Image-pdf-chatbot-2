# Model Comparison: Base vs Large

Here is the technical comparison between your current vision model and the recommended professional upgrade:

| Feature | Current (`clip-vit-base-patch32`) | Upgrade (`clip-vit-large-patch14`) |
| :--- | :--- | :--- |
| **Resolution** | **Low (Patch 32)**<br>Like seeing a 144p video. It sees "shapes" but misses fine details. | **High (Patch 14)**<br>Like seeing a 1080p video. It sees small text, axis labels, and subtle lines in blueprints. |
| **Parameters** | **88 Million**<br>Fast, light, but "dumber". | **307 Million**<br>3.5x more "brain power". Understands complex relationships better. |
| **Disk Size** | **~350 MB**<br>Small download. | **~1.7 GB**<br>Significant download, requires more RAM to run. |
| **Accuracy** | **~63% Zero-Shot**<br>Good for general photos (cats, dogs), okay for diagrams. | **~75% Zero-Shot**<br>Excellent for technical diagrams, charts, and specialized documents. |
| **Speed** | **Fast**<br>Instant processing. | **Slower**<br>Takes ~2-3x longer to index, but retrieval is still instant. |
| **Use Case** | General consumer apps. | **Professional/Medical/Technical Analysis**. |

### 💡 Why "Patch 14" matters?
The "Patch" size is how the AI chops up the image.
*   **Patch 32 (Current)**: Chops image into big 32x32 pixel blocks. Small details get lost inside these big blocks.
*   **Patch 14 (Upgrade)**: Chops image into tiny 14x14 pixel blocks. This captures much finer detail, crucial for reading graphs and technical PDFs.

### Recommendation
For a **professional-grade** document assistant where accuracy is critical (e.g., reading charts, blueprints), the **Large-Patch14** model is the industry standard. The speed trade-off during upload is negligible compared to the massive gain in answer quality.

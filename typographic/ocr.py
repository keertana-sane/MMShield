from paddleocr import PaddleOCR


class OCRExtractor:
    """
    OCR module for MMShield.
    Responsible only for extracting text regions
    from financial document images.
    """

    def __init__(self):

        print("Loading PaddleOCR...")

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

        print("OCR Ready!")

    def extract_text_regions(self, image_path):
        """
        Extract all text regions from an image.
        """

        results = self.ocr.predict(str(image_path))

        page = results[0]

        texts = page["rec_texts"]
        scores = page["rec_scores"]
        polygons = page["dt_polys"]

        regions = []

        for text, score, polygon in zip(texts, scores, polygons):
            regions.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "polygon": polygon.tolist(),
                }
            )

        return regions

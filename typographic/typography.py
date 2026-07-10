import string


class TypographyAnalyzer:
    """
    Extracts typographic features from OCR regions.
    """

    def __init__(self):
        pass

    # -------------------------------------------------
    # Geometry Features
    # -------------------------------------------------

    def geometry_features(self, polygon):

        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]

        min_x = min(xs)
        max_x = max(xs)

        min_y = min(ys)
        max_y = max(ys)

        width = max_x - min_x
        height = max_y - min_y

        area = width * height

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        aspect_ratio = width / height if height != 0 else 0

        return {
            "width": width,
            "height": height,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
            "aspect_ratio": aspect_ratio,
        }

    # -------------------------------------------------
    # Text Statistics
    # -------------------------------------------------

    def text_statistics(self, text):

        words = text.split()

        character_count = len(text)

        word_count = len(words)

        avg_word_length = (
            sum(len(word) for word in words) / word_count
            if word_count > 0 else 0
        )

        return {
            "character_count": character_count,
            "word_count": word_count,
            "avg_word_length": round(avg_word_length, 2),
        }

    # -------------------------------------------------
    # Character Composition
    # -------------------------------------------------

    def character_statistics(self, text):

        total = len(text)

        if total == 0:
            total = 1

        alphabet = sum(c.isalpha() for c in text)

        numeric = sum(c.isdigit() for c in text)

        uppercase = sum(c.isupper() for c in text)

        whitespace = sum(c.isspace() for c in text)

        special = sum(
            c in string.punctuation
            for c in text
        )

        return {

            "alphabet_ratio": round(alphabet / total, 3),

            "numeric_ratio": round(numeric / total, 3),

            "uppercase_ratio": round(uppercase / total, 3),

            "whitespace_ratio": round(whitespace / total, 3),

            "special_character_ratio": round(special / total, 3),
        }

    # -------------------------------------------------
    # Visual Density
    # -------------------------------------------------

    def visual_density(self, text, area):

        if area == 0:
            return 0

        return round(len(text) / area, 5)

    # -------------------------------------------------
    # Final Feature Vector
    # -------------------------------------------------

    def extract_features(self, region):

        geometry = self.geometry_features(region["polygon"])

        statistics = self.text_statistics(region["text"])

        characters = self.character_statistics(region["text"])

        density = self.visual_density(
            region["text"],
            geometry["area"]
        )

        return {

            "text": region["text"],

            "confidence": region["confidence"],

            **geometry,

            **statistics,

            **characters,

            "character_density": density,

            "estimated_font_size": geometry["height"],
        }

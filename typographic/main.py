from semantic import SemanticAnalyzer
from feature_vector import FeatureVectorBuilder
from typography import TypographyAnalyzer
from config import TRAIN_IMAGES
from ocr import OCRExtractor

TEST_IMAGE = "X00016469612.jpg"

image_path = TRAIN_IMAGES / TEST_IMAGE

print(f"Reading : {image_path.name}")

ocr = OCRExtractor()
typography = TypographyAnalyzer()
builder = FeatureVectorBuilder()
semantic = SemanticAnalyzer()

regions = ocr.extract_text_regions(image_path)

print(f"\nDetected {len(regions)} text regions\n")

for region in regions:

    features = typography.extract_features(region)
    semantic_features = semantic.extract_features(
    region["text"]
    )

    features.update(semantic_features)

    builder.add_region(features)

df = builder.save_csv()

print(df.head())

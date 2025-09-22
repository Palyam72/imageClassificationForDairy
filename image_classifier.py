from transformers import pipeline
from PIL import Image

class ImageClassification:
    def __init__(self):
        # Load zero-shot image classification pipeline
        self.classifier = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch16"
        )
        # Candidate labels
        self.labels = ["cow", "buffalo"]
        self.last_confidence = 0.0

    def image_classification(self, image_path: str) -> str:
        """
        Perform zero-shot classification on the input image.
        Returns the label with the highest score.
        """
        # Open image
        image = Image.open(image_path)

        # Run zero-shot classification
        results = self.classifier(image, candidate_labels=self.labels)

        # Pick the highest score label and store confidence
        if results and len(results) > 0:
            self.last_confidence = results[0]["score"]
            return results[0]["label"]
        else:
            return "Unknown"
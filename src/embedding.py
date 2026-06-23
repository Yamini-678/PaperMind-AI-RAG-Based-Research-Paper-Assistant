from sentence_transformers import SentenceTransformer

class EmbeddingManager:

    def __init__(self , model_name = "all-miniLM-L6-v2"):
        self.model_name = model_name
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            self.model_name
        )
        print("Model loaded")

    def generate_embedding(self , text):
        embedding = self.model.encode(text , show_progress_bar=True)
        return embedding
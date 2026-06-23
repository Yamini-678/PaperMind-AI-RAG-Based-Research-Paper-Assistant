from src.embedding import EmbeddingManager
from src.vector_store import VectorStore


class RAGRetriever:

    def __init__(self , embedding_manager , vector_store):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store


    def retrieve(self , query , top_k=3):

        query_embedding = self.embedding_manager.generate_embedding(
            [query]
        )[0]

        results = self.vector_store.query(
            query_embedding.tolist(),
            top_k
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
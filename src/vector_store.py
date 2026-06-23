import chromadb

class VectorStore:

    def __init__(self , persist_directory="data/vector_db" , collection_name = 'rag_collection'):

        self.client = chromadb.PersistentClient(path = persist_directory)

        self.collection = self.client.get_or_create_collection(
            name = collection_name
        )

    def reset_collection(self):

        try:
            self.client.delete_collection("rag_collection")
        except:
            pass

        self.collection = self.client.get_or_create_collection(
                name = 'rag_collection'
        )

    def add_documents(self , ids , documents , embeddings , metadatas):
        self.collection.add(
            ids = ids,
            documents = documents,
            embeddings = embeddings,
            metadatas = metadatas
        )

    def query(self , query_embeddings , top_k=3):

        results = self.collection.query(
            query_embeddings = [query_embeddings],
            n_results = top_k
        )

        return results

        
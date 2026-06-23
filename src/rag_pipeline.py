from src.llm import llm
from src.prompt import RAG_PROMPT

class RAGPipeline:

    def __init__(self, retriever):
        self.retriever = retriever

    def generate_answer(self, query):

        retrieved_data = self.retriever.retrieve(
            query=query,
            top_k=3
        )

        docs = retrieved_data["documents"]

        if not docs:
            return "I could not find relevant information in the uploaded document."

        context = "\n\n".join(docs)

        print("\n===== RETRIEVED CONTEXT =====")
        print(context[:1000])
        print("=============================\n")

        prompt = RAG_PROMPT.format(
            context=context,
            query=query
        )

        response = llm.invoke(prompt)

        answer = response.content

        if "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        return answer
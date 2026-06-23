from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate(
    input_variables = ['context', 'query'],
    template = """ 
You are an expert research paper assistant.

Answer only using the provided context.

If the answer is not present in the context, say:

"I could not find this information in the uploaded paper."

Context:
{context}

Question:
{query}

Answer:
"""
)
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


llm = ChatGroq(
    groq_api_key = os.getenv("GROQ_API_KEY"),
    model = "qwen/qwen3-32b",
    temperature = 0.1
)

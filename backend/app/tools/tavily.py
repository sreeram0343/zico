# pyrefly: ignore [missing-import]
import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def tavily_search(query):
    return tavily_client.search(query=query, max_results=5)

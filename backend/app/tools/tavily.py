# pyrefly: ignore [missing-import]
from tavily import TavilyClient

import os

from dotenv import load_dotenv
load_dotenv()



tavily_client = TavilyClient(
    api_key= os.getenv("TAVILY_API_KEY")
)

def tavily_search(query):
    response = tavily_client.search(
        query = query,
        max_results = 5

    )

    results = []


    


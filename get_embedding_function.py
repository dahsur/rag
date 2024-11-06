# from langchain_ollama import OllamaLLM
# from langchain_ollama import OllamaEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_ollama import OllamaLLM

def get_embedding_function():
    # Use OpenAI's text-embedding-ada-002 model for embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="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")
    return embeddings

#def get_embedding_function():
 #   embeddings = OllamaEmbeddings(model="nomic-embed-text")
 #   return embeddings

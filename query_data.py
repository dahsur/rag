import argparse
import logging
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

# Constants
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---

Answer the question based on the above context: {question}
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Initialize the embedding function and Chroma DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Retrieve documents with similarity search
    logging.info("Retrieving relevant documents...")
    results = db.similarity_search_with_score(query_text, k=20)  # Experiment with k, 20 is often a good starting point

    # Filter low-relevance results
    results = filter_low_relevance(results, threshold=0.5)  # Define a relevance threshold to discard low-scoring docs

    # Summarize or reduce context to essential content
    context_text = summarize_context(results)

    # Create prompt with refined template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    logging.info("Generated prompt for model:\n" + prompt)

    # Invoke model with prompt
    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)

    # Post-process and validate response
    response_text = post_process_response(response_text, query_text, context_text)

    # Extract and format sources
    sources = [doc.metadata.get("id", "unknown") for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


def filter_low_relevance(results, threshold=0.5):
    """
    Filters out documents below a relevance score threshold.
    """
    filtered_results = [(doc, score) for doc, score in results if score >= threshold]
    logging.info(f"Filtered results to {len(filtered_results)} documents with scores above {threshold}")
    return filtered_results


def summarize_context(results):
    """
    Summarizes or limits the context length to the top relevant documents.
    """
    context_chunks = [doc.page_content for doc, _score in results[:10]]  # Limit to top 10 most relevant chunks
    context_text = "\n\n---\n\n".join(context_chunks)
    logging.info("Context summarized for prompt.")
    return context_text


def post_process_response(response, query, context):
    """
    Post-processes the model's response to enhance accuracy.
    """
    # Cross-check response with context to ensure it aligns with the retrieved information
    if not validate_response(response, context):
        logging.warning("Response may contain hallucinations or irrelevant information.")
        response = "The model's response may not be fully accurate based on available context."

    # Optionally, add any further formatting to the response here
    return response


def validate_response(response, context):
    """
    Validates that the response aligns with context to reduce hallucinations.
    """
    # Basic check: ensure the response contains key terms from the context
    context_terms = set(context.split())
    response_terms = set(response.split())
    common_terms = context_terms.intersection(response_terms)

    return len(common_terms) > 5  # A simple heuristic for context alignment


if __name__ == "__main__":
    main()
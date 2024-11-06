from flask import Flask, request, jsonify, render_template
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}

---

Answer the question based on the above context: {question}
"""

# Initialize Flask app
app = Flask(__name__)

# Initialize the embedding function and Chroma DB
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Define the route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the API route to handle queries
@app.route('/query', methods=['POST'])
def query():
    query_text = request.json.get("query_text")
    if not query_text:
        return jsonify({"error": "No query text provided"}), 400

    # Use the existing query logic
    results = db.similarity_search_with_score(query_text, k=20)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    # Return JSON response
    return jsonify({"response": response_text, "sources": sources})

if __name__ == "__main__":
    app.run(debug=True)
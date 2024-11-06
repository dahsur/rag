import argparse
import os
import shutil
import logging
import time
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from concurrent.futures import ThreadPoolExecutor

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

CHROMA_PATH = "chroma"
DATA_PATH = "data"
ROWS_PER_CHUNK = 50  # This applies only to CSV files
BATCH_SIZE = 250      # Reduce batch size to handle rate limiting
MAX_WORKERS = 5       # Limit concurrency to avoid rate limiting
DELAY_BETWEEN_BATCHES = 2  # Add delay between batches in seconds


def main():
    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()

    # Clear the database if --reset flag is passed
    if args.reset:
        logging.info("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    documents = []

    # Check for CSV files
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
    if csv_files:
        for filename in csv_files:
            file_path = os.path.join(DATA_PATH, filename)
            logging.info(f"Loading CSV file: {file_path}")
            loader = CSVLoader(file_path)
            documents.extend(loader.load())
    else:
        logging.info("No CSV files found.")

    # Check for PDF files
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if pdf_files:
        logging.info("Loading PDF files...")
        pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
        documents.extend(pdf_loader.load())

    logging.info(f"Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents: list[Document]):
    chunks = []

    for document in documents:
        source = document.metadata.get("source", "")

        # Apply row-based splitting for CSVs
        if source.endswith(".csv"):
            chunks.extend(split_csv_document_by_rows([document], rows_per_chunk=ROWS_PER_CHUNK))
        # Apply text splitting for PDFs
        elif source.endswith(".pdf"):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=80,
                length_function=len,
                is_separator_regex=False,
            )
            chunks.extend(text_splitter.split_documents([document]))

    logging.info(f"Total number of chunks created: {len(chunks)}")
    return chunks


def split_csv_document_by_rows(documents: list[Document], rows_per_chunk=5):
    chunks = []
    current_chunk = []
    
    for i, document in enumerate(documents):
        current_chunk.append(document)
        
        if (i + 1) % rows_per_chunk == 0:
            chunks.append(current_chunk)
            current_chunk = []  # Reset for the next chunk
    
    if current_chunk:
        chunks.append(current_chunk)
    
    logging.info(f"Total number of CSV chunks created: {len(chunks)}")
    return chunks


def write_batch_to_chroma(db, batch, batch_ids, batch_number, total_batches, max_retries=5):
    logging.info(f"Writing batch {batch_number} of {total_batches} to Chroma")
    retry_count = 0
    success = False
    
    while not success and retry_count < max_retries:
        try:
            db.add_documents(batch, ids=batch_ids)
            logging.info(f"Batch {batch_number} of {total_batches} written successfully")
            success = True
        except openai.error.RateLimitError as e:
            retry_count += 1
            wait_time = 2 ** retry_count  # Exponential backoff
            logging.warning(f"Rate limit error: retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    logging.info(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        logging.info(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

        # Parallel processing of batches
        total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for i in range(0, len(new_chunks), BATCH_SIZE):
                batch_number = (i // BATCH_SIZE) + 1
                batch = new_chunks[i:i + BATCH_SIZE]
                batch_ids = new_chunk_ids[i:i + BATCH_SIZE]
                futures.append(
                    executor.submit(write_batch_to_chroma, db, batch, batch_ids, batch_number, total_batches)
                )

                # Add delay between batches to throttle requests
                time.sleep(DELAY_BETWEEN_BATCHES)

            for future in futures:
                future.result()  # Wait for all futures to complete
    else:
        logging.info("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    for chunk_index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown_source")
        page = chunk.metadata.get("page", None)

        # Use page number and chunk index to generate unique IDs for PDFs
        if page is not None:
            chunk_id = f"{source}:{page}:{chunk_index}"
        else:
            row_index = chunk.metadata.get("row_index", None)
            chunk_id = f"{source}:{row_index}:{chunk_index}" if row_index is not None else f"{source}:{chunk_index}"

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        logging.info("Database cleared successfully.")


if __name__ == "__main__":
    main()
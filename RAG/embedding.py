import json
import sys
import psycopg2
from pgvector.psycopg2 import register_vector

# Attempt to import HuggingFaceEmbeddings and handle potential errors
try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
except ImportError:
    print("Error: langchain_huggingface.embeddings could not be imported.")
    print("Please ensure 'langchain-huggingface', 'transformers', and 'torch' are correctly installed.")
    sys.exit(1)

DB_NAME = "vector_db_new"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"

JSONL_FILE_PATH = "../training_data/ragqa_arena_tech_corpus.jsonl"
JSON_TEXT_FIELD = "text"
JSON_DOC_ID_FIELD = "doc_id"
JSON_AUTHOR_FIELD = "author"

TABLE_NAME = "rga_embeddings"
TEXT_COLUMN_NAME = "text_content"
EMBEDDING_COLUMN_NAME = "embedding"
DOC_ID_COLUMN_NAME = "doc_id"
AUTHOR_COLUMN_NAME = "author"

MODEL_NAME = "BAAI/bge-m3"

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        register_vector(conn)
        print("Successfully connected to the database and registered pgvector type.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to the database: {e}")
        sys.exit(1)

def load_embedding_model():
    print(f"Initializing embedding model: {MODEL_NAME}...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        print(f"Successfully loaded embedding model: {MODEL_NAME}")
        return embedding_model
    except Exception as e:
        print(f"Fatal Error: Could not load HuggingFace embedding model '{MODEL_NAME}': {e}")
        sys.exit(1)

def process_and_insert_data(conn, embedding_model):
    print(f"Starting to process '{JSONL_FILE_PATH}'...")
    lines_processed_count = 0
    try:
        with open(JSONL_FILE_PATH, 'r', encoding='utf-8') as infile:
            for line_number, line in enumerate(infile, 1):
                try:
                    record = json.loads(line.strip())
                    text_to_embed = record.get(JSON_TEXT_FIELD)
                    doc_id = record.get(JSON_DOC_ID_FIELD)
                    author = record.get(JSON_AUTHOR_FIELD)

                    if text_to_embed and isinstance(text_to_embed, str) and text_to_embed.strip():
                        text_to_embed = text_to_embed.strip()
                        embedding = embedding_model.embed_documents([text_to_embed])[0]

                        with conn.cursor() as cur:
                            cur.execute(
                                f"""INSERT INTO {TABLE_NAME}
                                    ({DOC_ID_COLUMN_NAME}, {AUTHOR_COLUMN_NAME}, {TEXT_COLUMN_NAME}, {EMBEDDING_COLUMN_NAME})
                                    VALUES (%s, %s, %s, %s)""",
                                (doc_id, author, text_to_embed, embedding)
                            )
                        conn.commit()
                        lines_processed_count += 1
                    else:
                        print(f"Warning: Skipping line {line_number}. Missing or empty text in field '{JSON_TEXT_FIELD}'. Content: {line.strip()[:100]}...")

                except json.JSONDecodeError:
                    print(f"Warning: Skipping line {line_number} due to JSON decoding error: {line.strip()[:100]}...")
                except Exception as e:
                    print(f"Warning: Skipping line {line_number} due to unexpected error: {e}. Content: {line.strip()[:100]}...")
                    conn.rollback()

        print(f"\nFinished processing. Total lines processed and inserted: {lines_processed_count}")

    except FileNotFoundError:
        print(f"Error: The file '{JSONL_FILE_PATH}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")
        sys.exit(1)


def main():
    embedding_model = load_embedding_model()
    conn = None
    try:
        conn = get_db_connection()
        process_and_insert_data(conn, embedding_model)
    except Exception as e:
        print(f"A critical error occurred in main execution: {e}")
    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

if __name__ == "__main__":
    main()
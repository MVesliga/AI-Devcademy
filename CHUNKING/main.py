import json
import sys
import os

from db.connection import get_db_connection
from embeddings.model import get_embedding_model
from chunking_db.db_utils import insert_chunks
from fixed_chunking import get_fixed_chunks
from recursive_chunking import get_recursive_chunks
from cluster_chunking import get_cluster_chunks
from sentence_transformers import SentenceTransformer


# --- CONFIG ---
TABLE_MAP = {
    "fixed": "fixed_chunk_embeddings",
    "recursive": "recursive_chunk_embeddings",
    "cluster": "cluster_chunk_embeddings"
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "training_data", "ragqa_arena_tech_corpus.jsonl")
CHUNKING_METHOD = sys.argv[1] if len(sys.argv) > 1 else "fixed"
BATCH_SIZE = 32

def main():
    if CHUNKING_METHOD not in TABLE_MAP:
        print(f"Invalid method: {CHUNKING_METHOD}")
        sys.exit(1)

    conn = get_db_connection("chunking_db")
    embedding_model = get_embedding_model()
    cluster_model = SentenceTransformer("BAAI/bge-m3") if CHUNKING_METHOD == "cluster" else None
    table = TABLE_MAP[CHUNKING_METHOD]

    batch_rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            doc_id = record.get("doc_id")
            author = record.get("author")
            text = record.get("text", "").strip()
            if not text:
                continue

            # --- Apply chunking ---
            if CHUNKING_METHOD == "fixed":
                chunks = get_fixed_chunks(text)
            elif CHUNKING_METHOD == "recursive":
                chunks = get_recursive_chunks(text)
            elif CHUNKING_METHOD == "cluster":
                chunks = get_cluster_chunks(text, cluster_model)

            # --- Embed and collect ---
            embeddings = embedding_model.embed_documents(chunks)
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                batch_rows.append((doc_id, author, chunk, emb, i, CHUNKING_METHOD))

            # --- Batch insert ---
            if len(batch_rows) >= BATCH_SIZE:
                insert_chunks(conn, table, batch_rows)
                batch_rows.clear()

    # Insert leftovers
    if batch_rows:
        insert_chunks(conn, table, batch_rows)

    conn.close()
    print(f"Done. Data inserted into: {table}")

if __name__ == "__main__":
    main()

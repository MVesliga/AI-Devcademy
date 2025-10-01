import json
from local_embedding_model import get_local_embedding_model
from db.connection import get_db_connection
from tqdm import tqdm
from chunking_strategies import *

embedding_model = get_local_embedding_model()
conn = get_db_connection('vector_db')
cursor = conn.cursor()

jsonl_path = "training_data/ragqa_arena_tech_corpus.jsonl"

# Load first 1000 entries from JSONL
def load_first_1000(jsonl_path):
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            entry = json.loads(line)
            entries.append(entry)
    return entries

# Main
data = load_first_1000(jsonl_path)

for entry in tqdm(data, desc="Processing entries"):
    doc_id = entry["doc_id"]
    author = entry["author"]
    text = entry["text"]

    # Fixed-size
    #fixed_chunks = get_fixed_chunks(text)
    #save_chunks_to_db(embedding_model, cursor, "embeddings_fixed", doc_id, author, fixed_chunks)
    #conn.commit()

    # Recursive
    #recursive_chunks = get_recursive_chunks(text)
    #save_chunks_to_db(embedding_model, cursor,"embeddings_recursive", doc_id, author, recursive_chunks)
    #conn.commit()

    # Semantic
    try:
        semantic_chunks = get_semantic_chunks(text)
        save_chunks_to_db(embedding_model, cursor,"embeddings_semantic", doc_id, author, semantic_chunks)
        conn.commit()
    except Exception as e:
        print(f"Semantic chunking failed for doc {doc_id}: {e}")

cursor.close()
conn.close()
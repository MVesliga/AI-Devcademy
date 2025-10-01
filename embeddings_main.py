import json
from local_embedding_model import get_local_embedding_model
from db.connection import get_db_connection
from tqdm import tqdm

embedding_model = get_local_embedding_model()
conn = get_db_connection('vector_db')
cursor = conn.cursor()

jsonl_path = "training_data/ragqa_arena_tech_corpus.jsonl"
insert_sql = "INSERT INTO rag_embeddings (doc_id, author, text_content, embedding) VALUES (%s, %s, %s, %s)"

BATCH_SIZE = 100
batch_entries = []
MAX_ENTRIES = 1000
total_processed = 0

with open(jsonl_path, 'r', encoding='utf-8') as file:
    total_lines = sum(1 for _ in file)

with open(jsonl_path, 'r', encoding='utf-8') as file, tqdm(total=total_lines, desc="Processing documents") as pbar:
    for line in file:
        if total_processed >= MAX_ENTRIES:
            break

        entry = json.loads(line)
        doc_id = entry['doc_id']
        author = entry['author']
        text = entry['text']

        batch_entries.append((doc_id, author, text))
        total_processed += 1
        pbar.update(1)

        if len(batch_entries) >= BATCH_SIZE:
            texts = [e[2] for e in batch_entries]
            with tqdm(total=len(texts), desc="Embedding + Insert", leave=False) as batch_bar:
                embeddings = embedding_model.embed_documents(texts)
                batch_bar.update(len(texts))

            db_rows = [
                (e[0], e[1], e[2], emb)
                for e, emb in zip(batch_entries, embeddings)
            ]
            cursor.executemany(insert_sql, db_rows)
            conn.commit()
            batch_entries = []

if batch_entries:
    texts = [e[2] for e in batch_entries]
    with tqdm(total=len(texts), desc="Final batch", leave=False) as batch_bar:
        embeddings = embedding_model.embed_documents(texts)
        batch_bar.update(len(texts))

    db_rows = [
        (e[0], e[1], e[2], emb)
        for e, emb in zip(batch_entries, embeddings)
    ]
    cursor.executemany(insert_sql, db_rows)
    conn.commit()

cursor.close()
conn.close()





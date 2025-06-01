from utils.parse_jsonl import parse_jsonl_line
from ..config import (
    JSONL_FILE_PATH, JSON_TEXT_FIELD, JSON_DOC_ID_FIELD, JSON_AUTHOR_FIELD,
    TABLE_NAME, TEXT_COLUMN_NAME, EMBEDDING_COLUMN_NAME, DOC_ID_COLUMN_NAME, AUTHOR_COLUMN_NAME
)

def process_and_insert_data(conn, embedding_model):
    inserted = 0
    print(f"Processing file: {JSONL_FILE_PATH}")

    with open(JSONL_FILE_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            doc_id, author, text = parse_jsonl_line(line, JSON_TEXT_FIELD, JSON_DOC_ID_FIELD, JSON_AUTHOR_FIELD)

            if not text:
                print(f"⚠Skipping line {line_num}: Empty or missing text.")
                continue

            try:
                embedding = embedding_model.embed_documents([text])[0]
                with conn.cursor() as cur:
                    cur.execute(
                        f"""INSERT INTO {TABLE_NAME} 
                            ({DOC_ID_COLUMN_NAME}, {AUTHOR_COLUMN_NAME}, {TEXT_COLUMN_NAME}, {EMBEDDING_COLUMN_NAME})
                            VALUES (%s, %s, %s, %s)""",
                        (doc_id, author, text, embedding)
                    )
                conn.commit()
                inserted += 1
            except Exception as e:
                print(f"Error on line {line_num}: {e}")
                conn.rollback()

    print(f"\nDone. Total rows inserted: {inserted}")
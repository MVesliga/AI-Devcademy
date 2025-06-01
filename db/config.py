# config.py
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"

MODEL_NAME = "BAAI/bge-m3"

JSONL_FILE_PATH = "data/ragqa_arena_tech_corpus.jsonl"
JSON_TEXT_FIELD = "text"
JSON_DOC_ID_FIELD = "doc_id"
JSON_AUTHOR_FIELD = "author"

TABLE_NAME = "rga_embeddings"
TEXT_COLUMN_NAME = "text_content"
EMBEDDING_COLUMN_NAME = "embedding"
DOC_ID_COLUMN_NAME = "doc_id"
AUTHOR_COLUMN_NAME = "author"
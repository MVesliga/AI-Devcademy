from embeddings.model import get_embedding_model
from rag_db.connection import get_db_connection
from rag_db.insert import process_and_insert_data

def main():
    model = get_embedding_model()
    conn = None
    try:
        conn = get_db_connection("vector_db_new")
        process_and_insert_data(conn, model)
    finally:
        if conn:
            conn.close()
            print("DB connection closed.")

if __name__ == "__main__":
    main()
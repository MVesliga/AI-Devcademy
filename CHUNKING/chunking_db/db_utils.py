from psycopg2.extras import execute_values

def insert_chunks(conn, table_name, rows):
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {table_name} 
                (doc_id, author, chunk, embedding, chunk_index, chunking_method, cmetadata)
            VALUES %s
            """,
            rows
        )
    conn.commit()
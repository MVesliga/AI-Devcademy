from local_embedding_model import get_local_embedding_model
from db.connection import get_db_connection
from psycopg2 import sql

embedding_model = get_local_embedding_model()

conn = get_db_connection('vector_db')
cursor = conn.cursor()

#-----Questions generated for doc_ids to test the fetching-----
#query = "How do I edit and rebuild a .deb package?" #--doc_id: 229377
#query = "How can I mute the Mac startup chime?" #--doc_id: 6
#query = "How can I convert a .cer file to .pfx for Android Wi-Fi setup?" #--doc_id: 131078
#query = "Can in-app purchases be shared with Family Sharing on iOS?" #--doc_id: 65554
query = "What is a good application launcher for macOS?" #--doc_id: 34

query_embedding = embedding_model.embed_query(query)

#Query for testing chunked data
query = sql.SQL("""
                SELECT doc_id, author, chunk_text
                FROM embeddings_semantic
                ORDER BY embedding <#> %s::vector
    LIMIT 5;
                """)

cursor.execute(query, (query_embedding,))
results = cursor.fetchall()

# Display results
for doc_id, author, text in results:
    print(f"\nDoc ID: {doc_id}")
    print(f"Author: {author}")
    print(f"Text Snippet: {text[:300]}...")

cursor.close()
conn.close()
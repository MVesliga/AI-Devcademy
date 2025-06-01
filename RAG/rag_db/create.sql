CREATE TABLE IF NOT EXISTS rga_embeddings (
                                                          id SERIAL PRIMARY KEY,
                                                          doc_id INTEGER,
                                                          author TEXT,
                                                          text_content TEXT,
                                                          embedding VECTOR(1024)
    );
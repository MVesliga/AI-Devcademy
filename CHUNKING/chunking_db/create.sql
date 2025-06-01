CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE fixed_chunk_embeddings (
                                        id SERIAL PRIMARY KEY,
                                        doc_id TEXT,
                                        author TEXT,
                                        chunk TEXT,
                                        embedding VECTOR(1024),
                                        chunk_index INTEGER,
                                        chunking_method TEXT DEFAULT 'fixed',
                                        created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE recursive_chunk_embeddings (
                                            id SERIAL PRIMARY KEY,
                                            doc_id TEXT,
                                            author TEXT,
                                            chunk TEXT,
                                            embedding VECTOR(1024),
                                            chunk_index INTEGER,
                                            chunking_method TEXT DEFAULT 'recursive',
                                            created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE cluster_chunk_embeddings (
                                          id SERIAL PRIMARY KEY,
                                          doc_id TEXT,
                                          author TEXT,
                                          chunk TEXT,
                                          embedding VECTOR(1024),
                                          chunk_index INTEGER,
                                          chunking_method TEXT DEFAULT 'cluster',
                                          created_at TIMESTAMP DEFAULT NOW()
);
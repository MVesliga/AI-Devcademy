from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import re

def get_fixed_chunks(text, chunk_size=200, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def get_recursive_chunks(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)

def naive_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def get_semantic_chunks(text, model_name="BAAI/bge-m3", num_chunks=5):
    sentences = naive_sent_tokenize(text)
    if len(sentences) <= 1:
        return [text]  # Not enough to chunk

    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, normalize_embeddings=True)

    n_clusters = min(len(sentences), num_chunks)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    labels = clustering.fit_predict(embeddings)

    # Group sentences by cluster label
    clusters = {}
    for label, sentence in zip(labels, sentences):
        clusters.setdefault(label, []).append(sentence)

    # Concatenate grouped sentences into chunks
    chunks = [" ".join(clusters[k]) for k in sorted(clusters.keys())]
    return chunks

def save_chunks_to_db(model,  cursor, table_name, doc_id, author, chunks):
    embeddings = model.embed_documents(chunks)
    records = [
        (idx, doc_id, author, chunk, emb)
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]
    insert_sql = f"""
        INSERT INTO {table_name} (chunk_index, doc_id, author, chunk_text, embedding)
        VALUES (%s, %s, %s, %s, %s)
    """
    cursor.executemany(insert_sql, records)
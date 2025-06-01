from sklearn.cluster import KMeans

def get_cluster_chunks(text, embed_fn, num_clusters=5):
    sentences = text.split(". ")
    if len(sentences) <= 1:
        return [text]

    sentence_embeddings = embed_fn(sentences)
    kmeans = KMeans(n_clusters=min(num_clusters, len(sentences)), random_state=42)
    kmeans.fit(sentence_embeddings)

    clusters = [[] for _ in range(kmeans.n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(sentences[i])

    return [" ".join(cluster) for cluster in clusters]
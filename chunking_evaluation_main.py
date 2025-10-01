from sklearn.metrics import precision_score, recall_score, f1_score
from local_embedding_model import get_local_embedding_model
from db.connection import get_db_connection
from tqdm import tqdm
from psycopg2 import sql

evaluation_set = [
    {
        "query": "How do I edit and rebuild a .deb package?",
        "relevant_doc_ids": [229377]
    },
    {
        "query": "How can I mute the Mac startup chime?",
        "relevant_doc_ids": [6]
    },
    {
        "query": "How can I convert a .cer file to .pfx for Android Wi-Fi setup?",
        "relevant_doc_ids": [131078]
    },
    {
        "query": "Can in-app purchases be shared with Family Sharing on iOS?",
        "relevant_doc_ids": [65554]
    },
    {
        "query": "What is a good application launcher for macOS?",
        "relevant_doc_ids": [34]
    },
]

def evaluate_chunking_strategy(
        chunk_table_name: str,
        embedding_model,
        eval_set,
        cursor,
        k=5
):
    results = []

    for example in tqdm(eval_set, desc=f"Evaluating {chunk_table_name}"):
        query = example["query"]
        relevant_ids = set(example["relevant_doc_ids"])

        # Get embedding for the query
        query_embedding = embedding_model.embed_query(query)

        # Run vector similarity search
        query = sql.SQL("""
                      SELECT id, doc_id
                      FROM {table_name}
                      ORDER BY embedding <#> %s::vector
                      LIMIT %s;
                      """).format(
            table_name=sql.Identifier(chunk_table_name)
        )
        cursor.execute(query, (query_embedding, k))
        rows = cursor.fetchall() 

        retrieved_doc_ids = {row[1] for row in rows}

        # Compute metrics
        true_positives = relevant_ids & retrieved_doc_ids
        precision = len(true_positives) / len(retrieved_doc_ids) if retrieved_doc_ids else 0
        recall = len(true_positives) / len(relevant_ids) if relevant_ids else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append((precision, recall, f1))

    # Compute average scores
    avg_precision = sum(r[0] for r in results) / len(results)
    avg_recall = sum(r[1] for r in results) / len(results)
    avg_f1 = sum(r[2] for r in results) / len(results)

    return {
        "strategy": chunk_table_name,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1
    }


def run_evaluation():
    embedding_model = get_local_embedding_model()

    conn = get_db_connection('vector_db')
    cursor = conn.cursor()

    strategies = ["embeddings_fixed", "embeddings_recursive", "embeddings_semantic"]
    for strategy in strategies:
        metrics = evaluate_chunking_strategy(strategy, embedding_model, evaluation_set, cursor)
        print(f"\n[{strategy}]")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1 Score:  {metrics['f1']:.3f}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    run_evaluation()
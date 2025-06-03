import json
import os
import random
from langchain_community.vectorstores import PGVector
from embeddings.model import get_embedding_model
from langchain.vectorstores.base import VectorStoreRetriever
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(SCRIPT_DIR, "..", "training_data", "ragqa_arena_tech_examples.jsonl")

def load_eval_set(file_path, size=50):
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))

    if size > len(examples):
        raise ValueError(f"Requested size {size} is larger than available examples ({len(examples)}).")

    return random.sample(examples, size)

def get_retriever(table_name):
    embeddings = get_embedding_model()

    return PGVector(
        connection_string="postgresql+psycopg2://postgres:postgres@localhost:5433/chunking_db",
        embedding_function=embeddings,
        collection_name=table_name,
        distance_strategy="COSINE"
    ).as_retriever(search_kwargs={"k": 5})

def evaluate_strategy(retriever, eval_set):
    metric = AnswerRelevancyMetric()
    test_cases = []

    for item in eval_set:
        retrieved_docs = retriever.get_relevant_documents(item["question"])
        context = " ".join([doc.page_content for doc in retrieved_docs])
        test_case = LLMTestCase(
            input=item["question"],
            actual_output=context,
            expected_output=item["response"],
            context=context
        )
        test_cases.append(test_case)

    results = evaluate(test_cases, [metric])
    return results

def main():
    eval_set = load_eval_set(EVAL_FILE)

    for strategy in ["fixed", "recursive", "cluster"]:
        print(f"\n--- Evaluating Strategy: {strategy} ---")
        table_name = f"{strategy}_chunk_embeddings"
        retriever = get_retriever(table_name)
        results = evaluate_strategy(retriever, eval_set)
        for result in results:
            print(f"Q: {result.input}")
            print(f"F1: {result.score}\n")

if __name__ == "__main__":
    main()
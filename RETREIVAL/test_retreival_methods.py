import os
import json
import random

from langchain_community.vectorstores import PGVector
from embeddings.model import get_embedding_model
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# --- Import your retrieval methods ---
from hyde import hyde_retrieval
from multi_query import multi_query_retrieval
from step_back import step_back_query
from query_decomposition import decompose_query

# --- Config ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = os.path.join(SCRIPT_DIR, "..", "training_data", "ragqa_arena_tech_examples.jsonl")

# --- Load Evaluation Data ---
def load_eval_set(file_path, size=5):
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))

    return random.sample(examples, size)

# --- Create retriever ---
def get_retriever(table_name):
    embeddings = get_embedding_model()
    return PGVector(
        connection_string="postgresql+psycopg2://postgres:postgres@localhost:5433/chunking_db",
        embedding_function=embeddings,
        collection_name=table_name,
        distance_strategy="COSINE"
    ).as_retriever(search_kwargs={"k": 5})

def get_llm():
    return HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        model_kwargs={"temperature": 0.7, "max_length": 300},
        device=-1  # CPU
    )

# --- Main logic ---
def main():
    llm = get_llm()
    eval_set = load_eval_set(EVAL_FILE)

    table_name = "fixed_chunk_embeddings"  # Or switch to "recursive_chunk_embeddings" etc.
    retriever = get_retriever(table_name)

    for example in eval_set:
        query = example["question"]
        print(f"\n=== Query: {query} ===")

        # --- HyDE ---
        hyde_docs = hyde_retrieval(query, retriever, llm)
        print("\n[HyDE]")
        for doc in hyde_docs:
            print("-", doc.page_content.strip()[:200])

        # --- Multi-query ---
        multi_docs = multi_query_retrieval(query, retriever, llm)
        print("\n[Multi-query Expansion]")
        for doc in multi_docs:
            print("-", doc.page_content.strip()[:200])

        # --- Step-back ---
        step_back_q = step_back_query(query, llm)
        step_docs = retriever.get_relevant_documents(step_back_q)
        print("\n[Step-back]")
        print("Generalized:", step_back_q)
        for doc in step_docs:
            print("-", doc.page_content.strip()[:200])

        # --- Query decomposition ---
        subquestions = decompose_query(query, llm)
        print("\n[Query Decomposition]")
        print("Sub-questions:", subquestions)
        for sub in subquestions:
            sub_docs = retriever.get_relevant_documents(sub)
            for doc in sub_docs:
                print("-", doc.page_content.strip()[:200])

if __name__ == "__main__":
    main()

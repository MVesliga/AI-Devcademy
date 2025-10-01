import json
import re

import psycopg2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from sympy import false

from db.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
import logging

from db.retreiver import CustomSQLRetriever

from retreival_strategies.multi_query import multi_query_strategy
from retreival_strategies.query_decomposition import query_decomposition_strategy
from retreival_strategies.hyde import hyde_strategy
from retreival_strategies.step_back import step_back_strategy

from compression import compress_document

logging.basicConfig()
#logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)
#logging.getLogger("langchain").setLevel(logging.DEBUG)
#logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

db_name = "vector_db"
connection_string = f"host={DB_HOST} dbname={db_name} user={DB_USER} password={DB_PASSWORD} port={DB_PORT}"

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
llm = Ollama(model="mistral:instruct")

conn = psycopg2.connect(connection_string)
retriever = CustomSQLRetriever(
    connection=conn,
    embedding_model=embedding_model,
    table_name="embeddings_fixed", #-- rename to test on other chunking strategies --
    embedding_column="embedding",
    text_column="chunk_text",
    k=0
)

# --- Router prompt ---
router_prompt = PromptTemplate.from_template("""
    You are a router that decides how to answer a query.
    
    Options:
    1. "semantic" → if the query is about content meaning, explanation, or general knowledge.
    2. "sql" → if the query is about structured information such as author, date, topic, or metadata.
    
    Question: {question}
    
    Answer ONLY "semantic" or "sql".
""")
router_chain = LLMChain(llm=llm, prompt=router_prompt)

def route_query(question: str) -> str:
    decision = router_chain.run({"question": question}).strip().lower()
    return "sql" if "sql" in decision else "semantic"

# --- Existing text2sql runner ---
def run_text_to_sql(nl_query: str, conn, llm):
    sql_prompt = PromptTemplate.from_template(
        """
        You are an expert Postgres SQL generator.
        The table is `embeddings_fixed` and it has the following relevant columns:

        - doc_id (INT) → unique document ID
        - author (TEXT) → document author
        - chunk_text (TEXT) → document content
        - embedding (VECTOR) → semantic embedding

        Write a Postgres SQL query (no explanation, just SQL) 
        to answer the natural language question below.
        Question: {question}
        """
    )

    chain = LLMChain(llm=llm, prompt=sql_prompt)
    sql_query = chain.run({"question": nl_query}).strip()

    print("\n[Generated SQL Query]\n", sql_query)

    try:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print("SQL Execution Error:", e)
        return None

# --- Few-shot structured output chain ---
structured_prompt = PromptTemplate.from_template("""
You are tasked with producing structured JSON output. 
Follow the style of the few examples below: 
- concise, precise answers
- short reasoning (1-2 sentences)
- minimal summarized context

### Example 1
Question: "Who wrote the Linux kernel?"
Context: "Linus Torvalds created the Linux kernel in 1991."
Output:
{{
  "question": "Who wrote the Linux kernel?",
  "summarized_context": "Linus Torvalds created Linux in 1991.",
  "answer": "Linus Torvalds",
  "reasoning": "The context directly states he created the Linux kernel."
}}

### Example 2
Question: "What is Python used for?"
Context: "Python is a programming language used for web development, AI, and data analysis."
Output:
{{
  "question": "What is Python used for?",
  "summarized_context": "Python is used for web, AI, and data analysis.",
  "answer": "Web development, AI, and data analysis",
  "reasoning": "Summarized from context without extra explanation."
}}

---

Now generate the JSON output for the given query and context, following the same short style.

Question: {question}
Context: {context}

Output:
""")

structured_prompt_list_form = PromptTemplate.from_template("""
You are tasked with producing structured JSON output. 
Follow the style of the examples below:
- concise, precise answers
- summarized_context MUST be a list of short bullet points (not one long string)
- short reasoning (1-2 sentences)
- IMPORTANT: Only answer if the context provides sufficient information.
- If there is not enough information to answer the question, set:
  "answer": "Insufficient information"
  and briefly explain why in "reasoning".

### Example 1
Question: "Who wrote the Linux kernel?"
Context: "Linus Torvalds created the Linux kernel in 1991."
Output:
{{
  "question": "Who wrote the Linux kernel?",
  "summarized_context": ["Linus Torvalds created Linux in 1991."],
  "answer": "Linus Torvalds",
  "reasoning": "The context explicitly states he created the Linux kernel."
}}

### Example 2
Question: "Who is the current president of Mars?"
Context: "Mars is a planet in the solar system."
Output:
{{
  "question": "Who is the current president of Mars?",
  "summarized_context": ["Mars is a planet in the solar system."],
  "answer": "Insufficient information",
  "reasoning": "The context does not contain any information about Mars’ leadership."
}}

---

Now generate the JSON output for the given query and context, following the same short style. 
Remember: summarized_context must be a list of strings and answer only if sufficient information exists.

Question: {question}
Context: {context}

Output:
""")

structured_chain = LLMChain(llm=llm, prompt=structured_prompt_list_form)

# Self-reflection / evaluation chain
reflection_prompt = PromptTemplate.from_template("""
You are asked to evaluate the provided answer concisely. DO NOT reveal internal chain-of-thought.
Given:
- Question: {question}
- Context: {context}
- Answer: {answer}

Provide a very short evaluation (1-2 sentences) describing:
1) any key assumptions made,
2) limitations or missing context that might affect the answer,
3) a confidence label: ("low", "medium", "high").

Return ONLY the evaluation text (no JSON).
""")
reflection_chain = LLMChain(llm=llm, prompt=reflection_prompt)

# --- MAIN EXECUTION ---
#query = "How do I edit and rebuild a .deb package?"
#query = "What are the documents with the author Scott?"
query = "Da li ti znas da sam ja opasan po toj?!?"

route = route_query(query)
print(f"\n[Router Decision] → {route.upper()}")

summarized_context = ""

if route == "semantic":
    isHyde = False
    docs = []
    if not isHyde:
        strategies = [
            multi_query_strategy
            # query_decomposition_strategy
            # step_back_strategy
        ]

        queries_to_search = []
        for strategy in strategies:
            queries_to_search.extend(strategy(llm, query))

        queries_to_search = list(set(queries_to_search))
        print("Queries to search after retrieval strategy prompt: ", queries_to_search)

        for q in queries_to_search:
            results = retriever.get_relevant_documents(q)
            docs.extend(results)
    else:
        print("\nRunning HyDE strategy...")
        hyde_queries = hyde_strategy(llm, query)
        for hypo in hyde_queries:
            results = retriever.get_relevant_documents(hypo)
            docs.extend(results)

    unique_docs = {doc.page_content: doc for doc in docs}.values()

    # Selection + compression
    N = 3
    selected_docs = list(unique_docs)[:N]
    compressed_docs = [compress_document(doc.page_content, query, llm) for doc in selected_docs]
    summarized_context = "\n".join(compressed_docs) if compressed_docs else "No context found."

    if not compressed_docs:
        print("No documents found for any rewritten query.")
    else:
        print("\nCompressed Retrieval Responses:\n")
        for i, snippet in enumerate(compressed_docs, 1):
            print(f"Snippet {i}:\n{snippet}\n")

elif route == "sql":
    sql_results = run_text_to_sql(query, conn, llm)
    summarized_context = str(sql_results) if sql_results else "No SQL results."

# --- Structured output ---
structured_json = structured_chain.run({"question": query, "context": summarized_context})

try:
    parsed = json.loads(structured_json)
except Exception:
    print("\n[Raw Structured Output]\n", structured_json)
else:
    print("\n[Final Structured JSON Output]\n", json.dumps(parsed, indent=2))
    answer_text = parsed.get("answer", "")
    reflection = reflection_chain.run({"question": query, "context": summarized_context, "answer": answer_text})
    parsed["reasoning"] = reflection.strip()

    print("\n[Final Structured JSON Output with reasoning]")
    print(json.dumps(parsed, indent=2))




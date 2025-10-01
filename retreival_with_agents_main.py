import json
import re
import psycopg2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama

from db.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
import logging

from db.retreiver import CustomSQLRetriever
from retreival_strategies.multi_query import multi_query_strategy
from retreival_strategies.query_decomposition import query_decomposition_strategy
from retreival_strategies.hyde import hyde_strategy
from retreival_strategies.step_back import step_back_strategy
from compression import compress_document

from langchain.agents import Tool, initialize_agent, AgentType

logging.basicConfig()

db_name = "vector_db"
connection_string = f"host={DB_HOST} dbname={db_name} user={DB_USER} password={DB_PASSWORD} port={DB_PORT}"

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
llm = Ollama(model="mistral:instruct")

conn = psycopg2.connect(connection_string)
retriever = CustomSQLRetriever(
    connection=conn,
    embedding_model=embedding_model,
    table_name="embeddings_fixed",
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
        - doc_id (INT)
        - author (TEXT)
        - chunk_text (TEXT)
        - embedding (VECTOR)
        Write a Postgres SQL query (no explanation, just SQL).
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

# --- Structured JSON output (list form + insufficient info guard) ---
structured_prompt_list_form = PromptTemplate.from_template("""
You are tasked with producing structured JSON output. 
- concise, precise answers
- summarized_context MUST be a list of short bullet points
- short reasoning (1-2 sentences)
- ONLY answer if the context provides sufficient info. 
If insufficient, set:
  "answer": "Insufficient information"
  and explain why in "reasoning".
Question: {question}
Context: {context}
Output:
""")
structured_chain = LLMChain(llm=llm, prompt=structured_prompt_list_form)

# Self-reflection / evaluation chain
reflection_prompt = PromptTemplate.from_template("""
You are asked to evaluate the provided answer concisely. 
Given:
- Question: {question}
- Context: {context}
- Answer: {answer}
Provide a short evaluation (1-2 sentences) about assumptions, limitations, and confidence ("low", "medium", "high").
Return ONLY the evaluation text.
""")
reflection_chain = LLMChain(llm=llm, prompt=reflection_prompt)


# -------------------------------------------------------------------
# Export Tools for ReAct Agent
# -------------------------------------------------------------------
def router_tool_fn(query: str) -> str:
    return route_query(query)

router_tool = Tool(
    name="RouterTool",
    func=router_tool_fn,
    description="Classifies query as 'semantic' or 'sql'. Input: user query."
)

def retriever_tool_fn(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "[]"
    unique_docs = list({doc.page_content: doc for doc in docs}.values())[:3]
    snippets = [compress_document(doc.page_content, query, llm) for doc in unique_docs]
    return json.dumps(snippets)

retriever_tool = Tool(
    name="RetrieverTool",
    func=retriever_tool_fn,
    description="Retrieve semantic context for a query. Returns JSON list of snippets."
)

def text2sql_tool_fn(nl_query: str) -> str:
    rows = run_text_to_sql(nl_query, conn, llm)
    return json.dumps(rows) if rows else "No results."

text2sql_tool = Tool(
    name="Text2SQL",
    func=text2sql_tool_fn,
    description="Run natural language → SQL → database results."
)

def structured_tool_fn(arg_str: str) -> str:
    # Expect "QUESTION|||CONTEXT"
    parts = arg_str.split("|||")
    question = parts[0].strip()
    context = parts[1].strip() if len(parts) > 1 else ""
    return structured_chain.run({"question": question, "context": context})

structured_tool = Tool(
    name="StructuredFormatter",
    func=structured_tool_fn,
    description="Format final JSON answer. Input format: 'QUESTION|||CONTEXT'."
)

tools = [router_tool, retriever_tool, text2sql_tool, structured_tool]

# -------------------------------------------------------------------
# ReAct Agent + Judge
# -------------------------------------------------------------------
from langchain_core.prompts import PromptTemplate

judge_prompt = PromptTemplate.from_template("""
You are a judge evaluating an agent's answer.
Criteria:
1. Grounding: answer must be supported by context.
2. Schema: must be valid JSON with keys question, summarized_context (list), answer, reasoning.
3. Precision: answer should be concise.
Question: {question}
Context: {context}
Answer: {answer}
Output JSON:
{{
  "valid": true/false,
  "schema_ok": true/false,
  "precision_ok": true/false,
  "feedback": "short feedback"
}}
""")
judge_chain = LLMChain(llm=llm, prompt=judge_prompt)

def run_agent_with_judge(user_query: str, max_rounds: int = 2):
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    for attempt in range(max_rounds):
        agent_output = agent.run(user_query)
        print("\n[Agent Output]\n", agent_output)
        try:
            parsed = json.loads(re.search(r"\{.*\}", agent_output, re.DOTALL).group(0))
        except:
            parsed = {"question": user_query, "summarized_context": [], "answer": agent_output, "reasoning": "Unstructured"}
        # Judge evaluation
        context = "\n".join(parsed.get("summarized_context", []))
        judge_eval = judge_chain.run({"question": user_query, "context": context, "answer": json.dumps(parsed)})
        print("\n[Judge Evaluation]\n", judge_eval)
        try:
            judge_parsed = json.loads(re.search(r"\{.*\}", judge_eval, re.DOTALL).group(0))
        except:
            judge_parsed = {"valid": False, "schema_ok": False, "precision_ok": False, "feedback": judge_eval}
        if judge_parsed.get("valid") and judge_parsed.get("schema_ok") and judge_parsed.get("precision_ok"):
            print("\nAccepted Final Answer\n")
            return parsed
        else:
            print("\nRefining based on Judge Feedback\n")
            user_query = f"{user_query}\nJudge Feedback: {judge_parsed.get('feedback','')}"
    return parsed

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------
if __name__ == "__main__":
    query = "What are the documents with the author Scott?"
    route = route_query(query)
    print(f"\n[Router Decision] → {route.upper()}")
    if route == "semantic":
        docs = retriever.get_relevant_documents(query)
        unique_docs = list({doc.page_content: doc for doc in docs}.values())[:3]
        compressed_docs = [compress_document(doc.page_content, query, llm) for doc in unique_docs]
        summarized_context = "\n".join(compressed_docs) if compressed_docs else "No context found."
    else:
        sql_results = run_text_to_sql(query, conn, llm)
        summarized_context = str(sql_results) if sql_results else "No SQL results."
    structured_json = structured_chain.run({"question": query, "context": summarized_context})
    print("\n[Structured JSON from pipeline]\n", structured_json)
    print("\n[ReAct Agent + Judge Execution]\n")
    final = run_agent_with_judge(query, max_rounds=2)
    print(json.dumps(final, indent=2))

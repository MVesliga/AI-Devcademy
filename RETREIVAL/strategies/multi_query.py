def generate_query_variants(query, llm):
    prompt = f"Generate 3 diverse ways to rephrase the query: '{query}'"
    response = llm.invoke(prompt)
    return response.split("\n")

def multi_query_retrieval(query, retriever, llm):
    variants = generate_query_variants(query, llm)
    all_docs = []
    for variant in variants:
        docs = retriever.get_relevant_documents(variant)
        all_docs.extend(docs)
    # Optionally deduplicate
    return list({doc.page_content: doc for doc in all_docs}.values())
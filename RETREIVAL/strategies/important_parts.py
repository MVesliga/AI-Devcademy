def extract_relevant_spans(docs, query, llm):
    results = []
    for doc in docs:
        prompt = f"Extract the part of the following document most relevant to: '{query}'\n\n{doc.page_content}"
        span = llm.invoke(prompt)
        results.append(span.strip())
    return results
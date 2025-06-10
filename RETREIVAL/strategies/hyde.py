def hyde_generate_doc(query, llm):
    prompt = f"Write a detailed, hypothetical answer to: {query}"
    return llm.invoke(prompt)

def hyde_retrieval(query, retriever, llm):
    hypothetical_answer = hyde_generate_doc(query, llm)
    return retriever.get_relevant_documents(hypothetical_answer)
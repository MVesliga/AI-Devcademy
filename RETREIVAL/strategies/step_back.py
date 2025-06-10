def step_back_query(query, llm):
    prompt = f"Rephrase the question to be more general: {query}"
    return llm.invoke(prompt)
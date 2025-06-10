def decompose_query(query, llm):
    prompt = f"Break down this complex question into simpler sub-questions: {query}"
    subquestions = llm.invoke(prompt).split("\n")
    return [q.strip() for q in subquestions if q.strip()]
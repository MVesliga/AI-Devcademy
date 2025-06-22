def route_query(query, llm):
    routing_prompt = f"""
Decide how to handle the following query:
"{query}"

Return one word: "metadata" or "vector".
"""
    response = llm.predict(routing_prompt).strip().lower()
    return "metadata" if "metadata" in response else "vector"
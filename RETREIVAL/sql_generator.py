def generate_sql_query(query, table_name, llm):
    prompt = f"""You are a SQL assistant. Given the user question: '{query}', generate a SQL query that selects the most relevant rows from the table '{table_name}' using metadata in a JSONB column called 'cmetadata'.
Schema: {table_name}(doc_id, author, text_content, embedding, chunk_index, chunking_method, cmetadata JSONB)
"""
    return llm.invoke(prompt)
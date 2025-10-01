from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

'''def hyde_strategy(llm, question: str, embedding_model) -> list[tuple[str, list[float]]]:
    prompt = PromptTemplate.from_template(
        """Write a detailed answer to the following question as if you were an expert.
        Question: {question}
        """
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    hypothetical_answer = llm_chain.run({"question": question})

    query_embedding = embedding_model.embed_query(hypothetical_answer)
    return [(hypothetical_answer, query_embedding)]'''

def hyde_strategy(llm, query):
    """
    Generate hypothetical answer(s) to the query using the LLM.
    Returns: list of hypothetical answer strings.
    """
    hyde_prompt = PromptTemplate.from_template(
        """You are an expert at answering questions.
        Provide a concise and factual hypothetical answer to the following question,
        even if you need to make reasonable assumptions:

        Question: {question}
        """
    )

    chain = LLMChain(llm=llm, prompt=hyde_prompt)

    # Always ensure we get a string
    result = chain.run({"question": query})
    if isinstance(result, tuple):
        result = " ".join(str(r) for r in result)
    elif not isinstance(result, str):
        result = str(result)

    clean_result = result.strip()
    clean_result = re.sub(r"\s+", " ", clean_result)  # normalize spaces

    return [clean_result] if clean_result else []

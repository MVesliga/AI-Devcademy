import re
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

def multi_query_strategy(llm, question: str) -> list[str]:
    prompt = PromptTemplate.from_template(
        """Rephrase the question below into exactly three alternative questions without any numbering or bullet points.
        Question: {question}
        """
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    llm_response = llm_chain.run({"question": question})

    queries = []
    for line in llm_response.splitlines():
        clean_line = re.sub(r"^\s*\d+[\.\)]?\s*", "", line.strip())
        if clean_line:
            queries.append(clean_line)
    return [question] + queries
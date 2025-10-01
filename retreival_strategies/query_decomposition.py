from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

def query_decomposition_strategy(llm, question: str) -> list[str]:
    prompt = PromptTemplate.from_template(
        """Decompose the following question into smaller, specific sub-questions that could help answer it.
        Return each sub-question on a new line without numbering or bullet points.
        Question: {question}
        """
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    response = llm_chain.run({"question": question})
    return [question] + [line.strip() for line in response.splitlines() if line.strip()]

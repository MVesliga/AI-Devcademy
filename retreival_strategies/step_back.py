from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

def step_back_strategy(llm, question: str) -> list[str]:
    prompt = PromptTemplate.from_template(
        """Rewrite the following question to be more general, so that broader context might be retrieved. The answer you return should just contain a single rewritten question.
        Question: {question}
        """
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    general_question = llm_chain.run({"question": question})
    return [question, general_question.strip()]

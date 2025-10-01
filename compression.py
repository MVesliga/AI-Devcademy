# --- Compression: Extract only relevant snippets ---
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate


def compress_document(doc_text: str, query: str, llm) -> str:
    compression_prompt = PromptTemplate.from_template(
        """
        Extract only the most relevant sentences from the text below that directly help answer the question.
        
        Question: {question}
        Text: {text}
        
        Return only the trimmed relevant sentences, no extra commentary.
        """
    )
    chain = LLMChain(llm=llm, prompt=compression_prompt)
    compressed = chain.run({"question": query, "text": doc_text})
    return compressed.strip()
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

custom_prompt = PromptTemplate.from_template(
    "Rewrite the following question in three different ways:\n\nQuestion: {query}"
)

query_rewrite_chain = LLMChain(llm=llm, prompt=custom_prompt)

retriever = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=query_rewrite_chain
)

pipe = pipeline("text2text-generation", model="google/flan-t5-base")

prompt = "Rewrite the question in 3 different ways: How can I rebuild a Debian package?"
output = pipe(prompt, max_new_tokens=100)
print(output[0]['generated_text'])
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

DB_NAME = "vector_db_new"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"
CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "rga_embeddings"
MODEL_NAME = "BAAI/bge-m3"


embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'}
)

vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
    content_column_name="text_content",
    embedding_column_name="embedding"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


hf_llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation", # Or "text-generation" depending on the model
    model_kwargs={"temperature": 0.7, "max_length": 300},
    device=-1 # -1 for CPU, 0 for first GPU, etc.
)


qa_chain = RetrievalQA.from_chain_type(
    llm=hf_llm,  # Use the Hugging Face generative model here
    retriever=retriever,
    return_source_documents=True
)


query = "how do you achieve a numeric versioning scheme with git?"
result = qa_chain(query)

print("Answer:", result["result"])
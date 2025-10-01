from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model_name = "BAAI/bge-m3"

model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

def get_local_embedding_model() :
    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
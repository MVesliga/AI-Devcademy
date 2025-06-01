from langchain.text_splitter import CharacterTextSplitter

def get_fixed_chunks(text, chunk_size=500, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
from langchain.text_splitter import CharacterTextSplitter

def get_fixed_chunks(text, chunk_size=200, chunk_overlap=50):
    splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
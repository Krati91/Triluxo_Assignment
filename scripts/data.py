import faiss
import numpy as np

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


db_name = 'technical-courses'
base_url = 'http://localhost:11434'
embed_model = 'nomic-embed-text'

def create_context() -> str:
    '''
    To fetch data from the url specified
    '''
    url = 'https://brainlox.com/courses/category/technical'

    loader = WebBaseLoader(web_path=url)

    docs = []
    for doc in loader.lazy_load():
        docs.append(doc)

    context = '\n\n'.join([x.page_content for x in docs])

    return context



def create_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    return chunks


def store_text_in_vector_store(chunks: list):
    embeddings = OllamaEmbeddings(model=embed_model,
                                  base_url=base_url)
    chunk_embeddings = embeddings.embed_documents(chunks)
    dim = len(chunk_embeddings[0]) 
    
    index = faiss.IndexFlatL2(dim)
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    vector_store.add_texts(texts=chunks)
    vector_store.save_local(db_name)


def retrieve_context_from_vector_store(user_input):
    embeddings = OllamaEmbeddings(model=embed_model,
                                  base_url=base_url)
    vector_store = FAISS.load_local(db_name, embeddings,
                                    allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_type='similarity')
    
    docs = retriever.invoke(user_input)
    return '\n\n'.join([x.page_content for x in docs])

    

if __name__ == '__main__':
    text = create_context()
    chunks = create_chunks(text)
    store_text_in_vector_store(chunks)
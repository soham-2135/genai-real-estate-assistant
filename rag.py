from uuid import uuid4

from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import shutil

load_dotenv()
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "Alibaba-NLP/gte-base-en-v1.5"
COLLECTION_NAME = "real_estate"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

llm = None
vector_store = None

def initilaize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature = 0.9, max_tokens=500)

    ef = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs ={'trust_remote_code' : True}
    )

    if vector_store is None:
        vector_store = Chroma(
            collection_name = COLLECTION_NAME,
            embedding_function = ef,
            persist_directory = str(VECTORSTORE_DIR)
        )


def process_urls(urls):
    '''
    This function scraps data from a url and stores it in a vectordb
    :param urls:
    :return:
    '''

    yield "Initializing components"
    initilaize_components()

    vector_store.reset_collection()

    yield "Loading Data"
    loader = UnstructuredURLLoader(
        urls= urls,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; RealEstateRAG/1.0; +https://example.com)"
        }
    )
    data = loader.load()

    if not data: ### took help from gpt(basically if url is incorrect or false then following will be throwned) ###
        raise RuntimeError(
            "No content could be loaded from the provided URLs."
        )

    yield "Splitting text"
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", "." ," "],
        chunk_size = CHUNK_SIZE
    )

    docs= text_splitter.split_documents(data)

    if not docs: ### took help from gpt(basically if url is incorrect or false then following will be throwned ) ###
        raise RuntimeError(
            "The provided URLs do not contain usable text."
        )

    yield "Adding Docs to vector store"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector DB is not initialised")

    docs = vector_store.similarity_search(query, k=4)

    if not docs:
        return "No relevant information found.", ""

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    sources = []

    for doc in docs:
        source = doc.metadata.get("source", "")
        if source:
            sources.append(source)
    '''
        sources = [
        doc.metadata.get("source", "")
        for doc in docs
        if doc.metadata.get("source")
    ]
    shortcut for above one 
    '''


    return response.content, ", ".join(set(sources))


if __name__ == "__main__":
    urls = ["https://www.investopedia.com/articles/mortgages-real-estate.asp",
            "https://www.hud.gov/topics/buying_a_home"]


    process_urls(urls)
    answer, sources = generate_answer("")
    print(f"Answer:{answer}")
    print(f"sources: {sources}")
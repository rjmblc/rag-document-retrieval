import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# load env variables to env
load_dotenv()

# configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# load default embedding model
embedding = HuggingFaceEmbeddings()

# intialize gemini llm
gemini_llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.0
)

# intialize openai llm
openai_llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
        temperature= 0,        
    )

# load vector store
vector_store = Chroma(
    collection_name = COLLECTION_NAME,
    embedding_function= embedding,
    persist_directory = VECTOR_DB_PATH 
)

# create retriever
retriever  = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm = gemini_llm,
    chain_type = "stuff",
    retriever = retriever,
    return_source_documents = True
)

query = "who is the chairman, advisory group for textbooks in science and mathematics?" 
response  = qa_chain.invoke({"query": query})
print(response)
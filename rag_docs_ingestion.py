# import dependenices
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import nltk
import os
from dotenv import load_dotenv


nltk.download("punkt_tab")

load_dotenv()

# configuration
DOCS_DIR_PATH = os.getenv("DOCS_DIR")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# loading the embedding model
embedding  = HuggingFaceEmbeddings()

# directory loader
loader = DirectoryLoader(
    path = DOCS_DIR_PATH,
    glob = "./*.pdf",
    loader_cls = PyMuPDFLoader
)

# loader = PyMuPDFLoader(
#     file_path = DOCS_DIR_PATH,
 
# )

#single file loader

# load the documents
documents = loader.load()

#print(documents[0])

# initialize text splitter
text_splitter = CharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

#splitting the text into smaller chunks
text_chunks = text_splitter.split_documents(documents)

# creating the vector store
vector_store = Chroma.from_documents(
    documents = text_chunks,
    embedding = embedding,
    persist_directory = VECTOR_DB_PATH,
    collection_name = COLLECTION_NAME
)
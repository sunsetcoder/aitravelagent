from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

# Set up environment variables
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize NVIDIA embeddings
embeddings = NVIDIAEmbeddings()

# Load a Wikipedia article
loader = WikipediaLoader(query="Visa requirements for Indian citizens", load_max_docs=10)
documents = loader.load()
print(documents)

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Initialize the OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Create a Chroma vector store and add the document chunks
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="wikipedia"
)
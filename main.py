import os 
from getpass import getpass
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv, dotenv_values 

load_dotenv()


llm_model = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

store = Chroma(embedding_function=embeddings)

# Indexing of the data
import bs4 
from langchain_community.document_loaders import WebBaseLoader

strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/", 
    bs_kwargs={"parse_only":strainer}
)
docs = loader.load()

# Text splitting to create chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200, 
    add_start_index=True,
)

all_splits = splitter.split_documents(docs)
print(f"splitted chunks are a total of {len(all_splits)}")

# Storing these chunks in our vector database
doc_ids = store.add_documents(documents=all_splits)

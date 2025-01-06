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
# Each chunk will have a seperate doc id in the vector database
doc_ids = store.add_documents(documents=all_splits)

# Retrieval and generation
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")
exm = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

initial_prompt_message = exm[0].content

# Defining the state of the application or the type of data our app plays with
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Our main two functions that get it done

def generate(state):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question":state["question"], "context":docs_content})
    response = llm_model.invoke(messages)
    return {"answer":response.content}

def retrieve(state):
    retrieved_docs = store.similarity_search(state["question"])
    return {"context":retrieved_docs}

# Compile our pipeline into a graph
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Use the RAG features from the graph
result = graph.invoke({"question": "What is Chain of Hindsight ?"})

print(f'Context: {result["context"]}\n\n')
print(f'Answer: {result["answer"]}')
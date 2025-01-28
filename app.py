import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import custom_embedding as ce
import chromadb
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, dotenv_values 
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile 
import uuid
import sqlite3

load_dotenv()

# Setting initial variables needed
llm_model = ChatOpenAI(model="gpt-4o-mini")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = ce.CustomOpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

template = """
Context: {context}
Question: {question}

Please provide a detailed and thorough response. Ensure the answer is no shorter than 200 words.
"""
prompt = ChatPromptTemplate.from_template(template)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100, 
    add_start_index=True,
    separators=["\n\n", "\n", ".", "?", "!", " ", ""],
)

def process_docs(uploaded_file):
    if not uploaded_file: return None
    # proceed if file upload successful
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)
    chunks = splitter.split_documents(docs)
    return chunks

def get_chroma_client():
    # Persist this in the local directory
    client = chromadb.PersistentClient(path="./demo-rag")
    return client.get_or_create_collection(
        name="rag-app", 
        embedding_function=embeddings,
        metadata={"hnsw:space": "cosine"}
    )

def add_embeddings(chunks):
    collection_client = get_chroma_client()
    docs, metadata, ids = [],[],[]
    filename = uuid.uuid4().hex[:5]
    for idx, split in enumerate(chunks):
        docs.append(split.page_content)
        metadata.append(split.metadata)
        ids.append(f"{filename}_{idx}")
    
    collection_client.upsert(
        documents=docs, 
        metadatas=metadata,
        ids=ids
    )
    st.success("data is embedded")

def get_context_from_stores(prompt, n):
    collection = get_chroma_client()
    result = collection.query(query_texts=[prompt],n_results=n)
    return result 

def ask_for_inference(context,prompt):
    context = "\n\n".join([str(cnt) for cnt in context])
    messages = prompt.invoke({"question":prompt, "context":context})
    response = llm_model.invoke(messages)
    return response.content

st.set_page_config(page_title="Simple PDF Summarizer", page_icon="👿")
st.title("Get simple summaries for your added documents")

if __name__ == "__main__":

    user_prompt = st.text_area("Shoot your question...")
    ask = st.button("Ask")
    file_chunks = None


    with st.sidebar:
        st.title("Add your documents below")
        uploaded_file = st.file_uploader("Choose a simple file",type="pdf")
        if uploaded_file:
            if st.button("Process"):
                file_chunks = process_docs(uploaded_file=uploaded_file)
                st.session_state.chunked = True
                if file_chunks:
                    # add them to vectors
                    add_embeddings(file_chunks)
                    st.session_state.embedded = True
    
    if user_prompt and ask:
       result = get_context_from_stores(user_prompt,10)
       context = result.get("documents")
       response = ask_for_inference(context=context, prompt=prompt)
       if response:
           st.write(response)

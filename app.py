import uuid
import base64
from base64 import b64decode
import streamlit as st

# Import global modules and custom embedding if needed
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import custom_embedding as ce  # Your custom embedding module
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader  # if needed
import uuid
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv


load_dotenv()
import os

# Global LLM and embedding initialization
llm_model = ChatOpenAI(model="gpt-4o-mini")
# Use your custom embedding implementation (or replace with OpenAIEmbeddings)
embeddings = ce.CustomOpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# Initialize Chroma vector store with persistence (global)
vectorstore = Chroma(
    collection_name="multimodal-rag",
    embedding_function=embeddings,
    persist_directory="./demo-rag"
)
docstore = InMemoryStore()
doc_id = "doc_id"

# Initialize the multi-vector retriever (global)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=doc_id,
)

# === Summarization Chains using ChatOpenAI ===
# For text and table summarization
prompt_text = """
You are an assistant tasked with summarizing the following content.
Provide a concise summary.

Content: {element}
"""
prompt_summarize = ChatPromptTemplate.from_template(prompt_text)
# This chain will automatically pass the element through the prompt and LLM
summarize_chain = {"element": lambda x: x} | prompt_summarize | llm_model | StrOutputParser()

# For image summarization
prompt_img_text = """Describe the image in detail. For context,
the image is part of a research paper explaining the transformers architecture.
Be specific about graphs, such as bar plots."""
# Here we build a prompt that expects a base64 image string replacement
messages = [
    (
        "user",
        [
            {"type": "text", "text": prompt_img_text},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
        ],
    )
]
prompt_img = ChatPromptTemplate.from_messages(messages)
image_chain = prompt_img | llm_model| StrOutputParser()

# === Utility Functions ===
def display_base64_image(b64_str):
    """Display a base64-encoded image using Streamlit."""
    image_data = base64.b64decode(b64_str)
    st.image(image_data, use_column_width=True)

def parse_docs(docs):
    """Separate base64-encoded images from text documents."""
    images_b64 = []
    texts_list = []
    for doc in docs:
        try:
            # Try to decode; if it succeeds, assume it's an image
            b64decode(doc)
            images_b64.append(doc)
        except Exception:
            texts_list.append(doc)
    return {"images": images_b64, "texts": texts_list}

def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    # Concatenate all text content from retrieved docs
    context_text = "".join([text_element.text for text_element in docs_by_type["texts"]])
    prompt_template = f"""
Answer the question based only on the following context, which may include text, tables, and images.
Context: {context_text}
Question: {user_question}
"""
    prompt_content = [{"type": "text", "text": prompt_template}]
    # Append any retrieved images
    for image in docs_by_type["images"]:
        prompt_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        )
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

# === Streamlit App Layout ===
st.set_page_config(page_title="Multi-modal RAG with LangChain", page_icon="📄")
st.title("Multi-modal RAG with LangChain")
st.markdown("Upload a PDF file to extract and summarize its content, then ask questions based on that content.")

# --- File Upload & Processing ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.info("Processing document...")
    # Save the uploaded PDF temporarily
    temp_pdf = "temp.pdf"
    with open(temp_pdf, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # --- Extract content from the PDF using unstructured ---
    from unstructured.partition.pdf import partition_pdf
    chunks = partition_pdf(
        filename=temp_pdf,
        infer_table_structure=True,
        strategy="fast",
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    
    # Separate extracted elements into texts, tables, and images.
    tables = []
    texts = []
    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)):
            texts.append(chunk)
    
    def get_images_base64(chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                for el in chunk.metadata.orig_elements:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64
    
    images = get_images_base64(chunks)
    
    # Optionally display the first extracted image
    if images:
        st.subheader("Extracted Image Sample:")
        display_base64_image(images[0])
    
    # --- Summarize the Content ---
    st.info("Summarizing document content...")
    # Summarize texts
    with st.spinner("Summarizing text content..."):
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})
    # Summarize tables (using table HTML if available)
    with st.spinner("Summarizing table content..."):
        tables_html = [table.metadata.text_as_html for table in tables]
        table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})
    # Summarize images
    with st.spinner("Summarizing image content..."):
        image_summaries = image_chain.batch(images)
    
    st.success("Summarization complete!")
    
    # --- Load Summaries into the Vector Store ---
    st.info("Loading summaries into vector store...")
    # Load text summaries
    text_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={doc_id: text_ids[i]})
        for i, summary in enumerate(text_summaries)
    ]
    if summary_texts:
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(text_ids, texts)))
    
    # Load table summaries
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={doc_id: table_ids[i]})
        for i, summary in enumerate(table_summaries)
    ]
    if summary_tables:
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, tables)))
    
    # Load image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_images = [
        Document(page_content=summary, metadata={doc_id: img_ids[i]})
        for i, summary in enumerate(image_summaries)
    ]
    if summary_images:
        retriever.vectorstore.add_documents(summary_images)
        retriever.docstore.mset(list(zip(img_ids, images)))
    
    st.success("Document vectors added successfully!")
    st.write("Vectorstore internal state:", vectorstore._collection.get())
    
    # --- RAG Pipeline & User Query ---
    st.info("You can now ask questions about the document.")
    user_question = st.text_input("Enter your question:")
    if st.button("Get Answer") and user_question:
        with st.spinner("Retrieving answer..."):
            # Build a simple chain:
            chain = (
                {
                    "context": retriever | RunnableLambda(parse_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(build_prompt)
                | ChatOpenAI(model="gpt-4o-mini")
                | StrOutputParser()
            )
            response = chain.invoke(user_question)
        st.subheader("Answer:")
        st.write(response)

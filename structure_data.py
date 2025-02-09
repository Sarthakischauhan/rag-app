#! base/bin/python3
from dotenv import load_dotenv
load_dotenv()
import tempfile 

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import *
# Summarizing the text and tables 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


output_path = "./content/"
file_path = "./content/sample.pdf"

 
def process_docs(file):
    # Save temp file
    if not file: return ([],[],[])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue()) 
        temp_filename = temp_file.name  
     
    elements = partition_pdf(
        filename=temp_filename,
        infer_table_structure=True,           
        strategy="fast",                     

        extract_image_block_types=["Image"],   
        # image_output_dir_path=output_path,   

        extract_image_block_to_payload=True,   

        chunking_strategy="by_title",         
        max_characters=10000,                
        combine_text_under_n_chars=2000,       
        new_after_n_chars=6000
    )

    assert elements, "No elements could be found, check again"
    # separate tables from texts
    tables = []
    texts = []

    for chunk in elements:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            text_content = " ".join([str(el) for el in chunk.metadata.orig_elements if "Text" in str(type(el))])
            texts.append(text_content)

    # Get the images from the CompositeElement objects
    def get_images_base64(chunks):
        images_b64 = []
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images_b64.append(el.metadata.image_base64)
        return images_b64

    images = get_images_base64(elements)
    return texts, tables, images



def summarize_chunks(text_chunks, tables_chunks, images_chunks):
    text_summaries, table_summaries, image_summaries = [],[],[]
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatOpenAI(model="gpt-4o-mini")

    # chaining tasks
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # Summarize texts
    text_summaries = summarize_chain.batch(text_chunks, {"max_concurrency": 3})

    # Summarize tables
    tables_html = [table.metadata.text_as_html for table in tables_chunks]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    if images_chunks:
        image_prompt_text = """
        You are an assistant tasked with summarizing images 
        Give a concise summary the following image that is a part of a pdf which can be the following:

        1. A research paper about math, science or technology
        2. Logos of companies

        Respond only with the summary, no additionnal comment.
        Do not start your message by saying "Here is a summary" or anything like that.
        Just give the summary as it is.
        """

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": image_prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)

        chain = prompt | model | StrOutputParser()

        image_summaries  = chain.batch(images_chunks)

    return (text_summaries, table_summaries, image_summaries)


def attach_Ids(text_summaries, table_summaries, image_summaries, texts, tables, images, doc_id="doc_id"):
    from langchain.schema.document import Document
    import uuid

    docs_ids = [str(uuid.uuid4()) for _ in texts] 
    # Setting docs and their ids
    summary_texts = [
        Document(page_content=summary, metadata={doc_id: docs_ids[i]}) for i, summary in enumerate(text_summaries)
    ]

    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={doc_id: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]

    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={doc_id: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]

    return (summary_texts, summary_tables, summary_img)


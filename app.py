
from PyPDF2 import PdfReader
from langchain_community.llms import HuggingFaceTextGenInference
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import joblib
import os
import gradio as gr
import warnings
from langchain_community.llms import CTransformers

# Suppress warnings
warnings.filterwarnings('ignore')


EMBEDDINGS_FILE = "C:/Users/ishan/OneDrive/Desktop/RAG(custom_pdf)/steve-jobs.joblib"
PDF_FILE = 'C:/Users/ishan/OneDrive/Desktop/RAG(custom_pdf)/steve-jobs.pdf'
TEMPLATE = """
<s>[INST] <<SYS>>
You are a helpful AI assistant.
Answer based on the context provided. If you cannot find the correct answer, say I don't know. Be concise and just include the response.
<</SYS>>
{context}
Question: {question}
Helpful Answer: [/INST]
"""

# Initialize large language model
llm=CTransformers(model="X:/llama-7b", 
                      model_type="llama-7b", config={"temperature":0.01, "context_length":4096, "max_new_tokens":4000})

# Create a prompt template
prompt = PromptTemplate.from_template(TEMPLATE)

# Function to read and extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    raw_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

# Function to get or create embeddings
def get_or_create_embeddings(texts, embeddings_file):
    if os.path.exists(embeddings_file):
        return joblib.load(embeddings_file)
    embeddings = HuggingFaceEmbeddings()
    joblib.dump(embeddings, embeddings_file)
    return embeddings

# Main chatbot function
def mychatbot(query, history):
    raw_text = extract_text_from_pdf(PDF_FILE)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    texts = text_splitter.split_text(raw_text)

    embeddings = get_or_create_embeddings(texts, EMBEDDINGS_FILE)
    vectorstore = Chroma.from_texts(texts, embeddings)
    retriever = vectorstore.as_retriever()

    retriever.get_relevant_documents(query)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm)
    return chain.invoke(query)

# Gradio interface setup
gr.ChatInterface(
    mychatbot,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question", container=False, scale=7),
    title="My Chatbot",
    description="Custom Chatbot for your PDF",
    theme="soft",
    examples=[
        "Where did Steve Jobs deliver this speech?",
        "Where did Steve Jobs start Apple?",
        "Apart from Apple, which other companies were founded by Steve Jobs?",
        "Did Steve Jobs graduate?",
        "Who created the The Whole Earth Catalog and when?",
        "What was in The Whole Earth Catalog?",
        "What are the 3 key takeaways from Steve Jobs speech?",
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear"
).launch(share=True)

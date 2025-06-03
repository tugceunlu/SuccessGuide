import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
import chromadb
import chardet
from PyPDF2 import PdfReader

# Wrapper for ChromaDB's EmbeddingFunction interface
class ChromaEmbeddingFunction:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function

    def embed_query(self, query: str) -> List[float]:
        return self.embedding_function.embed_query(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return self.embedding_function.embed_documents(documents)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedding_function.embed_documents(input)

# Initialize LangChain-compatible embedding function
langchain_embedding = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
embedding_function = ChromaEmbeddingFunction(langchain_embedding)

# Initialize ChromaDB client and collection
client1 = chromadb.PersistentClient(path="student_roadmap_db")
collection = client1.get_or_create_collection(
    name="student_course_docs", embedding_function=embedding_function
)

# Function to handle file uploads
def add_document_to_collection(file, collection):
    content = ""
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                content += page.extract_text()
        except Exception as e:
            st.error(f"Error reading the PDF file: {e}")
            return
    elif file.name.endswith(".txt"):
        raw_data = file.read()
        detected = chardet.detect(raw_data)
        encoding = detected.get("encoding")
        if encoding is None:
            st.error("Unable to detect encoding for the TXT file.")
            return
        try:
            content = raw_data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            st.error("Error decoding the TXT file.")
            return
    else:
        st.error("Unsupported file type.")
        return

    # Split into chunks and add to collection
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)
    ids = [f"doc_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)
    st.success(f"Uploaded and processed {file.name} successfully!")

# GPT MODEL 
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.0,
    openai_api_key="") #ENTER YOUR API

# Memory for ongoing conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# RAG !

# Retriever function
def create_retriever(persistent_client_path):
    return Chroma(
        collection_name="student_course_docs",
        embedding_function=embedding_function,
        persist_directory=persistent_client_path
    ).as_retriever()

retriever = create_retriever(persistent_client_path="student_roadmap_db")

# Conversational retrieval chain creation function
def create_conversational_chain(llm, retriever):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

chain = create_conversational_chain(llm=llm, retriever=retriever)

# INTERFACE
st.title("Your Success Guide")

# Sidebar
st.sidebar.title("Upload Course Materials")
uploaded_file = st.sidebar.file_uploader("Upload your course syllabi, objectives, or other materials, and ask questions about succeeding in your courses or careers! (PDF or TXT)", type=["pdf", "txt"])
if uploaded_file is not None:
    add_document_to_collection(uploaded_file, collection)

if 'user_interest' not in st.session_state:
    st.session_state.user_interest = ''

user_interest = st.text_input('Enter your interests (e.g., programming, art, design):')
if user_interest:
    st.session_state.user_interest = user_interest

# Chat interface 
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Enter your question (e.g., How can I excel in this course? What careers can I pursue?):")
if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    
    response_context = f"User interest: {st.session_state.user_interest}. "
    
    response = chain({"question": response_context + user_question})
    
    with st.chat_message("assistant"):
        st.markdown(response['answer'])
    
    # Update chat history with new messages
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})

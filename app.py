import streamlit as st
from pypdf import PdfReader
from type import FilePDF
from typing import List
from config import EMBEDDING_MODEL_PATH, EMBEDDING_DOC_PATH
from html_template import css, bot_template, user_template

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

st.write(css, unsafe_allow_html=True)

def embed_texts(splitted_texts: List[FilePDF]):
  if len(splitted_texts):
    # texts exist
    # do embedding
    embed_paths = set()
    for text in splitted_texts:
      embedded_text = FAISS.from_texts(text.text, st.session_state.embedding_model)
      embedding_path = f"{EMBEDDING_DOC_PATH}/{text.title}"
      embed_paths.add(embedding_path)

      embedded_text.save_local(embedding_path)
      print(f"success embed into {embedding_path}")
      st.session_state.embedding_paths.append(text.title)

    for i in embed_paths:
      fais_local = FAISS.load_local(folder_path=f"{i}", embeddings=st.session_state.embedding_model, allow_dangerous_deserialization=True)
      st.session_state.vs_db.merge_from(fais_local)
      print(f"{embedding_path} success merge into FAISS Db")
    st.session_state.pdf_files_ref = []
    update_conversation_chain()

def get_embedding():
  for embedding_path in st.session_state.embedding_paths:
    # load and merge embedding_paths
    embedding_stored = FAISS.load_local(folder_path=f"{EMBEDDING_DOC_PATH}/{embedding_path}", embeddings=st.session_state.embedding_model, allow_dangerous_deserialization=True)
    st.session_state.vs_db.merge_from(embedding_stored)

def get_conversation_chain():
  return ConversationalRetrievalChain.from_llm (
    llm=ChatOpenAI(),
    retriever=st.session_state.vs_db.as_retriever(),
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
  )

def update_conversation_chain():
  st.session_state.conversation_chain = get_conversation_chain()

def split_pdf_files(pdf_files):
    for file in pdf_files:
        text = ""
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
        title_file = file.name
        splitted_text = st.session_state.text_splitter.split_text(text)

        st.write(f"{title_file} success to split")
        new_pdf = FilePDF(title=title_file, text=splitted_text)
        st.session_state.pdf_files_ref.append(new_pdf)

def handle_question(user_question):
    response = st.session_state.conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

if "counter" not in st.session_state:
  st.session_state.counter = 0

if "text_splitter" not in st.session_state:
  st.session_state.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    separators=[
      "\n\n",
      "\n",
      " ",
      ".",
      ",",
      "\u200b",  # Zero-width space
      "\uff0c",  # Fullwidth comma
      "\u3001",  # Ideographic comma
      "\uff0e",  # Fullwidth full stop
      "\u3002",  # Ideographic full stop
      "",
    ]
  )

if "pdf_files_ref" not in st.session_state:
  st.session_state.pdf_files_ref = [] 

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs={'trust_remote_code': True})

if "embedding_paths" not in st.session_state:
  st.session_state.embedding_paths = []

if "vs_db" not in st.session_state:
  st.session_state.vs_db = FAISS.load_local(folder_path=f"{EMBEDDING_DOC_PATH}/7.-Panduan-PKM-KI-2024.pdf", embeddings=st.session_state.embedding_model, allow_dangerous_deserialization=True) # as faiss init

if "conversation_chain" not in st.session_state:
  update_conversation_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

st.session_state.counter += 1
st.title("chatbot")
st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")
user_question = st.text_input("Ask a question about your documents:")

if user_question:
  handle_question(user_question)

with st.sidebar:
  pdf_files = st.file_uploader("Upload PDF file", accept_multiple_files=True, label_visibility="visible", type=("PDF"))
  st.button("split", on_click=split_pdf_files(pdf_files))
  st.button("embed", on_click=embed_texts(st.session_state.pdf_files_ref))
 

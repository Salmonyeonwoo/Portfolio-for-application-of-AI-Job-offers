# ================================
# Streamlit App: 개인 맞춤형 AI 학습 코치
# LangChain + Gemini + LSTM + RAG
# ================================

import os
import subprocess
import streamlit as st
import tempfile
import time
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV3Large

# ================================
# 0️⃣ 환경별 자동 설치
# ================================
if not os.environ.get("STREAMLIT_RUNTIME"):
    try:
        subprocess.check_call([
            "pip", "install",
            "tensorflow==2.13.0",
            "unstructured-inference==0.7.11"
        ])
        print("✅ Local mode detected: Installed TensorFlow & unstructured-inference")
    except Exception as e:
        print("⚠️ Local install skipped:", e)
else:
    print("🌐 Streamlit Cloud mode detected: Skipping heavy installs")

# ================================
# 0️⃣-1. NLTK punkt_tab 자동 설치
# ================================
import nltk

@st.cache_resource
def ensure_nltk_punkt_tab():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

ensure_nltk_punkt_tab()

# ================================
# 1️⃣ 환경 설정 및 LLM 초기화
# ================================
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

if 'client' not in st.session_state:
    try:
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=API_KEY)
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        st.session_state.is_llm_ready = True
    except Exception as e:
        st.error(f"LLM 초기화 오류: API 키를 확인해 주세요. {e}")
        st.session_state.is_llm_ready = False

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ================================
# 2️⃣ LSTM 모델 정의
# ================================
@st.cache_resource
def load_or_train_lstm():
    np.random.seed(42)
    data = np.cumsum(np.random.normal(loc=5, scale=5, size=50)) + 60
    data = np.clip(data, 50, 95)

    def create_dataset(dataset, look_back=3):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back)])
            Y.append(dataset[i + look_back])
        return np.array(X), np.array(Y)

    look_back = 5
    X, Y = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=10, batch_size=1, verbose=0)

    return model, data

# ================================
# 3️⃣ RAG 관련 함수
# ================================
def get_document_chunks(files):
    documents = []
    temp_dir = tempfile.mkdtemp()

    for uploaded_file in files:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.name.endswith(".html"):
            loader = UnstructuredHTMLLoader(temp_filepath)
        else:
            loader = TextLoader(temp_filepath, encoding="utf-8")

        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def get_vector_store(text_chunks):
    vector_store = FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)
    return vector_store

def get_rag_chain(vector_store):
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=

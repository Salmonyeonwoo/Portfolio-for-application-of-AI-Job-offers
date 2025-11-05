# ================================
# Streamlit App: 개인 맞춤형 AI 학습 코치
# Features: RAG, LSTM 예측, 맞춤 콘텐츠 생성
# Auto Local / Cloud Installation
# ================================

import os
import subprocess
import streamlit as st
import tempfile
import time
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# =====================================
# 🌐 환경에 따라 TensorFlow & unstructured-inference 자동 설치
# =====================================
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

# =====================================
# NLTK punkt 다운로드 (RAG에서 필요)
# =====================================
nltk.download('punkt', quiet=True)

# =====================================
# Gemini API 키
# =====================================
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

# =====================================
# LangChain LLM & Embedding 초기화
# =====================================
if 'client' not in st.session_state:
    try:
        st.session_state.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=API_KEY
        )
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY
        )
        st.session_state.is_llm_ready = True
    except Exception as e:
        st.error(f"LLM 초기화 오류: {e}")
        st.session_state.is_llm_ready = False

# =====================================
# LangChain Memory 초기화
# =====================================
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# =====================================
# LSTM 모델 생성 함수
# =====================================
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

# =====================================
# RAG 관련 함수
# =====================================
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)


def get_vector_store(text_chunks):
    return FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)


def get_rag_chain(vector_store):
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

# =====================================
# Streamlit UI
# =====================================
st.set_page_config(page_title="개인 맞춤형 AI 학습 코치", layout="wide")

with st.sidebar:
    st.title("📚 AI Study Coach 설정")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "학습 자료 업로드 (PDF, TXT, HTML)",
        type=["pdf", "txt", "html"],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.is_llm_ready:
        if st.button("자료 분석 시작 (RAG Indexing)", key="start_analysis"):
            with st.spinner("자료를 분석하고 학습 데이터베이스를 구축 중입니다..."):
                try:
                    text_chunks = get_document_chunks(uploaded_files)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                    st.success(f"총 {len(text_chunks)}개 청크로 학습 데이터베이스 구축 완료!")
                except Exception as e:
                    st.error(f"RAG 구축 오류: {e}")
                    st.session_state.is_rag_ready = False
    else:
        st.session_state.is_rag_ready = False
        st.warning("먼저 학습 자료를 업로드하고 분석을 시작하세요.")

    st.markdown("---")
    feature_selection = st.radio(
        "기능 선택",
        ["RAG 지식 챗봇", "맞춤형 학습 콘텐츠 생성", "LSTM 성취도 예측 대시보드"]
    )

st.title("✨ 개인 맞춤형 AI 학습 코치")

# =====================================
# 기능별 구현
# =====================================
if feature_selection == "RAG 지식 챗봇":
    st.header("RAG 지식 챗봇 (문서 기반 Q&A)")
    st.markdown("업로드된 문서(포트폴리오, PDF 등)의 내용을 기반으로 질문에 답변합니다.")

    if st.session_state.is_rag_ready:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("학습 자료에 대해 질문해 보세요 (예: 이 문서의 핵심 기술은 무엇인가요?)"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변을 생성 중입니다..."):
                    try:
                        response = st.session_state.conversation_chain.invoke({"question": prompt})
                        answer = response.get('answer', '응답을 생성할 수 없습니다.')
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"챗봇 응답 오류: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": "처리 중 오류가 발생했습니다."})
    else:
        st.info("사이드바에서 학습 자료를 업로드하고 '자료 분석 시작' 버튼을 눌러주세요.")

elif feature_selection == "맞춤형 학습 콘텐츠 생성":
    st.header("맞춤형 학습 콘텐츠 생성")
    st.markdown("원하는 학습 주제, 난이도, 형식을 입력하시면 LLM이 맞춤형 콘텐츠를 생성합니다.")

    if st.session_state.is_llm_ready:
        topic = st.text_input("학습 주제 (예: Transformer의 Self-Attention 메커니즘)")
        level = st.selectbox("난이도", ["초급", "중급", "고급"])
        content_type = st.selectbox("콘텐츠 형식", ["핵심 요약 노트", "객관식 퀴즈 3문항", "실습 예제 아이디어"])

        if st.button("콘텐츠 생성"):
            if topic:
                system_prompt = f"당신은 {level} 수준의 전문 AI 코치입니다. 요청 주제에 대해 {content_type} 형식으로 교육적인 콘텐츠를 생성해 주세요. 한국어로만 답변."
                user_prompt = f"주제: {topic}. 요청 형식: {content_type}."
                with st.spinner(f"{topic}에 대한 {content_type} 생성 중..."):
                    try:
                        response = st.session_state.llm.invoke(user_prompt, system_instruction=system_prompt)
                        st.success(f"**{topic}** 에 대한 **{content_type}** 결과:")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("학습 주제를 입력해 주세요.")
    else:
        st.error("LLM 초기화 실패. API 키 확인 필요.")

elif feature_selection == "LSTM 성취도 예측 대시보드":
    st.header("LSTM 기반 학습 성취도 예측 대시보드")
    st.markdown("가상의 과거 퀴즈 점수 데이터를 바탕으로 미래 성취도를 예측합니다.")

    with st.spinner("LSTM 모델 로드/학습 중..."):
        try:
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("LSTM 모델 준비 완료!")

            look_back = 5
            last_sequence = historical_scores[-look_back:]
            input_sequence = np.reshape(last_sequence, (1, look_back, 1))

            future

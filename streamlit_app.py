# ================================
# Streamlit AI 학습 코치 최종 버전
# ================================

import streamlit as st

# ⚠️ 가장 먼저 페이지 설정 (StreamlitAPIException 방지)
st.set_page_config(
    page_title="개인 맞춤형 AI 학습 코치",
    layout="wide"
)

import os
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# LangChain & Google Gemini
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# TensorFlow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.applications import MobileNetV3Large

# NLTK punkt 리소스 자동 설치
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

# 🌐 로컬 환경 전용 설치
if not os.environ.get("STREAMLIT_RUNTIME"):
    try:
        subprocess.check_call([
            "pip", "install",
            "tensorflow==2.13.0",
            "unstructured-inference==0.7.11"
        ])
        print("✅ Local mode: TensorFlow & unstructured-inference 설치 완료")
    except Exception as e:
        print("⚠️ Local install skipped:", e)
else:
    print("🌐 Streamlit Cloud mode: Heavy install skipped")

# ================================
# LLM & Memory 초기화
# ================================

API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

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

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ================================
# LSTM 모델 정의 (심화 기능)
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
# RAG 구축 관련 함수
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
    return FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)


def get_rag_chain(vector_store):
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

# ================================
# Streamlit UI
# ================================

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
            with st.spinner("자료 분석 중..."):
                try:
                    text_chunks = get_document_chunks(uploaded_files)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                    st.success(f"총 {len(text_chunks)}개 청크 분석 완료!")
                except Exception as e:
                    st.error(f"RAG 구축 오류: {e}")
                    st.session_state.is_rag_ready = False
    else:
        st.session_state.is_rag_ready = False

    st.markdown("---")
    feature_selection = st.radio(
        "기능 선택",
        ["RAG 지식 챗봇", "맞춤형 학습 콘텐츠 생성", "LSTM 성취도 예측 대시보드"]
    )

st.title("✨ 개인 맞춤형 AI 학습 코치")

# ================================
# 기능별 페이지
# ================================

if feature_selection == "RAG 지식 챗봇":
    st.header("RAG 지식 챗봇 (문서 기반 Q&A)")
    st.markdown("업로드된 문서 기반 질문에 답변합니다.")
    if st.session_state.is_rag_ready:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("질문을 입력하세요:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        response = st.session_state.conversation_chain.invoke({"question": prompt})
                        answer = response.get('answer', '응답을 생성할 수 없습니다.')
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"챗봇 응답 오류: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": "오류 발생"})

elif feature_selection == "맞춤형 학습 콘텐츠 생성":
    st.header("맞춤형 학습 콘텐츠 생성")
    if st.session_state.is_llm_ready:
        topic = st.text_input("학습 주제")
        level = st.selectbox("난이도", ["초급", "중급", "고급"])
        content_type = st.selectbox("콘텐츠 형식", ["핵심 요약 노트", "객관식 퀴즈 3문항", "실습 예제 아이디어"])

        if st.button("콘텐츠 생성"):
            if topic:
                system_prompt = f"당신은 {level} 수준의 전문 AI 코치입니다. 요청 주제: {topic}. 형식: {content_type}. 한국어로 작성."
                user_prompt = f"주제: {topic}. 형식: {content_type}."
                with st.spinner("콘텐츠 생성 중..."):
                    try:
                        response = st.session_state.llm.invoke(user_prompt, system_instruction=system_prompt)
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("학습 주제를 입력해주세요.")
    else:
        st.error("LLM이 초기화되지 않았습니다.")

elif feature_selection == "LSTM 성취도 예측 대시보드":
    st.header("LSTM 기반 학습 성취도 예측")
    with st.spinner("LSTM 모델 학습 중..."):
        try:
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("LSTM 모델 준비 완료!")

            look_back = 5
            last_sequence = historical_scores[-look_back:]
            input_sequence = np.reshape(last_sequence, (1, look_back, 1))

            future_predictions = []
            current_input = input_sequence

            for _ in range(5):
                next_score = lstm_model.predict(current_input, verbose=0)[0]
                future_predictions.append(next_score[0])
                next_input = np.append(current_input[:, 1:, :], next_score[0]).reshape(1, look_back, 1)
                current_input = next_input

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(len(historical_scores)), historical_scores, label="과거 점수", marker='o')
            future_indices = range(len(historical_scores), len(historical_scores) + len(future_predictions))
            ax.plot(future_indices, future_predictions, label="예측 성취도 (다음 5일)", marker='x', linestyle='--', color='red')



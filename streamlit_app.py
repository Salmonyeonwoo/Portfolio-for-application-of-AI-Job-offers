# ================================
# Streamlit Cloud + Local Dual Mode
# LangChain + Gemini (2025-11)
# ================================

import os
import subprocess
import streamlit as st
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# --- 0. 로컬 환경 자동 설치 ---
if not os.environ.get("STREAMLIT_RUNTIME"):
    try:
        subprocess.check_call([
            "pip", "install",
            "tensorflow==2.13.0",
            "unstructured-inference==0.7.11"
        ])
        print("✅ Local mode: Installed TensorFlow & unstructured-inference")
    except Exception as e:
        print("⚠️ Local install skipped:", e)
else:
    print("🌐 Streamlit Cloud detected: Skipping heavy installs")

# ================================
# 1. LLM 초기화
# ================================
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

if 'llm' not in st.session_state:
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

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ================================
# 2. LSTM 모델 정의
# ================================
@st.cache_resource
def load_or_train_lstm():
    np.random.seed(42)
    data = np.cumsum(np.random.normal(5, 5, 50)) + 60
    data = np.clip(data, 50, 95)

    def create_dataset(dataset, look_back=3):
        X, Y = [], []
        for i in range(len(dataset)-look_back):
            X.append(dataset[i:i+look_back])
            Y.append(dataset[i+look_back])
        return np.array(X), np.array(Y)

    look_back = 5
    X, Y = create_dataset(data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=10, batch_size=1, verbose=0)
    return model, data

# ================================
# 3. RAG 함수
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

def get_vector_store(chunks):
    return FAISS.from_documents(chunks, embedding=st.session_state.embeddings)

def get_rag_chain(vector_store):
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

# ================================
# 4. Streamlit UI
# ================================
st.set_page_config(page_title="AI 학습 코치", layout="wide")

with st.sidebar:
    st.title("📚 AI Study Coach 설정")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "학습 자료 업로드 (PDF, TXT, HTML)",
        type=["pdf", "txt", "html"],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.is_llm_ready:
        if st.button("자료 분석 시작 (RAG Indexing)"):
            with st.spinner("RAG 데이터베이스 구축 중..."):
                try:
                    chunks = get_document_chunks(uploaded_files)
                    vector_store = get_vector_store(chunks)
                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                    st.success(f"{len(chunks)}개 청크로 RAG 구축 완료!")
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
# 5. 기능별 페이지
# ================================
if feature_selection == "RAG 지식 챗봇":
    st.header("RAG 챗봇")
    if st.session_state.is_rag_ready:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("질문 입력"):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        response = st.session_state.conversation_chain.invoke({"question": prompt})
                        answer = response.get("answer", "응답 생성 실패")
                        st.markdown(answer)
                        st.session_state.messages.append({"role":"assistant","content":answer})
                    except Exception as e:
                        st.error(f"챗봇 오류: {e}")
                        st.session_state.messages.append({"role":"assistant","content":"오류 발생"})
    else:
        st.info("사이드바에서 자료를 업로드 후 분석 시작 버튼 클릭")

elif feature_selection == "맞춤형 학습 콘텐츠 생성":
    st.header("맞춤형 학습 콘텐츠 생성")
    topic = st.text_input("주제 입력")
    level = st.selectbox("난이도", ["초급","중급","고급"])
    content_type = st.selectbox("콘텐츠 형식", ["핵심 요약 노트","퀴즈 3문항","실습 예제"])

    if st.button("콘텐츠 생성"):
        if topic:
            system_prompt = f"당신은 {level} 수준의 전문 AI 코치입니다. 요청한 주제에 대해 {content_type} 생성. 한국어로."
            user_prompt = f"주제: {topic}. 형식: {content_type}."
            with st.spinner("생성 중..."):
                try:
                    response = st.session_state.llm.invoke(user_prompt, system_instruction=system_prompt)
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"생성 오류: {e}")
        else:
            st.warning("주제 입력 필요")

elif feature_selection == "LSTM 성취도 예측 대시보드":
    st.header("LSTM 학습 성취도 예측")
    with st.spinner("모델 준비 중..."):
        try:
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("모델 준비 완료!")

            look_back = 5
            input_seq = np.reshape(historical_scores[-look_back:], (1, look_back, 1))

            future_preds = []
            current_input = input_seq
            for _ in range(5):
                next_score = lstm_model.predict(current_input, verbose=0)[0]
                future_preds.append(next_score[0])
                current_input = np.append(current_input[:,1:,:], next_score[0]).reshape(1, look_back, 1)

            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(range(len(historical_scores)), historical_scores, label="과거 점수", marker='o')
            ax.plot(range(len(historical_scores), len(historical_scores)+5), future_preds, label="예측 점수", marker='x', linestyle='--', color='red')
            ax.set_title("LSTM 성취도 예측")
            ax.set_xlabel("주기")
            ax.set_ylabel("점수")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            avg_recent = np.mean(historical_scores[-5:])
            avg_predict = np.mean(future_preds)

            if avg_predict > avg_recent:
                comment = "앞으로 성취도가 향상될 것으로 예측됩니다."
            elif avg_predict < avg_recent-5:
                comment = "성취도가 다소 하락할 수 있습니다."
            else:
                comment = "성취도는 현재 수준을 유지할 것으로 보입니다."

            st.info(comment)

        except Exception as e:
            st.error(f"LSTM 처리 오류: {e}")

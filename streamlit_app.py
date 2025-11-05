# =======================================
# Streamlit + LangChain + Gemini 안정화
# 무료 티어 임베딩 캐시 포함
# =======================================

import os
import pickle
import tempfile
import subprocess
import streamlit as st
import numpy as np
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

import nltk

# =======================================
# 0. NLTK 리소스 자동 다운로드
# =======================================
if "nltk_downloaded" not in st.session_state:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    st.session_state["nltk_downloaded"] = True

# =======================================
# 1. LLM 및 Embeddings 초기화
# =======================================
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

if "llm" not in st.session_state:
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
        st.error(f"LLM 초기화 실패: {e}")
        st.session_state.is_llm_ready = False

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# =======================================
# 2. LSTM 모델 로드/학습
# =======================================
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
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([LSTM(50, activation="relu", input_shape=(look_back, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, batch_size=1, verbose=0)
    return model, data

# =======================================
# 3. RAG 문서 처리 + VectorStore 캐시
# =======================================
CACHE_PATH = "vectorstore_cache.pkl"

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

def get_vector_store(text_chunks):
    try:
        vector_store = FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)
        # 캐시 저장
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(vector_store, f)
        return vector_store
    except Exception as e:
        st.warning(f"임베딩 실패: 무료 티어 한도 초과 또는 네트워크 문제.\n{e}")
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                st.info("⚡ 캐시된 VectorStore 사용")
                return pickle.load(f)
        else:
            st.error("RAG 구축 실패: 학습 자료를 업로드 후 다시 시도하세요.")
            return None

def get_rag_chain(vector_store):
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

# =======================================
# 4. Streamlit UI
# =======================================
st.set_page_config(page_title="AI 학습 코치", layout="wide")

with st.sidebar:
    st.title("📚 설정")
    uploaded_files = st.file_uploader(
        "자료 업로드 (PDF/TXT/HTML)", type=["pdf","txt","html"], accept_multiple_files=True
    )

    if uploaded_files and st.session_state.is_llm_ready:
        if st.button("자료 분석 (RAG 구축)"):
            with st.spinner("자료 분석 중..."):
                chunks = get_document_chunks(uploaded_files)
                vector_store = get_vector_store(chunks)
                if vector_store:
                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                    st.success(f"RAG 구축 완료! 총 {len(chunks)} 청크.")
                else:
                    st.session_state.is_rag_ready = False
    else:
        st.session_state.is_rag_ready = False

    st.markdown("---")
    feature_selection = st.radio(
        "기능 선택",
        ["RAG 챗봇", "맞춤형 콘텐츠 생성", "LSTM 예측 대시보드"]
    )

st.title("✨ 개인 맞춤형 AI 학습 코치")

# =======================================
# 5. 기능별 구현
# =======================================
if feature_selection == "RAG 챗봇":
    st.header("RAG 기반 문서 Q&A")
    if st.session_state.is_rag_ready:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("질문을 입력하세요"):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        resp = st.session_state.conversation_chain.invoke({"question":prompt})
                        answer = resp.get("answer","응답 생성 실패")
                        st.markdown(answer)
                        st.session_state.messages.append({"role":"assistant","content":answer})
                    except Exception as e:
                        st.error(f"RAG 오류: {e}")

elif feature_selection == "맞춤형 콘텐츠 생성":
    st.header("맞춤형 학습 콘텐츠 생성")
    if st.session_state.is_llm_ready:
        topic = st.text_input("학습 주제")
        level = st.selectbox("난이도", ["초급","중급","고급"])
        content_type = st.selectbox("형식", ["핵심 요약 노트","객관식 퀴즈 3문항","실습 예제 아이디어"])
        if st.button("콘텐츠 생성"):
            if topic:
                system_prompt = f"""당신은 {level} 수준의 전문 AI 코치입니다.
요청 주제에 대해 {content_type} 형식에 맞춰 명확하고 교육적인 콘텐츠를 제공하세요.
답변은 한국어로만 작성합니다."""
                user_prompt = f"주제: {topic}. 형식: {content_type}."
                with st.spinner("생성 중..."):
                    try:
                        resp = st.session_state.llm.invoke(user_prompt, system_instruction=system_prompt)
                        st.success(f"**{topic}** - {content_type} 결과:")
                        st.markdown(resp.content)
                    except Exception as e:
                        st.error(f"콘텐츠 생성 실패: {e}")
            else:
                st.warning("학습 주제를 입력하세요.")

elif feature_selection == "LSTM 예측 대시보드":
    st.header("LSTM 학습 성취도 예측")
    with st.spinner("LSTM 모델 로드 중..."):
        try:
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("LSTM 모델 준비 완료!")

            look_back = 5
            last_seq = historical_scores[-look_back:]
            input_seq = np.reshape(last_seq, (1, look_back, 1))

            future_preds = []
            curr_input = input_seq
            for _ in range(5):
                next_score = lstm_model.predict(curr_input, verbose=0)[0]
                future_preds.append(next_score[0])
                curr_input = np.append(curr_input[:,1:,:], next_score[0]).reshape(1, look_back, 1)

            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(range(len(historical_scores)), historical_scores, label="과거 점수", marker='o', linestyle='-')
            future_idx = range(len(historical_scores), len(historical_scores)+len(future_preds))
            ax.plot(future_idx, future_preds, label="예측 점수", marker='x', linestyle='--', color='red')
            ax.set_title("LSTM 성취도 예측")
            ax.set_xlabel("주기")
            ax.set_ylabel("점수")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            avg_recent = np.mean(historical_scores[-5:])
            avg_future = np.mean(future_preds)
            if avg_future > avg_recent:
                comment = "앞으로 성취도 향상 예상"
            elif avg_future < avg_recent-5:
                comment = "성취도 하락 가능. RAG 챗봇 활용 추천"
            else:
                comment = "성취도 유지 예상"
            st.info(comment)

        except Exception as e:
            st.error(f"LSTM 처리 실패: {e}")

# ========================================
# 안정화 Streamlit AI 학습 코치 (RAG + LSTM + 콘텐츠 생성)
# Gemini API 무료 티어 한도 및 임베딩 캐시 대응
# ========================================

import os
import subprocess
import tempfile
import hashlib
import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

import nltk

# ========================================
# 0. 환경별 설치
# ========================================
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
    print("🌐 Streamlit Cloud mode: Skipping heavy installs")

# ========================================
# 0-1. NLTK 리소스 자동 다운로드
# ========================================
if "nltk_downloaded" not in st.session_state:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    st.session_state["nltk_downloaded"] = True

# ========================================
# 1. Gemini / LangChain 초기화
# ========================================
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

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ========================================
# 2. LSTM 모델 학습
# ========================================
@st.cache_resource
def load_or_train_lstm():
    np.random.seed(42)
    data = np.cumsum(np.random.normal(5,5,50)) + 60
    data = np.clip(data, 50, 95)

    def create_dataset(dataset, look_back=5):
        X, Y = [], []
        for i in range(len(dataset)-look_back):
            X.append(dataset[i:i+look_back])
            Y.append(dataset[i+look_back])
        return np.array(X), np.array(Y)

    look_back = 5
    X, Y = create_dataset(data, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,Y,epochs=10,batch_size=1,verbose=0)
    return model, data

# ========================================
# 3. RAG 구축 + 임베딩 캐시 대응
# ========================================
CACHE_DIR = ".rag_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file):
    data = file.getvalue()
    return hashlib.md5(data).hexdigest()

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

def get_vector_store_with_cache(files):
    file_hash = hashlib.md5("".join([get_file_hash(f) for f in files]).encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.pkl")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                vector_store = pickle.load(f)
            st.info("⚡ 캐시된 임베딩 로드 완료")
            return vector_store
        except Exception:
            st.warning("⚠️ 캐시 로드 실패, 새로 임베딩 시도")
    
    try:
        chunks = get_document_chunks(files)
        vector_store = FAISS.from_documents(chunks, embedding=st.session_state.embeddings)
        with open(cache_path, "wb") as f:
            pickle.dump(vector_store, f)
        st.success(f"총 {len(chunks)} 청크 임베딩 완료")
        return vector_store
    except Exception as e:
        st.error(f"RAG 구축 오류: {e}")
        return None

def get_rag_chain(vector_store):
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

# ========================================
# 4. Streamlit UI
# ========================================
st.set_page_config(page_title="개인 맞춤형 AI 학습 코치", layout="wide")

with st.sidebar:
    st.title("📚 AI Study Coach 설정")
    uploaded_files = st.file_uploader(
        "학습 자료 업로드 (PDF, TXT, HTML)",
        type=["pdf","txt","html"],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.is_llm_ready:
        if st.button("자료 분석 시작 (RAG Indexing)"):
            with st.spinner("자료 분석 중..."):
                vector_store = get_vector_store_with_cache(uploaded_files)
                if vector_store:
                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                else:
                    st.session_state.is_rag_ready = False
    else:
        st.session_state.is_rag_ready = False

    st.markdown("---")
    feature_selection = st.radio(
        "기능 선택",
        ["RAG 지식 챗봇", "맞춤형 학습 콘텐츠 생성", "LSTM 성취도 예측 대시보드"]
    )

st.title("✨ 개인 맞춤형 AI 학습 코치")

# ========================================
# 5-1. RAG 지식 챗봇
# ========================================
if feature_selection == "RAG 지식 챗봇":
    st.header("RAG 지식 챗봇")
    if st.session_state.is_rag_ready:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("질문 입력"):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        response = st.session_state.conversation_chain.invoke({"question":prompt})
                        answer = response.get("answer","응답 없음")
                        st.markdown(answer)
                        st.session_state.messages.append({"role":"assistant","content":answer})
                    except Exception as e:
                        st.error(f"RAG 오류: {e}")
    else:
        st.info("자료 업로드 후 분석 시작 필요")

# ========================================
# 5-2. 맞춤형 학습 콘텐츠 생성
# ========================================
elif feature_selection == "맞춤형 학습 콘텐츠 생성":
    st.header("맞춤형 학습 콘텐츠 생성")
    if st.session_state.is_llm_ready:
        topic = st.text_input("학습 주제")
        level = st.selectbox("난이도", ["초급","중급","고급"])
        content_type = st.selectbox("콘텐츠 형식", ["핵심 요약 노트","객관식 퀴즈 3문항","실습 예제 아이디어"])
        if st.button("콘텐츠 생성"):
            if topic:
                system_prompt = f"""당신은 {level} 수준의 전문 AI 코치입니다.
요청받은 주제에 대해 {content_type} 형식에 맞춰 명확하고 교육적인 콘텐츠를 생성해 주세요.
답변은 한국어로만 제공해야 합니다."""
                user_prompt = f"주제: {topic}. 요청 형식: {content_type}"
                with st.spinner("콘텐츠 생성 중..."):
                    try:
                        response = st.session_state.llm.invoke(user_prompt, system_instruction=system_prompt)
                        st.success("콘텐츠 생성 완료!")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("학습 주제를 입력하세요.")
    else:
        st.error("LLM 초기화 실패")

# ========================================
# 5-3. LSTM 성취도 예측
# ========================================
elif feature_selection == "LSTM 성취도 예측 대시보드":
    st.header("LSTM 기반 학습 성취도 예측")
    with st.spinner("LSTM 모델 로드 및 학습 중..."):
        try:
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("LSTM 모델 준비 완료!")

            look_back = 5
            last_sequence = historical_scores[-look_back:]
            input_sequence = np.reshape(last_sequence,(1,look_back,1))

            future_predictions = []
            current_input = input_sequence
            for _ in range(5):
                next_score = lstm_model.predict(current_input, verbose=0)[0]
                future_predictions.append(next_score[0])
                # 다음 입력 시퀀스 갱신
                current_input = np.append(current_input[:, 1:, :], next_score[0]).reshape(1, look_back, 1)

            # 그래프 시각화
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(range(len(historical_scores)), historical_scores, label="과거 점수", marker='o', linestyle='-', color='blue')
            future_indices = range(len(historical_scores), len(historical_scores) + len(future_predictions))
            ax.plot(future_indices, future_predictions, label="예측 성취도", marker='x', linestyle='--', color='red')
            ax.set_title("LSTM 기반 학습 성취도 예측")
            ax.set_xlabel("주기")
            ax.set_ylabel("성취도 점수 (0-100)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # 최근 평균 vs 예측 평균 비교
            avg_recent = np.mean(historical_scores[-5:])
            avg_predict = np.mean(future_predictions)
            if avg_predict > avg_recent:
                comment = "앞으로 성취도가 긍정적으로 향상될 것으로 예측됩니다."
            elif avg_predict < avg_recent - 5:
                comment = "성취도가 다소 하락할 수 있습니다. RAG 챗봇으로 기초 개념 확인을 추천합니다."
            else:
                comment = "성취도는 현재 수준 유지 예상. 새로운 콘텐츠로 학습 활력 추가 고려."
            st.info(comment)

        except Exception as e:
            st.error(f"LSTM 처리 중 오류: {e}")


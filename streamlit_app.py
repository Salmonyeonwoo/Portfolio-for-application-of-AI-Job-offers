# ========================================
# Streamlit Full App: AI 학습 코치 (2025-11)
# RAG + 맞춤 콘텐츠 생성 + LSTM 성취도 예측
# ========================================

import os
import tempfile
import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===============================
# 1. Gemini API 초기화
# ===============================
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

# 메모리 초기화
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ===============================
# 2. LSTM 모델 생성 (예측용)
# ===============================
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

# ===============================
# 3. RAG 구축 함수
# ===============================
def get_document_chunks(files, max_files=2):
    documents = []
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in files[:max_files]:
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path,"wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_path) if uploaded_file.name.endswith(".pdf") else TextLoader(temp_path)
        documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def get_vector_store(text_chunks):
    cache_file = "vector_store.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    try:
        vector_store = FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)
        with open(cache_file, "wb") as f:
            pickle.dump(vector_store, f)
        return vector_store
    except Exception as e:
        st.error(f"임베딩 오류: {e}")
        return None

def get_rag_chain(vector_store):
    return ConversationalRetrievalChain.from_llm(
        llm=None,  # LLM 연결 시 여기에 st.session_state.llm
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

# ===============================
# 4. Streamlit UI
# ===============================
st.set_page_config(page_title="개인 맞춤형 AI 학습 코치", layout="wide")
st.sidebar.title("📚 AI Study Coach 설정")

uploaded_files = st.sidebar.file_uploader(
    "학습 자료 업로드 (PDF, TXT)", type=["pdf","txt"], accept_multiple_files=True
)

if uploaded_files and st.session_state.is_llm_ready:
    if st.sidebar.button("자료 분석 시작 (RAG Indexing)"):
        with st.spinner("RAG 분석 중..."):
            text_chunks = get_document_chunks(uploaded_files)
            vector_store = get_vector_store(text_chunks)
            if vector_store:
                st.session_state.conversation_chain = get_rag_chain(vector_store)
                st.session_state.is_rag_ready = True
                st.success(f"RAG 구축 완료! ({len(text_chunks)} 청크)")
else:
    st.session_state.is_rag_ready = False

feature_selection = st.sidebar.radio(
    "기능 선택",
    ["RAG 지식 챗봇","맞춤형 학습 콘텐츠 생성","LSTM 성취도 예측 대시보드"]
)

st.title("✨ 개인 맞춤형 AI 학습 코치")

# ===============================
# 5-1. RAG 챗봇
# ===============================
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
        st.info("자료 업로드 후 분석 시작 버튼 클릭 필요")

# ===============================
# 5-2. 맞춤형 학습 콘텐츠 생성
# ===============================
elif feature_selection == "맞춤형 학습 콘텐츠 생성":
    st.header("맞춤형 학습 콘텐츠 생성")
    st.markdown("학습 주제와 난이도, 형식을 입력하세요.")
    if st.session_state.is_llm_ready:
        topic = st.text_input("주제 입력")
        level = st.selectbox("난이도",["초급","중급","고급"])
        content_type = st.selectbox("형식",["요약 노트","객관식 퀴즈","실습 아이디어"])
        if st.button("생성"):
            if topic:
                system_prompt = f"당신은 {level} 수준 AI 코치입니다. {content_type} 형식으로 콘텐츠 생성. 한국어로만."
                user_prompt = f"주제: {topic}, 형식: {content_type}"
                with st.spinner("콘텐츠 생성 중..."):
                    try:
                        response = st.session_state.llm.invoke(user_prompt, system_instruction=system_prompt)
                        st.success("생성 완료!")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("주제를 입력하세요.")
    else:
        st.error("LLM 초기화 실패")

# ===============================
# 5-3. LSTM 성취도 예측
# ===============================
elif feature_selection == "LSTM 성취도 예측 대시보드":
    st.header("LSTM 기반 학습 성취도 예측")
    with st.spinner("LSTM 모델 로드 및 학습 중..."):
        try:
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("LSTM 모델 준비 완료!")
            look_back = 5
            last_seq = historical_scores[-look_back:]
            input_seq = np.reshape(last_seq,(1,look_back,1))

            future_preds = []
            current_input = input_seq
            for i in range(5):
                next_score = lstm_model.predict(current_input, verbose=0)[0]
                future_preds.append(next_score[0])
                current_input = np.append(current_input[:,1:,:],next_score[0]).reshape(1,look_back,1)

            # 시각화
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(range(len(historical_scores)), historical_scores,label="과거 점수",marker='o',color='blue')
            future_idx = range(len(historical_scores),len(historical_scores)+len(future_preds))
            ax.plot(future_idx, future_preds,label="예측 점수",marker='x',linestyle='--',color='red')
            ax.set_title("LSTM 학습 성취도 예측")
            ax.set_xlabel("Day/Week")
            ax.set_ylabel("점수")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # 코멘트
            avg_recent = np.mean(historical_scores[-5:])
            avg_predict = np.mean(future_preds)
            if avg_predict > avg_recent:
                comment = "앞으로 성취도가 향상될 것으로 예측됩니다. 현재 학습 방식을 유지하세요."
            elif avg_predict < avg_recent-5:
                comment = "성취도가 다소 하락할 수 있습니다. RAG 챗봇 활용 추천."
            else:
                comment = "성취도는 유지될 것으로 예상됩니다. 새로운 학습 콘텐츠 활용 고려."
            st.info(comment)

        except Exception as e:
            st.error(f"LSTM 처리 오류: {e}")

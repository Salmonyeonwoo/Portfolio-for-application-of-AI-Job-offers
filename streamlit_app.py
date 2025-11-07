# ========================================
# Streamlit AI 학습 코치 (RAG 최종 수정)
# ========================================
import streamlit as st
import os
import tempfile
import time
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import tensorflow as tf # ⭐필수: LSTM 복구를 위해 임포트
from tensorflow.keras.models import Sequential # ⭐필수: LSTM 복구를 위해 임포트
from tensorflow.keras.layers import LSTM, Dense # ⭐필수: LSTM 복구를 위해 임포트

# ================================
# 1. LLM 및 임베딩 초기화 + 임베딩 캐시
# (이전 코드와 동일)
# ================================
API_KEY = os.environ.get("GEMINI_API_KEY")

if 'llm' not in st.session_state:
    if not API_KEY:
        st.error("⚠️ 경고: GEMINI API 키가 설정되지 않았습니다. Streamlit Secrets에 'GEMINI_API_KEY'를 설정해주세요.")
        st.session_state.is_llm_ready = False
    else:
        try:
            st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=API_KEY)
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
            st.session_state.is_llm_ready = True
        except Exception as e:
            st.error(f"LLM 초기화 오류: API 키를 확인해 주세요. {e}")
            st.session_state.is_llm_ready = False

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

# ================================
# 2. LSTM 모델 정의 (⭐복구된 영역⭐)
# ================================
@st.cache_resource
def load_or_train_lstm():
    """가상의 학습 성취도 예측을 위한 LSTM 모델을 생성하고 학습합니다."""
    # 1. 가상 데이터 생성: 10주간의 퀴즈 점수 (0-100)
    np.random.seed(42)
    data = np.cumsum(np.random.normal(loc=5, scale=5, size=50)) + 60
    data = np.clip(data, 50, 95)  # 점수 범위 제한

    # 2. 시계열 데이터 전처리
    def create_dataset(dataset, look_back=3):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back)])
            Y.append(dataset[i + look_back])
        return np.array(X), np.array(Y)

    look_back = 5
    X, Y = create_dataset(data, look_back)

    # LSTM 입력 형태 맞추기: [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 3. LSTM 모델 정의
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, 1)),
        Dense(1)
    ])

    # 4. 모델 학습 (빠른 시연을 위해 최소한의 epoch만 설정)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, Y, epochs=10, batch_size=1, verbose=0)

    return model, data

# --- RAG 관련 함수 (이전 코드와 동일) ---
def get_document_chunks(files):
    documents = []
    temp_dir = tempfile.mkdtemp()

    for uploaded_file in files:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == "pdf":
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            documents.extend(loader.load())
        
        elif file_extension == "html":
            from bs4 import BeautifulSoup
            raw_html = uploaded_file.getvalue().decode('utf-8')
            soup = BeautifulSoup(raw_html, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)
            
            from langchain.schema.document import Document
            documents.append(Document(page_content=text_content, metadata={"source": uploaded_file.name}))


        elif file_extension == "txt":
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = TextLoader(temp_filepath, encoding="utf-8")
            documents.extend(loader.load())
            
        else:
            st.warning(f"'{uploaded_file.name}' 파일은 현재 PDF, TXT, HTML만 지원하여 로딩할 수 없습니다.")
            continue

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)


def get_vector_store(text_chunks):
    cache_key = tuple(doc.page_content for doc in text_chunks)
    if cache_key in st.session_state.embedding_cache:
        st.info("✅ 임베딩 캐시가 발견되어 재사용합니다. (API 한도 절약)")
        return st.session_state.embedding_cache[cache_key]
    
    if not st.session_state.is_llm_ready:
        return None

    try:
        vector_store = FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)
        st.session_state.embedding_cache[cache_key] = vector_store
        return vector_store
    
    except Exception as e:
        if "429" in str(e):
             st.error("⚠️ **API 임베딩 한도 초과 (429 Error)**: Google Gemini API의 무료 임베딩 요청 한도를 초과했습니다. 내일 다시 시도하거나 API 사용량 대시보드를 확인하세요.")
        else:
            st.error(f"Vector Store 생성 중 오류 발생: {e}")
        return None


def get_rag_chain(vector_store):
    if vector_store is None:
        return None
        
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )

# ================================
# 4. Streamlit UI (이전 코드와 동일)
# ================================
st.set_page_config(page_title="개인 맞춤형 AI 학습 코치", layout="wide")

with st.sidebar:
    st.title("📚 AI Study Coach 설정")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "학습 자료 업로드 (PDF, TXT, HTML)",
        type=["pdf","txt","html"],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.is_llm_ready:
        if st.button("자료 분석 시작 (RAG Indexing)", key="start_analysis"):
            with st.spinner("자료 분석 및 학습 DB 구축 중..."):
                text_chunks = get_document_chunks(uploaded_files)
                vector_store = get_vector_store(text_chunks)
                
                if vector_store:
                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                    st.success(f"총 {len(text_chunks)}개 청크로 학습 DB 구축 완료!")
                else:
                    st.session_state.is_rag_ready = False
                    st.error("임베딩 실패: 무료 티어 한도 초과 또는 네트워크 문제.")

    else:
        st.session_state.is_rag_ready = False
        st.warning("먼저 학습 자료를 업로드하세요.")

    st.markdown("---")
    feature_selection = st.radio(
        "기능 선택",
        ["RAG 지식 챗봇", "맞춤형 학습 콘텐츠 생성", "LSTM 성취도 예측 대시보드"]
    )

st.title("✨ 개인 맞춤형 AI 학습 코치")

# ================================
# 5. 기능별 페이지 구현 (일부만 복구됨)
# ================================
if feature_selection == "RAG 지식 챗봇":
    # RAG 챗봇 로직 (이전과 동일)
    st.header("RAG 지식 챗봇 (문서 기반 Q&A)")
    st.markdown("업로드된 문서 기반으로 질문에 답변합니다.")
    if st.session_state.is_rag_ready and st.session_state.conversation_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("학습 자료에 대해 질문해 보세요"):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        response = st.session_state.conversation_chain.invoke({"question":prompt})
                        answer = response.get('answer','응답을 생성할 수 없습니다.')
                        st.markdown(answer)
                        st.session_state.messages.append({"role":"assistant","content":answer})
                    except Exception as e:
                        st.error(f"챗봇 오류: {e}")
                        st.session_state.messages.append({"role":"assistant","content":"오류 발생"})
    else:
        st.warning("RAG가 준비되지 않았습니다. 학습 자료를 업로드하고 분석하세요.")

# ================================
# 맞춤형 학습 콘텐츠 생성
# ================================
elif feature_selection == "맞춤형 학습 콘텐츠 생성":
    # 콘텐츠 생성 로직 (이전과 동일)
    st.header("맞춤형 학습 콘텐츠 생성")
    st.markdown("학습 주제와 난이도에 맞춰 콘텐츠 생성")

    if st.session_state.is_llm_ready:
        topic = st.text_input("학습 주제")
        level = st.selectbox("난이도", ["초급","중급","고급"])
        content_type = st.selectbox("콘텐츠 형식", ["핵심 요약 노트","객관식 퀴즈 3문항","실습 예제 아이디어"])

        if st.button("콘텐츠 생성"):
            if topic:
                full_prompt = f"""당신은 {level} 수준의 전문 AI 코치입니다.
요청받은 주제에 대해 {content_type} 형식에 맞춰 명확하고 교육적인 콘텐츠를 생성해 주세요. 답변은 한국어로만 제공해야 합니다.

주제: {topic}
요청 형식: {content_type}"""

                with st.spinner(f"{topic}에 대한 {content_type} 생성 중..."):
                    try:
                        response = st.session_state.llm.invoke(full_prompt)
                        st.success(f"**{topic}** - **{content_type}** 결과:")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("학습 주제를 입력해 주세요.")
    else:
        st.error("LLM이 초기화되지 않았습니다. API 키를 확인해 주세요.")

# ================================
# LSTM 성취도 예측 대시보드 (⭐복구된 영역⭐)
# ================================
elif feature_selection == "LSTM 성취도 예측 대시보드":
    st.header("LSTM 기반 학습 성취도 예측 대시보드")
    st.markdown("가상의 과거 퀴즈 점수 데이터를 바탕으로 LSTM 모델을 훈련하고 미래 성취도를 예측하여 보여줍니다.")

    with st.spinner("LSTM 모델을 로드/학습 중입니다..."):
        try:
            # 1. 모델 로드 및 데이터 생성
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("LSTM 모델 준비 완료!")

            # 2. 예측 로직
            look_back = 5
            last_sequence = historical_scores[-look_back:]
            input_sequence = np.reshape(last_sequence, (1, look_back, 1))
            
            future_predictions = []
            current_input = input_sequence

            for i in range(5):
                next_score = lstm_model.predict(current_input, verbose=0)[0]
                future_predictions.append(next_score[0])

                next_input = np.append(current_input[:, 1:, :], next_score[0]).reshape(1, look_back, 1)
                current_input = next_input

            # 3. 시각화
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(range(len(historical_scores)), historical_scores, label="과거 퀴즈 점수 (가상)", marker='o', linestyle='-', color='blue')
            future_indices = range(len(historical_scores), len(historical_scores) + len(future_predictions))
            ax.plot(future_indices, future_predictions, label="예측 성취도 (다음 5일)", marker='x', linestyle='--', color='red')

            ax.set_title("LSTM 기반 학습 성취도 시계열 예측")
            ax.set_xlabel("주기 (Day/Week)")
            ax.set_ylabel("성취도 점수 (0-100)")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # 4. LLM 분석 코멘트
            st.markdown("---")
            st.markdown("#### AI 코치의 성취도 분석 코멘트")
            avg_recent = np.mean(historical_scores[-5:])
            avg_predict = np.mean(future_predictions)

            if avg_predict > avg_recent:
                comment = "최근 학습 데이터와 LSTM 예측 결과에 따르면, **앞으로의 학습 성취도가 긍정적으로 향상될 것으로 예측**됩니다. 현재 학습 방식을 유지하시거나, 난이도를 한 단계 높여 도전해 보세요!"
            elif avg_predict < avg_recent - 5:
                comment = "LSTM 예측 결과, **성취도가 다소 하락할 수 있다는 신호**가 보입니다. 학습에 사용된 자료나 방법론에 대한 깊은 이해가 부족할 수 있습니다. RAG 챗봇 기능을 활용하여 기초 개념을 다시 확인해 보는 것을 추천합니다."
            else:
                comment = "성취도는 현재 수준을 유지할 것으로 예측됩니다. 정체기가 될 수 있으니, **새로운 학습 콘텐츠 형식(예: 실습 예제 아이디어)을 생성**하여 학습에 활력을 더하는 것을 고려해 보세요."

            st.info(comment)

        except Exception as e:
            st.error(f"LSTM 모델 처리 중 오류가 발생했습니다: {e}")
            st.info("⚠️ 오류 원인: TensorFlow 및 관련 라이브러리 설치 실패.")

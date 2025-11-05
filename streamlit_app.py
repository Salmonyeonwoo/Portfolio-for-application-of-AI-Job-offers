import streamlit as st
import os
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

# --- 1. 환경 설정 및 모델 초기화 ---

# Gemini API 키 설정 (실제 환경에서는 secrets.toml 또는 환경 변수 사용)
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

if 'client' not in st.session_state:
    try:
        # LLM 및 Embedding 모델 초기화
        st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=API_KEY)
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        st.session_state.is_llm_ready = True
    except Exception as e:
        st.error(f"LLM 초기화 오류: API 키를 확인해 주세요. {e}")
        st.session_state.is_llm_ready = False

# LangChain 메모리 초기화
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# --- 2. LSTM 모델 정의 (심화 기능) ---
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


# --- 3. RAG 관련 함수 ---
def get_document_chunks(files):
    """업로드된 파일에서 텍스트를 로드하고 청킹합니다."""
    documents = []
    temp_dir = tempfile.mkdtemp()

    for uploaded_file in files:
        # 임시 파일로 저장 (LangChain Loader가 파일 경로를 필요로 함)
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 파일 형식에 따른 로더 선택
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.name.endswith(".html"):
            # 기존 GitHub 포트폴리오 HTML 파일을 로드하기 위한 UnstructuredHTMLLoader 사용
            loader = UnstructuredHTMLLoader(temp_filepath)
        else:  # .txt, 기타 일반 텍스트 파일
            loader = TextLoader(temp_filepath, encoding="utf-8")

        documents.extend(loader.load())

    # 텍스트 분할 (청킹)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)


def get_vector_store(text_chunks):
    """텍스트 청크를 임베딩하고 FAISS Vector Store를 생성합니다."""
    # Google Generative AI Embeddings 사용
    vector_store = FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)
    return vector_store


def get_rag_chain(vector_store):
    """검색 체인(ConversationalRetrievalChain)을 생성합니다."""
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )


# --- 4. Streamlit UI 구성 ---

st.set_page_config(page_title="개인 맞춤형 AI 학습 코치", layout="wide")

# 사이드바: 설정 및 파일 업로드
with st.sidebar:
    st.title("📚 AI Study Coach 설정")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "학습 자료 업로드 (PDF, TXT, HTML)",
        type=["pdf", "txt", "html"],
        accept_multiple_files=True
    )

    # LLM 및 RAG 상태 관리
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

    # 기능 선택 라디오 버튼
    st.markdown("---")
    feature_selection = st.radio(
        "기능 선택",
        ["RAG 지식 챗봇", "맞춤형 학습 콘텐츠 생성", "LSTM 성취도 예측 대시보드"]
    )

# 메인 화면 제목
st.title("✨ 개인 맞춤형 AI 학습 코치")

# --- 5. 기능별 페이지 구현 ---

if feature_selection == "RAG 지식 챗봇":
    st.header("RAG 지식 챗봇 (문서 기반 Q&A)")
    st.markdown("업로드된 문서(포트폴리오, PDF 등)의 내용을 기반으로 질문에 답변합니다.")

    if st.session_state.is_rag_ready:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 이전 대화 내용 출력
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력 처리
        if prompt := st.chat_input("학습 자료에 대해 질문해 보세요 (예: 이 문서의 핵심 기술은 무엇인가요?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변을 생성 중입니다..."):
                    try:
                        # LangChain RAG 체인 호출
                        response = st.session_state.conversation_chain.invoke({"question": prompt})

                        # 응답 텍스트와 근거 문서 추출
                        answer = response.get('answer', '응답을 생성할 수 없습니다.')
                        # source_documents가 있다면 출력 (Retrieval-Augmented)
                        # NOTE: ConversationalRetrievalChain의 출처 추출은 별도 커스터마이징이 필요할 수 있으나, 여기서는 기본 응답을 사용

                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                    except Exception as e:
                        st.error(f"챗봇 응답 오류: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": "죄송합니다. 처리 중 오류가 발생했습니다."})
    else:
        st.info("사이드바에서 학습 자료를 업로드하고 '자료 분석 시작' 버튼을 눌러주세요.")

elif feature_selection == "맞춤형 학습 콘텐츠 생성":
    st.header("맞춤형 학습 콘텐츠 생성")
    st.markdown("원하는 학습 주제, 난이도, 형식을 입력하시면 LLM이 맞춤형 콘텐츠(요약, 퀴즈)를 생성해 드립니다.")

    if st.session_state.is_llm_ready:
        topic = st.text_input("학습 주제 (예: Transformer의 Self-Attention 메커니즘)")
        level = st.selectbox("난이도", ["초급", "중급", "고급"])
        content_type = st.selectbox("콘텐츠 형식", ["핵심 요약 노트", "객관식 퀴즈 3문항", "실습 예제 아이디어"])

        if st.button("콘텐츠 생성"):
            if topic:
                # LLM 프롬프트 구성 (AI Studio에서 테스트한 최적 프롬프트 적용)
                system_prompt = f"""당신은 {level} 수준의 전문 AI 코치입니다. 요청받은 주제에 대해 {content_type} 형식에 맞춰 명확하고 교육적인 콘텐츠를 생성해 주세요. 답변은 한국어로만 제공해야 합니다."""

                user_prompt = f"주제: {topic}. 요청 형식: {content_type}."

                with st.spinner(f"{topic}에 대한 {content_type}을 생성 중입니다..."):
                    try:
                        # LLM에 요청 (LangChain 없이 단순 생성 기능)
                        response = st.session_state.llm.invoke(user_prompt, system_instruction=system_prompt)
                        st.success(f"**{topic}** 에 대한 **{content_type}** 결과:")
                        st.markdown(response.content)

                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("학습 주제를 입력해 주세요.")
    else:
        st.error("LLM이 초기화되지 않았습니다. API 키를 확인해 주세요.")

elif feature_selection == "LSTM 성취도 예측 대시보드":
    st.header("LSTM 기반 학습 성취도 예측 대시보드")
    st.markdown("가상의 과거 퀴즈 점수 데이터를 바탕으로 LSTM 모델을 훈련하고 미래 성취도를 예측하여 보여줍니다.")

    with st.spinner("LSTM 모델을 로드/학습 중입니다..."):
        try:
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("LSTM 모델 준비 완료!")

            # 예측 준비 (마지막 look_back 기간의 데이터 사용)
            look_back = 5
            last_sequence = historical_scores[-look_back:]
            # 입력 형태 맞추기: [1, time steps, features]
            input_sequence = np.reshape(last_sequence, (1, look_back, 1))

            # 다음 5일 예측
            future_predictions = []
            current_input = input_sequence

            for i in range(5):
                next_score = lstm_model.predict(current_input, verbose=0)[0]
                future_predictions.append(next_score[0])

                # 다음 예측을 위한 입력 시퀀스 갱신
                next_input = np.append(current_input[:, 1:, :], next_score[0]).reshape(1, look_back, 1)
                current_input = next_input

            # 시각화
            fig, ax = plt.subplots(figsize=(10, 6))

            # 과거 데이터
            ax.plot(range(len(historical_scores)), historical_scores, label="과거 퀴즈 점수 (가상)", marker='o', linestyle='-',
                    color='blue')

            # 예측 데이터
            future_indices = range(len(historical_scores), len(historical_scores) + len(future_predictions))
            ax.plot(future_indices, future_predictions, label="예측 성취도 (다음 5일)", marker='x', linestyle='--', color='red')

            ax.set_title("LSTM 기반 학습 성취도 시계열 예측")
            ax.set_xlabel("주기 (Day/Week)")
            ax.set_ylabel("성취도 점수 (0-100)")
            ax.legend()
            ax.grid(True)

            st.pyplot(fig)

            st.markdown("---")
            st.markdown("#### AI 코치의 성취도 분석 코멘트")
            # LLM에게 예측 결과를 기반으로 코멘트를 요청
            avg_recent = np.mean(historical_scores[-5:])
            avg_predict = np.mean(future_predictions)

            if avg_predict > avg_recent:
                comment = "최근 학습 데이터와 LSTM 예측 결과에 따르면, **앞으로의 학습 성취도가 긍정적으로 향상될 것으로 예측**됩니다. 현재 학습 방식을 유지하시거나, 난이도를 한 단계 높여 도전해 보세요!"
            elif avg_predict < avg_recent - 5:  # 5점 이상 하락 시
                comment = "LSTM 예측 결과, **성취도가 다소 하락할 수 있다는 신호**가 보입니다. 학습에 사용된 자료나 방법론에 대한 깊은 이해가 부족할 수 있습니다. RAG 챗봇 기능을 활용하여 기초 개념을 다시 확인해 보는 것을 추천합니다."
            else:
                comment = "성취도는 현재 수준을 유지할 것으로 예측됩니다. 정체기가 될 수 있으니, **새로운 학습 콘텐츠 형식(예: 실습 예제 아이디어)을 생성**하여 학습에 활력을 더하는 것을 고려해 보세요."

            st.info(comment)

        except Exception as e:
            st.error(f"LSTM 모델 처리 중 오류가 발생했습니다: {e}")
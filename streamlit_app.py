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
import pandas as pd
import matplotlib.pyplot as plt

# --- 4. Streamlit UI 구성 (최상단으로 이동) ---
st.set_page_config(page_title="개인 맞춤형 AI 학습 코치", layout="wide")

# --- TensorFlow/LSTM 관련 코드 임시 제거 ---
# Streamlit Cloud 배포 성공을 위해 관련 라이브러리 및 로직을 모두 비활성화합니다.

# --- 1. 환경 설정 및 모델 초기화 ---

# Gemini API 키 설정 (secrets.toml에서 로드)
API_KEY = os.environ.get("GEMINI_API_KEY")

if 'client' not in st.session_state:
    if not API_KEY: # API_KEY가 빈 문자열이거나 None인 경우
        st.error("⚠️ 경고: GEMINI API 키가 설정되지 않았습니다. Streamlit Secrets에 'GEMINI_API_KEY'를 설정해주세요.")
        st.session_state.is_llm_ready = False
    else:
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


# --- 2. LSTM 모델 정의 (기능 임시 주석 처리) ---
# LSTM 관련 함수는 모두 주석 처리하여 에러 방지
# def load_or_train_lstm():
#     return None, None 


# --- 3. RAG 관련 함수 ---
def get_document_chunks(files):
    """업로드된 파일에서 텍스트를 로드하고 청킹합니다."""
    documents = []
    temp_dir = tempfile.mkdtemp()

    for uploaded_file in files:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 파일 형식에 따른 로더 선택
        if uploaded_file.name.endswith(".pdf"):
            # PDF 로딩 시 unstructured가 NLTK 대신 PaddlePaddle을 사용하도록 기대합니다.
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.name.endswith(".html"):
            loader = UnstructuredHTMLLoader(temp_filepath)
        else:
            loader = TextLoader(temp_filepath, encoding="utf-8")

        documents.extend(loader.load())

    # 텍스트 분할 (청킹)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)


def get_vector_store(text_chunks):
    """텍스트 청크를 임베딩하고 Vector Store를 생성합니다."""
    # FAISS가 requirements.txt에 포함되어 있어야 합니다.
    try:
        vector_store = FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)
        return vector_store
    except ImportError:
        # FAISS 설치 오류 시 사용자에게 안내
        st.error("FAISS 라이브러리를 찾을 수 없습니다. requirements.txt에 'faiss-cpu'가 포함되어 있는지 확인해 주세요.")
        return None
    except Exception as e:
        st.error(f"Vector Store 생성 중 오류 발생: {e}")
        return None


def get_rag_chain(vector_store):
    """검색 체인(ConversationalRetrievalChain)을 생성합니다."""
    if vector_store is None:
        return None
        
    return ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=vector_store.as_retriever(),
        memory=st.session_state.memory
    )


# --- 4. Streamlit UI (사이드바 및 기능 선택) ---

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
    if uploaded_files and st.session_state.is_llm_ready: # is_nltk_ready 체크 제거
        if st.button("자료 분석 시작 (RAG Indexing)", key="start_analysis"):
            with st.spinner("자료를 분석하고 학습 데이터베이스를 구축 중입니다..."):
                try:
                    text_chunks = get_document_chunks(uploaded_files)
                    vector_store = get_vector_store(text_chunks)
                    
                    if vector_store:
                        st.session_state.conversation_chain = get_rag_chain(vector_store)
                        st.session_state.is_rag_ready = True
                        st.success(f"총 {len(text_chunks)}개 청크로 학습 데이터베이스 구축 완료!")
                    else:
                         st.session_state.is_rag_ready = False
                         
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
        ["맞춤형 학습 콘텐츠 생성", "RAG 지식 챗봇", "LSTM 성취도 예측 대시보드"]
    )

# 메인 화면 제목
st.title("✨ 개인 맞춤형 AI 학습 코치")

# --- 5. 기능별 페이지 구현 ---
# (이하 생략 - 이전 코드와 동일)

if feature_selection == "RAG 지식 챗봇":
    # RAG 챗봇 기능 
    st.header("RAG 지식 챗봇 (문서 기반 Q&A)")
    st.markdown("업로드된 문서(포트폴리오, PDF 등)의 내용을 기반으로 질문에 답변합니다.")

    if st.session_state.is_rag_ready:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("학습 자료에 대해 질문해 보세요 (예: 이 문서의 핵심 기술은 무엇인가요?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("답변을 생성 중입니다..."):
                    try:
                        # RAG 체인은 ConversationalRetrievalChain을 사용하므로 system_instruction 문제 없음
                        response = st.session_state.conversation_chain.invoke({"question": prompt})
                        answer = response.get('answer', '응답을 생성할 수 없습니다.')
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"챗봇 응답 오류: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": "죄송합니다. 처리 중 오류가 발생했습니다."})
    else:
        st.error("RAG 기능을 사용하려면, 사이드바에서 학습 자료를 업로드하고 '자료 분석 시작' 버튼을 눌러주세요.")


elif feature_selection == "맞춤형 학습 콘텐츠 생성":
    # 콘텐츠 생성 기능
    st.header("맞춤형 학습 콘텐츠 생성")
    st.markdown("원하는 학습 주제, 난이도, 형식을 입력하시면 LLM이 맞춤형 콘텐츠(요약, 퀴즈)를 생성해 드립니다.")

    if st.session_state.is_llm_ready:
        topic = st.text_input("학습 주제 (예: Transformer의 Self-Attention 메커니즘)")
        level = st.selectbox("난이도", ["초급", "중급", "고급"])
        content_type = st.selectbox("콘텐츠 형식", ["핵심 요약 노트", "객관식 퀴즈 3문항", "실습 예제 아이디어"])

        if st.button("콘텐츠 생성"):
            if topic:
                # 🛠️ 수정된 부분: system_instruction을 user_prompt에 통합합니다.
                system_prompt = f"""당신은 {level} 수준의 전문 AI 코치입니다. 요청받은 주제에 대해 {content_type} 형식에 맞춰 명확하고 교육적인 콘텐츠를 생성해 주세요. 답변은 한국어로만 제공해야 합니다."""
                
                # 프롬프트 통합 (System + User)
                full_prompt = f"{system_prompt}\n\n[사용자 요청]\n주제: {topic}. 요청 형식: {content_type}."

                with st.spinner(f"{topic}에 대한 {content_type}을 생성 중입니다..."):
                    try:
                        # LLM에 요청: system_instruction 인수를 제거하고 통합된 프롬프트만 전달
                        response = st.session_state.llm.invoke(full_prompt)
                        st.success(f"**{topic}** 에 대한 **{content_type}** 결과:")
                        st.markdown(response.content)

                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("학습 주제를 입력해 주세요.")
    else:
        st.error("LLM이 초기화되지 않았습니다. API 키를 확인해 주세요.")

elif feature_selection == "LSTM 성취도 예측 대시보드":
    # LSTM 기능 비활성화 메시지 출력 (TensorFlow 오류 방지)
    st.header("LSTM 기반 학습 성취도 예측 대시보드")
    st.markdown("LSTM 기능을 사용하려면 TensorFlow 및 관련 라이브러리의 설치가 선행되어야 합니다.")
    st.error("현재 빌드 환경 문제로 인해 LSTM 기능은 잠정적으로 비활성화되었습니다. '맞춤형 학습 콘텐츠 생성' 기능을 먼저 사용해 주세요.")

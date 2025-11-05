# ========================================
# Streamlit AI 학습 코치 (NLTK 의존성 우회)
# ========================================
import streamlit as st
import os
import tempfile
import time
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader # UnstructuredHTMLLoader 다시 추가
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# [⭐핵심 수정⭐] NLTK 의존성 우회를 위한 환경 변수 설정
# unstructured가 NLTK 대신 다른 파서를 사용하도록 강제합니다.
# NLTK를 완전히 사용하지 않도록 환경 변수를 설정합니다.
os.environ["UNSTRUCTURED_FORCE_ANON_USER_AGENT"] = "true" # 사용자 에이전트 익명화
os.environ["NLTK_DATA"] = "/tmp/nltk_data" # NLTK 데이터 경로를 임시 경로로 설정

# NLTK를 명시적으로 임포트하지 않도록 코드를 정리합니다.

# ================================
# 1. LLM 및 임베딩 초기화 + 임베딩 캐시
# (이전 코드와 동일, LLM 초기화 로직 유지)
# ================================
API_KEY = os.environ.get("GEMINI_API_KEY")

if 'client' not in st.session_state:
    if not API_KEY: # API_KEY가 빈 문자열이거나 None인 경우
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

# LangChain 메모리 초기화
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# --- RAG 관련 함수 ---
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
            # PDF는 PyPDFLoader 또는 UnstructuredPDFLoader를 사용해야 하지만, 
            # PyPDFLoader는 NLTK를 쓰지 않음. HTML은 UnstructuredHTML Loader로 복구합니다.
            loader = PyPDFLoader(temp_filepath)
        elif uploaded_file.name.endswith(".html"):
            # HTML 처리를 위해 UnstructuredHTML Loader를 다시 사용합니다.
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
    key = tuple(doc.page_content for doc in text_chunks)
    if key in st.session_state.embedding_cache:
        return st.session_state.embedding_cache[key]

    try:
        vector_store = FAISS.from_documents(text_chunks, embedding=st.session_state.embeddings)
        st.session_state.embedding_cache[key] = vector_store
        return vector_store
    except Exception as e:
        st.warning(f"임베딩 요청 실패 (무료 티어 한도 초과 가능): {e}")
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
# 4. Streamlit UI
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
# 5. 기능별 페이지 구현
# ================================
if feature_selection == "RAG 지식 챗봇":
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
    st.header("맞춤형 학습 콘텐츠 생성")
    st.markdown("학습 주제와 난이도에 맞춰 콘텐츠 생성")

    if st.session_state.is_llm_ready:
        topic = st.text_input("학습 주제")
        level = st.selectbox("난이도", ["초급","중급","고급"])
        content_type = st.selectbox("콘텐츠 형식", ["핵심 요약 노트","객관식 퀴즈 3문항","실습 예제 아이디어"])

        if st.button("콘텐츠 생성"):
            if topic:
                system_prompt = f"""당신은 {level} 수준의 전문 AI 코치입니다.
요청받은 주제에 대해 {content_type} 형식에 맞춰 명확하고 교육적인 콘텐츠를 생성해 주세요.
답변은 한국어로만 제공해야 합니다."""

                user_prompt = f"주제: {topic}. 요청 형식: {content_type}."

                with st.spinner(f"{topic}에 대한 {content_type} 생성 중..."):
                    try:
                        response = st.session_state.llm.invoke(
                            user_prompt,
                            system_instruction=system_prompt
                        )
                        st.success(f"**{topic}** - **{content_type}** 결과:")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("학습 주제를 입력해 주세요.")



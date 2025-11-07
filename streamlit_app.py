# ========================================
# Streamlit AI 학습 코치 (RAG 최종 수정)
# ========================================
import streamlit as st
import os
import tempfile
import time
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
# [⭐삭제] UnstructuredHTMLLoader는 NLTK 의존성 문제로 삭제합니다.
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
# [⭐추가] 필요한 모듈 임포트 누락 수정
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup 

# ================================
# 1. LLM 및 임베딩 초기화 + 임베딩 캐시
# ================================
API_KEY = os.environ.get("GEMINI_API_KEY")

if 'llm' not in st.session_state: # 'client' 대신 'llm' 상태를 확인합니다.
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

# [⭐추가⭐] 세션 임베딩 캐시 초기화
if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}

# --- RAG 관련 함수 ---
def get_document_chunks(files):
    """업로드된 파일에서 텍스트를 로드하고 청킹합니다."""
    documents = []
    temp_dir = tempfile.mkdtemp()

    for uploaded_file in files:
        temp_filepath = os.path.join(temp_dir, uploaded_file.name)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # 파일 형식에 따른 로더 선택 (BeautifulSoup 사용 로직 유지)
        if file_extension == "pdf":
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_filepath)
            documents.extend(loader.load())
        
        elif file_extension == "html":
            # BeautifulSoup을 사용하여 HTML 태그를 제거하고 텍스트만 추출합니다.
            raw_html = uploaded_file.getvalue().decode('utf-8')
            soup = BeautifulSoup(raw_html, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)
            
            # LangChain Document 객체로 변환
            from langchain.schema.document import Document
            documents.append(Document(page_content=text_content, metadata={"source": uploaded_file.name}))


        elif file_extension == "txt": # TXT 파일 처리
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = TextLoader(temp_filepath, encoding="utf-8")
            documents.extend(loader.load())
            
        else:
            st.warning(f"'{uploaded_file.name}' 파일은 현재 PDF, TXT, HTML만 지원하여 로딩할 수 없습니다.")
            continue

    # 텍스트 분할 (청킹)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    # TextLoader/PyPDFLoader의 결과는 이미 Document 객체이므로 바로 split
    return text_splitter.split_documents(documents)


def get_vector_store(text_chunks):
    """텍스트 청크를 임베딩하고 Vector Store를 생성합니다."""
    
    # [⭐핵심 수정⭐] 문서 내용의 해시값을 키로 사용하여 캐시를 확인합니다.
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
        # 429 오류가 발생하면 사용자에게 정확하게 안내
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
                # [⭐수정 로직⭐] system_instruction 대신 프롬프트를 통합하여 LLM에 전달
                full_prompt = f"""당신은 {level} 수준의 전문 AI 코치입니다.
요청받은 주제에 대해 {content_type} 형식에 맞춰 명확하고 교육적인 콘텐츠를 생성해 주세요. 답변은 한국어로만 제공해야 합니다.

주제: {topic}
요청 형식: {content_type}"""

                with st.spinner(f"{topic}에 대한 {content_type} 생성 중..."):
                    try:
                        response = st.session_state.llm.invoke(full_prompt) # system_instruction 인수를 제거
                        st.success(f"**{topic}** - **{content_type}** 결과:")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"콘텐츠 생성 오류: {e}")
            else:
                st.warning("학습 주제를 입력해 주세요.")
    else:
        st.error("LLM이 초기화되지 않았습니다. API 키를 확인해 주세요.")

# ================================
# LSTM 성취도 예측 대시보드 (비활성화)
# ================================
elif feature_selection == "LSTM 성취도 예측 대시보드":
    st.header("LSTM 기반 학습 성취도 예측 대시보드")
    st.markdown("LSTM 기능을 사용하려면 TensorFlow 및 관련 라이브러리의 설치가 선행되어야 합니다.")
    st.error("현재 빌드 환경 문제로 인해 LSTM 기능은 잠정적으로 비활성화되었습니다. '맞춤형 학습 콘텐츠 생성' 기능을 먼저 사용해 주세요.")

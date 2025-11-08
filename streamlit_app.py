# ========================================
# Streamlit AI 학습 코치 (최종 다국어/RAG/JSON 안정화)
# ========================================
import streamlit as st
import os
import tempfile 
import time
import json # JSON 처리를 위해 추가
import re # 정규표현식(Regex)을 위해 추가

from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema.document import Document
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# ================================
# 1. JSON 안정화 함수 정의 (최상단) ⭐⭐⭐
# ================================
def clean_and_load_json(text):
    """LLM 응답 텍스트에서 JSON 객체만 정규표현식으로 추출하여 로드"""
    # 응답 텍스트에서 첫 '{'부터 마지막 '}'까지를 찾습니다.
    match = re.search(r'\{.*\}', text, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            # 추출된 문자열을 파싱 시도
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 추출된 문자열도 유효하지 않으면 None 반환
            return None
    return None

def render_interactive_quiz(quiz_data, current_lang):
    """생성된 퀴즈 데이터를 Streamlit UI로 렌더링하고 피드백을 제공합니다."""
    L = LANG[current_lang]
    
    if not quiz_data or 'quiz_questions' not in quiz_data:
        st.error(L.get("quiz_fail_structure", "퀴즈 데이터 구조가 올바르지 않습니다."))
        return

    questions = quiz_data['quiz_questions']
    num_questions = len(questions)

    if "current_question" not in st.session_state or st.session_state.current_question >= num_questions:
        st.session_state.current_question = 0
        st.session_state.quiz_results = [None] * num_questions
        st.session_state.quiz_submitted = False
        
    
    q_index = st.session_state.current_question
    q_data = questions[q_index]
    
    st.subheader(f"{q_index + 1}. {q_data['question']}")
    
    # 옵션 생성 (옵션 A, B, C, D)
    options_dict = {f"{opt['option']}": f"{opt['option']}) {opt['text']}" for opt in q_data['options']}
    options_list = list(options_dict.values())
    
    # 사용자가 선택한 답변
    selected_answer = st.radio(
        L.get("select_answer", "정답을 선택하세요"),
        options=options_list,
        key=f"q_radio_{q_index}"
    )

    col1, col2 = st.columns(2)

    # 제출 버튼
    if col1.button(L.get("check_answer", "정답 확인"), key=f"check_btn_{q_index}", disabled=st.session_state.quiz_submitted):
        # 선택된 옵션 문자(A, B, C...) 추출
        user_choice_letter = selected_answer.split(')')[0] if selected_answer else None
        correct_answer_letter = q_data['correct_answer']

        is_correct = (user_choice_letter == correct_answer_letter)
        
        st.session_state.quiz_results[q_index] = is_correct
        st.session_state.quiz_submitted = True
        
        if is_correct:
            st.success(L.get("correct_answer", "정답입니다!"))
        else:
            st.error(L.get("incorrect_answer", "오답입니다."))
        
        # 해설 표시
        st.markdown(f"**{L.get('correct_is', '정답')}: {correct_answer_letter}**")
        st.info(f"**{L.get('explanation', '해설')}:** {q_data['explanation']}")

    # 다음/결과 버튼
    if st.session_state.quiz_submitted:
        if q_index < num_questions - 1:
            if col2.button(L.get("next_question", "다음 문항"), key=f"next_btn_{q_index}"):
                st.session_state.current_question += 1
                st.session_state.quiz_submitted = False
                st.rerun()
        else:
            # 최종 결과 표시
            total_correct = st.session_state.quiz_results.count(True)
            total_questions = len(st.session_state.quiz_results)
            st.success(f"**{L.get('quiz_complete', '퀴즈 완료!')}** {L.get('score', '점수')}: {total_correct}/{total_questions}")
            if st.button(L.get("retake_quiz", "퀴즈 다시 풀기"), key="retake"):
                st.session_state.current_question = 0
                st.session_state.quiz_results = [None] * num_questions
                st.session_state.quiz_submitted = False
                st.rerun()


# ================================
# 1. RAG 핵심 함수 정의 (최상단)
# (이전 코드와 동일)
# ================================

def get_document_chunks(files):
    """업로드된 파일에서 텍스트를 로드하고 청킹합니다."""
    documents = []
    temp_dir = tempfile.mkdtemp()
    # (함수 로직 중략) ...

def get_vector_store(text_chunks):
    """텍스트 청크를 임베딩하고 Vector Store를 생성합니다."""
    # (함수 로직 중략) ...

def get_rag_chain(vector_store):
    """검색 체인(ConversationalRetrievalChain)을 생성합니다."""
    # (함수 로직 중략) ...

# ================================
# 2. LSTM 모델 정의 (최상단)
# (이전 코드와 동일)
# ================================
@st.cache_resource
def load_or_train_lstm():
    """가상의 학습 성취도 예측을 위한 LSTM 모델을 생성하고 학습합니다."""
    # (함수 로직 중략) ...
    pass # 실제 함수 로직은 여기에 위치

# ================================
# 3. 다국어 지원 딕셔너리 (Language Dictionary)
# (NameError 방지: 모든 함수 정의 후 딕셔너리 정의)
# ================================
LANG = {
    "ko": {
        "title": "개인 맞춤형 AI 학습 코치",
        "sidebar_title": "📚 AI Study Coach 설정",
        "file_uploader": "학습 자료 업로드 (PDF, TXT, HTML)",
        "button_start_analysis": "자료 분석 시작 (RAG Indexing)",
        "rag_tab": "RAG 지식 챗봇",
        "content_tab": "맞춤형 학습 콘텐츠 생성",
        "lstm_tab": "LSTM 성취도 예측 대시보드",
        "rag_header": "RAG 지식 챗봇 (문서 기반 Q&A)",
        "rag_desc": "업로드된 문서 기반으로 질문에 답변합니다。",
        "rag_input_placeholder": "학습 자료에 대해 질문해 보세요",
        "llm_error_key": "⚠️ 경고: GEMINI API 키가 설정되지 않았습니다. Streamlit Secrets에 'GEMINI_API_KEY'를 설정해주세요。",
        "llm_error_init": "LLM 초기화 오류: API 키를 확인해 주세요。",
        "content_header": "맞춤형 학습 콘텐츠 생성",
        "content_desc": "학습 주제와 난이도에 맞춰 콘텐츠 생성",
        "topic_label": "학습 주제",
        "level_label": "난이도",
        "content_type_label": "콘텐츠 형식",
        "level_options": ["초급", "중급", "고급"],
        "content_options": ["핵심 요약 노트", "객관식 퀴즈 3문항", "실습 예제 아이디어"],
        "button_generate": "콘텐츠 생성",
        "warning_topic": "학습 주제를 입력해 주세요。",
        "lstm_header": "LSTM 기반 학습 성취도 예측 대시보드",
        "lstm_desc": "가상의 과거 퀴즈 점수 데이터를 바탕으로 LSTM 모델을 훈련하고 미래 성취도를 예측하여 보여줍니다。",
        "lstm_disabled_error": "현재 빌드 환경 문제로 인해 LSTM 기능은 잠정적으로 비활성화되었습니다. '맞춤형 학습 콘텐츠 생성' 기능을 먼저 사용해 주세요。",
        "lang_select": "언어 선택",
        "embed_success": "총 {count}개 청크로 학습 DB 구축 완료!",
        "embed_fail": "임베딩 실패: 무료 티어 한도 초과 또는 네트워크 문제。",
        "warning_no_files": "먼저 학습 자료를 업로드하세요。",
        "warning_rag_not_ready": "RAG가 준비되지 않았습니다. 학습 자료를 업로드하고 분석하세요。",
        "quiz_fail_structure": "퀴즈 데이터 구조가 올바르지 않습니다.",
        "select_answer": "정답을 선택하세요",
        "check_answer": "정답 확인",
        "next_question": "다음 문항",
        "correct_answer": "정답입니다! 🎉",
        "incorrect_answer": "오답입니다. 😞",
        "correct_is": "정답",
        "explanation": "해설",
        "quiz_complete": "퀴즈 완료!",
        "score": "점수",
        "retake_quiz": "퀴즈 다시 풀기"
    },

    
    "en": {
        "title": "Personalized AI Study Coach",
        "sidebar_title": "📚 AI Study Coach Settings",
        "file_uploader": "Upload Study Materials (PDF, TXT, HTML)",
        "button_start_analysis": "Start Analysis (RAG Indexing)",
        "rag_tab": "RAG Knowledge Chatbot",
        "content_tab": "Custom Content Generation",
        "lstm_tab": "LSTM Achievement Prediction",
        "rag_header": "RAG Knowledge Chatbot (Document Q&A)",
        "rag_desc": "Answers questions based on the uploaded documents.",
        "rag_input_placeholder": "Ask a question about your study materials",
        "llm_error_key": "⚠️ Warning: GEMINI API Key is not set. Please set 'GEMINI_API_KEY' in Streamlit Secrets.",
        "llm_error_init": "LLM initialization error: Please check your API key.",
        "content_header": "Custom Learning Content Generation",
        "content_desc": "Generate content tailored to your topic and difficulty.",
        "topic_label": "Learning Topic",
        "level_label": "Difficulty",
        "content_type_label": "Content Type",
        "level_options": ["Beginner", "Intermediate", "Advanced"],
        "content_options": ["Key Summary Note", "3 Multiple-Choice Questions", "Practical Example Idea"],
        "button_generate": "Generate Content",
        "warning_topic": "Please enter a learning topic.",
        "lstm_header": "LSTM Based Achievement Prediction",
        "lstm_desc": "Trains an LSTM model on hypothetical past quiz scores to predict future achievement.",
        "lstm_disabled_error": "The LSTM feature is temporarily disabled due to build environment issues. Please use the 'Custom Content Generation' feature first.",
        "lang_select": "Select Language",
        "embed_success": "Learning DB built with {count} chunks!",
        "embed_fail": "Embedding failed: Free tier quota exceeded or network issue.",
        "warning_no_files": "Please upload study materials first.",
        "warning_rag_not_ready": "RAG is not ready. Upload materials and click Start Analysis."
        "quiz_fail_structure": "Loops for quiz datas are not correct.",
        "select_answer": "Select answer",
        "check_answer": "Confirm answer",
        "next_question": "Next Quiz",
        "correct_answer": "Correct! 🎉",
        "incorrect_answer": "Incorrect. 😞",
        "correct_is": "Correct answer",
        "explanation": "Details",
        "quiz_complete": "Quiz completed!",
        "score": "Scores",
        "retake_quiz": "Retake quize"
    },
    "ja": {
        "title": "パーソナライズAI学習コーチ",
        "sidebar_title": "📚 AI学習コーチ設定",
        "file_uploader": "学習資料をアップロード (PDF, TXT, HTML)",
        "button_start_analysis": "資料分析開始 (RAGインデックス作成)",
        "rag_tab": "RAG知識チャットボット",
        "content_tab": "カスタムコンテンツ生成",
        "lstm_tab": "LSTM達成度予測ダッシュボード",
        "rag_header": "RAG知識チャットボット (ドキュメントQ&A)",
        "rag_desc": "アップロードされたドキュメントに基づいて質問に回答します。",
        "rag_input_placeholder": "学習資料について質問してください",
        "llm_error_key": "⚠️ 警告: GEMINI APIキーが設定されていません。Streamlit Secretsに'GEMINI_API_KEY'を設定してください。",
        "llm_error_init": "LLM初期化エラー：APIキーを確認してください。",
        "content_header": "カスタム学習コンテンツ生成",
        "content_desc": "学習テーマと難易度に合わせてコンテンツを生成します。",
        "topic_label": "学習テーマ",
        "level_label": "難易度",
        "content_type_label": "コンテンツ形式",
        "level_options": ["初級", "中級", "上級"],
        "content_options": ["核心要約ノート", "選択式クイズ3問", "実践例のアイデア"],
        "button_generate": "コンテンツ生成",
        "warning_topic": "学習テーマを入力してください。",
        "lstm_header": "LSTMベース達成度予測ダッシュボード",
        "lstm_desc": "仮想の過去クイズスコアデータに基づき、LSTMモデルを訓練して将来の達成度を予測し表示します。",
        "lstm_disabled_error": "現在、ビルド環境の問題によりLSTM機能は一時的に無効化されています。「カスタムコンテンツ生成」機能を先にご利用ください。",
        "lang_select": "言語選択",
        "embed_success": "全{count}チャンクで学習DB構築完了!",
        "embed_fail": "埋め込み失敗: フリーティアのクォータ超過またはネットワークの問題。",
        "warning_no_files": "まず学習資料をアップロードしてください。",
        "warning_rag_not_ready": "RAGの準備ができていません。資料をアップロードし、分析開始ボタンを押してください。"
        "quiz_fail_structure": "クイズのデーターの構造が正しくありません。",
        "select_answer": "正解を選んでください",
        "check_answer": "正解を確認する",
        "next_question": "次のクイズ",
        "correct_answer": "正解です! 🎉",
        "incorrect_answer": "不正解です。 😞",
        "correct_is": "正解は。。",
        "explanation": "解説",
        "quiz_complete": "すべてのクイズを完了しました!",
        "score": "点数",
        "retake_quiz": "クイズを再挑戦する"
    }
}

# ================================
# 4. 세션 상태 및 LLM 초기화 로직
# ================================
# 초기 세션 상태 설정
if 'language' not in st.session_state:
    st.session_state.language = 'ko'
L = LANG[st.session_state.language] 
if 'uploaded_files_state' not in st.session_state:
    st.session_state.uploaded_files_state = None
if 'is_llm_ready' not in st.session_state:
    st.session_state.is_llm_ready = False
if 'is_rag_ready' not in st.session_state:
    st.session_state.is_rag_ready = False

# LLM 및 임베딩 초기화
API_KEY = os.environ.get("GEMINI_API_KEY")

if 'llm' not in st.session_state: 
    if not API_KEY:
        st.error(L["llm_error_key"])
    else:
        try:
            st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=API_KEY)
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
            st.session_state.is_llm_ready = True
        except Exception as e:
            st.error(f"{L['llm_error_init']} {e}")
            st.session_state.is_llm_ready = False

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "embedding_cache" not in st.session_state:
    st.session_state.embedding_cache = {}


# ================================
# 5. Streamlit UI (최종 NameError 해결)
# ================================
st.set_page_config(page_title=L["title"], layout="wide") 

with st.sidebar:
    selected_lang_key = st.selectbox(
        L["lang_select"],
        options=['ko', 'en', 'ja'],
        index=['ko', 'en', 'ja'].index(st.session_state.language), 
        format_func=lambda x: {"ko": "한국어", "en": "English", "ja": "日本語"}[x],
    )
    
    if selected_lang_key != st.session_state.language:
        st.session_state.language = selected_lang_key
        st.rerun() 
    
    # L 변수 재설정 (언어 선택 후 UI 업데이트)
    L = LANG[st.session_state.language] 
    
    st.title(L["sidebar_title"])
    st.markdown("---")
    
    uploaded_files_widget = st.file_uploader(
        L["file_uploader"],
        type=["pdf","txt","html"],
        accept_multiple_files=True
    )
    
    # 세션 상태 업데이트
    if uploaded_files_widget:
        st.session_state.uploaded_files_state = uploaded_files_widget
    elif 'uploaded_files_state' not in st.session_state:
        st.session_state.uploaded_files_state = None
    
    files_to_process = st.session_state.uploaded_files_state if st.session_state.uploaded_files_state else []
    
    if files_to_process and st.session_state.is_llm_ready:
        if st.button(L["button_start_analysis"], key="start_analysis"):
            with st.spinner(f"자료 분석 및 학습 DB 구축 중..."):
                # ⭐ NameError 해결: 함수 정의가 최상단에 있어 이제 안전함 ⭐
                text_chunks = get_document_chunks(files_to_process)
                vector_store = get_vector_store(text_chunks)
                
                if vector_store:
                    st.session_state.conversation_chain = get_rag_chain(vector_store)
                    st.session_state.is_rag_ready = True
                    st.success(L["embed_success"].format(count=len(text_chunks)))
                else:
                    st.session_state.is_rag_ready = False
                    st.error(L["embed_fail"])

    else:
        st.session_state.is_rag_ready = False
        st.warning(L.get("warning_no_files", "먼저 학습 자료를 업로드하세요.")) 

    st.markdown("---")
    feature_selection = st.radio(
        L["content_tab"], 
        [L["rag_tab"], L["content_tab"], L["lstm_tab"]]
    )

st.title(L["title"])

# ================================
# 6. 기능별 페이지 구현 (⭐텍스트 요소 모두 L[]로 변경⭐)
# ================================
if feature_selection == L["rag_tab"]:
    st.header(L["rag_header"])
    st.markdown(L["rag_desc"])
    if st.session_state.is_rag_ready and st.session_state.conversation_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(L["rag_input_placeholder"]):
            st.session_state.messages.append({"role":"user","content":prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(f"답변 생성 중..." if st.session_state.language == 'ko' else "Generating response..."):
                    try:
                        response = st.session_state.conversation_chain.invoke({"question":prompt})
                        answer = response.get('answer','응답을 생성할 수 없습니다.' if st.session_state.language == 'ko' else 'Could not generate response.')
                        st.markdown(answer)
                        st.session_state.messages.append({"role":"assistant","content":answer})
                    except Exception as e:
                        st.error(f"챗봇 오류: {e}")
                        st.session_state.messages.append({"role":"assistant","content":"오류 발생" if st.session_state.language == 'ko' else "An error occurred"})
    else:
        st.warning(L["warning_rag_not_ready"])

elif feature_selection == L["content_tab"]:
    st.header(L["content_header"])
    st.markdown(L["content_desc"])

    if st.session_state.is_llm_ready:
        topic = st.text_input(L["topic_label"])
        
        level = st.selectbox(L["level_label"], L["level_options"])
        content_type = st.selectbox(L["content_type_label"], L["content_options"])

        if st.button(L["button_generate"]):
            if topic:
                target_lang = {"ko": "Korean", "en": "English", "ja": "Japanese"}[st.session_state.language]
                
                full_prompt = f"""You are a professional AI coach at the {level} level.
Please generate clear and educational content in the requested {content_type} format based on the topic.
The response MUST be strictly in {target_lang}.

Topic: {topic}
Requested Format: {content_type}"""

                with st.spinner(f"Generating {content_type} for {topic}..."):
                    try:
                        response = st.session_state.llm.invoke(full_prompt)
                        st.success(f"**{topic}** - **{content_type}** Result:")
                        st.markdown(response.content)
                    except Exception as e:
                        st.error(f"Content Generation Error: {e}")
            else:
                st.warning(L["warning_topic"])
    else:
        st.error(L["llm_error_init"])

elif feature_selection == L["lstm_tab"]:
    st.header(L["lstm_header"])
    st.markdown(L["lstm_desc"])

    with st.spinner(f"LSTM model loading/training..." if st.session_state.language != 'ko' else "LSTM 모델을 로드/학습 중입니다..."):
        try:
            # 1. 모델 로드 및 데이터 생성
            lstm_model, historical_scores = load_or_train_lstm()
            st.success("LSTM Model Ready!")

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

            ax.plot(range(len(historical_scores)), historical_scores, label="Past Quiz Scores (Hypothetical)", marker='o', linestyle='-', color='blue')
            future_indices = range(len(historical_scores), len(historical_scores) + len(future_predictions))
            ax.plot(future_indices, future_predictions, label="Predicted Achievement (Next 5 Days)", marker='x', linestyle='--', color='red')

            ax.set_title(L["lstm_header"])
            ax.set_xlabel(L["topic_label"])
            ax.set_ylabel("Achievement Score (0-100)")
            ax.legend()
            st.pyplot(fig)

            # 4. LLM 분석 코멘트
            st.markdown("---")
            st.markdown("#### AI Coach Analysis Comment")
            
            avg_recent = np.mean(historical_scores[-5:])
            avg_predict = np.mean(future_predictions)
            
            # (이 로직은 언어 딕셔너리로 대체하기가 복잡하여 임시로 영어/한국어/일본어 분기로 처리)
            if st.session_state.language == 'ko':
                if avg_predict > avg_recent:
                    comment = "최근 학습 데이터와 LSTM 예측 결과에 따르면, **앞으로의 학습 성취도가 긍정적으로 향상될 것으로 예측**됩니다. 현재 학습 방식을 유지하시거나, 난이도를 한 단계 높여 도전해 보세요!"
                elif avg_predict < avg_recent - 5:
                    comment = "LSTM 예측 결과, **성취도가 다소 하락할 수 있다는 신호**가 보입니다. 학습에 사용된 자료나 방법론에 대한 깊은 이해가 부족할 수 있습니다. RAG 챗봇 기능을 활용하여 기초 개념을 다시 확인해 보는 것을 추천합니다."
                else:
                    comment = "성취도는 현재 수준을 유지할 것으로 예측됩니다. 정체기가 될 수 있으니, **새로운 학습 콘텐츠 형식(예: 실습 예제 아이디어)을 생성**하여 학습에 활력을 더하는 것을 고려해 보세요。"
            elif st.session_state.language == 'en': # English
                if avg_predict > avg_recent:
                    comment = "Based on recent learning data and LSTM prediction, **your achievement is projected to improve positively**. Maintain your current study methods or consider increasing the difficulty level."
                elif avg_predict < avg_recent - 5:
                    comment = "LSTM prediction suggests a **potential drop in achievement**. Your understanding of fundamental concepts may be lacking. Use the RAG Chatbot to review foundational knowledge."
                else:
                    comment = "Achievement is expected to remain stable. Consider generating **new content types (e.g., Practical Example Ideas)** to revitalize your learning during this plateau."
            else: # Japanese
                 if avg_predict > avg_recent:
                    comment = "最近の学習データとLSTM予測結果に基づき、**今後の達成度はポジティブに向上すると予測**されます。現在の学習方法を維持するか、難易度を一段階上げて挑戦することを検討してください。"
                 elif avg_predict < avg_recent - 5:
                    comment = "LSTM予測の結果、**達成度がやや低下する可能性**が示されました。学習資料や方法論の基礎理解が不足しているかもしれません。RAGチャットボット機能を利用して、基本概念を再確認することをお勧めします。"
                 else:
                    comment = "達成度は現状維持と予測されます。停滞期になる可能性があるため、**新しいコンテンツ形式（例：実践例のアイデア）を生成**し、学習に活力を与えることを検討してください。"


            st.info(comment)

        except Exception as e:
            st.error(f"LSTM Model Processing Error: {e}")
            st.markdown(f'<div style="background-color: #fce4e4; color: #cc0000; padding: 10px; border-radius: 5px;">{L["lstm_disabled_error"]}</div>', unsafe_allow_html=True)




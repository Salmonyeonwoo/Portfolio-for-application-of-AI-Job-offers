# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

# -------------------------------
# GPU / Platform 설정
riva_target_gpu_family="non-tegra"
riva_tegra_platform="orin"

# -------------------------------
# Riva 서비스 활성화 (Step1~6 영어 전용 기준)
service_enabled_asr=true
service_enabled_nlp=true
service_enabled_tts=true

# -------------------------------
# Riva Enterprise (선택)
# RIVA_API_KEY=<본인 NGC API KEY>
# RIVA_API_NGC_ORG=<본인 NGC ORG>
# RIVA_EULA=accept

# -------------------------------
# 언어 코드 (영어 전용)
language_code=("en-US")

# ASR 모델
asr_acoustic_model=("conformer")

# GPU 지정
gpus_to_use="device=0"

# 모델 배포 키
MODEL_DEPLOY_KEY="tlt_encode"

# -------------------------------
# 모델 저장 위치
# 과제에서 생성한 새 모델 리포지토리 경로를 지정해야 Step1 점수 정상
riva_model_loc="/dli_workspace/riva-assessment-model-repo"  # ← 과제 지시대로 수정
if [[ $riva_target_gpu_family == "tegra" ]]; then
    riva_model_loc="`pwd`/model_repository"
fi

# 기존 RMIR 사용 여부
use_existing_rmirs=false

# API 포트
riva_speech_api_port="50051"

# -------------------------------
# NGC 조직 설정
riva_ngc_org="nvidia"
riva_ngc_team="riva"
riva_ngc_image_version="2.8.1"
riva_ngc_model_version="2.8.0"

# -------------------------------
# ASR 모델 정의
models_asr=()
for lang_code in ${language_code[@]}; do
    modified_lang_code="${lang_code/-/_}"
    modified_lang_code=${modified_lang_code,,}
    decoder=""
    models_asr+=(
        "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_${asr_acoustic_model}_${modified_lang_code}_str${decoder}:${riva_ngc_model_version}"
        "${riva_ngc_org}/${riva_ngc_team}/rmir_asr_${asr_acoustic_model}_${modified_lang_code}_ofl${decoder}:${riva_ngc_model_version}"
    )
done

# -------------------------------
# NLP 모델 정의
models_nlp=(
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_punctuation_bert_base_en_us:${riva_ngc_model_version}"
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_named_entity_recognition_bert_base:${riva_ngc_model_version}"
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_intent_slot_bert_base:${riva_ngc_model_version}"
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_question_answering_bert_base:${riva_ngc_model_version}"
    "${riva_ngc_org}/${riva_ngc_team}/rmir_nlp_text_classification_bert_base:${riva_ngc_model_version}"
)

# -------------------------------
# TTS 모델 정의
models_tts=(
    "${riva_ngc_org}/${riva_ngc_team}/rmir_tts_fastpitch_hifigan_en_us_ipa:${riva_ngc_model_version}"
)

# -------------------------------
# Docker 이미지
NGC_TARGET=${riva_ngc_org}
if [[ ! -z ${riva_ngc_team} ]]; then
  NGC_TARGET="${NGC_TARGET}/${riva_ngc_team}"
fi

image_speech_api="nvcr.io/${NGC_TARGET}/riva-speech:${riva_ngc_image_version}"
image_init_speech="nvcr.io/${NGC_TARGET}/riva-speech:${riva_ngc_image_version}-servicemaker"

# -------------------------------
# Daemon 이름
riva_daemon_speech="riva-speech"
if [[ $riva_target_gpu_family != "tegra" ]]; then
    riva_daemon_client="riva-client"
fi

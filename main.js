// Voice_of_AI_features.html에서 옮겨온 JavaScript 코드
// 이 파일에서는 더 이상 API 키를 직접 관리하지 않습니다.

document.addEventListener('DOMContentLoaded', () => {
    // 프레젠테이션 슬라이드 로직 등 기존 JavaScript 코드가 여기에 위치합니다.
    const slides = document.querySelectorAll('.slide');
    let currentSlideIndex = 0;

    function showSlide(index) {
        // 슬라이드 표시 로직
    }

    // ... 기타 이벤트 리스너 및 함수들 ...

    // 예시: API를 호출하는 함수
    async function callGeminiAPI() {
        // apiKey 변수는 이제 config.js 파일에서 전역으로 가져옵니다.
        // 이 파일에는 더 이상 키가 존재하지 않습니다.
        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;

        // fetch 로직...
        console.log("API를 호출합니다. 사용된 키:", apiKey ? "키 있음" : "키 없음");
    }

    // 함수 호출 예시
    callGeminiAPI();
});

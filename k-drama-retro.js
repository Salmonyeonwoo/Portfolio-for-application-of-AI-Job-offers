// K-드라마 복고 전략 프레젠테이션의 모든 JavaScript 로직을 담고 있는 파일입니다.

document.addEventListener('DOMContentLoaded', () => {
    // --- 슬라이드 네비게이션 로직 ---
    const slides = document.querySelectorAll('.slide');
    let currentSlideIndex = 0;

    function updateSlideVisibility() {
        slides.forEach((slide, index) => {
            slide.classList.remove('active');
            // active가 아닐 때의 스타일 초기화
            slide.style.position = 'absolute';
            slide.style.transform = 'translateX(100%)';
            slide.style.opacity = '0';
            slide.style.visibility = 'hidden';

            if (index === currentSlideIndex) {
                slide.classList.add('active');
                // active일 때의 스타일 적용
                slide.style.position = 'relative'; 
                slide.style.transform = 'translateX(0)';
                slide.style.opacity = '1';
                slide.style.visibility = 'visible';
            }
        });
        
        // 버튼 상태 업데이트
        const prevButton = document.getElementById('prev-slide');
        const nextButton = document.getElementById('next-slide');

        if(prevButton && nextButton) {
            prevButton.disabled = currentSlideIndex === 0;
            prevButton.style.opacity = currentSlideIndex === 0 ? 0.5 : 1;
            
            nextButton.disabled = currentSlideIndex === slides.length - 1;
            nextButton.style.opacity = currentSlideIndex === slides.length - 1 ? 0.5 : 1;
        }
    }

    function nextSlide() {
        if (currentSlideIndex < slides.length - 1) {
            currentSlideIndex++;
            updateSlideVisibility();
        }
    }

    function prevSlide() {
        if (currentSlideIndex > 0) {
            currentSlideIndex--;
            updateSlideVisibility();
        }
    }

    const nextButton = document.getElementById('next-slide');
    const prevButton = document.getElementById('prev-slide');

    if (nextButton && prevButton) {
        nextButton.addEventListener('click', nextSlide);
        prevButton.addEventListener('click', prevSlide);
    }

    // 키보드 네비게이션
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight') {
            nextSlide();
        } else if (e.key === 'ArrowLeft') {
            prevSlide();
        }
    });

    // 프레젠테이션 초기화
    updateSlideVisibility(); 

    // --- Gemini API 호출 로직 ---
    // 전역 스코프에 generateVibe 함수를 노출시켜 HTML의 onclick에서 찾을 수 있도록 함
    window.generateVibe = async function() {
        // API 키는 config.js에 정의된 API_KEY 변수를 사용합니다.
        // 이 파일이 config.js 보다 나중에 로드되므로 API_KEY에 접근 가능합니다.
        if (typeof API_KEY === 'undefined' || API_KEY === "") {
            console.error("API key is not defined. Please check config.js");
            const outputArea = document.getElementById('output-area');
            outputArea.innerHTML = '<p class="text-red-400">API 키가 설정되지 않았습니다. config.js 파일을 확인해주세요.</p>';
            return;
        }

        const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${API_KEY}`;
        const year = document.getElementById('input-year').value;
        const theme = document.getElementById('input-theme').value;
        const outputArea = document.getElementById('output-area');
        const sourcesArea = document.getElementById('sources-area');
        
        outputArea.innerHTML = '<div class="flex items-center justify-center h-full"><div class="loading-spinner"></div><span class="ml-4 text-pink-400">AI가 그 시절 감성을 찾고 있습니다...</span></div>';
        sourcesArea.innerHTML = '';

        const systemPrompt = "당신은 한국의 레트로 감성을 전문적으로 재현하는 카피라이터입니다. 선택된 연도와 테마에 맞는, 당시 유행하던 말투와 감성을 사용하여 톡톡 튀는 문구(2문장 이내)를 생성해 주세요. 당시 유행어나 표현을 적극적으로 사용해야 합니다. 결과는 텍스트만 출력해야 합니다.";
        const userQuery = `${year}년, 주제: ${theme}. 이 두 가지를 반영한 레트로 감성 문구를 만들어 줘. 특히 ${year}년에 유행했던 말투와 표현을 반드시 포함해줘.`;

        const payload = {
            contents: [{ parts: [{ text: userQuery }] }],
            tools: [{ "google_search": {} }],
            systemInstruction: { parts: [{ text: systemPrompt }] },
        };

        try {
            const response = await fetchWithBackoff(apiUrl, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            const candidate = result.candidates?.[0];

            if (candidate && candidate.content?.parts?.[0]?.text) {
                const text = candidate.content.parts[0].text;
                let sources = [];
                const groundingMetadata = candidate.groundingMetadata;

                if (groundingMetadata && groundingMetadata.groundingAttributions) {
                    sources = groundingMetadata.groundingAttributions
                        .map(attr => ({ uri: attr.web?.uri, title: attr.web?.title }))
                        .filter(s => s.uri && s.title);
                }
                
                outputArea.innerHTML = `<p class="text-xl font-medium text-white">${text}</p>`;

                if (sources.length > 0) {
                    const sourcesHtml = sources.map((s, i) => 
                        `<a href="${s.uri}" target="_blank" class="text-blue-400 hover:text-blue-300 underline block">${i + 1}. ${s.title}</a>`
                    ).join('');
                    sourcesArea.innerHTML = `<strong>참고 자료 (고증):</strong><br>${sourcesHtml}`;
                }
            } else {
                outputArea.innerHTML = '<p class="text-red-400">응답 생성에 실패했습니다. 다시 시도해 주세요.</p>';
            }
        } catch (error) {
            console.error("Gemini API Error:", error);
            outputArea.innerHTML = `<p class="text-red-400">API 호출 중 오류 발생: ${error.message}</p>`;
        }
    }
    
    // Exponential Backoff 유틸리티 함수
    async function fetchWithBackoff(url, options, maxRetries = 5) {
        for (let i = 0; i < maxRetries; i++) {
            try {
                const response = await fetch(url, options);
                if (response.status !== 429) {
                    return response;
                }
                const delay = Math.pow(2, i) * 1000 + Math.random() * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            } catch (error) {
                console.error(`Fetch attempt ${i + 1} failed:`, error);
                const delay = Math.pow(2, i) * 1000 + Math.random() * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
        throw new Error("API call failed after maximum retries.");
    }
});

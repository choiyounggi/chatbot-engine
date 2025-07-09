# LASA 기반 챗봇 엔진

이 프로젝트는 LASA(Listen-Analyze-Solve-Answer) 프레임워크를 적용한 챗봇 엔진입니다.

## LASA 프레임워크란?

LASA는 효과적인 챗봇 시스템을 구축하기 위한 프레임워크로, 다음 4단계로 구성됩니다:

1. **Listen (듣기)**: 사용자 입력을 수신하고 전처리합니다.
2. **Analyze (분석)**: 사용자 의도와 관련 엔티티를 추출합니다.
3. **Solve (해결)**: 분석된 의도에 따라 문제를 해결하고 응답 전략을 수립합니다.
4. **Answer (응답)**: 최종 응답을 생성하여 사용자에게 전달합니다.

## 시스템 구조

```
com.yk.chatbot
├── controller
│   └── ChatbotController.java    # REST API 컨트롤러
├── dto
│   ├── ChatRequest.java          # 요청 DTO
│   └── ChatResponse.java         # 응답 DTO
├── lasa                          # LASA 프레임워크 인터페이스
│   ├── Listen.java               # 듣기 인터페이스
│   ├── Analyze.java              # 분석 인터페이스
│   ├── AnalysisResult.java       # 분석 결과 클래스
│   ├── Solve.java                # 해결 인터페이스
│   ├── SolutionResult.java       # 해결 결과 클래스
│   ├── Answer.java               # 응답 인터페이스
│   └── impl                      # 구현체 패키지
│       ├── SimpleListener.java   # 기본 리스너 구현
│       ├── SimpleAnalyzer.java   # 기본 분석기 구현
│       ├── SimpleSolver.java     # 기본 해결기 구현
│       └── SimpleAnswerer.java   # 기본 응답기 구현
└── service
    └── LasaChatbotService.java   # LASA 기반 챗봇 서비스
```

## API 사용 방법

### 요청

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "오늘 날씨 어때?"
}
```

### 응답

```json
{
  "reply": "현재 서울의 날씨는 맑음입니다. 기온은 23도 입니다.",
  "intent": "weather"
}
```

## 확장 방법

새로운 의도를 추가하거나 기능을 확장하고 싶다면:

1. `SimpleAnalyzer` 클래스에서 새로운 의도 패턴을 추가
2. `SimpleSolver` 클래스에서 새로운 의도에 대한 처리 로직 구현
3. 필요시 새로운 DTO, 엔티티 타입 등을 추가

## 의존성

- Spring Boot
- Lombok

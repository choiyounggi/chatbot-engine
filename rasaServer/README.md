# Rasa 한국어 NLU 서버

이 디렉토리는 Spring Boot 챗봇 엔진과 연동되는 Rasa NLU 서버를 포함하고 있습니다. 
기존 KoalaNLP 내부 분석기 대신 외부 Rasa 서버를 활용하여 한국어 자연어 처리를 수행합니다.

## 설치 방법

### 1. Python 환경 설정

Python 3.8 이상이 필요합니다. 가상 환경 사용을 권장합니다.

```bash
# 가상 환경 생성 (Windows)
python -m venv venv

# 가상 환경 활성화 (Windows)
venv\Scripts\activate

# 가상 환경 활성화 (Linux/Mac)
# source venv/bin/activate
```

### 2. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

## Rasa 모델 학습

학습 데이터를 기반으로 Rasa 모델을 학습합니다.

```bash
rasa train
```

## Rasa 서버 실행

서버를 5005 포트에서 실행합니다.

```bash
rasa run --enable-api --cors "*" --port 5005
```

또는 제공된 배치 스크립트를 사용할 수도 있습니다.

```bash
start_rasa.bat
```

## API 테스트

서버가 실행되면 다음 엔드포인트로 메시지를 전송하여 테스트할 수 있습니다:

```
POST http://localhost:5005/model/parse
Content-Type: application/json

{
  "text": "서울 날씨 어때요?"
}
```

예상되는 응답:

```json
{
  "intent": {
    "name": "weather",
    "confidence": 0.98
  },
  "entities": [
    {
      "entity": "location",
      "value": "서울",
      "start": 0,
      "end": 2
    }
  ],
  "text": "서울 날씨 어때요?"
}
```

## 주요 파일 설명

- `config.yml`: NLU 파이프라인 및 정책 설정
- `domain.yml`: 봇의 의도, 엔티티, 응답 등 정의
- `data/nlu.yml`: 학습을 위한 예시 문장
- `data/stories.yml`: 대화 흐름 정의
- `data/rules.yml`: 규칙 기반 응답 정의
- `endpoints.yml`: 외부 엔드포인트 설정

## Spring Boot 연동

이 Rasa 서버는 `RasaAnalyzer` 클래스를 통해 Spring Boot 애플리케이션과 연동됩니다. 
`RasaClientService`에서 이 서버의 `/model/parse` 엔드포인트로 요청을 보내 의도와 엔티티를 추출합니다.

# chatcore

[English](README.md) | **한국어**

내 컴퓨터에서 돌아가는, LLM 없는 시나리오형 챗봇 엔진이에요. 봇은 데이터예요. 예문이 달린 의도,
동의어가 달린 엔티티, 시나리오, 응답 템플릿으로 이뤄져요. 엔진은 한국어 입력을 문장 임베딩 모델(학습하지 않음)과
작은 분류기로 이해하고, 시나리오를 정해진 대로 진행해요. 이 데이터를 편집하는 빌더 UI는 나중에 만들어요.

> 상태: 빌더 단계 E-E — 한국어 NLU를 붙인 시나리오 엔진을 HTTP API(`make serve`)와 터미널 채팅(`make chat`)으로 쓸 수 있어요. 빌더 API(`/v1/builder/...`)로 초안 저장(ETag로 동시 편집 보호), 백그라운드 학습(SSE로 진행 상황), 테스트 빌드와 테스트 대화, 메모·버전 번호가 붙는 배포, 되돌리기, 작업이력, 사용자 발화를 문장별로 묶은 학습대기(예문으로 추가·무시), HTTP 스킬(URL로 정하는 액션, 쓰기 전용 비밀값, 내부 주소 차단, 테스트 실행·오류 내역), 지식베이스(FAQ·문서·PDF, 블록에 없는 질문에 답하거나 답안 추천, 재순위 모델로 답할지 판단, 봇과 따로 버전을 만들어 함께 배포), 분석(사용자·세션·블록 호출·지식베이스 응답률·잔존율, 한국 시간 일별), 봇 만들기(빈 봇·기존 봇 복사)와 7일 대기 후 영구 삭제, 머신러닝 설정(봇마다 임계값을 정하는 유사도 5단계, 정확 일치만, 헷갈리는 의도 경고, 핵심어 일치)을 써요. 봇에는 카카오식 웰컴·탈출 블록, 세션 시간 초과, 선택·검증 파라미터, 블록으로 바로 가는 바로연결·버튼, 이미지·카드·리스트·케로셀 응답, 시스템 엔티티 16종(날짜·시각·숫자·기간·서수·나이·퍼센트·URL·단위·임의 텍스트)을 쓸 수 있어요. 빌더 API는 `CHATCORE_ADMIN_TOKEN`을 설정하면 그 토큰으로만 열려요. 빌더 화면은 [chatbot-builder](https://github.com/choiyounggi/chatbot-builder)에서 만들어요.
> 설계와 로드맵: [docs/engine-core-plan.md](docs/engine-core-plan.md)

## 시작하기

[uv](https://docs.astral.sh/uv/)와 `make`가 필요해요. Python 3.14는 uv가 설치해요.

```sh
make setup   # uv sync --locked
make check   # ruff + mypy(strict, src/) + pytest·커버리지 — CI와 같음
make test-db # Postgres 저장소 테스트까지 (5432가 쓰이고 있으면 CHATCORE_DB_PORT=…)
```

선택:

```sh
make db-up   # Postgres 17 + pgvector, 127.0.0.1:5432 (Docker), Phase 2부터 사용
make bench   # Phase 0 NLU 벤치마크 — Apple Silicon, 모델 다운로드, 약 10분
```

## 구조

| 경로 | 내용 |
|---|---|
| `src/chatcore/` | 엔진 패키지: `bot/`(봇 정의·로더), `dialogue/`(명령·이벤트·상태·시나리오 엔진·턴 처리), `store/`(메모리·Postgres) |
| `bots/sample/` | 샘플 봇: `bot.yaml`(시나리오·응답), `nlu.yaml`(의도·엔티티) |
| `bench/` | NLU 벤치마크와 테스트셋 |
| `docs/` | 계획과 벤치마크 보고서 |

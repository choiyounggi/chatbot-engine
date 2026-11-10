# chatcore

[English](README.md) | **한국어**

내 컴퓨터에서 돌아가는, LLM 없는 시나리오형 챗봇 엔진이에요. 봇은 데이터예요. 예문이 달린 의도,
동의어가 달린 엔티티, 시나리오, 응답 템플릿으로 이뤄져요. 엔진은 한국어 입력을 문장 임베딩 모델(학습하지 않음)과
작은 분류기로 이해하고, 시나리오를 정해진 대로 진행해요. 이 데이터를 편집하는 빌더 UI는 나중에 만들어요.

> 상태: 빌더 단계 E-A — 한국어 NLU를 붙인 시나리오 엔진을 HTTP API(`make serve`)와 터미널 채팅(`make chat`)으로 쓸 수 있어요. 빌더 API(`/v1/builder/...`)로 초안 저장(ETag로 동시 편집 보호), 백그라운드 학습(SSE로 진행 상황), 테스트 빌드와 테스트 대화, 메모·버전 번호가 붙는 배포, 되돌리기, 오인식 문장 조회를 해요. 빌더 API는 `CHATCORE_ADMIN_TOKEN`을 설정하면 그 토큰으로만 열려요. 빌더 화면은 [chatbot-builder](https://github.com/choiyounggi/chatbot-builder)에서 만들어요.
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

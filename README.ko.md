# chatcore

[English](README.md) | **한국어**

내 컴퓨터에서 돌아가는, LLM 없는 시나리오형 챗봇 엔진이에요. 봇은 데이터예요. 예문이 달린 의도,
동의어가 달린 엔티티, 시나리오, 응답 템플릿으로 이뤄져요. 엔진은 한국어 입력을 문장 임베딩 모델(학습하지 않음)과
작은 분류기로 이해하고, 시나리오를 정해진 대로 진행해요. 이 데이터를 편집하는 빌더 UI는 나중에 만들어요.

> 상태: Phase 1(프로젝트 뼈대). 엔진 코어는 Phase 2부터 만들어요.
> 설계와 로드맵: [docs/engine-core-plan.md](docs/engine-core-plan.md)

## 시작하기

[uv](https://docs.astral.sh/uv/)와 `make`가 필요해요. Python 3.14는 uv가 설치해요.

```sh
make setup   # uv sync --locked
make check   # ruff + mypy(strict, src/) + pytest — CI와 같음
```

선택:

```sh
make db-up   # Postgres 17 + pgvector, 127.0.0.1:5432 (Docker), Phase 2부터 사용
make bench   # Phase 0 NLU 벤치마크 — Apple Silicon, 모델 다운로드, 약 10분
```

## 구조

| 경로 | 내용 |
|---|---|
| `src/chatcore/` | 엔진 패키지 (Phase 2 전까지 비어 있음) |
| `bots/sample/nlu.yaml` | 샘플 봇 NLU 데이터 (원본) |
| `bench/` | NLU 벤치마크와 테스트셋 |
| `docs/` | 계획과 벤치마크 보고서 |

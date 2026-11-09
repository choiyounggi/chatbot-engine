# Phase 0b — 독립 테스트셋 평가

`uv run python bench/eval_independent.py`가 생성한 파일이다. 손으로 고치지 말 것.

- 테스트셋 `bench/data/test_independent.yaml`은 학습 데이터와 이전 실패 사례를 보지 않은 별도 에이전트가 썼다(범위 안 135, 범위 밖 hard 60 / easy 40, 대화 20개).
- 구성과 규칙(인사/작별 위치 규칙 margin 0.5)은 이 테스트를 보기 전에 정했다. 임계값은 `bench/phase0.py` 교차검증에서 고른 값이다.
- '학습에 없는 문장만'은 실제 학습 행(지역을 채운 예문 포함)과 글자까지 같은 문장("안녕" 등)을 뺀 값이다.
- '학습과 먼 문장만'은 `arctic` 임베딩으로 모든 학습 행과의 최대 코사인 < 0.8인 문장만 센 값이다. hard negative 학습 문장과 테스트 문장은 같은 방식(의도 핵심어 + 다른 요구)으로 같은 날 쓰였다. 그래서 전체 수치는 **분포 안 성능**이고, 처음 보는 유형으로의 일반화는 이 열이 더 가깝다. 다만 구성마다 학습 데이터가 달라 n도 다르다.

| 구성 | macro-F1 (전체) | 범위 안 정확도 | hard 범위 밖 거절률 | easy 범위 밖 거절률 | macro-F1 (학습에 없는 문장만) | 학습과 먼 문장만: macro-F1 / hard 거절률 |
|---|---|---|---|---|---|---|
| A. 기준 (커밋 c626440 데이터) | 0.849 | 0.978 | 0.33 | 1.00 | 0.829 (n=215) | 0.627 / 0.33 (n=137) |
| B. 기준 + 위치 규칙 | 0.849 | 0.978 | 0.33 | 1.00 | 0.829 (n=215) | 0.627 / 0.33 (n=137) |
| C. hard negative 추가 | 0.947 | 0.978 | 0.83 | 1.00 | 0.935 (n=215) | 0.828 / 0.82 (n=134) |
| D. hard negative 추가 + 위치 규칙 | 0.947 | 0.978 | 0.83 | 1.00 | 0.935 (n=215) | 0.828 / 0.82 (n=134) |

## 대화 평가 (인사/작별 위치 규칙)

- 기본 대화: 테스트셋의 대화 20개. 대부분 '첫 턴 인사, 마지막 턴 작별' 구조라 규칙에 유리하다.
- 경계 대화: `bench/data/test_dialogues_edge.yaml`(대화 중간 재인사, 첫 턴 작별, 같은 말의 두 뜻, 섞인 경우). 규칙을 정한 뒤 별도 에이전트가 썼다.

| 구성 | 기본: 턴 정확도 | 기본: 인사·작별 턴 | 경계: 턴 정확도 | 경계: 인사·작별 턴 | 경계 사례별 인사·작별 턴 정확도 |
|---|---|---|---|---|---|
| A. 기준 (커밋 c626440 데이터) | 0.904 (n=73) | 0.875 (n=24) | 0.907 (n=75) | 0.833 (n=30) | ambiguous_both 0.75 (n=12), first_turn_bye 0.83 (n=6), mid_regreet 0.83 (n=6), unambiguous_mixed 1.00 (n=6) |
| B. 기준 + 위치 규칙 | 0.932 (n=73) | 0.958 (n=24) | 0.893 (n=75) | 0.800 (n=30) | ambiguous_both 0.83 (n=12), first_turn_bye 0.83 (n=6), mid_regreet 0.50 (n=6), unambiguous_mixed 1.00 (n=6) |
| C. hard negative 추가 | 0.932 (n=73) | 0.875 (n=24) | 0.907 (n=75) | 0.833 (n=30) | ambiguous_both 0.75 (n=12), first_turn_bye 0.83 (n=6), mid_regreet 0.83 (n=6), unambiguous_mixed 1.00 (n=6) |
| D. hard negative 추가 + 위치 규칙 | 0.959 (n=73) | 0.958 (n=24) | 0.893 (n=75) | 0.800 (n=30) | ambiguous_both 0.83 (n=12), first_turn_bye 0.83 (n=6), mid_regreet 0.50 (n=6), unambiguous_mixed 1.00 (n=6) |

## 구성

- **A. 기준 (커밋 c626440 데이터)**: 데이터 `c626440:bots/sample/nlu.yaml`, `expand/emb:arctic/logreg/C10`, 확률 임계값 0.35, 유사도 임계값 0.38, 위치 규칙 끔
- **B. 기준 + 위치 규칙**: 데이터 `c626440:bots/sample/nlu.yaml`, `expand/emb:arctic/logreg/C10`, 확률 임계값 0.35, 유사도 임계값 0.38, 위치 규칙 켬
- **C. hard negative 추가**: 데이터 `bots/sample/nlu.yaml`, `expand/emb:arctic/logreg/C10`, 확률 임계값 0.3, 유사도 임계값 0.39, 위치 규칙 끔
- **D. hard negative 추가 + 위치 규칙**: 데이터 `bots/sample/nlu.yaml`, `expand/emb:arctic/logreg/C10`, 확률 임계값 0.3, 유사도 임계값 0.39, 위치 규칙 켬

## 틀린 문장 — A. 기준 (커밋 c626440 데이터) (43개)

| 문장 | 정답 | 예측 |
|---|---|---|
| 야, 거기 있어? | greeting | out_of_scope |
| 실례합니다 | greeting | out_of_scope |
| 몇 신지 궁금해 | time | out_of_scope |
| 날씨 좋은 날 듣기 좋은 플레이리스트 추천해줘 | out_of_scope | weather |
| 지난달 날씨 통계 자료 어디서 찾아? | out_of_scope | weather |
| 날씨 때문에 등산 계획 짜는 법 알려줘 | out_of_scope | weather |
| 날씨 예보 앱 중에 뭐가 제일 정확해? | out_of_scope | weather |
| 날씨처럼 변덕스러운 직장 상사 때문에 힘들어 | out_of_scope | weather |
| 다음 주 날씨 기반으로 여행지 추천해줄 수 있어? | out_of_scope | weather |
| 날씨 좋으면 가기 좋은 공원 알려줘 | out_of_scope | weather |
| 이 커피 온도가 너무 뜨거운 것 같아 | out_of_scope | temperature |
| 체온이 몇 도부터 열이라고 하지? | out_of_scope | temperature |
| 냉장고 적정 온도가 몇 도야? | out_of_scope | temperature |
| 어항 물 온도 맞추는 법 알려줘 | out_of_scope | temperature |
| 와인은 몇 도에서 보관해야 돼? | out_of_scope | temperature |
| 작년 여름 평균 기온이 어느 정도였는지 궁금해 | out_of_scope | temperature |
| 시간 관리 잘하는 방법 알려줘 | out_of_scope | time |
| 비행기 도착 시간이 언제인지 확인해줘 | out_of_scope | time |
| 식당 영업 시간이 몇 시까지야? | out_of_scope | time |
| 회의 시간 바꾸는 법 좀 알려줄래? | out_of_scope | time |
| 안녕이라는 인사말의 유래가 뭐야? | out_of_scope | greeting |
| 안녕이란 단어 영어로 어떻게 번역해? | out_of_scope | greeting |
| 「다들 안녕히 주무셨어요」는 무슨 뜻이야? | out_of_scope | bye |
| 안녕이라는 드라마 줄거리 알려줄래? | out_of_scope | greeting |
| 감사 인사말 문자 예시 하나 써줘 | out_of_scope | thanks |
| 고맙다는 표현 영어로 여러 개 알려줘 | out_of_scope | thanks |
| 감사 인사 타이밍은 언제가 적절해? | out_of_scope | thanks |
| 추석에 어른들께 드릴 감사 선물 뭐가 좋을까? | out_of_scope | thanks |
| 감사 인사를 너무 늦게 전하면 실례일까? | out_of_scope | thanks |
| 이 수학 문제 푸는 거 도와줄 수 있어? | out_of_scope | help |
| 친구가 힘들어하는데 어떻게 도와줘야 할지 모르겠어 | out_of_scope | help |
| 이력서 쓰는 거 도와줄 수 있는 사이트 추천해줘 | out_of_scope | help |
| 이 계산 결과가 틀렸는지 확인해줄 수 있어? | out_of_scope | negative_feedback |
| 친구가 내 말이 틀렸다고 우기는데 어떻게 설득하지? | out_of_scope | negative_feedback |
| 이 번역이 틀렸는지 검토해줘 | out_of_scope | negative_feedback |
| 시험에서 틀린 문제 다시 풀어보고 싶어 | out_of_scope | negative_feedback |
| 발표 자료에 틀린 통계 수치가 있는지 검토해줘 | out_of_scope | negative_feedback |
| 칭찬 잘하는 방법 알려줘 | out_of_scope | compliment |
| 아이 칭찬해주는 좋은 말 몇 개 추천해줘 | out_of_scope | compliment |
| 동료한테 칭찬하는 멘트 좀 써줘 | out_of_scope | compliment |
| 상사한테 칭찬받는 보고서 쓰는 법 알려줘 | out_of_scope | compliment |
| 칭찬은 고래도 춤추게 한다는 책 내용 요약해줘 | out_of_scope | compliment |
| 후배한테 칭찬하는 말 좀 추천해줘 | out_of_scope | compliment |

## 틀린 문장 — B. 기준 + 위치 규칙 (43개)

| 문장 | 정답 | 예측 |
|---|---|---|
| 야, 거기 있어? | greeting | out_of_scope |
| 실례합니다 | greeting | out_of_scope |
| 몇 신지 궁금해 | time | out_of_scope |
| 날씨 좋은 날 듣기 좋은 플레이리스트 추천해줘 | out_of_scope | weather |
| 지난달 날씨 통계 자료 어디서 찾아? | out_of_scope | weather |
| 날씨 때문에 등산 계획 짜는 법 알려줘 | out_of_scope | weather |
| 날씨 예보 앱 중에 뭐가 제일 정확해? | out_of_scope | weather |
| 날씨처럼 변덕스러운 직장 상사 때문에 힘들어 | out_of_scope | weather |
| 다음 주 날씨 기반으로 여행지 추천해줄 수 있어? | out_of_scope | weather |
| 날씨 좋으면 가기 좋은 공원 알려줘 | out_of_scope | weather |
| 이 커피 온도가 너무 뜨거운 것 같아 | out_of_scope | temperature |
| 체온이 몇 도부터 열이라고 하지? | out_of_scope | temperature |
| 냉장고 적정 온도가 몇 도야? | out_of_scope | temperature |
| 어항 물 온도 맞추는 법 알려줘 | out_of_scope | temperature |
| 와인은 몇 도에서 보관해야 돼? | out_of_scope | temperature |
| 작년 여름 평균 기온이 어느 정도였는지 궁금해 | out_of_scope | temperature |
| 시간 관리 잘하는 방법 알려줘 | out_of_scope | time |
| 비행기 도착 시간이 언제인지 확인해줘 | out_of_scope | time |
| 식당 영업 시간이 몇 시까지야? | out_of_scope | time |
| 회의 시간 바꾸는 법 좀 알려줄래? | out_of_scope | time |
| 안녕이라는 인사말의 유래가 뭐야? | out_of_scope | greeting |
| 안녕이란 단어 영어로 어떻게 번역해? | out_of_scope | greeting |
| 「다들 안녕히 주무셨어요」는 무슨 뜻이야? | out_of_scope | bye |
| 안녕이라는 드라마 줄거리 알려줄래? | out_of_scope | greeting |
| 감사 인사말 문자 예시 하나 써줘 | out_of_scope | thanks |
| 고맙다는 표현 영어로 여러 개 알려줘 | out_of_scope | thanks |
| 감사 인사 타이밍은 언제가 적절해? | out_of_scope | thanks |
| 추석에 어른들께 드릴 감사 선물 뭐가 좋을까? | out_of_scope | thanks |
| 감사 인사를 너무 늦게 전하면 실례일까? | out_of_scope | thanks |
| 이 수학 문제 푸는 거 도와줄 수 있어? | out_of_scope | help |
| 친구가 힘들어하는데 어떻게 도와줘야 할지 모르겠어 | out_of_scope | help |
| 이력서 쓰는 거 도와줄 수 있는 사이트 추천해줘 | out_of_scope | help |
| 이 계산 결과가 틀렸는지 확인해줄 수 있어? | out_of_scope | negative_feedback |
| 친구가 내 말이 틀렸다고 우기는데 어떻게 설득하지? | out_of_scope | negative_feedback |
| 이 번역이 틀렸는지 검토해줘 | out_of_scope | negative_feedback |
| 시험에서 틀린 문제 다시 풀어보고 싶어 | out_of_scope | negative_feedback |
| 발표 자료에 틀린 통계 수치가 있는지 검토해줘 | out_of_scope | negative_feedback |
| 칭찬 잘하는 방법 알려줘 | out_of_scope | compliment |
| 아이 칭찬해주는 좋은 말 몇 개 추천해줘 | out_of_scope | compliment |
| 동료한테 칭찬하는 멘트 좀 써줘 | out_of_scope | compliment |
| 상사한테 칭찬받는 보고서 쓰는 법 알려줘 | out_of_scope | compliment |
| 칭찬은 고래도 춤추게 한다는 책 내용 요약해줘 | out_of_scope | compliment |
| 후배한테 칭찬하는 말 좀 추천해줘 | out_of_scope | compliment |

## 틀린 문장 — C. hard negative 추가 (13개)

| 문장 | 정답 | 예측 |
|---|---|---|
| 야, 거기 있어? | greeting | bye |
| 실례합니다 | greeting | out_of_scope |
| 몇 신지 궁금해 | time | help |
| 날씨 좋으면 가기 좋은 공원 알려줘 | out_of_scope | weather |
| 이 커피 온도가 너무 뜨거운 것 같아 | out_of_scope | temperature |
| 체온이 몇 도부터 열이라고 하지? | out_of_scope | temperature |
| 냉장고 적정 온도가 몇 도야? | out_of_scope | temperature |
| 작년 여름 평균 기온이 어느 정도였는지 궁금해 | out_of_scope | temperature |
| 안녕이라는 인사말의 유래가 뭐야? | out_of_scope | greeting |
| 「다들 안녕히 주무셨어요」는 무슨 뜻이야? | out_of_scope | bye |
| 이 수학 문제 푸는 거 도와줄 수 있어? | out_of_scope | help |
| 친구가 힘들어하는데 어떻게 도와줘야 할지 모르겠어 | out_of_scope | help |
| 이 번역이 틀렸는지 검토해줘 | out_of_scope | negative_feedback |

## 틀린 문장 — D. hard negative 추가 + 위치 규칙 (13개)

| 문장 | 정답 | 예측 |
|---|---|---|
| 야, 거기 있어? | greeting | bye |
| 실례합니다 | greeting | out_of_scope |
| 몇 신지 궁금해 | time | help |
| 날씨 좋으면 가기 좋은 공원 알려줘 | out_of_scope | weather |
| 이 커피 온도가 너무 뜨거운 것 같아 | out_of_scope | temperature |
| 체온이 몇 도부터 열이라고 하지? | out_of_scope | temperature |
| 냉장고 적정 온도가 몇 도야? | out_of_scope | temperature |
| 작년 여름 평균 기온이 어느 정도였는지 궁금해 | out_of_scope | temperature |
| 안녕이라는 인사말의 유래가 뭐야? | out_of_scope | greeting |
| 「다들 안녕히 주무셨어요」는 무슨 뜻이야? | out_of_scope | bye |
| 이 수학 문제 푸는 거 도와줄 수 있어? | out_of_scope | help |
| 친구가 힘들어하는데 어떻게 도와줘야 할지 모르겠어 | out_of_scope | help |
| 이 번역이 틀렸는지 검토해줘 | out_of_scope | negative_feedback |

@echo off
echo Rasa 서버를 시작합니다 (포트: 5005)...

REM 가상 환경이 있는 경우 활성화 (없으면 주석 처리하세요)
REM call venv\Scripts\activate.bat

REM Rasa 모델이 없는 경우 학습
IF NOT EXIST "models" (
    echo Rasa 모델을 학습합니다...
    rasa train
)

REM Rasa 서버 시작 (포트 5005)
echo Rasa 서버를 5005 포트에서 시작합니다...
rasa run --enable-api --cors "*" --port 5005 --debug

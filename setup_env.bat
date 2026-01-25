@echo off
setlocal enabledelayedexpansion

REM ====== 0) 프로젝트 루트 ======
cd /d %~dp0

REM ====== 1) venv 만들기 ======
if not exist ".venv" (
  echo [1/6] Create venv...
  python -m venv .venv
) else (
  echo [1/6] venv exists. skip
)

REM ====== 2) venv 활성화 ======
call .venv\Scripts\activate

REM ====== 3) pip 업그레이드 ======
echo [2/6] Upgrade pip...
python -m pip install --upgrade pip

REM ====== 4) 패키지 설치 (이미 있으면 pip가 스킵/업데이트) ======
echo [3/6] Install requirements...
pip install -r requirements.txt

REM ====== 5) nnU-Net 환경변수 세팅 (현재 쉘 세션용) ======
echo [4/6] Set nnU-Net env vars...

set "NNUNET_RAW=%cd%\nnUNet_raw"
set "NNUNET_PREP=%cd%\nnUNet_preprocessed"
set "NNUNET_RES=%cd%\models\nnUNet_results"

REM nnU-Net이 읽는 환경변수명 (중요)
set "nnUNet_raw=!NNUNET_RAW!"
set "nnUNet_preprocessed=!NNUNET_PREP!"
set "nnUNet_results=!NNUNET_RES!"

REM 폴더 생성
if not exist "!nnUNet_raw!" mkdir "!nnUNet_raw!"
if not exist "!nnUNet_preprocessed!" mkdir "!nnUNet_preprocessed!"

REM 모델 폴더 체크
if not exist "!nnUNet_results!" (
  echo [ERROR] models\nnUNet_results not found.
  echo Put model folders under: %cd%\models\nnUNet_results\
  exit /b 1
)

echo nnUNet_raw=!nnUNet_raw!
echo nnUNet_preprocessed=!nnUNet_preprocessed!
echo nnUNet_results=!nnUNet_results!

REM ====== 6) 간단 체크: nnUNetv2_predict 호출 가능? ======
echo [5/6] Check nnUNet install...
where nnUNetv2_predict >nul 2>nul
if errorlevel 1 (
  echo [ERROR] nnUNetv2_predict not found in venv.
  exit /b 1
)

echo [6/6] Setup done!
echo Now run: streamlit run app.py
endlocal

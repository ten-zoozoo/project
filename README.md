<div align="center">

# MIMICare

### MIMIC SICU 중환자 임상 데이터 기반 실시간 사망률 예측 모니터링 서비스

<br>

> **"중환자실의 골든타임을 지키는 AI 기반 임상 의사결정 지원 시스템"**

<br>

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logoColor=white)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

<br>

**프로젝트 기간** : 2025.12.29 ~ 2026.01.19 (3주)

[시연 영상 보기](https://www.youtube.com/watch?v=ey2c4cSZLpw)

</div>

<br>

---

## 목차

1. [프로젝트 개요](#1--프로젝트-개요)
2. [문제 정의 및 기획 배경](#2--문제-정의-및-기획-배경)
3. [핵심 기능](#3--핵심-기능)
4. [서비스 아키텍처](#4--서비스-아키텍처)
5. [도메인 분석 및 AI 변수 설계](#5--도메인-분석-및-ai-변수-설계)
6. [기술 스택](#6--기술-스택)
7. [팀 구성 및 역할](#7--팀-구성-및-역할)
8. [시연 영상](#8--시연-영상)

<br>

---

## 1. 프로젝트 개요

**MIMICare**는 MIT에서 공개한 대규모 중환자 임상 데이터베이스 **MIMIC-IV**의 SICU(외과 중환자실) 데이터를 활용하여, **실시간으로 환자의 사망 위험도를 예측**하고 **다장기 위험 이벤트를 감지**하는 **AI 기반 임상 의사결정 지원 시스템(CDSS)** 입니다.

### 프로젝트 핵심 가치

| 가치 | 설명 |
|:---:|:---|
| **조기 경보** | 6 / 12 / 18 / 24시간 단위 사망률 예측으로 골든타임 확보 |
| **다장기 모니터링** | 순환계 / 호흡계 / 신장 / 신경계 / 간 5개 장기 위험 이벤트 실시간 감지 |
| **의료 표준 점수** | Apache II, SOFA, CCI 등 국제 표준 중증도 점수 차용 |
| **영상 진단 보조** | CT 영상 기반 간 종양 / 췌장 자동 분할(Segmentation) 뷰어 제공 |

<br>

---

## 2. 문제 정의 및 기획 배경

### 해결하고자 한 문제

```
중환자실 환자의 상태는 짧은 시간 내 급변할 수 있으며,
의료진이 다수의 환자를 동시에 모니터링하는 데에는 물리적 한계가 존재합니다.
```

- 국내 중환실 **외과 전문의 비중 7%**, 수술과 중환자 동시에 관리하기 부담
- 바이탈 사인, 검사 결과, 투약 이력 등 **방대한 데이터가 실시간으로 발생**하지만, 이를 종합적으로 분석하기 어려움
- 환자 상태 악화의 **조기 징후를 놓칠 경우** 치명적 결과로 이어질 수 있음

### 기획 방향

| 단계 | 접근 방식 |
|:---:|:---|
| **도메인 분석** | 중환자의학 문헌 조사, Apache II / SOFA / CCI 등 임상 중증도 지표 연구 |
| **변수 설계** | 임상적 근거에 기반한 AI 입력 변수 선정 및 시간 윈도우(6/12/18/24h) 설계 |
| **서비스 설계** | 의료진의 워크플로우를 고려한 대시보드 UI/UX 기획 |
| **검증** | 시계열 기반 시뮬레이션을 통한 실시간 모니터링 시나리오 검증 |

<br>

---

## 3. 핵심 기능

### 3-1. 실시간 SICU 환자 모니터링 대시보드

<table>
<tr>
<td width="50%">

**환자 위험도 랭킹**
- 전체 환자를 사망 위험도 기준 정렬
- 위험 등급별 시각적 구분 (Critical / Warning / Stable)
- 원클릭 환자 상세 조회

</td>
<td width="50%">

**실시간 바이탈 사인 트렌드**
- MAP, HR, SpO₂, Lactate, 소변량, 호흡수 6개 지표
- 시간 흐름에 따른 추이 차트
- 임상 정상 범위 대비 상태 표시

</td>
</tr>
<tr>
<td width="50%">

**AI 사망률 예측**
- XGBoost 기반 6/12/18/24h 사망 위험도 예측
- 시간대별 위험도 변화 추이 시각화
- 실시간 시뮬레이션 기반 예측 갱신

</td>
<td width="50%">

**다장기 위험 이벤트 감지**
- 순환계, 호흡계, 신장, 신경계, 간 5대 장기
- 근거 기반 중증도 판정 (Evidence Accumulation)
- 위험 이벤트 발생 시 즉시 알림

</td>
</tr>
</table>

### 3-2. 임상 중증도 스코어링

| 스코어 | 설명 | 활용 |
|:---:|:---|:---|
| **Apache II** | 11개 생리학적 변수 기반 급성 중증도 평가 | ICU 입실 후 24h 내 사망률 예측 |
| **SOFA** | 6개 장기계 기능부전 평가 | 장기별 기능부전 정도 모니터링 |
| **CCI** | ICD-9/10 코드 기반 동반질환 가중치 평가 | 기저질환에 따른 예후 보정 |

### 3-3. CT 영상 세그멘테이션 뷰어

- **nnU-Net** 기반 간 종양 및 췌장 자동 분할 결과 시각화
- DICOM → NIfTI 자동 변환 파이프라인
- 병변 위치 자동 탐색 (First / Center / Last Lesion 네비게이션)
- 다중 라벨 오버레이 (간=초록, 종양=마젠타)

<br>

---

## 4. 서비스 아키텍처

<div align="center">
<img width="800" alt="MIMICare 서비스 구성도" src="https://github.com/user-attachments/assets/3211f20c-5b66-42e3-a5ec-f8a0cabb743c" />
</div>

<br>

### 데이터 파이프라인

```
MIMIC-IV Raw Data
    │
    ├── 전처리 & 피처 엔지니어링
    │       ├── 시간 윈도우별 집계 (6h / 12h / 18h / 24h)
    │       ├── Apache II 변수 추출 (체온, MAP, HR, RR, PaO₂, pH 등)
    │       ├── SOFA 구성 요소 산출 (호흡, 응고, 간, 심혈관, CNS, 신장)
    │       └── 인공호흡기 / 승압제 / RRT 플래그 생성
    │
    ├── PostgreSQL 적재
    │       ├── chartevents (바이탈 사인)
    │       ├── labevents (검사 결과)
    │       ├── inputevents (투약 이력)
    │       ├── outputevents (배출량)
    │       └── diagnoses_icd / procedures_icd
    │
    └── AI Prediction
            ├── XGBoost Classifier → 사망률 예측
            └── nnU-Net → CT 세그멘테이션
```

<br>

---

## 5. 도메인 분석 및 AI 변수 설계

### 도메인 분석 프로세스

프로젝트 기획 단계에서 중환자의학 도메인에 대한 심층 분석을 수행하여, AI 모델에 투입할 **임상적으로 유의미한 변수**를 선정했습니다.

**1단계 : 문헌 기반 변수 후보 도출**
- Apache II, SOFA, SAPS 등 국제 표준 중증도 평가 체계 조사
- 중환자 사망률 예측 관련 논문 리뷰
- MIMIC-IV 데이터셋 내 활용 가능 변수 매핑

**2단계 : 변수 그룹핑 및 기준 수립**

| 변수 그룹 | 포함 변수 | 임상적 의미 |
|:---:|:---|:---|
| **활력징후** | MAP, HR, SpO₂, RR, 체온 | 기본 생체 신호 |
| **호흡 지표** | PaO₂, FiO₂, P/F ratio, 인공호흡기 여부 | 호흡 기능 평가 |
| **혈액 검사** | WBC, HCT, PLT, Lactate, pH | 전신 상태 및 장기 기능 |
| **전해질** | Na⁺, K⁺, Creatinine, BUN | 신장 기능 및 전해질 균형 |
| **간기능** | Bilirubin, AST, ALT | 간 기능 평가 |
| **신경계** | GCS | 의식 수준 평가 |
| **투약** | 승압제 용량, 진정제, 항생제, 수액 | 치료 강도 지표 |
| **합병증** | CCI (ICD-9/10 기반) | 기저질환 보정 |

**3단계 : 시간 윈도우 설계**
- ICU 입실 후 **6h / 12h / 18h / 24h** 4개 시점에서 변수 집계
- 각 윈도우별 **최솟값, 최댓값, 평균값** 추출로 환자 상태 변화 포착
- 시간 경과에 따른 악화 / 호전 추이를 AI 모델이 학습할 수 있도록 설계

<br>

---

## 6. 기술 스택

| 분류 | 기술 |
|:---:|:---|
| **AI / ML** | XGBoost Classifier, nnU-Net (PyTorch 기반) |
| **데이터 처리** | Pandas, Polars, NumPy |
| **의료 영상** | nibabel (NIfTI), SimpleITK (DICOM), Matplotlib |
| **프론트엔드** | Streamlit, Altair (인터랙티브 시각화) |
| **데이터베이스** | PostgreSQL (psycopg2) |
| **인프라** | Python 3.x, dotenv |

<br>

---

## 7. 팀 구성 및 역할

| 이름 | 역할 | 담당 업무 |
|:---:|:---:|:---|
| 문예진 | PM | 프로젝트 총괄, DB 구축 및 적재, AI 모델링 총괄 |
| 권민정 | Data Engineer | 빅데이터 전처리, DB 구축 & 적재, Streamlit 기반 서비스 구현 |
| **윤지우** | **Domain Analyst** | **도메인 분석, AI 변수 기준 수립, AI 모델 시각화** |
| 이시은 | Domain Analyst | 도메인 분석, AI 변수 기준 수립, 문서 관리 |
| 배순은 | Data Scientist | 데이터 전처리, AI 변수 가공, 통계 분석 |

<br>

---

## 8. 시연 영상

<div align="center">

[![MIMICare 시연 영상](https://img.youtube.com/vi/ey2c4cSZLpw/0.jpg)](https://www.youtube.com/watch?v=ey2c4cSZLpw)

**클릭하면 YouTube에서 시연 영상을 확인할 수 있습니다.**

</div>

<br>

---

<div align="center">

#### 본 프로젝트는 MIMIC-IV 데이터셋을 활용한 학술 목적의 프로젝트입니다.

<br>

`MIMICare` `MIMIC-IV` `SICU` `중환자 모니터링` `사망률 예측` `XGBoost` `nnU-Net` `CT Segmentation` `Apache II` `SOFA` `Streamlit`

</div>

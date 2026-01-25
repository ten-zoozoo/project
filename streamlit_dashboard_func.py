import streamlit as st
import pandas as pd
import altair as alt
import polars as pl
from predict_data_preprocessing import * 
from model_loader import ICU24hRiskModel
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv 
import os
import psycopg2

streamlit_path = 'C:\mimic_analysis\\for_streamlit\\'

load_dotenv('.env')
db_name = os.getenv('db_name')
user_name = os.getenv('db_admin')
password = os.getenv('password')

# postgresql DB 연결
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database=db_name,
    user=user_name,
    password=password
)

cur = conn.cursor()

VASOPRESSOR_ITEMIDS = [221906,221289,222315,221749, 221662]
SEDATION_ITEMIDS = [222168,221668,223257,221712,221385]
FLUID_ITEMIDS = [225158,225828,225159,225161,225166,225160,220864,220862]
ANTIBIO_ITEMIDS = [225798,225970,225942,225936,225931,225948,225945,225913,225952,
                    225946,225934,225950,225930,225912,225929,225947,225932]

# 데이터에서 stay_id로 데이터 필터링
def stay_filtering(table_name, stay_id):
    return pd.read_sql(
        f"""
        SELECT *
        FROM {table_name}
        WHERE stay_id = {stay_id};
        """, conn)

# 환자들 데이터에서 subject_id로 데이터 필터링
def subject_filtering(table_name, subject_id, hadm_id):
    return pd.read_sql(
    f"""
        SELECT *
        FROM {table_name}
        WHERE subject_id = {subject_id} and hadm_id = {hadm_id};
    """, conn)

def preprocessing_mimic(mimic_df, intime):
    mimic_df = mimic_df.sort_values('starttime')

    mimic_df['hours_from_icu'] = (
        (mimic_df['starttime'] - intime).dt.total_seconds() / 3600
    )

    mimic_df['window'] = pd.cut(
        mimic_df['hours_from_icu'],
        bins=[0, 6, 12, 18, 24],
        labels=[6,12,18,24],
        right=False,  # 왼쪽 포함, 오른쪽 미포함 [0, 6)
        include_lowest=True
    )

    mimic_df = mimic_df.drop('hours_from_icu',axis=1)
    return mimic_df

def basic_css():
    return st.markdown("""
    <style>
    /* 사이드바 버튼을 카드처럼 스타일링 */
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent;
        border: none;
        padding: 0 0 0 5px;
        margin: 0;
        transition: all 0.3s;
        height: 50px;
        font-size : 20px;
        justify-content: flex-start;   /* 핵심 */
        text-align: left;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateX(3px);
    }
    
    section[data-testid="stSidebar"] .stButton > button:active {
        background: #fff5f5;
    }
    
    /* 버튼 내 텍스트 스타일 */
    section[data-testid="stSidebar"] .stButton > button p {
        font-size : 20px;
        margin: 0;
        padding: 0;
        color : black;
        text-align: left;
    }
                       
    section[data-testid="stSidebar"] .stButton > button:hover p {
        color : red;
    }
            
    .st-emotion-cache-1uvhpyl p {
        text-overflow: ellipsis;
        overflow: hidden;
        padding: 0;
    }
            
    .st-emotion-cache-n9eile p {
        text-overflow: ellipsis;
        overflow: hidden;
        padding: 0;
    }
    
    p {
        padding: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def return_metric(df, metric_label,unit):
    now = df.iloc[-1]['valuenum']
    previous = df.iloc[-2]['valuenum']

    st.metric(
                label=metric_label,
                value=f"{round(now,2)}",
                delta=f"{round(now - previous,2)}{unit}(몇 시간 전인지)",
                delta_color="inverse"
            )

@st.cache_data(show_spinner=False)
def load_patient_data(selected_patient):
    """병렬 처리로 빠르게 로드"""
    patient = stay_filtering('patients', selected_patient)
    input = stay_filtering('inputevents', selected_patient)
    output = stay_filtering('outputevents', selected_patient)
    lab = stay_filtering('labevents', selected_patient)
    chart = stay_filtering('chartevents', selected_patient)

    mimic_df = pd.concat([input, output, lab, chart],ignore_index=True)
    mimic_df = preprocessing_mimic(mimic_df, patient['intime'].iloc[0])
    
    return patient, mimic_df

def top_5_css():
    return st.sidebar.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            display: flex !important;
            justify-content: flex-start !important;
            align-items: center;
            text-align: left !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button * {
            justify-content: flex-start !important;
            text-align: left !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button span {
            display: block !important;
            width: 100%;
            text-align: left !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stButton"] > button p {
            padding: 4px 8px;   /* 위아래 4px, 좌우 8px */
        }

        """,
        unsafe_allow_html=True
    )

def predict_die_css():
    st.markdown(
        """
        <style>
        .stock-card {
            align-items: center;
            justify-content: space-between;
            background-color: white;
            padding: 14px 18px;
            border-radius: 12px;
            width: 100%;
            box-shadow: rgba(0, 0, 0, 0.02) 0px 1px 3px 0px, rgba(27, 31, 35, 0.15) 0px 0px 0px 1px;
            margin-bottom : 15px;
            text-align: left;
        }

        .name {
            font-size: 36px;
            font-weight: 600;
            line-height: 1.2;
            color : black;
        }

        .hour {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
            align-items: center;
            font-size: 20px;
            font-weight: 600;
            line-height: 1.2;
        }
        </style>
        """, unsafe_allow_html=True)

# 사망률 예측 카드
def predict_die_data_yes_css(hour, pred_time, pred_percent, diff, color, symbol):
    return st.markdown(
        f"""
        <div class="stock-card">
            <div class="hour">
                <span style="color:#00396F">{str(hour)}H</span>
                <span style="color:gray;">{str(pred_time)}</span>
            </div>
            <div class="display:flex;">
                <span class="name">{pred_percent:.1f}%</span>
                <span style="color:{color}; font-size : 20px">({symbol}{diff:.1f}%)</span>
            <div>
        </div>
        """,
        unsafe_allow_html=True
    )

def predict_die_data_no_css(hour):
    return st.markdown(
        f"""
        <div class="stock-card">
            <div class="hour">
                <span style="color:#00396F">{str(hour)}H</span>
                <span style="color:gray;">{''}</span>
            </div>
            <div class="display:flex;">
                <span class="name">-%</span>
                <span style="color:red; font-size : 20px">(- %)</span>
            <div>
        </div>
        """,
        unsafe_allow_html=True
    )

# 사망률 예측 코드
def pred_die_percent(patient, mimic_df, diagnoses_icd_df, procedures_icd_df, model):
    sofa_vaso_df = sofa_vaso(mimic_df)
    sofa_var_df = sofa_var(mimic_df)
    sofa_vent_df = sofa_vent(mimic_df)
    sofa_rrt_df = sofa_rrt(mimic_df)
    apache_score_df = calc_apache(mimic_df)
    df_sofa_final = return_sofa_score(sofa_vaso_df, sofa_var_df, sofa_vent_df, sofa_rrt_df)
    vent_flag = return_vaso_yes(mimic_df)
    cci_result = return_CCI_score(patient, diagnoses_icd_df)
    postop_24h = return_surgery_yes(mimic_df,procedures_icd_df)
    gender_age = return_patient_info(patient)
    total = return_merge(apache_score_df, df_sofa_final,vent_flag,cci_result,postop_24h,gender_age)
    TEMP_COLS = [
    "apache_temp_score_min_w6", "apache_temp_score_max_w6",
    "apache_temp_score_min_w12", "apache_temp_score_max_w12",
    "apache_temp_score_min_w18", "apache_temp_score_max_w18",
    "apache_temp_score_min_w24", "apache_temp_score_max_w24",
    ]

    GCS_COLS = [
        "apache_gcs_score_min_w6", "apache_gcs_score_max_w6",
        "apache_gcs_score_min_w12", "apache_gcs_score_max_w12",
        "apache_gcs_score_min_w18", "apache_gcs_score_max_w18",
        "apache_gcs_score_min_w24", "apache_gcs_score_max_w24",
        "sofa_cns_w6", "sofa_cns_w12",
        "sofa_cns_w18", "sofa_cns_w24",
    ]

    drop_cols =TEMP_COLS + GCS_COLS
    total = total.drop(drop_cols, axis=1)
    risk_df = model.predict_from_db(total)
    return round(risk_df['risk_score_24h'].iloc[0],1)

def insert_data_db(df, conn):
    cur = conn.cursor()
    try:
        for _, row in df.iterrows():
            cur.execute(
                """
                INSERT INTO mortality_prediction (
                    stay_id, admission_time, check_time,
                    hours, pred_die_percent, diff
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    int(row['selected_patient']),
                    row['admission_time'],
                    row['check_time'],
                    int(row['hours']),
                    float(row['pred_die_percent']),
                    float(row['diff'])
                )
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e

def should_run_prediction(intime, current_time):
    hours = int((current_time - intime).total_seconds() / 3600)
    return hours in (6, 12, 18, 24)

def pred_all_phase(selected_patient, current_time, patient_df, mimic_df, diagnoses_icd_df, procedures_icd_df, model):
    
    def should_predict(selected_patient, hours):
        """이 환자의 이 시간대에 예측이 필요한지 확인"""
        # 6시간 단위이고, 1시간 이상이며, 아직 예측하지 않은 시간대인지 확인
        if hours < 1 or hours % 6 != 0:
            return False
        
        if selected_patient not in st.session_state.last_prediction_hours:
            return True
        
        return hours not in st.session_state.last_prediction_hours[selected_patient]
    
    def get_hours_since_admission(admission_time, current_time):
        """입실 시간으로부터 경과 시간 계산"""
        elapsed = current_time - admission_time
        hours = int(elapsed.total_seconds() / 3600)
        return hours
    
    def predict_for_patient(selected_patient, patient_df, mimic_df, diagnoses_icd_df, procedures_icd_df, model):
        """특정 환자에 대한 예측 수행"""
        # 환자가 등록되어 있는지 확인
        if selected_patient not in st.session_state.patient_admission_times:
            st.warning(f"환자 {selected_patient}의 입실시간이 등록되지 않았습니다.")
            return False
        
        admission_time = st.session_state.patient_admission_times[selected_patient]
        hours = get_hours_since_admission(admission_time, current_time) 
        
        # 예측이 필요한 시간인지 확인
        if not should_predict(selected_patient, hours):
            return False
        
        # 예측 수행
        percent = pred_die_percent(patient_df, mimic_df, diagnoses_icd_df, procedures_icd_df, model) * 100
        
        # 이전 예측값과 비교 (같은 환자의 이전 기록)
        patient_history = st.session_state.pred_store[
            st.session_state.pred_store['selected_patient'] == selected_patient
        ]
        
        if patient_history.empty:
            diff = percent
        else:
            prev = patient_history['pred_die_percent'].iloc[-1]
            diff = round(percent - prev, 1)
        
        # 새 예측 저장
        new_row = pd.DataFrame([{
            'selected_patient': selected_patient,
            'admission_time': admission_time.strftime('%Y-%m-%d %H:%M'),
            'check_time': current_time.strftime('%Y-%m-%d %H:%M'),
            'hours': hours,
            'pred_die_percent': round(percent, 1),
            'diff': round(diff, 1)
        }])
        
        st.session_state.pred_store = pd.concat(
            [st.session_state.pred_store, new_row],
            ignore_index=True
        )
        
        # 예측 완료 기록 (리스트 초기화 필요)
        if selected_patient not in st.session_state.last_prediction_hours:
            st.session_state.last_prediction_hours[selected_patient] = []
        
        st.session_state.last_prediction_hours[selected_patient].append(hours)
        
        if (
            24 in st.session_state.last_prediction_hours[selected_patient]
            and selected_patient not in st.session_state.flushed_patients
        ):
            will_insert = st.session_state.pred_store[
                st.session_state.pred_store['selected_patient'] == selected_patient
            ]

            insert_data_db(will_insert, conn)

            # 임시 저장소에서 제거
            st.session_state.pred_store = st.session_state.pred_store[
                st.session_state.pred_store['selected_patient'] != selected_patient
            ]

            # 🔴 핵심: 완료 환자 등록
            st.session_state.completed_patients.add(selected_patient)
            st.session_state.flushed_patients.add(selected_patient)
                
    # 실행
    return predict_for_patient(selected_patient, patient_df, mimic_df, diagnoses_icd_df, procedures_icd_df, model)

def patient_info_css(selected_patient, patient, intime):
    st.markdown(
        f"""
        <style>
        .patient-card {{
            display : flex;
            align-items: center;
            height: 100%;
            border-radius: 10px;
            padding: 0;
            background-color: white;
        }}
        </style>

        <div class="patient-card">
            <h1 style="margin:0; color:#c92a2a; font-size:36px; font-weight:700; padding : 10px 0;">
                <span style="color:black;">ID_{selected_patient}</span>
            </h1>
            <p style="margin:0; color:gray; font-size:22px; font-weight:500;">
                {patient.iloc[0]["환자실제나이"]} / {patient.iloc[0]["gender"]}
            </p>
        </div>
        <p style="margin:0; color:gray; font-size:18px; font-weight:500; margin-bottom:10px">
            ICU 입실시간 : {intime}
        </p>
        """,
        unsafe_allow_html=True)

def map_status(map_val):
    if map_val < 55:
        return "red", "Critically low"
    elif map_val < 65:
        return "orange", "Below target"
    else:
        return "black", "Stable"

def lactate_status(lac):
    if lac >= 4.0:
        return "red", "Severely elevated"
    elif lac >= 2.0:
        return "orange", "Elevated"
    else:
        return "black", "Normal"

def spo2_status(spo2):
    if spo2 < 90:
        return "red", "Severe hypoxia"
    elif spo2 < 94:
        return "orange", "Low"
    else:
        return "black", "Adequate"

def hr_status(hr):
    if hr >= 130:
        return "red", "Severe tachycardia"
    elif hr >= 110:
        return "orange", "Tachycardia"
    else:
        return "black", "Normal"

def rr_status(rr):
    if rr >= 30:
        return "red", "Severe tachypnea"
    elif rr >= 22:
        return "orange", "Tachypnea"
    else:
        return "black", "Normal"

def uop_status(uop):
    if uop < 0.3:
        return "red", "Severe oliguria"
    elif uop < 0.5:
        return "orange", "Oliguria"
    else:
        return "black", "Normal"

def return_card(now, title, unit,standard_message, color):
    st.markdown(f"""
                <div class = 'card'>
                    <strong>{title}</strong>
                    <h1 style='margin:8px 0; color:#c92a2a; font-size:38px; font-weight:700; padding:0;'>
                        <span style='color:{color};'>{now}</span>
                        <span style='color:#000000; font-size:22px; font-weight:500; margin-left:4px;'>
                            {unit}
                        </span>
                    </h1>
                    <p style='margin:0; color:gray; font-size:20px; font-weight:500;'>
                            {standard_message}
                    </p>
                </div>
            """, unsafe_allow_html=True)
    
def return_card2(now, title, unit,standard_message, color):
    st.markdown(f"""
                <div style="
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    padding: 0;
                    width: 100%;
                    min-height: 96px;
                    margin-bottom:15px;
                ">
                    <strong style='font-size:20px;'>{title}</strong>
                    <h1 style='margin:8px -5px 8px 0px; color:#c92a2a; font-size:36px; font-weight:700; padding:0;'>
                        <span style='color:{color};'>{now}</span>
                        <span style='color:#000000; font-size:20px; font-weight:500; margin-left:2px;'>
                            {unit}
                        </span>
                    </h1>
                    <p style='margin:0; color:gray; font-size:20px; font-weight:500;'>
                            {standard_message}
                    </p>
                </div>
            """, unsafe_allow_html=True)


# vital sign (수치 + 그래프)
def metric_card_with_trend(df,last_value,title,unit,status,color):
    with st.container():
        col1, col2 = st.columns([1, 1.3], gap=None)

        # 왼쪽: 상태 카드
        with col1:
            return_card2(last_value, title, unit, status, color)

        # 오른쪽: 트렌드 그래프
        with col2:
            st.markdown(
                """
                <div style="
                    width: 85%;
                    margin-left: -5px;
                ">
                """,
                unsafe_allow_html=True
            )
    
            chart = (
                alt.Chart(df)
                .mark_line(strokeWidth=2, interpolate="monotone", color=color)
                .encode(
                    x=alt.X(
                        f"starttime:T",
                        axis=alt.Axis(labels=False, ticks=False, title=None)
                    ),
                    y=alt.Y(
                        f"valuenum:Q",
                        axis=alt.Axis(labels=False, ticks=False, title=None),
                        scale=alt.Scale(zero=False)
                    )
                )
                .properties(
                    height=96,
                    width=230,
                    padding={"top": 0, "bottom": 15, "left": 0, "right": 10}
                )
                .configure_view(strokeWidth=0)
                .configure_axis(grid=False)
            )

            st.altair_chart(chart, use_container_width=False)

# 약물 투여 중 - 약물 대분류 시각화를 위한 색상 매칭
def medication_category(itemid, dose):
    if itemid in ANTIBIO_ITEMIDS:
        return "Antibiotic", int(dose), "#1f77b4"
    elif itemid in VASOPRESSOR_ITEMIDS:
        return "Vasopressor", round(dose,2), "#d62728"
    elif itemid in SEDATION_ITEMIDS:
        return "Sedative", round(dose,1), "#9467bd"
    elif itemid in FLUID_ITEMIDS:
        return "Fluid", int(dose), "#2ca02c"
    else:
        return "Other", round(dose,2), "#7f7f7f"

# 약물 투여 완료 - 약물 대분류 시각화를 위한 색상 매칭
def medication_category_completed(category):
    if category == "Antibiotic":
        return 2, "#1f77b4"
    elif category == "Vasopressor":
        return 2, "#d62728"
    elif category == "Sedative":
        return 1, "#9467bd"
    elif category == "Fluid":
        return 0, "#2ca02c"
    else:
        return 2,"#7f7f7f"

# 약물 투여 기본 CSS
def drug_alarm_css():
    st.markdown(
        """
        <style>
        .event-card {
            display: flex;
            align-items: center;
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 10px 0px;
            width: 100%;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            margin-bottom: 10px;
        }

        .event-bar {
            width: 4px;
            align-self: stretch;
            border-radius: 4px;
            margin-right: 12px;
        }

        .event-content {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .event-title {
            font-size: 25px;
            font-weight: 600;
            color: #111827;
            line-height: 1.2;
        }

        .event-meta {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 17px;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# 약물 투여 진행중 표시 카드 
def drug_alarm(drug_name, category, color, starttime, endtime, duration):
    st.markdown(
        f"""
        <div class="event-card">
            <div class="event-bar"></div>
            <div class="event-content">
                <div class="event-title">
                    {drug_name}
                </div>
                <div class="event-meta">
                    <span style='color: {color}'>{category}</span>
                    <span>🕛{starttime}-{endtime} {duration}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 약물 투여 완료 표시 카드 
def drug_alarm_completed(title, category, color, time_range, r):
    time_spans = "".join(
            f"<span style='font-size:17px;color: #6b7280;'>🕛{'TIME '+ str(cnt)} {starttime}-{endtime} ({round(valuenum,r):,}{valueuom})</span>"
            for cnt,starttime,endtime,valuenum,valueuom in time_range
    )
    
    st.markdown(
        f"""
        <div class="event-card">
            <div class="event-bar"></div>
            <div class="event-content">
                <div class="event-title">
                    {title}
                </div>
                <div class="event-meta">
                    <span style='color: {color}'>{category}</span>
                </div>
                {time_spans}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 긴급 이벤트

SEVERITY_RANK = {
    None: -1,
    "YELLOW": 0,
    "ORANGE": 1,
    "RED": 2
}

SEVERITY_FONT_COLOR = {
    None: "white",
    "YELLOW": "#FFCD19",
    "ORANGE": "#FF9900",
    "RED": "#FF0000"
}

SEVERITY_BG_COLOR = {
    None: "white",
    "YELLOW": "#FFF4CC",
    "ORANGE": "#FFE0B2",
    "RED": "#FFD6D6"
}

def make_event(organ, severity, evidence):
    return {
        "organ": organ,
        "severity": severity,
        "font-color" : SEVERITY_FONT_COLOR[severity],
        "background-color" : SEVERITY_BG_COLOR[severity],
        "evidence": [
        evidence
    ]
    }

def update_event(prev_event, new_event, now):
    if prev_event is None and new_event is None:
        return None

    # 새 이벤트 발생
    if prev_event is None and new_event is not None:
        new_event["since"] = now
        return new_event

    # 이벤트 해제
    if prev_event is not None and new_event is None:
        return None

    prev_rank = SEVERITY_RANK[prev_event["severity"]]
    new_rank = SEVERITY_RANK[new_event["severity"]]

    # 🔴 악화: evidence 누적
    if new_rank > prev_rank:
        new_event["since"] = now

        prev_evs = prev_event.get("evidence", [])
        new_evs = new_event.get("evidence", [])

        # 이미 존재하는 증상(e) 집합
        existing = {ev["e"] for ev in prev_evs}

        merged = prev_evs.copy()
        for ev in new_evs:
            if ev["e"] not in existing:
                merged.append(ev)

        new_event["evidence"] = merged
        return new_event

    # 🟠🟡 유지 / 호전: evidence 유지, since 유지
    new_event["since"] = prev_event["since"]
    new_event["evidence"] = prev_event["evidence"]
    return new_event

def critical_circulation(df):
    map_itemids = [220052] # MAP
    lactate_itemids = [50813] # Lactate
    vaso_itemids = [221906, 221289] # Vasopressor

    map_df = df[(df["itemid"].isin(map_itemids)) & (df["valuenum"].notna())]
    lact_df = df[(df["itemid"].isin(lactate_itemids)) & (df["valuenum"].notna())]
    vaso_df = df[(df["itemid"].isin(vaso_itemids))]

    if len(map_df) >= 3:
        recent = map_df.sort_values("starttime").tail(3)
        if (recent["valuenum"] < 55).all():
            return make_event("Circulation", "RED", {'e':"Sustained MAP < 55", 'time':recent["starttime"].iloc[0]})
        if (recent["valuenum"] < 65).all():
            return make_event("Circulation", "ORANGE", {'e':"Sustained MAP < 65", 'time':recent["starttime"].iloc[0]})

    if not vaso_df.empty:
        return make_event("Circulation", "RED", {'e':"Vasopressor initiated", 'time' : vaso_df["starttime"].min()})

    if not lact_df.empty:
        v = lact_df["valuenum"].iloc[-1]
        if v >= 4.0:
            return make_event("Circulation", "RED", {'e':"Lactate ≥ 4.0", 'time':lact_df["starttime"].iloc[-1]})
        if v >= 2.0:
            return make_event("Circulation", "ORANGE", {'e':"Lactate ≥ 2.0", 'time':lact_df["starttime"].iloc[-1]})

    return None

def critical_respiration(df, icu_admit_time):
    spo2_itemids = [220277, 646]
    fio2_itemids = [223835, 190]
    vent_itemids = [224687, 226732] # Ventilator 

    spo2_df = df[(df["itemid"].isin(spo2_itemids)) & (df["valuenum"].notna()) & (df["tablename"] == "chartevents")]
    fio2_df = df[(df["itemid"].isin(fio2_itemids)) & (df["valuenum"].notna()) & (df["tablename"] == "chartevents")]
    vent_df = df[(df["itemid"].isin(vent_itemids)) & (df["tablename"] == "chartevents")]

    if not vent_df.empty and vent_df["starttime"].min() >= icu_admit_time:
        return make_event("Respiration", "RED", {'e':"New invasive mechanical ventilation", 'time':vent_df["starttime"].min()})
    
    if len(spo2_df) >= 3:
        recent = spo2_df.sort_values("starttime").tail(3)
        if (recent["valuenum"] < 90).all():
            return make_event("Respiration", "RED", {'e':"SpO₂ < 90% sustained", 'time':recent["starttime"].iloc[0]})
        if (recent["valuenum"] < 94).all():
            return make_event("Respiration", "ORANGE", {'e':"SpO₂ < 94% sustained", 'time':recent["starttime"].iloc[0]})

    if not fio2_df.empty:
        fio2_curr = fio2_df["valuenum"].iloc[-1]
        if fio2_curr >= 80:
            return make_event("Respiration", "RED", {'e':"High FiO₂ requirement", 'time':fio2_df["starttime"].iloc[-1]})       
        if fio2_curr >= 50: 
            return make_event("Respiration", "ORANGE", {'e':"Elevated FiO₂ requirement", 'time':fio2_df["starttime"].iloc[-1]})

    return None

def critical_kidney(df):
    uop_itemids = [226559, 226560, 226561, 226584] # Urine Output : 소변 배출량
    cr_itemids = [50912] # Creatinine
    rrt_itemids = [225802] # RRT 

    uop_df = df[(df["itemid"].isin(uop_itemids)) & (df["valuenum"].notna())]
    cr_df = df[(df["itemid"].isin(cr_itemids)) & (df["valuenum"].notna())]
    rrt_df = df[(df["itemid"].isin(rrt_itemids))]

    if not rrt_df.empty:
        return make_event("Kidney", "RED", {'e':"Renal replacement therapy initiated", 'time':rrt_df["starttime"].min()})

    if len(uop_df) >= 3:
        recent = uop_df.sort_values("starttime").tail(3)
        if (recent["valuenum"] < 0.3).all():
            return make_event("Kidney", "RED", {'e':"Severe oliguria", 'time':recent["starttime"].iloc[0]})

        if (recent["valuenum"] < 0.5).all():
            return make_event("Kidney", "ORANGE", {'e':"Oliguria", 'time':recent["starttime"].iloc[0]})

    if not cr_df.empty:
        cr_curr = cr_df["valuenum"].iloc[-1]
        if cr_curr >= 3.0:
            return make_event("Kidney", "RED", {'e':"Creatinine ≥ 3.0", 'time':cr_df["starttime"].iloc[-1]})
        if cr_curr >= 1.5:
            return make_event("Kidney", "ORANGE", {'e':"Creatinine ≥ 1.5", 'time':cr_df["starttime"].iloc[-1]})

    return None

def critical_neurologic(df):
    gcs_itemids = [220739] # GCS
    sed_itemids = [221744] # Sedation : 진정제
    seizure_itemids = [225401] # Seizure : 발작 (간질 발작이나 경련 발생 기록)

    gcs_df = df[(df["itemid"].isin(gcs_itemids)) & (df["valuenum"].notna())]
    sed_df = df[(df["itemid"].isin(sed_itemids))]
    seiz_df = df[(df["itemid"].isin(seizure_itemids))]

    if not seiz_df.empty:
        return make_event("Neurologic", "RED", {'e':"New seizure activity", 'time' : seiz_df["starttime"].iloc[-1]})

    if len(gcs_df) >= 2:
        gcs_curr = gcs_df["valuenum"].iloc[-1]
        if gcs_curr <= 8 and sed_df.empty:
            return make_event("Neurologic", "RED", {'e':"GCS ≤ 8", 'time':gcs_df["starttime"].iloc[-1]})
        if gcs_curr <= 12:
            return make_event("Neurologic", "ORANGE", {'e':"GCS ≤ 12", 'time':gcs_df["starttime"].iloc[-1]})

    return None

def critical_liver(df):
    bili_itemids = [50885] # Bilirubin
    inr_itemids = [51237] # 혈액 응고 지표
    ast_itemids = [50878] # 간세포 손상 시 혈액으로 누출되는 효소

    bili_df = df[(df["itemid"].isin(bili_itemids)) & (df["valuenum"].notna())]
    inr_df = df[(df["itemid"].isin(inr_itemids)) & (df["valuenum"].notna())]
    ast_df = df[(df["itemid"].isin(ast_itemids)) & (df["valuenum"].notna())]

    if not bili_df.empty:
        bili_curr = bili_df["valuenum"].iloc[-1]
        if bili_curr >= 6.0:
            return make_event("Liver", "RED", {'e': "Bilirubin ≥ 6.0", 'time': bili_df["starttime"].iloc[-1]})
        if bili_curr >= 2.0:
            return make_event("Liver", "ORANGE", {'e': "Bilirubin ≥ 2.0", 'time': bili_df["starttime"].iloc[-1]})
 
    if not inr_df.empty:
        inr_curr = inr_df["valuenum"].iloc[-1]
        if inr_curr >= 2.0:
            return make_event("Liver", "RED", {'e': "INR ≥ 2.0", 'time': inr_df["starttime"].iloc[-1]})
        if inr_curr >= 1.5:
            return make_event("Liver", "ORANGE", {'e': "INR ≥ 1.5", 'time': inr_df["starttime"].iloc[-1]})

    if not ast_df.empty:
        ast_curr = ast_df["valuenum"].iloc[-1]
        if ast_curr >= 1000:
            return make_event("Liver", "RED", {'e': "AST ≥ 1000", 'time': ast_df["starttime"].iloc[-1]})
        if ast_curr >= 200:
            return make_event("Liver", "ORANGE", {'e': "AST ≥ 200", 'time': ast_df["starttime"].iloc[-1]})
    
    return None

def update_all_events(df, prev_events, icu_admit_time, now):
    return {
        "Circulation": update_event(prev_events.get("Circulation"),
                                    critical_circulation(df), now),

        "Respiration": update_event(prev_events.get("Respiration"),
                                    critical_respiration(df, icu_admit_time), now),

        "Kidney": update_event(prev_events.get("Kidney"),
                               critical_kidney(df), now),

        "Neurologic": update_event(prev_events.get("Neurologic"),
                                   critical_neurologic(df), now),

        "Liver": update_event(prev_events.get("Liver"),
                               critical_liver(df), now),
    }

def format_since(since, now):
    if since is None:
        return ""

    if isinstance(since, str):
        since = pd.to_datetime(since)

    delta = now - since
    total_minutes = int(delta.total_seconds() // 60)

    hours = total_minutes // 60
    minutes = total_minutes % 60

    if hours > 0:
        return f"{hours}h {minutes}m ago"
    else:
        return f"{minutes}m ago"

def organ_event_css():
    st.markdown(
        """
        <style>
        .sicucard {
            display: flex;
            align-items: center;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 15px;
            width: 100%;
            box-shadow: rgba(60, 64, 67, 0.3) 0px 1px 2px 0px, rgba(60, 64, 67, 0.15) 0px 2px 6px 2px;
            margin-bottom: 10px;
        }
        
        .sicuevent-title {
            font-size: 25px;
            font-weight: 600;
            line-height: 1.2;
            padding: 0 0 10px 0;
        }

        .evidence {
            gap: 12px;
            font-size: 20px;
            color: #6b7280;
        }
        
        </style>
        """,
        unsafe_allow_html=True
    )

def organ_event_card(card_info, current_time):
    evi_spans = "".join(
            f"<span>{evi['e']} ({format_since(evi['time'], current_time)})</span><br>"
            for evi in card_info['evidence']
    )

    icon_color = card_info['font-color'].replace('#','')

    st.markdown(
    f"""
    <div class="sicucard" style="border: 3px solid {card_info['font-color']}; background-color:{card_info['background-color']}">
        <div class="sicuevent-content">
            <div class="sicuevent-title" style = "color: {card_info['font-color']};">
                <img src="https://img.icons8.com/?size=100&id=59782&format=png&color={icon_color}" style="height:23px;">
                {card_info['organ']}
            </div>
            <div class="evidence">
                {evi_spans}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True)
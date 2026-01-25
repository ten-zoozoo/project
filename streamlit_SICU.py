import streamlit as st
import time
from datetime import timedelta
import pandas as pd
import altair as alt
import numpy as np
import polars as pl
from streamlit_dashboard_func import *
from predict_data_preprocessing import *
from dotenv import load_dotenv
import os
import psycopg2
import io
from streamlit_autorefresh import st_autorefresh

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

# ML 모델 로드하기
@st.cache_resource
def load_model():
    return ICU24hRiskModel(streamlit_path + "icu_xgb_24h_calibrated.pkl")
model = load_model()

st.set_page_config(layout="wide")
basic_css()

# 1. 현재 입실 중인 환자 (시작 시간 : 2176-02-24 03:03:54)

now = datetime(2176,2,24,3,3,54)

patients_list = pd.read_sql(
    f"""
    SELECT *
    FROM patients
    WHERE
        intime >= TIMESTAMP '{str(now)}'
    AND intime <= TIMESTAMP '{str(now)}' + INTERVAL '24 hours'  -- 앞으로 24시간
    AND (
        outtime IS NULL
        OR outtime > TIMESTAMP '{str(now)}'
    )
    ORDER BY intime;
""", conn)

# 2. session_state 초기화
if 'selected_patient' not in st.session_state:
    if len(patients_list) > 0: 
        st.session_state.selected_patient = patients_list['stay_id'].iloc[0]
    else:
        st.session_state.selected_patient = None
        st.warning('입실한 환자가 없습니다.')
if 'stay_id_input' not in st.session_state: # 입력받은 stay_id_input값
    st.session_state.stay_id_input = ""
if 'last_search' not in st.session_state:
    st.session_state.last_search = ""
if 'force_clear' not in st.session_state:
    st.session_state.force_clear = False
if st.session_state.force_clear:
    st.session_state.stay_id_input = ""
    st.session_state.force_clear = False

cols = st.columns(3)

# stay_id 입력받기
with cols[0]:
    with st.expander("🔍 STAY_ID 검색", expanded=False):
        st.caption("환자 ID를 입력하고 Enter를 누르세요")
        input_stay_id = st.text_input(
            "STAY_ID를 입력하세요",
            key="stay_id_input",
            label_visibility="collapsed",
            placeholder="예: 31488097"
        )

# 검색어가 입력되면 selected_patient 업데이트
if input_stay_id and input_stay_id != st.session_state.last_search:
    try:
        st.session_state.selected_patient = int(input_stay_id)
        st.session_state.last_search = input_stay_id
        st.session_state.force_clear = True
        st.rerun()
    except ValueError:
        st.error("유효한 숫자를 입력하세요")

# selected_patient 값 가져오기
selected_patient = st.session_state.selected_patient
patient, mimic_df = load_patient_data(selected_patient)
intime = patient['intime'].iloc[0] # 환자가 ICU 입실한 시간

hadm_id, subject_id = patient['hadm_id'].iloc[0], patient['subject_id'].iloc[0]
weight = mimic_df[(mimic_df['itemid'].isin([226512, 224639, 226531])) & (mimic_df['tablename'] == 'chartevents')]
height = mimic_df[(mimic_df['itemid'].isin([226730, 226707])) & (mimic_df['tablename'] == 'chartevents')]
p_weight = weight['valuenum'].iloc[0] if not weight.empty else '-'
p_height = height['valuenum'].iloc[0] if not height.empty else '-'

diagnoses_icd_df = subject_filtering('diagnoses_icd',subject_id, hadm_id)
procedures_icd_df = subject_filtering('procedures_icd',subject_id, hadm_id)

# 세션 상태 초기화

# 현재 시각이 없으면
if 'start_time' not in st.session_state:
    st.session_state.start_time = now
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'elapsed_seconds' not in st.session_state:
    st.session_state.elapsed_seconds = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'patient_admission_times' not in st.session_state:
    st.session_state.patient_admission_times = dict(zip(patients_list['stay_id'],patients_list['intime']))  # {selected_patient: 입장시간}
if 'last_prediction_hours' not in st.session_state:
    st.session_state.last_prediction_hours = {}  # {selected_patient: [6, 12, 18, ...]}
if 'pred_store' not in st.session_state:
    st.session_state.pred_store = pd.DataFrame(
        columns=['selected_patient', 'admission_time', 'check_time', 'hours', 'pred_die_percent', 'diff'])
if "completed_patients" not in st.session_state: # UI/로직에서 더 이상 추적하지 않을 환자
    st.session_state.completed_patients = set()
if 'flushed_patients' not in st.session_state: # DB에 이미 INSERT한 환자
    st.session_state.flushed_patients = set()

# 컨트롤 버튼
col1, col2, col3 = st.columns(3)
st.session_state.is_running = True

# 현재 시간 계산
current_time = st.session_state.start_time + timedelta(seconds=st.session_state.elapsed_seconds)

latest_predictions = (
    st.session_state.pred_store[
        (st.session_state.pred_store['hours'] < 24)          # 24시간 미만
    ]
    .sort_values('hours')                    # 시간순 정렬
    .groupby('selected_patient')
    .tail(1)                                 # 각 환자의 마지막 기록만
    .sort_values('pred_die_percent', ascending=False)  # 사망률 높은 순
)

# 최근에 입장한 사람
currently_admitted = patients_list[patients_list['intime'] <= current_time].copy()

if len(latest_predictions) > 0:
    sorted_stay_ids = []
    for sid in latest_predictions['selected_patient'].tolist():
        if sid in st.session_state.completed_patients:
            continue
        if sid in st.session_state.patient_admission_times:
            elapsed_hours = (current_time - st.session_state.patient_admission_times[sid]).total_seconds() / 3600
            if elapsed_hours < 24:
                sorted_stay_ids.append(sid)
else:
    sorted_stay_ids = []

# 현재 시간 기준 입실한 환자 중 예측 없는 환자도 추가
for stay_id in currently_admitted['stay_id']:
    if stay_id not in sorted_stay_ids and stay_id not in st.session_state.completed_patients:
        if stay_id in st.session_state.patient_admission_times:
            elapsed_hours = (current_time - st.session_state.patient_admission_times[stay_id]).total_seconds() / 3600
            if elapsed_hours < 24:
                sorted_stay_ids.append(stay_id)

st.sidebar.markdown("<h2 style='padding : 0'>EMERGENCY PATIENT</h2>",unsafe_allow_html=True)

if st.session_state.is_running:
    filtered_data = mimic_df[mimic_df['starttime'] <= current_time]
    pred_all_phase(
        int(selected_patient),
        current_time,
        patient,
        filtered_data,
        diagnoses_icd_df,
        procedures_icd_df,
        model
    )
    
    for stay_id in sorted_stay_ids:
        patient_temp, mimic_df_temp = load_patient_data(stay_id)
        hadm_id_temp = patient_temp['hadm_id'].iloc[0]
        subject_id_temp = patient_temp['subject_id'].iloc[0]
        intime_temp = patient_temp['intime'].iloc[0]
        diagnoses_icd_df_temp = subject_filtering('diagnoses_icd', subject_id_temp, hadm_id_temp)
        procedures_icd_df_temp = subject_filtering('procedures_icd', subject_id_temp, hadm_id_temp)

        filtered_data_temp = mimic_df_temp[mimic_df_temp['starttime'] <= current_time]

        pred_all_phase(
            int(stay_id),
            current_time,
            patient_temp,
            filtered_data_temp,
            diagnoses_icd_df_temp,
            procedures_icd_df_temp,
            model
        )

# 사이드바에 정렬된 순서대로 표시
if len(latest_predictions) > 0:
    sorted_stay_ids = []
    for sid in latest_predictions['selected_patient'].tolist():
        if sid in st.session_state.completed_patients:
            continue
        
        # 24시간 경과 체크
        if sid in st.session_state.patient_admission_times:
            elapsed_hours = (current_time - st.session_state.patient_admission_times[sid]).total_seconds() / 3600
            if elapsed_hours < 24:  # 24시간 미만만 포함
                sorted_stay_ids.append(sid)
else:
    sorted_stay_ids = []
    st.sidebar.warning('최근 입실한 환자가 없습니다.')

if len(sorted_stay_ids) > 0:
    same_period_patients = [i for i in sorted_stay_ids if i != selected_patient]
    for stay_id in same_period_patients:
        row = patients_list[patients_list['stay_id'] == stay_id].iloc[0]
        intime_str = row['intime'].strftime("%Y-%m-%d")
        
        top_5_css()
        
        # 현재 환자의 최신 예측 데이터
        latest_per_df = st.session_state.pred_store[
            st.session_state.pred_store['selected_patient'] == stay_id
        ]

        if len(latest_per_df) > 0:
            latest_per = str(int(latest_per_df.sort_values('hours')['pred_die_percent'].iloc[-1]))
            latest_per += '%'
        else:
            latest_per = '-'

        # 사이드바 카드 렌더링
        st.sidebar.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                gap: 14px;
                padding: 14px 16px;
                border: 2px solid #1f77ff;
                border-radius: 14px;
                max-width: 420px;
                margin-bottom: 10px;
                background-color: #ffffff;
            ">
                <div style="flex: 1;">
                    <div style="
                        font-weight: 700;
                        font-size: 20px;
                        line-height: 1.2;
                    ">
                        STAY_ID: {stay_id}
                    </div>
                    <div style="
                        color: #6b7280;
                        font-size: 13px;
                        margin-top: 4px;
                    ">
                        Register: {intime_str}
                    </div>
                </div>
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 60px;
                    font-weight: 700;
                    font-size: 28px;
                    color: #374151;
                ">
                    {latest_per}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 대시보드 이동 버튼
        if st.sidebar.button(
            f" -> View Dashboard (ID: {stay_id})",
            key=f"btn_{stay_id}", # key값에 중복 방지를 위해 stay_id 활용
            use_container_width=True
        ):
            st.session_state.selected_patient = stay_id
            st.session_state.last_search = ""
            st.session_state.start_time = current_time
            st.session_state.force_clear = False
            st.rerun()


# 메인 화면
if st.session_state.selected_patient:
    st.title(f"SICU Patient Monitoring & AI Mortality Risk Dashboard")
    total_seconds = (current_time - intime).total_seconds()
    hours = int(total_seconds // 3600) if total_seconds > 0 else 0
    minutes = int((total_seconds % 3600) // 60)
    st.subheader(f"현재 시간 : {current_time.strftime('%Y-%m-%d %H:%M:%S')} (ICU 입실 후 {hours}시간 {minutes}분 경과)")
    filtered_data = mimic_df[mimic_df['starttime'] <= current_time]  # current_time 사용!
    
    if st.session_state.selected_patient in st.session_state.patient_admission_times:
        elapsed_hours = (current_time - st.session_state.patient_admission_times[st.session_state.selected_patient]).total_seconds() / 3600
        if elapsed_hours >= 24:
            st.session_state.completed_patients.add(st.session_state.selected_patient)
            if len(sorted_stay_ids) > 0:
                st.session_state.selected_patient = sorted_stay_ids[0]
                st.rerun()
            else:
                st.warning("모든 환자 모니터링 완료")
                st.stop()
    
    # 환자 정보
    cols = st.columns(1)
    with cols[0].container(border=True,):
        st.markdown(f'<h4 style="padding:0">Patient Info</h4>',unsafe_allow_html=True)
        patient_info_css(selected_patient, patient, intime)
    

    # 환자 사망 예측률
    cols = st.columns(1)
    
    with cols[0].container(border=True, height="stretch"):
        st.markdown(f'<h4 style="padding:0">AI-Based Mortality Prediction</h4>',unsafe_allow_html=True)
        
        # pred_store에서 현재 선택된 환자의 데이터만 필터링
        if not st.session_state.pred_store.empty:
            pred_store_filtered = st.session_state.pred_store[
                st.session_state.pred_store['selected_patient'] == selected_patient
            ].copy()
            pred_store_filtered = pred_store_filtered.sort_values('hours').reset_index(drop=True)
        else:
            pred_store_filtered = pd.DataFrame()
    
        predict_die_css()
        predict_die_col = st.columns(4)

        for i in range(4):
            with predict_die_col[i]:
                if i < len(pred_store_filtered):
                    var = pred_store_filtered.iloc[i]

                    symbol = '▲' if var['diff'] > 0 else '▼'
                    color = 'red' if var['diff'] > 0 else 'green'

                    predict_die_data_yes_css(
                        var['hours'],  # window 대신 hours 사용
                        var['check_time'],  # pred_time 대신 check_time 사용
                        var['pred_die_percent'],
                        var['diff'],
                        color,
                        symbol
                    )
                else:
                    predict_die_data_no_css((i + 1) * 6)

    # Live Vitals
    cols = st.columns(1)
    with cols[0].container(border=True, height="stretch"):
        "#### Live Vitals"
        shared_x = alt.X("starttime:T", title="", scale=alt.Scale(domain=[intime, patient['intime24'].iloc[0]]))
        
        cols = st.columns(3)

        # MAP + graph
        with cols[0].container(height="stretch", border=True):
            try: 
                map_art = filtered_data[(filtered_data['itemid'] == 220052) & (filtered_data['tablename'] == 'chartevents')].copy()
                last_value = map_art.iloc[-1]['valuenum']
                color, status = map_status(last_value) 
                metric_card_with_trend(map_art,last_value,"MAP (ART)","mmHg",status,color)
            except:
                return_card2('-',"MAP (ART)","mmHg", "Not Recorded", "gray")
        
        # heart_rate + graph
        with cols[1].container(height="stretch", border=True):
            try: 
                heart_rate = filtered_data[(filtered_data['itemid'] == 220045) & (filtered_data['tablename'] == 'chartevents')].copy()
                last_value = heart_rate.iloc[-1]['valuenum']
                color, status = map_status(last_value) 

                metric_card_with_trend(heart_rate,last_value,"Heart Rate (HR)","bpm",status,color)

            except:
                return_card2('-',"Heart Rate (HR)","bpm", "Not Recorded", "gray")
        
        # spo2 vital sign + graph
        with cols[2].container(height="stretch", border=True):
            try: 
                spo2 = filtered_data[(filtered_data['itemid'] == 220277) & (filtered_data['tablename'] == 'chartevents')].copy()
                last_value = spo2.iloc[-1]['valuenum']
                color, status = map_status(last_value) 

                metric_card_with_trend(spo2,last_value,"SpO₂","%",status,color)

            except:
                return_card2('-',"SpO₂","%", "Not Recorded", "gray")

        cols = st.columns(3)

        # lactate + graph
        with cols[0].container(height="stretch", border=True):
            try: 
                lactate = filtered_data[(filtered_data['itemid'] == 50813) & (filtered_data['tablename'] == 'labevents')].copy()
                last_value = lactate.iloc[-1]['valuenum']
                color, status = lactate_status(last_value) 
                metric_card_with_trend(lactate,last_value,"Lactate","mmol/L",status,color)

            except:
                return_card2('-',"Lactate","mmol/L","Not Recorded", "gray")
        
        # urine_output + graph
        with cols[1].container(height="stretch", border=True):
            try: 
                urine_itemid = [226559, 226560, 226561, 226584]
                urine_output = filtered_data[(filtered_data['itemid'].isin(urine_itemid)) & (filtered_data['tablename'] == 'outputevents')].copy()
                last_value = int(urine_output.iloc[-1]['valuenum'])
                color, status = uop_status(last_value) 
                metric_card_with_trend(urine_output,last_value,"Urine Output","mL/kg/hr",status,color)
            except:
                return_card2('-',"Urine Output","mL/kg/hr", "Not Recorded", "gray")
        
        # Respiratory Rate (RR) + graph
        with cols[2].container(height="stretch", border=True):
            try: 
                rr = filtered_data[(filtered_data['itemid'].isin([618, 220210])) & (filtered_data['tablename'] == 'chartevents')].copy()
                last_value = rr.iloc[-1]['valuenum']
                color, status = rr_status(last_value) 
                metric_card_with_trend(rr,last_value,"RR","insp/min",status,color)
            except:
                return_card2('-',"RR","insp/min", "Not Recorded", "gray")

    # Events
    drug_alarm_css()
    organ_event_css()
    
    cols = st.columns(2)
    if "prev_events" not in st.session_state:
            st.session_state.prev_events = {
                "Circulation": None,
                "Respiration": None,
                "Kidney": None,
                "Neurologic": None,
                "Liver": None
            }

    # # Critical Event
    with cols[0].container(border=True, height=600):
        "#### Critical Event"

        organ_status = update_all_events(
            filtered_data,
            st.session_state.prev_events,
            patient['intime'].astype('datetime64[ns]').iloc[0],
            current_time
        )

        # None 제거
        l = [v for v in organ_status.values() if v]

        SEVERITY_RANK = {
            None: -1,
            "YELLOW": 0,
            "ORANGE": 1,
            "RED": 2
        }

        SEVERITY_COLOR = {
            None: "white",
            "YELLOW": "#FFF4CC",
            "ORANGE": "#FFE0B2",
            "RED": "#FFD6D6"
        }
        
        df_organs = pd.DataFrame(l).reset_index(drop=True)
        if df_organs.shape[0] > 0:        
            df_organs["severity_rank"] = df_organs["severity"].map(SEVERITY_RANK)
            df_organs = df_organs.sort_values('severity_rank')

        for _, row in df_organs.iterrows():
            organ_event_card({
                "organ": row["organ"],
                "severity": row["severity"],
                "font-color" : row["font-color"],
                "background-color" : row["background-color"],
                "evidence": row["evidence"]   # 그대로 넘김 (list[dict])
            } ,current_time)

        # 🔑 반드시 마지막에 저장
        st.session_state.prev_events = organ_status

    # Medication in Progress
    with cols[1].container(border=True, height=600):
        "#### Medication in Progress"
        VASOPRESSOR_ITEMIDS = [221906,221289,222315,221749, 221662]
        SEDATION_ITEMIDS = [222168,221668,223257,221712,221385]
        FLUID_ITEMIDS = [225158,225828,225159,225161,225166,225160,220864,220862]
        ANTIBIO_ITEMIDS = [225798,225970,225942,225936,225931,225948,225945,225913,225952,
                        225946,225934,225950,225930,225912,225929,225947,225932]
        
        now_status = st.pills(
            "STATUS", ['🟢 processing', '⚫ complete'], default='🟢 processing', selection_mode="single"
        )
        drug_itemid = VASOPRESSOR_ITEMIDS + SEDATION_ITEMIDS + FLUID_ITEMIDS + ANTIBIO_ITEMIDS
        drug_df = filtered_data[(filtered_data['itemid'].isin(drug_itemid)) & (filtered_data['tablename'] == 'inputevents')].copy()

        if len(drug_df) > 0:
            # 상태별로 데이터 미리 분리
            processing_df = drug_df[drug_df["endtime"].isna() | (drug_df["endtime"] > current_time)].copy()
            complete_df = drug_df[drug_df["endtime"].notna() & (drug_df["endtime"] <= current_time)].copy()

            # Processing 탭
            if now_status == '🟢 processing':
                for _, row in processing_df.iterrows():
                    start_dt = pd.to_datetime(row['starttime'])
                    start_str = start_dt.strftime("%H:%M")
                    drug_name = row['label']
                    category, valuenum, color = medication_category(row['itemid'], row['valuenum'])
                    
                    elapsed = current_time - start_dt
                    elapsed_str = f"{elapsed.seconds//3600}h {(elapsed.seconds%3600)//60}m"
                    drug_alarm(
                        drug_name=drug_name,
                        category=category,
                        color=color,
                        starttime=f"{start_str}",
                        endtime="",
                        duration=''
                    )
                
            # Complete 탭
            elif now_status == '⚫ complete':
                complete_df["category"] = np.select(
                    [
                        complete_df["itemid"].isin(ANTIBIO_ITEMIDS).copy(),
                        complete_df["itemid"].isin(VASOPRESSOR_ITEMIDS).copy(),
                        complete_df["itemid"].isin(SEDATION_ITEMIDS).copy(),
                        complete_df["itemid"].isin(FLUID_ITEMIDS).copy()
                    ],
                    ["Antibiotic", "Vasopressor", "Sedative", "Fluid"],
                    default="Other"
                )
                IMPORTANT_ORDER = [
                    "Vasopressor",
                    "Sedative",
                    "Antibiotic",
                    "Fluid"
                ]

                complete_df["category"] = pd.Categorical(
                    complete_df["category"],
                    categories=IMPORTANT_ORDER,
                    ordered=True
                )

                complete_df = complete_df.sort_values('category')
                for drug_name in list(complete_df['label'].unique()):
                    one_drug = complete_df[complete_df['label'] == drug_name]
                    one_drug["round"] = one_drug.groupby(["label"]).cumcount() + 1
                    unit = one_drug['valueuom'].iloc[0]
                    r, color = medication_category_completed(one_drug['category'].iloc[0])
                    one_drug['valuenum'] = round(one_drug['valuenum'],r)
                    drug_dose_sum = one_drug['valuenum'].sum()
                    one_drug[['starttime', 'endtime']] = (
                        one_drug[['starttime', 'endtime']]
                        .apply(lambda col: col.dt.strftime("%H:%M"))
                    )
                    title = f"{drug_name} 누적 투여량 {round(drug_dose_sum,r):,}{unit}"
                    time_range = one_drug[['round','starttime','endtime','valuenum','valueuom']].values.tolist()

                    drug_alarm_completed(title, one_drug['category'].iloc[0], color, time_range, r)

# 타이머 실행 (맨 마지막에!)
if st.session_state.is_running:
    if st.session_state.elapsed_seconds < 24 * 3600:
        st.session_state.elapsed_seconds += 3600
        st.rerun()
    else:
        st.session_state.is_running = False  # 24시간 도달 시 정지


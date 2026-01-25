import pandas as pd 
import numpy as np

# patient_df : 환자 데이터 한 줄
# mimic_df : 이벤트 다 들어있는거

def apache_temp_score(val):
    if pd.isna(val): return np.nan
    if val >= 41.0: return 4
    if 39.0 <= val <= 40.9: return 3
    if 38.5 <= val <= 38.9: return 1
    if 36.0 <= val <= 38.4: return 0
    if 34.0 <= val <= 35.9: return 1
    if 32.0 <= val <= 33.9: return 2
    if 30.0 <= val <= 31.9: return 3
    return 4

def apache_map_score(val):
    if pd.isna(val): return np.nan
    if val >= 160: return 4
    if 130 <= val <= 159: return 3
    if 110 <= val <= 129: return 2
    if 70 <= val <= 109: return 0
    if 50 <= val <= 69: return 2
    return 4

def apache_hr_score(val):
    if pd.isna(val): return np.nan
    if val >= 180: return 4
    if 140 <= val <= 179: return 3
    if 110 <= val <= 139: return 2
    if 70 <= val <= 109: return 0
    if 55 <= val <= 69: return 2
    if 40 <= val <= 54: return 3
    return 4

def apache_rr_score(val):
    if pd.isna(val): return np.nan
    if val >= 50: return 4
    if 35 <= val <= 49: return 3
    if 25 <= val <= 34: return 1
    if 12 <= val <= 24: return 0
    if 10 <= val <= 11: return 1
    if 6 <= val <= 9: return 2
    return 4

def apache_pao2_score(val):
    if pd.isna(val): return np.nan
    if val < 55: return 4
    if 55 <= val <= 60: return 3
    if 61 <= val <= 70: return 1
    return 0

def apache_ph_score(val):
    if pd.isna(val): return np.nan
    if val >= 7.70: return 4
    if 7.60 <= val <= 7.69: return 3
    if 7.50 <= val <= 7.59: return 1
    if 7.33 <= val <= 7.49: return 0
    if 7.25 <= val <= 7.32: return 2
    if 7.15 <= val <= 7.24: return 3
    return 4

def apache_sodium_score(val):
    if pd.isna(val): return np.nan
    if val >= 180: return 4
    if 160 <= val <= 179: return 3
    if 155 <= val <= 159: return 2
    if 150 <= val <= 154: return 1
    if 130 <= val <= 149: return 0
    if 120 <= val <= 129: return 2
    if 111 <= val <= 119: return 3
    return 4

def apache_potassium_score(val):
    if pd.isna(val): return np.nan
    if val >= 7.0: return 4
    if 6.0 <= val <= 6.9: return 3
    if 5.5 <= val <= 5.9: return 1
    if 3.5 <= val <= 5.4: return 0
    if 3.0 <= val <= 3.4: return 1
    if 2.5 <= val <= 2.9: return 2
    return 4

def apache_creatinine_score(val):
    if pd.isna(val): return np.nan
    if val >= 3.5: return 4
    if 2.0 <= val <= 3.4: return 3
    if 1.5 <= val <= 1.9: return 2
    if 0.6 <= val <= 1.4: return 0
    return 2

def apache_hct_score(val):
    if pd.isna(val): return np.nan
    if val >= 60: return 4
    if 50 <= val <= 59.9: return 2
    if 46 <= val <= 49.9: return 1
    if 30 <= val <= 45.9: return 0
    if 20 <= val <= 29.9: return 2
    return 4

def apache_wbc_score(val):
    if pd.isna(val): return np.nan
    if val >= 40: return 4
    if 20 <= val <= 39.9: return 2
    if 15 <= val <= 19.9: return 1
    if 3 <= val <= 14.9: return 0
    if 1 <= val <= 2.9: return 2
    return 4

def apache_gcs_score(val):
    if pd.isna(val): return np.nan
    return 15 - val

def calc_apache(mimic_df):
    # windows = [6, 12, 18, 24]

    # # 매핑 설정 (변수명: [ItemID들], 스코어 함수)
    # item_config = {
    #     'temp': ([223761, 223762], apache_temp_score),
    #     'map': ([220052], apache_map_score),
    #     'hr': ([220045], apache_hr_score),
    #     'rr': ([220210], apache_rr_score),
    #     'pao2': ([220224], apache_pao2_score),
    #     'ph': ([50820], apache_ph_score),
    #     'sodium': ([50983], apache_sodium_score),
    #     'potassium': ([50971], apache_potassium_score),
    #     'creatinine': ([50912], apache_creatinine_score),
    #     'hct': ([51221], apache_hct_score),
    #     'wbc': ([51301], apache_wbc_score)
    # }

    # # 최종 데이터프레임 초기화 (CSV의 모든 stay_id 포함)
    # apache_score_df = pd.DataFrame({'stay_id': sorted(mimic_df['stay_id'].unique())})

    # # A. 일반 항목 처리
    # for var_name, (ids, score_func) in item_config.items():
    #     # 해당 아이템 필터링 및 stay_id, window별 집계
    #     filtered = mimic_df[mimic_df['itemid'].isin(ids)]
    #     agg = filtered.groupby(['stay_id', 'window'])['valuenum'].agg(['min', 'max']).unstack('window')

    #     for w in windows:
    #         # 해당 윈도우 데이터가 있는 경우에만 처리
    #         if w in agg.columns.levels[1]:
    #             # Score 계산 컬럼 한 줄로 생성
    #             apache_score_df[f"apache_{var_name}_score_min_w{w}"] = apache_score_df['stay_id'].map(agg[('min', w)]).apply(score_func)
    #             apache_score_df[f"apache_{var_name}_score_max_w{w}"] = apache_score_df['stay_id'].map(agg[('max', w)]).apply(score_func)
    #         else:
    #             # 데이터가 없는 경우 NaN 컬럼 생성
    #             apache_score_df[f"apache_{var_name}_score_min_w{w}"] = np.nan
    #             apache_score_df[f"apache_{var_name}_score_max_w{w}"] = np.nan

    # # B. GCS 특수 항목 처리 (3개 요소 동시 존재 시 합산)
    # gcs_ids = [220739, 223900, 223901]
    # gcs_df = mimic_df[mimic_df['itemid'].isin(gcs_ids)]

    # # stay_id, starttime, window별로 GCS 3개 항목이 모두 있는 기록만 필터링
    # gcs_complete = gcs_df.groupby(['stay_id', 'starttime', 'window']).filter(lambda x: x['itemid'].nunique() == 3)
    # # 시점별 합산
    # gcs_summed = gcs_complete.groupby(['stay_id', 'starttime', 'window'])['valuenum'].sum().reset_index()

    # # 윈도우별 GCS 집계
    # gcs_agg = gcs_summed.groupby(['stay_id', 'window'])['valuenum'].agg(['min', 'max']).unstack('window')

    # for w in windows:
    #     if w in gcs_agg.columns.levels[1]:
    #         apache_score_df[f"apache_gcs_score_min_w{w}"] = apache_score_df['stay_id'].map(gcs_agg[('min', w)]).apply(apache_gcs_score)
    #         apache_score_df[f"apache_gcs_score_max_w{w}"] = apache_score_df['stay_id'].map(gcs_agg[('max', w)]).apply(apache_gcs_score)
    #     else:
    #         apache_score_df[f"apache_gcs_score_min_w{w}"] = np.nan
    #         apache_score_df[f"apache_gcs_score_max_w{w}"] = np.nan
    
    # return apache_score_df

    windows = [6, 12, 18, 24]

    item_config = {
        'temp': ([223761, 223762], apache_temp_score),
        'map': ([220052], apache_map_score),
        'hr': ([220045], apache_hr_score),
        'rr': ([220210], apache_rr_score),
        'pao2': ([220224], apache_pao2_score),
        'ph': ([50820], apache_ph_score),
        'sodium': ([50983], apache_sodium_score),
        'potassium': ([50971], apache_potassium_score),
        'creatinine': ([50912], apache_creatinine_score),
        'hct': ([51221], apache_hct_score),
        'wbc': ([51301], apache_wbc_score)
    }

    # 결과 DF (모든 stay_id 유지)
    apache_score_df = pd.DataFrame({
        'stay_id': sorted(mimic_df['stay_id'].unique())
    })

    # 공통 expected 컬럼
    expected_cols = pd.MultiIndex.from_product(
        [['min', 'max'], windows]
    )

    # =========================
    # A. 일반 항목 처리
    # =========================
    for var_name, (ids, score_func) in item_config.items():

        filtered = mimic_df[mimic_df['itemid'].isin(ids)]

        if filtered.empty:
            # 데이터 자체가 없으면 전부 NaN 컬럼 생성
            for w in windows:
                apache_score_df[f"apache_{var_name}_score_min_w{w}"] = np.nan
                apache_score_df[f"apache_{var_name}_score_max_w{w}"] = np.nan
            continue

        agg = (
            filtered
            .groupby(['stay_id', 'window'])['valuenum']
            .agg(['min', 'max'])
            .unstack('window')
        )

        # ⭐ 컬럼 강제 생성 (없으면 NaN)
        agg = agg.reindex(columns=expected_cols)

        for w in windows:
            apache_score_df[f"apache_{var_name}_score_min_w{w}"] = (
                apache_score_df['stay_id']
                .map(agg[('min', w)])
                .apply(score_func)
            )

            apache_score_df[f"apache_{var_name}_score_max_w{w}"] = (
                apache_score_df['stay_id']
                .map(agg[('max', w)])
                .apply(score_func)
            )

    # =========================
    # B. GCS 특수 항목
    # =========================
    gcs_ids = [220739, 223900, 223901]
    gcs_df = mimic_df[mimic_df['itemid'].isin(gcs_ids)]

    if gcs_df.empty:
        for w in windows:
            apache_score_df[f"apache_gcs_score_min_w{w}"] = np.nan
            apache_score_df[f"apache_gcs_score_max_w{w}"] = np.nan
    else:
        gcs_complete = (
            gcs_df
            .groupby(['stay_id', 'starttime', 'window'])
            .filter(lambda x: x['itemid'].nunique() == 3)
        )

        if gcs_complete.empty:
            for w in windows:
                apache_score_df[f"apache_gcs_score_min_w{w}"] = np.nan
                apache_score_df[f"apache_gcs_score_max_w{w}"] = np.nan
        else:
            gcs_summed = (
                gcs_complete
                .groupby(['stay_id', 'starttime', 'window'])['valuenum']
                .sum()
                .reset_index()
            )

            gcs_agg = (
                gcs_summed
                .groupby(['stay_id', 'window'])['valuenum']
                .agg(['min', 'max'])
                .unstack('window')
            )

            gcs_agg = gcs_agg.reindex(columns=expected_cols)

            for w in windows:
                apache_score_df[f"apache_gcs_score_min_w{w}"] = (
                    apache_score_df['stay_id']
                    .map(gcs_agg[('min', w)])
                    .apply(apache_gcs_score)
                )

                apache_score_df[f"apache_gcs_score_max_w{w}"] = (
                    apache_score_df['stay_id']
                    .map(gcs_agg[('max', w)])
                    .apply(apache_gcs_score)
                )

    return apache_score_df

# SOFA

# 인공호흡 여부 Flag
def sofa_vent(mimic_df):
    vent_itemids = [
        224688, 224689, 224690, 224687, 224685, 224684, 224686,
        224696, 220339, 224700, 223835, 223849, 229314, 223848, 224691
    ]
    
    # (1) 인공호흡기 관련 아이템만 필터링
    # value가 null이 아닌 기록이 하나라도 있는지 확인하기 위함
    vent_data = mimic_df[mimic_df['itemid'].isin(vent_itemids) & mimic_df['value'].notna()].copy()

    # (2) 해당 기록이 있는 경우 flag를 1로 설정
    vent_data['vent_flag'] = 1

    # (3) stay_id와 window별로 그룹화하여 1이 하나라도 있는지 확인 (max 사용)
    vent_grouped = vent_data.groupby(['stay_id', 'window'])['vent_flag'].max().unstack('window')

    # (4) 모든 stay_id를 포함하는 기본 프레임 생성
    all_stay_ids = mimic_df['stay_id'].unique()
    final_vent_df = pd.DataFrame({'stay_id': sorted(all_stay_ids)})

    # (5) 윈도우별(6, 12, 18, 24) 컬럼 생성 및 결측치 0 채우기
    windows = [6, 12, 18, 24]
    for w in windows:
        col_name = f"vent_flag_w{w}"
        if w in vent_grouped.columns:
            # 해당 윈도우에 기록이 있으면 1, 없으면 0
            final_vent_df[col_name] = final_vent_df['stay_id'].map(vent_grouped[w]).fillna(0).astype(int)
        else:
            # 데이터 자체가 없는 윈도우는 모두 0
            final_vent_df[col_name] = 0

    return final_vent_df

# vaso rate sum
def sofa_vaso(mimic_df):
    filtered_vaso = mimic_df[
        (mimic_df['itemid'].isin([221906, 221289])) &
        (mimic_df['valuenum'].notna())
    ].copy()

    # 4. stay_id, window, itemid별로 min, max 집계
    agg_vaso = filtered_vaso.groupby(['stay_id', 'window', 'itemid'])['valuenum'].agg(['min', 'max']).unstack('itemid')

    vaso_index = agg_vaso.index
    vaso = pd.DataFrame(index=vaso_index)

    def get_vaso_series(df, stat, itemid):
        if stat in df.columns and itemid in df[stat].columns:
            # 데이터가 있으면 NaN만 0으로 채워서 반환
            return df[stat][itemid].fillna(0)
        else:
            # 해당 약물 기록이 아예 없으면 인덱스가 일치하는 0점짜리 시리즈 반환
            return pd.Series(0.0, index=vaso_index)

    # (1) rate_min_sum 계산 (SQL의 NVL(MIN(...), 0) + NVL(MIN(...), 0))
    vaso['rate_min_sum'] = get_vaso_series(agg_vaso, 'min', 221906) + \
                        get_vaso_series(agg_vaso, 'min', 221289)

    # (2) rate_max_sum 계산 (SQL의 NVL(MAX(...), 0) + NVL(MAX(...), 0))
    vaso['rate_max_sum'] = get_vaso_series(agg_vaso, 'max', 221906) + \
                        get_vaso_series(agg_vaso, 'max', 221289)

    # 인덱스(stay_id, window)를 일반 컬럼으로 변환
    vaso = vaso.reset_index()

    return vaso

# RRT 투석 여부
def sofa_rrt(mimic_df):
    dialysis_data = mimic_df[mimic_df['itemid'].isin([225802, 225441, 227536, 227525])].copy()

    # (2) 투석 기록이 있는 행에 flag 1 부여
    dialysis_data['dialysis_flag'] = 1

    # (3) stay_id와 window별로 그룹화하여 하나라도 있으면 1 (max 사용)
    # window 컬럼의 6, 12, 18, 24 값을 기준으로 집계
    dialysis_grouped = dialysis_data.groupby(['stay_id', 'window'])['dialysis_flag'].max().unstack('window')

    # (4) 전체 stay_id를 포함하는 기본 프레임 생성 (모든 환자 포함)
    all_stay_ids = mimic_df['stay_id'].unique()
    final_dialysis_df = pd.DataFrame({'stay_id': sorted(all_stay_ids)})

    # (5) 윈도우별(6, 12, 18, 24) 컬럼 생성 및 결측치 0 채우기
    windows = [6, 12, 18, 24]
    for w in windows:
        col_name = f"dialysis_flag_w{int(w)}" # 예: dialysis_flag_w6
        if w in dialysis_grouped.columns:
            # 기록이 있으면 1, 없으면 0
            final_dialysis_df[col_name] = final_dialysis_df['stay_id'].map(dialysis_grouped[w]).fillna(0).astype(int)
        else:
            # 해당 윈도우에 데이터가 아예 없는 경우 모두 0
            final_dialysis_df[col_name] = 0

    return final_dialysis_df


# SOFA 구성변수 추출
def sofa_var(mimic_df):
    # 2. 항목별 Item ID 매핑 정의
    sofa_raw_items = {
        220224: 'pao2',
        223835: 'fio2',
        50885: 'bilirubin',
        220052: 'map',
        50912: 'creatinine',
        51265: 'platelet'
    }

    # 해당 itemid를 필터링하고 레이블을 매핑합니다.
    df_filtered = mimic_df[mimic_df['itemid'].isin(sofa_raw_items.keys())].copy()
    df_filtered['label'] = df_filtered['itemid'].map(sofa_raw_items)

    # stay_id, window, label별로 최소값과 최대값을 집계합니다.
    agg_df = df_filtered.groupby(['stay_id', 'window', 'label'])['valuenum'].agg(['min', 'max']).unstack('label')

    # 4. 가로 변환 (Wide Format) - 기본 데이터프레임 생성
    sofa_feature_df = pd.DataFrame({'stay_id': sorted(mimic_df['stay_id'].unique())})
    windows = [6, 12, 18, 24]

    # 일반 지표들에 대한 윈도우별 Min/Max 컬럼 생성
    for label in sofa_raw_items.values():
        for w in windows:
            if (w in agg_df.index.get_level_values('window')) and (label in agg_df.columns.levels[1]):
                # 최소값 매핑
                min_series = agg_df.xs(w, level='window')[('min', label)]
                sofa_feature_df[f"{label}_min_w{w}"] = sofa_feature_df['stay_id'].map(min_series)

                # 최대값 매핑
                max_series = agg_df.xs(w, level='window')[('max', label)]
                sofa_feature_df[f"{label}_max_w{w}"] = sofa_feature_df['stay_id'].map(max_series)
            else:
                # 데이터가 없는 경우 NaN 처리
                sofa_feature_df[f"{label}_min_w{w}"] = np.nan
                sofa_feature_df[f"{label}_max_w{w}"] = np.nan


    gcs_df = mimic_df[mimic_df['itemid'].isin([220739, 223900, 223901])].copy()

    # stay_id, starttime, window별로 GCS 3개 항목이 모두 있는 기록만 필터링
    gcs_complete = gcs_df.groupby(['stay_id', 'starttime', 'window']).filter(lambda x: x['itemid'].nunique() == 3)

    # 시점별 합산 (Eye + Verbal + Motor)
    gcs_summed = gcs_complete.groupby(['stay_id', 'starttime', 'window'])['valuenum'].sum().reset_index()

    # 윈도우별 GCS 합계의 MAX값 집계
    # (요청하신 대로 score 변환 전의 합계 중 가장 높은 값을 가져옵니다)
    gcs_max_agg = gcs_summed.groupby(['stay_id', 'window'])['valuenum'].max().unstack('window')

    for w in windows:
        col_name = f"gcs_sum_max_w{w}"
        if w in gcs_max_agg.columns:
            # 윈도우별 Max 합계값을 stay_id에 매핑
            sofa_feature_df[col_name] = sofa_feature_df['stay_id'].map(gcs_max_agg[w])
        else:
            # 해당 윈도우에 기록이 없는 경우 NaN 처리
            sofa_feature_df[col_name] = np.nan

    return sofa_feature_df

# SOFA Score 변환
def return_sofa_score(sofa_vaso_df, sofa_var_df, sofa_vent_df, sofa_rrt_df):
    # ---------------------------------------------------------
    # 1. SOFA 항목별 점수 변환 함수 (의학적 기준)
    # ---------------------------------------------------------
    def score_resp(pao2, fio2, vent_flag):
        if pd.isna(pao2) or pd.isna(fio2): return np.nan
        pf_ratio = pao2 / (fio2 / 100.0)
        if pf_ratio <= 75 and vent_flag == 1: return 4
        if pf_ratio <= 150 and vent_flag == 1: return 3
        if pf_ratio <= 225: return 2
        if pf_ratio <= 300: return 1
        return 0

    def score_coag(plt):
        if pd.isna(plt): return np.nan
        if plt <= 50: return 4
        if plt <= 80: return 3
        if plt <= 100: return 2
        if plt <= 150: return 1
        return 0

    def score_liver(bili):
        if pd.isna(bili): return np.nan
        if bili > 12.0: return 4
        if bili > 6.0: return 3
        if bili > 3.0: return 2
        if bili >= 1.2: return 1
        return 0

    def score_cardio(map_val, vaso_rate):
        rate = 0 if pd.isna(vaso_rate) else vaso_rate
        if rate > 0.4: return 4
        if rate > 0.2: return 3
        if rate > 0: return 2
        if map_val < 70: return 1
        return 0

    def score_cns(gcs_sum):
        if pd.isna(gcs_sum): return np.nan
        if gcs_sum < 6: return 4
        if 6 <= gcs_sum <= 8: return 3
        if 9 <= gcs_sum <= 12: return 2
        if 13 <= gcs_sum <= 14: return 1
        return 0

    def score_renal(cre, rrt_flag):
        if rrt_flag == 1: return 4
        if pd.isna(cre): return np.nan
        if cre >= 3.5: return 3
        if cre >= 2.0: return 2
        if cre >= 1.2: return 1
        return 0

    # ---------------------------------------------------------
    # 2. 데이터프레임 병합 및 전처리 (Safe Merge)
    # ---------------------------------------------------------

    # vaso 데이터 보호 및 Wide 변환
    temp_vaso = sofa_vaso_df.copy()
    if 'window' not in temp_vaso.columns:
        temp_vaso = temp_vaso.reset_index()

    vaso_wide = temp_vaso.pivot(index='stay_id', columns='window', values='rate_max_sum').reset_index()
    vaso_wide.columns = ['stay_id'] + [f'vaso_rate_w{int(c)}' for c in vaso_wide.columns[1:]]

    # 최종 병합
    df_sofa = pd.merge(sofa_var_df, sofa_vent_df, on='stay_id', how='left')
    df_sofa = pd.merge(df_sofa, vaso_wide, on='stay_id', how='left')
    df_sofa = pd.merge(df_sofa, sofa_rrt_df, on='stay_id', how='left')

    # 결측치 처리 (기록 없음 = 미시행/정상)
    fill_cols = [c for c in df_sofa.columns if 'flag' in c or 'vaso_rate' in c]
    df_sofa[fill_cols] = df_sofa[fill_cols].fillna(0)

    windows = [6, 12, 18, 24]

    for w in windows:
        # 1) Respiratory Score(호흡기)
        df_sofa[f'sofa_resp_w{w}'] = df_sofa.apply(
            lambda x: score_resp(x.get(f'pao2_min_w{w}'), x.get(f'fio2_max_w{w}'), x.get(f'vent_flag_w{w}', 0)), axis=1)

        # 2) Coagulation Score(응고계)
        df_sofa[f'sofa_coag_w{w}'] = df_sofa.get(f'platelet_min_w{w}', pd.Series(np.nan)).apply(score_coag)

        # 3) Liver Score(간)
        df_sofa[f'sofa_liver_w{w}'] = df_sofa.get(f'bilirubin_max_w{w}', pd.Series(np.nan)).apply(score_liver)

        # 4) Cardiovascular Score(순환기)
        df_sofa[f'sofa_cardio_w{w}'] = df_sofa.apply(
            lambda x: score_cardio(x.get(f'map_min_w{w}'), x.get(f'vaso_rate_w{w}', 0)), axis=1)

        # 5) CNS Score (GCS 합계 기반)
        df_sofa[f'sofa_cns_w{w}'] = df_sofa.get(f'gcs_sum_max_w{w}', pd.Series(np.nan)).apply(score_cns)

        # 6) Renal Score(신장)
        df_sofa[f'sofa_renal_w{w}'] = df_sofa.apply(
            lambda x: score_renal(x.get(f'creatinine_max_w{w}'), x.get(f'dialysis_flag_w{w}', 0)), axis=1)

    
    score_cols = [c for c in df_sofa.columns if c.startswith('sofa_')]
    df_sofa_final = df_sofa[['stay_id'] + score_cols]

    return df_sofa_final

# 찰슨
def return_CCI_score(patient_df, diagnoses_icd_df):
    # 분석 대상 hadm_id만 필터링하여 메모리 절약
    valid_hadm_ids = patient_df['hadm_id'].unique()
    diagnoses_icd_df = diagnoses_icd_df[diagnoses_icd_df['hadm_id'].isin(valid_hadm_ids)].copy()

    # =====================================================
    # 2. 나이 점수 계산 (벡터화 연산)
    # =====================================================
    # age_score: <=50(0), 51-60(1), 61-70(2), 71-80(3), >80(4)

    patient_df['age_score'] = pd.cut(
        patient_df['환자실제나이'],
        bins=[-np.inf, 50, 60, 70, 80, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # =====================================================
    # 3. ICD 코드 슬라이싱
    # =====================================================
    is9 = diagnoses_icd_df['icd_version'].isin([9])
    is10 = diagnoses_icd_df['icd_version'].isin([10])

    diagnoses_icd_df['c3'] = diagnoses_icd_df['icd_code'].str[:3]
    diagnoses_icd_df['c4'] = diagnoses_icd_df['icd_code'].str[:4]
    diagnoses_icd_df['c5'] = diagnoses_icd_df['icd_code'].str[:5]

    # =====================================================
    # 4. 질환 플래그 생성 (괄호 추가로 TypeError 해결)
    # =====================================================
    # 모든 비교 연산은 반드시 ()로 감싸야 비트 연산(&, |) 오류가 발생하지 않습니다.
    conds = {
        'mi': (is9 & diagnoses_icd_df['c3'].isin(['410','412'])) | (is10 & (diagnoses_icd_df['c3'].isin(['I21','I22']) | (diagnoses_icd_df['c4'] == 'I252'))),
        'chf': (is9 & ((diagnoses_icd_df['c3'] == '428') | diagnoses_icd_df['c5'].isin(['39891','40201','40211','40291','40401','40403','40411','40413','40491','40493']) | diagnoses_icd_df['c4'].between('4254','4259'))) | (is10 & diagnoses_icd_df['c3'].isin(['I43','I50'])),
        'pvd': (is9 & (diagnoses_icd_df['c3'].isin(['440','441']) | diagnoses_icd_df['c4'].isin(['0930','4373','4471','5571','5579','V434']))) | (is10 & diagnoses_icd_df['c3'].isin(['I70','I71'])),
        'cvd': (is9 & diagnoses_icd_df['c3'].between('430','438')) | (is10 & (diagnoses_icd_df['c3'].isin(['G45','G46']) | diagnoses_icd_df['c3'].between('I60','I69'))),
        'dementia': (is9 & ((diagnoses_icd_df['c3'] == '290') | diagnoses_icd_df['c4'].isin(['2941','3312']))) | (is10 & (diagnoses_icd_df['c3'].isin(['F00','F01','F02','F03']) | (diagnoses_icd_df['c3'] == 'G30'))),
        'cpd': (is9 & diagnoses_icd_df['c3'].between('490','505')) | (is10 & diagnoses_icd_df['c3'].between('J40','J47')),
        'rheum': (is9 & (diagnoses_icd_df['c3'] == '725')) | (is10 & diagnoses_icd_df['c3'].isin(['M05','M06','M32','M33','M34'])),
        'pud': (is9 & diagnoses_icd_df['c3'].isin(['531','532','533','534'])) | (is10 & diagnoses_icd_df['c3'].isin(['K25','K26','K27','K28'])),
        'mild_liver': (is9 & diagnoses_icd_df['c3'].isin(['570','571'])) | (is10 & diagnoses_icd_df['c3'].isin(['B18','K73','K74'])),
        'severe_liver': (is9 & diagnoses_icd_df['c4'].isin(['5722','5723','5724','5728'])) | (is10 & diagnoses_icd_df['c4'].isin(['K704','K711','K721','K729','K765','K766','K767'])),
        'dm_no': (is9 & diagnoses_icd_df['c4'].isin(['2500','2501','2502','2503','2508','2509'])) | (is10 & diagnoses_icd_df['c4'].isin(['E100','E101'])),
        'dm_cc': (is9 & diagnoses_icd_df['c4'].isin(['2504','2505','2506','2507'])) | (is10 & diagnoses_icd_df['c4'].isin(['E102','E103','E112','E113'])),
        'para': (is9 & diagnoses_icd_df['c3'].isin(['342','343'])) | (is10 & diagnoses_icd_df['c3'].isin(['G81','G82'])),
        'renal': (is9 & (diagnoses_icd_df['c3'].isin(['582','585','586','V56']) | diagnoses_icd_df['c4'].isin(['5880','V420','V451']) | diagnoses_icd_df['c4'].between('5830','5837') | diagnoses_icd_df['c5'].isin(['40301','40311','40391','40402','40403','40412','40413','40492','40493']))) | (is10 & (diagnoses_icd_df['c3'].isin(['N18','N19']) | diagnoses_icd_df['c4'].isin(['I120','I131','N032','N033','N034','N035','N036','N037','N052','N053','N054','N055','N056','N057','N250','Z490','Z491','Z492','Z940','Z992']))),
        'cancer': (is9 & (diagnoses_icd_df['c3'].between('140','172') | diagnoses_icd_df['c4'].between('1740','1958') | diagnoses_icd_df['c3'].between('200','208') | (diagnoses_icd_df['c4'] == '2386'))) | (is10 & diagnoses_icd_df['c3'].between('C00','C97')),
        'mets': (is9 & diagnoses_icd_df['c3'].isin(['196','197','198','199'])) | (is10 & diagnoses_icd_df['c3'].isin(['C77','C78','C79','C80'])),
        'aids': (is9 & diagnoses_icd_df['c3'].isin(['042','043','044'])) | (is10 & diagnoses_icd_df['c3'].isin(['B20','B21','B22','B24']))
    }

    # 루프를 통한 플래그 할당
    for name, cond in conds.items():
        diagnoses_icd_df[name] = cond.astype('int8')

    # =====================================================
    # 5. 한 번의 GroupBy로 모든 질환 집계 (속도 핵심)
    # =====================================================
    com_df = diagnoses_icd_df.groupby('hadm_id')[list(conds.keys())].max().reset_index()

    # =====================================================
    # 6. 최종 결합 및 CCI 계산
    # =====================================================
    final = patient_df.merge(com_df, on='hadm_id', how='left').fillna(0)

    # CCI 산출 공식 적용
    final['cci_score'] = (
        final['age_score'] +
        final['mi'] + final['chf'] + final['pvd'] + final['cvd'] +
        final['dementia'] + final['cpd'] + final['rheum'] + final['pud'] +
        np.maximum(final['mild_liver'], 3 * final['severe_liver']) + # 간질환 가중치 적용
        np.maximum(final['dm_no'], 2 * final['dm_cc']) +            # 당뇨 가중치 적용
        2 * final['para'] +
        2 * final['renal'] +
        np.maximum(2 * final['cancer'], 6 * final['mets']) +        # 암 가중치 적용
        6 * final['aids']
    ).astype('int16')

    # 최종 결과 저장 및 확인
    cci_result = final[['stay_id', 'cci_score']]
    return cci_result

# 수술 여부
def return_surgery_yes(mimic_df,procedures_icd_df):
    # 2. stay_base 생성 (ICU 입실 시점 추출)
    mimic_df['starttime'] = pd.to_datetime(mimic_df['starttime'])

    # STAY_ID별 가장 빠른 starttime을 ICU 입실 시점(ICU_INTIME)으로 정의합니다.
    stay_base = mimic_df.groupby(['stay_id', 'hadm_id'])['starttime'].min().reset_index()
    stay_base.rename(columns={'starttime': 'icu_intime'}, inplace=True)

    # 3. 데이터 결합 (HADM_ID 기준 Left Join)
    # 환자의 입원(HADM_ID)별 수술 기록을 매칭합니다.
    merged = pd.merge(stay_base, procedures_icd_df, on='hadm_id', how='left')

    # 4. 날짜 형식 변환 및 수술 시점 조건 확인
    # 수술일(chartdate)을 datetime 형식으로 변환합니다.
    merged['chartdate'] = pd.to_datetime(merged['chartdate'])

    # SQL 로직: 수술일이 ICU 입실 전 24시간 이내인지 확인
    # (icu_intime - 1일) <= chartdate <= icu_intime
    condition = (merged['chartdate'] >= merged['icu_intime'] - pd.Timedelta(days=1)) & \
                (merged['chartdate'] <= merged['icu_intime'])

    # 조건 만족 시 1, 미만 시 0 부여
    merged['is_postop_24h_flag'] = condition.astype(int)

    postop_24h = merged.groupby('stay_id')['is_postop_24h_flag'].max().reset_index()
    postop_24h.rename(columns={'is_postop_24h_flag': 'IS_POSTOP_24H'}, inplace=True)

    return postop_24h

# 승압제 사용여부
def return_vaso_yes(mimic_df):
    # 2. 약물 투여 여부 플래그 생성
    # 각 itemid에 해당하는 승압제 투여 여부를 0 또는 1로 표시
    mimic_df['Norepinephrine'] = np.where(mimic_df['itemid'] == 221906, 1, 0)
    mimic_df['Epinephrine'] = np.where(mimic_df['itemid'] == 221289, 1, 0)
    mimic_df['Vasopressin'] = np.where(mimic_df['itemid'] == 222315, 1, 0)

    # 3. 1차 집계 (stay_id와 window별로 max값 추출)
    # 해당 환자가 특정 윈도우(6, 12, 18, 24) 내에서 약물을 투여받았는지 결정
    agg_mimic_df = mimic_df.groupby(['stay_id', 'window']).agg({
        'Norepinephrine': 'max',
        'Epinephrine': 'max',
        'Vasopressin': 'max'
    }).reset_index()

    # 4. 데이터 재구조화 (Long to Wide)
    # stay_id를 행(index)으로, window를 열(columns)로 변환
    pivot_mimic_df = agg_mimic_df.pivot(index='stay_id', columns='window',
                            values=['Norepinephrine', 'Epinephrine', 'Vasopressin'])

    # 5. 컬럼명 변경 (변수명_w시간 형태)
    # MultiIndex로 생성된 컬럼을 'Norepinephrine_w6'와 같은 형태로 평탄화
    # window 값이 6.0 등 실수형으로 올 경우를 대비해 int() 처리를 포함
    pivot_mimic_df.columns = [f"{drug}_w{int(time)}" for drug, time in pivot_mimic_df.columns]

    # 6. 인덱스 초기화 및 최종 결과 확인
    # stay_id를 다시 일반 컬럼으로 변환하여 최종 데이터셋을 완성
    vent_flag = pivot_mimic_df.reset_index()
    return vent_flag

def return_patient_info(patient_df):
    patient_df["gender"] = patient_df["gender"].map({"M": 1, "F": 0})
    gender_age = patient_df[['stay_id','gender','환자실제나이']]
    return gender_age

# sofa_vaso_df = sofa_vent(mimic_df)
# sofa_var_df = sofa_vaso(mimic_df)
# sofa_vent_df = sofa_rrt(mimic_df)
# sofa_rrt_df = sofa_var(mimic_df)
# apache_score_df = calc_apache(mimic_df)
# df_sofa_final = return_sofa_score(sofa_vaso_df, sofa_var_df, sofa_vent_df, sofa_rrt_df)
# vent_flag = return_vaso_yes(mimic_df)
# cci_result = return_CCI_score(patient_df, diagnoses_icd_df)
# postop_24h = return_surgery_yes(mimic_df,procedures_icd_df)
# gender_age = return_patient_info(patient_df)


def return_merge(apache_score_df, df_sofa_final,vent_flag,cci_result,postop_24h,gender_age):
    total = pd.merge(apache_score_df,df_sofa_final,on='stay_id', how='left')
    vaso_cols = [
        'Norepinephrine_w6', 'Norepinephrine_w12', 'Norepinephrine_w18', 'Norepinephrine_w24',
        'Epinephrine_w6', 'Epinephrine_w12', 'Epinephrine_w18', 'Epinephrine_w24',
        'Vasopressin_w6', 'Vasopressin_w12', 'Vasopressin_w18', 'Vasopressin_w24'
    ]

    # =====================================================
    # 3. 데이터프레임 최종 병합
    # =====================================================
    try:
        # A. 약물 윈도우 데이터 병합
        if vent_flag is not None:
            # 실제 존재하는 컬럼만 필터링
            existing_vaso = [c for c in vaso_cols if c in vent_flag.columns]
            # 기존 total에 중복 컬럼이 있으면 삭제 후 병합
            total = total.drop(columns=[c for c in existing_vaso if c in total.columns], errors='ignore')
            total = pd.merge(total, vent_flag[['stay_id'] + existing_vaso], on='stay_id', how='left')

        # B. CCI 점수 병합
        if cci_result is not None:
            total = total.drop(columns=['cci_score'], errors='ignore')
            total = pd.merge(total, cci_result[['stay_id', 'cci_score']], on='stay_id', how='left')

        # C. 수술 여부 (IS_POSTOP_24H) 병합
        if postop_24h is not None:
            # 원본 컬럼명 postop_24h를 표준 명칭인 IS_POSTOP_24H로 변경
            temp_postop = postop_24h.rename(columns={'postop_24h': 'IS_POSTOP_24H'})
            total = total.drop(columns=['IS_POSTOP_24H'], errors='ignore')
            total = pd.merge(total, temp_postop[['stay_id', 'IS_POSTOP_24H']], on='stay_id', how='left')

        # D. 성별 및 나이 병합 (중요: 환자실제나이 -> age로 변경)
        if gender_age is not None:
            # 중복 방지를 위해 기존 컬럼 삭제
            total = total.drop(columns=['gender', 'age'], errors='ignore')
            # '환자실제나이' 컬럼명을 'age'로 변경하여 가져옴
            temp_gender = gender_age[['stay_id', 'gender', '환자실제나이']].rename(columns={'환자실제나이': 'age'})
            total = pd.merge(total, temp_gender, on='stay_id', how='left')

    except Exception as e:
        print(f"!! 병합 중 오류 발생: {e}")

    fill_cols = vaso_cols + ['cci_score', 'IS_POSTOP_24H', 'gender', 'age']

    for col in fill_cols:
        if col in total.columns:
            total[col] = total[col].fillna(0).astype(int)

    return total



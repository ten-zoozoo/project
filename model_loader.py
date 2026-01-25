import pickle
import pandas as pd


class ICU24hRiskModel:
    """
    ICU 24h Mortality Risk Model
    - Input: DB-style dataframe (stay_id + features)
    - Missing values allowed (XGBoost native handling)
    - Output: risk probability per stay_id
    """

    def __init__(self, model_pkl_path: str):
        with open(model_pkl_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.target = bundle.get("target", "dod_whithin_24h")

    def predict_from_db(
        self,
        df: pd.DataFrame,
        stay_id_col: str = "stay_id"
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            DB-style dataframe (1 row per stay_id)
            - must include stay_id
            - must include all feature columns
            - NaN values allowed

        stay_id_col : str
            Column name for stay_id

        Returns
        -------
        pd.DataFrame
            stay_id | risk_score_24h
        """

        # =========================
        # 1) stay_id 분리
        # =========================
        stay_ids = df[stay_id_col]

        # =========================
        # 2) feature contract 적용
        # =========================
        X = df.reindex(columns=self.feature_columns)

        # (중요) 결측치 처리 없음 → 그대로 통과
        # XGBoost는 NaN을 내부적으로 처리함

        # =========================
        # 3) 확률 예측
        # =========================
        proba = self.model.predict_proba(X)[:, 1]

        # =========================
        # 4) 결과 정리
        # =========================
        result = pd.DataFrame({
            stay_id_col: stay_ids,
            "risk_score_24h": proba
        })

        return result

    def predict_with_snapshot_time(
        self,
        df: pd.DataFrame,
        stay_id_col: str = "stay_id",
        time_col: str = None
    ) -> pd.DataFrame:
        """
        Optional:
        전자차트 느낌으로 '언제 시점의 예측인지' 같이 반환하고 싶을 때
        """

        res = self.predict_from_db(df, stay_id_col)

        if time_col and time_col in df.columns:
            res["snapshot_time"] = df[time_col].values

        return res
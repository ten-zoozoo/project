# =========================
# app.py (Full)
# - QueryParam 기반 커스텀 탭(SICU/CT)
# - CT 페이지: Pancreas + LiverTumor "미리 계산된 예측 NIfTI" 오버레이
# - nnUNet 실행 없음(대체: precomputed_preds에서 파일 로드)
# - UI는 원본 유지, z는 pred 기준으로 동작
# =========================

import os
import re
import zipfile
import tempfile
import shutil
import subprocess
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import SimpleITK as sitk
except Exception:
    sitk = None


# =========================================================
# 0) 프로젝트 루트/경로 세팅
# =========================================================
ROOT = Path(__file__).resolve().parent

# ✅ 미리 계산된 prediction 폴더 (네 스샷 구조 그대로)
PRECOMP_ROOT = ROOT / "precomputed_preds"
PRECOMP_LIVER_DIR = PRECOMP_ROOT / "LiverTumor_out"
PRECOMP_PANC_DIR  = PRECOMP_ROOT / "Pancreas_out"


# -------------------------
# Page config (must be first)
# -------------------------
st.set_page_config(layout="wide")


# =========================================================
# Helpers
# =========================================================
def require_nib_or_warn() -> bool:
    if nib is None:
        st.error("NIfTI(.nii/.nii.gz) 로드를 위해 nibabel이 필요해요. `pip install nibabel`")
        return False
    return True


def ensure_workdir() -> Path:
    """
    ✅ /tmp 대신 프로젝트 폴더 아래에 작업 폴더를 고정으로 생성
    - ct_service/_runtime 아래에 입력/출력/마스크가 계속 남음
    """
    work = ROOT / "_runtime"
    (work / "inputs").mkdir(parents=True, exist_ok=True)
    (work / "pred_masks").mkdir(parents=True, exist_ok=True)
    (work / "nnunet_in").mkdir(parents=True, exist_ok=True)
    (work / "nnunet_out").mkdir(parents=True, exist_ok=True)
    return work


def dicom_zip_to_nifti(dicom_zip: Path, out_nii_gz: Path) -> Path:
    if sitk is None:
        raise RuntimeError("SimpleITK가 없어 DICOM ZIP 변환 불가. `pip install SimpleITK` 필요")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(dicom_zip, "r") as zf:
            zf.extractall(td)

        chosen_series = None
        chosen_dir = None
        for d in [td] + [p for p in td.rglob("*") if p.is_dir()]:
            try:
                series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(d))
                if series_ids:
                    chosen_series = series_ids[0]
                    chosen_dir = d
                    break
            except Exception:
                continue

        if chosen_series is None:
            raise RuntimeError("ZIP에서 DICOM series를 찾지 못했어요. (압축 구조/파일 확인 필요)")

        file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(chosen_dir), chosen_series)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_names)
        img = reader.Execute()

        out_nii_gz.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(out_nii_gz))
        return out_nii_gz


def hu_window_to_uint8(x: np.ndarray, wl: float, ww: float) -> np.ndarray:
    lo = wl - ww / 2.0
    hi = wl + ww / 2.0
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)
    return (x * 255.0).astype(np.uint8)


def overlay_one(ct_u8: np.ndarray, mask2d: np.ndarray, alpha: float, color) -> np.ndarray:
    rgb = np.stack([ct_u8, ct_u8, ct_u8], axis=-1).astype(np.float32)
    m = (mask2d > 0).astype(np.float32)[..., None]
    color = np.array(color, dtype=np.float32)
    out = rgb * (1 - alpha * m) + color * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_liver_tumor(ct_u8: np.ndarray, seg2d: np.ndarray, alpha: float) -> np.ndarray:
    """
    LiverTumor 라벨이 (1=liver, 2=tumor)인 경우를 우선 지원.
    만약 0/1만 있으면 'tumor만'으로 간주해서 한 색으로 칠함.
    """
    out = np.stack([ct_u8, ct_u8, ct_u8], axis=-1).astype(np.float32)

    seg2d = np.asarray(seg2d)
    uniq = set(np.unique(seg2d).tolist())

    if 2 in uniq and 1 in uniq:
        liver = (seg2d == 1).astype(np.float32)[..., None]
        tumor = (seg2d == 2).astype(np.float32)[..., None]

        green = np.array([0, 255, 0], dtype=np.float32)
        mag = np.array([255, 0, 255], dtype=np.float32)

        out = out * (1 - alpha * liver) + green * (alpha * liver)
        out = out * (1 - alpha * tumor) + mag * (alpha * tumor)
        return np.clip(out, 0, 255).astype(np.uint8)

    return overlay_one(ct_u8, (seg2d > 0).astype(np.uint8), alpha=alpha, color=(255, 0, 255))


def _z_map_pred_to_ct(z_pred: int, pred_zmax: int, ct_zmax: int) -> int:
    """pred z 기준으로 CT z를 비례 매핑 (shape 불일치용)"""
    if pred_zmax <= 0 or ct_zmax <= 0:
        return 0
    z_ct = int(round((z_pred / pred_zmax) * ct_zmax))
    return int(np.clip(z_ct, 0, ct_zmax))


def render_slice(ct_vol: np.ndarray, z_pred: int, wl: float, ww: float, alpha: float,
                 mask_path: Path | None, mode: str):
    """
    ✅ z는 'pred 기준'
    - mask가 있으면: pred z를 mask에서 쓰고, CT는 비례매핑해서 씀
    - mask가 없으면: 그냥 CT z로 사용(=z_pred를 CT 범위로 clip)
    """
    ct_zmax = int(ct_vol.shape[-1] - 1)

    if mask_path is None:
        z_ct = int(np.clip(z_pred, 0, ct_zmax))
        ct_slice = np.asarray(ct_vol[..., z_ct]).astype(np.float32)
        ct_u8 = hu_window_to_uint8(ct_slice, wl=wl, ww=ww)
        out = np.stack([ct_u8, ct_u8, ct_u8], axis=-1)
    else:
        mv = np.asanyarray(nib.load(str(mask_path)).dataobj)
        pred_zmax = int(mv.shape[-1] - 1)

        z_m = int(np.clip(z_pred, 0, pred_zmax))
        z_ct = _z_map_pred_to_ct(z_m, pred_zmax, ct_zmax)

        ct_slice = np.asarray(ct_vol[..., z_ct]).astype(np.float32)
        ct_u8 = hu_window_to_uint8(ct_slice, wl=wl, ww=ww)

        seg2d = np.asarray(mv[..., z_m])

        if mode == "liver_tumor":
            out = overlay_liver_tumor(ct_u8, seg2d, alpha=alpha)
        else:
            out = overlay_one(ct_u8, (seg2d > 0).astype(np.uint8), alpha=alpha, color=(255, 180, 0))

    fig = plt.figure(figsize=(6.2, 6.2), dpi=120)
    ax = fig.add_subplot(111)
    ax.imshow(out)
    ax.set_axis_off()
    st.pyplot(fig, clear_figure=True)


def download_button(path: Path, label: str):
    with open(path, "rb") as f:
        st.download_button(
            label=f"⬇️ {label} 다운로드",
            data=f,
            file_name=path.name,
            mime="application/gzip" if str(path).endswith(".gz") else "application/octet-stream",
            use_container_width=True,
        )


def _empty_masks():
    # UI는 유지하지만, "all"은 사용 안 함(항상 None로 둘 것)
    return {"pancreas": None, "liver_tumor": None, "all": None}

def _get_pred_zmax(p: Path | None) -> int | None:
    if p is None:
        return None
    try:
        mv = np.asanyarray(nib.load(str(p)).dataobj)
        if mv.ndim == 3:
            return int(mv.shape[-1] - 1)
    except Exception:
        return None
    return None

def _mask_z_stats(mask_path: Path):
    """
    mask_path의 3D 마스크에서 foreground가 존재하는 z 리스트/범위 반환
    return: (z_list, zmin, zmax, count)
    """
    try:
        mv = np.asanyarray(nib.load(str(mask_path)).dataobj)
        if mv.ndim != 3:
            return [], None, None, 0
        z_has = np.where((mv > 0).sum(axis=(0, 1)) > 0)[0]
        if len(z_has) == 0:
            return [], None, None, 0
        return z_has.tolist(), int(z_has.min()), int(z_has.max()), int(len(z_has))
    except Exception:
        return [], None, None, 0
    

# =========================================================
# ✅ precomputed prediction loader
# =========================================================
def _extract_case_number_from_path(p: Path) -> str | None:
    """
    업로드한 CT 파일명에서 숫자를 뽑음
    예: 3.nii / 79.nii.gz / case_03_xxx.nii -> "3" / "79" / "03"
    """
    m = re.search(r"(\d+)", p.stem)
    return m.group(1) if m else None


def _extract_case_number_from_active(active_case: str) -> str | None:
    """
    UI 케이스명(case_01 등)에서 숫자 뽑음
    """
    m = re.search(r"(\d+)", active_case or "")
    return m.group(1) if m else None


def _find_precomputed_pred(model: str, ct_nii_path: Path, active_case: str) -> Path | None:
    """
    model: "pancreas" | "liver_tumor"
    - 1순위: 업로드 CT 파일명에서 숫자 추출해서 {num}.nii(.gz) 찾기
    - 2순위: active_case에서 숫자 추출해서 {num}.nii(.gz) 찾기
    """
    if model == "pancreas":
        base = PRECOMP_PANC_DIR
    else:
        base = PRECOMP_LIVER_DIR

    # 후보 번호들
    nums = []
    n1 = _extract_case_number_from_path(ct_nii_path) if ct_nii_path else None
    n2 = _extract_case_number_from_active(active_case)
    for n in [n1, n2]:
        if n is not None:
            nums.append(str(int(n)))  # "03" -> "3" 통일

    # 중복 제거
    nums = list(dict.fromkeys(nums))

    # 폴더 자체가 없으면 None
    if not base.exists():
        return None

    # 확장자 후보
    exts = [".nii.gz", ".nii"]

    for num in nums:
        for ext in exts:
            cand = base / f"{num}{ext}"
            if cand.exists():
                return cand

    # 마지막 fallback: stem에 num 포함된 파일
    for num in nums:
        hits = list(base.glob(f"*{num}*.nii*"))
        if hits:
            return hits[0]

    return None


def run_predict(model: str, ct_img, ct_vol: np.ndarray, ct_nii_path: Path | None) -> Path:
    """
    ✅ nnUNet 실행 대신, 미리 계산된 예측 파일을 찾아서 반환
    model: "pancreas" | "liver_tumor"
    """
    if ct_nii_path is None:
        raise RuntimeError("CT 파일 경로가 없습니다.")

    active_case = st.session_state.get("active_case", "case_01")
    pred_path = _find_precomputed_pred(model, ct_nii_path, active_case)

    if pred_path is None:
        st.error(
            f"미리 계산된 예측 파일을 못 찾았어.\n\n"
            f"- model={model}\n"
            f"- ct={ct_nii_path.name}\n"
            f"- expected dir={ (PRECOMP_PANC_DIR if model=='pancreas' else PRECOMP_LIVER_DIR) }\n\n"
            f"예: {PRECOMP_PANC_DIR}/3.nii  또는  {PRECOMP_LIVER_DIR}/3.nii"
        )
        raise FileNotFoundError("precomputed prediction not found")

    return pred_path


# =========================================================
# Session init
# =========================================================
if "ct_cases" not in st.session_state:
    st.session_state["ct_cases"] = {}
if "active_case" not in st.session_state:
    st.session_state["active_case"] = None


# =========================================================
# ✅ "탭" 상태: query param 기반
# =========================================================
qp = st.query_params
page = qp.get("page", "SICU")
if page not in ["SICU", "CT"]:
    page = "SICU"


def go(page_name: str):
    st.query_params["page"] = page_name


# =========================================================
# ✅ Top "Tabs" UI (버튼 2개로 탭처럼 보이게)  (원본 유지)
# =========================================================
st.markdown(
    """
    <style>
    div[data-testid="column"] button {
        height: 42px;
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

t1, t2 = st.columns([1, 1], gap="small")
with t1:
    if page == "SICU":
        st.button("🩺 SICU Mortality Dashboard", use_container_width=True, disabled=True)
    else:
        st.button("🩺 SICU Mortality Dashboard", use_container_width=True, on_click=go, args=("SICU",))
with t2:
    if page == "CT":
        st.button("🩻 CT Segmentation", use_container_width=True, disabled=True)
    else:
        st.button("🩻 CT Segmentation", use_container_width=True, on_click=go, args=("CT",))



# =========================================================
# ✅ Sidebar: page에 따라 자동 변경  (원본 유지)
# =========================================================
with st.sidebar:
    if page == "SICU":
        st.markdown("## 🩺 SICU")
        st.caption("여기는 SICU용 사이드바 자리(기존 코드 넣기).")
        st.text_input("STAY_ID", key="sicu_stay_id")
        st.caption("TOP5/필터/환자 리스트 등 여기에 배치하면 됩니당.")
    else:
        st.markdown("## 🩻 CT 케이스")
        panel_title = st.text_input("외과 중환자실", value="📁 간 CT", key="ct_panel_title")

        with st.expander(panel_title, expanded=True):
            st.caption("저장된 케이스를 선택/삭제/초기화")

            case_names = list(st.session_state["ct_cases"].keys())
            if len(case_names) == 0:
                st.info("저장된 케이스가 없습니다.\n메인에서 CT 업로드 후 저장하세요.")
            else:
                if st.session_state["active_case"] not in case_names:
                    st.session_state["active_case"] = case_names[0]

                sel = st.selectbox(
                    "케이스 선택",
                    options=case_names,
                    index=case_names.index(st.session_state["active_case"]),
                    key="ct_case_selectbox"
                )
                st.session_state["active_case"] = sel

                c1, c2 = st.columns([1, 1], gap="small")
                with c1:
                    if st.button("🧹 마스크 초기화", use_container_width=True):
                        st.session_state["ct_cases"][sel]["masks"] = _empty_masks()
                        st.success("초기화 완료")
                with c2:
                    if st.button("🗑️ 케이스 삭제", use_container_width=True):
                        del st.session_state["ct_cases"][sel]
                        remain = list(st.session_state["ct_cases"].keys())
                        st.session_state["active_case"] = remain[0] if remain else None
                        st.success("삭제 완료")

        st.caption("※ 케이스가 많아지면 메모리가 커질 수 있습니다. 사용하지 않은 케이스들은 삭제해주세요.")


# =========================================================
# ✅ Main content
# =========================================================
if page == "SICU":
    st.markdown("# 🩺 SICU Mortality Dashboard")
    sid = st.session_state.get("sicu_stay_id", "")
    if sid:
        st.info(f"현재 STAY_ID: {sid}")
    else:
        st.info("왼쪽 사이드바에서 STAY_ID를 입력하면 여기에 반영됩니다.")
    st.caption("👉 여기에 기존 SICU 대시보드 UI를 그대로 붙이면 됩니다.")

else:
    # -------------------------
    # CT Page
    # -------------------------
    if not require_nib_or_warn():
        st.stop()

    st.markdown("# 🩻 CT Segmentation")

    # (원본 코드 유지) nnUNet_results 체크는 UI 흐름상 남겨두되,
    # 실제로는 precomputed_preds만 있으면 돌아가게 완화
    if not PRECOMP_ROOT.exists():
        st.error(f"precomputed_preds 폴더가 없어요: {PRECOMP_ROOT}")
        st.stop()

    work = ensure_workdir()
    in_dir = work / "inputs"

    col_up, col_set = st.columns([1.2, 1.8], gap="large")

    with col_up:
        st.markdown("#### CT 업로드")
        ct_file = st.file_uploader(
            "CT (.nii/.nii.gz 또는 DICOM.zip)",
            type=["nii", "gz", "zip"],
            key="ct_uploader",
            help="NIfTI 또는 DICOM ZIP 업로드",
        )

        default_case_name = f"case_{len(st.session_state['ct_cases']) + 1:02d}"
        case_name = st.text_input("케이스 이름(환자명/ID 등)", value=default_case_name, key="ct_case_name")
        save_case = st.button("💾 이 업로드를 케이스로 저장", use_container_width=True)

    with col_set:
        st.markdown("#### 표시 설정")
        wl = st.slider("Window Level", -200, 200, 50, 10, key="ct_wl")
        ww = st.slider("Window Width", 50, 2000, 350, 50, key="ct_ww")
        alpha = st.slider("Overlay Alpha", 0.0, 0.9, 0.35, 0.05, key="ct_alpha")

    # 업로드 파일 로드
    if ct_file is not None:
        uploaded_path = in_dir / ct_file.name
        with open(uploaded_path, "wb") as f:
            f.write(ct_file.getbuffer())

        if uploaded_path.suffix.lower() == ".zip":
            try:
                ct_nii_path = dicom_zip_to_nifti(uploaded_path, in_dir / f"{case_name}_converted.nii.gz")
                st.success("DICOM ZIP → NIfTI 변환 완료")
            except Exception as e:
                st.error(f"DICOM 변환 실패: {e}")
                st.stop()
        else:
            ct_nii_path = uploaded_path

        ct_img = nib.load(str(ct_nii_path))
        ct_vol = np.asanyarray(ct_img.dataobj)
        if ct_vol.ndim != 3:
            st.error(f"CT가 3D가 아닙니다. shape={ct_vol.shape}")
            st.stop()

        if save_case:
            if case_name.strip() == "":
                st.warning("케이스 이름을 입력해주세요.")
            else:
                st.session_state["ct_cases"][case_name] = {
                    "ct_path": ct_nii_path,
                    "ct_img": ct_img,
                    "ct_vol": ct_vol,
                    "masks": _empty_masks(),
                }
                st.session_state["active_case"] = case_name
                st.success(f"케이스 저장 완료: {case_name}")

    # 활성 케이스
    active = st.session_state["active_case"]
    if active is None:
        st.info("왼쪽 CT 케이스 패널에 케이스가 없어요. CT 업로드 후 저장해 주세요.")
        st.stop()

    if active not in st.session_state["ct_cases"]:
        keys = list(st.session_state["ct_cases"].keys())
        st.session_state["active_case"] = keys[0] if keys else None

    ct_img = st.session_state["ct_cases"][active]["ct_img"]
    ct_vol = st.session_state["ct_cases"][active]["ct_vol"]
    ct_nii_path = st.session_state["ct_cases"][active]["ct_path"]
    masks = st.session_state["ct_cases"][active]["masks"]

    st.markdown("---")
    st.markdown(f"#### CT Overlay 결과  ·  활성 케이스: `{active}`")

    mode = st.radio(
        "표시할 마스크",
        ["전체(ALL)", "Pancreas", "종양(Liver Tumor)"],
        horizontal=True,
        label_visibility="collapsed",
        key="ct_mode_radio",
    )

    b1, b2, b3, b4 = st.columns([1, 1, 1, 1], gap="small")
    run_all = b1.button("전체 예측", use_container_width=True)
    run_selected = b2.button("선택 예측", use_container_width=True)
    clear_masks = b3.button("이 케이스 마스크 초기화", use_container_width=True)
    dl_area = b4

    if clear_masks:
        st.session_state["ct_cases"][active]["masks"] = _empty_masks()
        st.success("마스크 초기화 완료")

    # ✅ 이제 "예측" 버튼은 nnUNet이 아니라 "미리 계산된 pred 파일 로드" 역할
    if run_selected:
        if mode.startswith("전체"):
            # UI는 유지하되, ALL은 union 안 만들고 "둘 다 로드"만 해둠
            masks["pancreas"] = run_predict("pancreas", ct_img, ct_vol, ct_nii_path)
            masks["liver_tumor"] = run_predict("liver_tumor", ct_img, ct_vol, ct_nii_path)
            masks["all"] = None
        elif "Pancreas" in mode:
            masks["pancreas"] = run_predict("pancreas", ct_img, ct_vol, ct_nii_path)
        else:
            masks["liver_tumor"] = run_predict("liver_tumor", ct_img, ct_vol, ct_nii_path)

        st.session_state["ct_cases"][active]["masks"] = masks
        st.success("완료!")

    if run_all:
        masks["pancreas"] = run_predict("pancreas", ct_img, ct_vol, ct_nii_path)
        masks["liver_tumor"] = run_predict("liver_tumor", ct_img, ct_vol, ct_nii_path)
        masks["all"] = None  # ALL은 사용 안 함 (UI만 유지)
        st.session_state["ct_cases"][active]["masks"] = masks
        st.success("전체 완료!")

    with dl_area:
        # all 다운로드는 없음(만들지 않으니까)
        if masks.get("pancreas"):
            download_button(Path(masks["pancreas"]), "PANCREAS")
        if masks.get("liver_tumor"):
            download_button(Path(masks["liver_tumor"]), "LIVER_TUMOR")

    # =========================
    # ✅ z 슬라이더: pred 기준
    # - mode/보여줄 마스크에 따라 pred의 zmax로 슬라이더 범위 결정
    # - pred가 없으면 CT zmax로 fallback
    # =========================
    def _get_pred_zmax(p: Path | None) -> int | None:
        if p is None:
            return None
        try:
            mv = np.asanyarray(nib.load(str(p)).dataobj)
            if mv.ndim == 3:
                return int(mv.shape[-1] - 1)
        except Exception:
            return None
        return None

    # 어떤 마스크를 현재 "보기"로 선택할지 (원본 로직 유지)
    if mode.startswith("전체"):
        mpath = masks.get("liver_tumor") or masks.get("pancreas")  # all은 없음
        view_mode = "liver_tumor" if (masks.get("liver_tumor") is not None) else "pancreas"
    elif "Pancreas" in mode:
        mpath = masks.get("pancreas")
        view_mode = "pancreas"
    else:
        mpath = masks.get("liver_tumor")
        view_mode = "liver_tumor"

    # =========================
    # ✅ z 컨트롤: 병변 slice로만 이동 (slider는 표시만)
    # - 핵심: ct_z_state(진짜 값) / ct_z_view(표시용 slider) 키 분리
    # - 마스크 있으면 첫 진입 시 자동으로 첫 병변 slice로 스냅
    # =========================

    pred_zmax = _get_pred_zmax(mpath)
    ct_zmax = int(ct_vol.shape[-1] - 1)

    if pred_zmax is None:
        zmax = ct_zmax
        st.caption(f"ℹ️ pred 없음 → CT 기준 z (0~{zmax})")
    else:
        zmax = pred_zmax
        st.caption(f"✅ pred 기준 z (0~{zmax})  |  CT zmax={ct_zmax} (비례 매핑)")

    # --- state 초기화 (진짜 z는 ct_z_state) ---
    if "ct_z_state" not in st.session_state:
        st.session_state["ct_z_state"] = 0

    # zmax 바뀌면 클램프
    st.session_state["ct_z_state"] = int(np.clip(st.session_state["ct_z_state"], 0, int(zmax)))

    def _set_z(v: int):
        st.session_state["ct_z_state"] = int(np.clip(v, 0, int(zmax)))
        st.rerun()

    # --- 병변 z 통계 (선택된 마스크 기준) ---
    z_has, zmin, zmax_mask, zcount = [], None, None, 0
    if mpath is not None:
        z_has, zmin, zmax_mask, zcount = _mask_z_stats(Path(mpath))

    # ✅ 마스크가 있고 병변이 존재하면: 첫 진입 시 자동 스냅(회색 화면 방지)
    if mpath is not None and zcount > 0:
        z_list = np.array(z_has, dtype=int)
        z_list.sort()

        # 현재 z가 병변 범위 밖이면 가장 가까운 병변으로 스냅
        z_cur = int(st.session_state["ct_z_state"])
        if z_cur < int(z_list[0]) or z_cur > int(z_list[-1]):
            st.session_state["ct_z_state"] = int(z_list[0])
            st.rerun()

        # 현재 위치 인덱스(병변 리스트 기준)
        z_cur = int(st.session_state["ct_z_state"])
        idx = int(np.searchsorted(z_list, z_cur, side="right") - 1)
        idx = int(np.clip(idx, 0, len(z_list) - 1))

        def _set_idx(i: int):
            i = int(np.clip(i, 0, len(z_list) - 1))
            _set_z(int(z_list[i]))

        st.info(f"🧭 **마스크 위치 안내** | z 범위 **{zmin}~{zmax_mask}** | slice **{zcount}개**")

        c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1,1,1,1], gap="small")
        with c1:
            st.button("⏮ 첫 병변", use_container_width=True, on_click=_set_idx, args=(0,))
        with c2:
            st.button("⏪ -10", use_container_width=True, on_click=_set_idx, args=(idx - 10,))
        with c3:
            st.button("◀ -1", use_container_width=True, on_click=_set_idx, args=(idx - 1,))
        with c4:
            st.button("🎯 중앙", use_container_width=True, on_click=_set_idx, args=(len(z_list)//2,))
        with c5:
            st.button("+1 ▶", use_container_width=True, on_click=_set_idx, args=(idx + 1,))
        with c6:
            st.button("+10 ⏩", use_container_width=True, on_click=_set_idx, args=(idx + 10,))
        with c7:
            st.button("마지막 ⏭", use_container_width=True, on_click=_set_idx, args=(len(z_list)-1,))

    else:
        st.caption("ℹ️ 선택된 마스크가 없거나 foreground가 없어 병변 이동 버튼을 표시하지 않았어요.")

    # --- 표시용 slider (사람이 못 움직이게) ---
    st.slider(
        "Slice (Z)",
        0, int(zmax),
        value=int(st.session_state["ct_z_state"]),
        key="ct_z_view",
        disabled=True
    )

    # 최종 z (렌더링에 사용할 값)
    z = int(st.session_state["ct_z_state"])

    cL, cR = st.columns([1, 1], gap="large")
    with cL:
        st.caption("원본 CT")
        # 왼쪽은 그냥 CT만 보여주되, z는 pred 기준으로 들어오니까 render_slice에서 매핑 처리됨
        render_slice(ct_vol, z, wl, ww, alpha=0.0, mask_path=None, mode=view_mode)

    with cR:
        st.caption("오버레이")
        if mpath is None:
            st.warning("위에서 예측(=미리 계산된 pred 로드)을 실행하면 오버레이가 나옵니다.")
        else:
            render_slice(ct_vol, z, wl, ww, alpha=alpha, mask_path=Path(mpath), mode=view_mode)

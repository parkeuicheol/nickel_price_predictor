import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from PIL import Image
import os
import psutil

# ── 페이지 설정 및 헤더 이미지 ─────────────────────────────────────
st.set_page_config(page_title="LME Nickel Price Predictor", layout="wide")
img = Image.open("history_kv.png")
st.image(img, use_container_width=True)
st.title("LME Nickel Price Predicting App")

# Google 스프레드시트 정보
SPREADSHEET_ID = "1TuDjoOtuZHP7xe3WplvIqwIJXTpJVY0k6JHo6ZQktc8"
SHEET_NAME     = "Sheet1"
SPREADSHEET_URL = (
    f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}"
    "/edit?usp=sharing"
)
st.markdown(
    f"- 📋 **LME Nickel Price 원본 데이터셋(스프레드시트) 열기:** "
    f"[여기를 클릭하세요]({SPREADSHEET_URL})"
)

# ── 사이드바: 기준 날짜 & 예측 기간 ─────────────────────────────────
with st.sidebar:
    st.header("입력 설정")
    date_str = st.date_input(
        "기준 날짜를 선택하세요",
        value=pd.to_datetime("2025-01-31")
    ).strftime('%Y-%m-%d')
    shift_set = st.radio(
        "예측 기간(일)을 선택하세요",
        [30, 60, 90],
        format_func=lambda x: f"+{x}일({x//30}달)"
    )
    run = st.button("RUN")

# 메모리 사용량 표시 함수
@st.cache_data

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    return mem

# ── 캐시: 스프레드시트 로드 및 전처리 ─────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        f"/export?format=csv&sheet={sheet_name}"
    )
    df = pd.read_csv(csv_url)
    # 접미사 단위 처리
    def parse_si(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        suffix = s[-1].upper()
        num = s[:-1] if suffix in ['K','M','B'] else s
        try:
            return float(num) * {'K':1e3,'M':1e6,'B':1e9}.get(suffix,1)
        except:
            return np.nan
    for col in ['Gold_Trading_Volume','Dollar_Trading_Volume','NASDAQ_Trading_Volume']:
        df[col] = df[col].apply(parse_si).astype('float64')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df.set_index('date', inplace=True)
    return df

# ── 캐시: 모델 학습 및 분할 ──────────────────────────────────────────
@st.cache_resource
def train_pipeline(
    spreadsheet_id: str,
    sheet_name: str,
    date_str: str,
    shift_set: int
):
    df = load_data(spreadsheet_id, sheet_name)
    # 기준 날짜 이후 데이터
    pos = df.index.get_loc(date_str)
    df2 = df.iloc[pos:].copy()
    df2.bfill(inplace=True)
    df2.dropna(inplace=True)
    # 타깃 생성
    df2['Ni_price_Y'] = df2['Ni_price'].shift(shift_set)
    shift_df = df2.dropna()
    # 학습/검증 분할
    X = shift_df.drop(columns=['Ni_price_Y'])
    y = shift_df['Ni_price_Y']
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    train_n = int(len(X) * 0.99)
    train_idx, val_idx = perm[:train_n], perm[train_n:]
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]
    # XGBoost 학습
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_val,   label=y_val)
    params = {
        'objective': 'reg:squarederror',
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain,'train'),(dtest,'eval')],
        early_stopping_rounds=20,
        verbose_eval=False
    )
    # 예측용 데이터 준비 (첫 shift_set일)
    X_test = df2[df2['Ni_price_Y'].isna()].copy()
    X_test.drop(columns=['Ni_price_Y'], inplace=True)
    X_test.index = pd.to_datetime(X_test.index) + pd.DateOffset(days=shift_set)

    return bst, X_val, y_val, X_test

# ── RUN 버튼 클릭 시 ────────────────────────────────────────────────
if run:
    bst, X_val, y_val, X_test = train_pipeline(
        SPREADSHEET_ID, SHEET_NAME, date_str, shift_set
    )
    # 메모리 사용량 표시
    mem_mb = get_memory_usage_mb()
    st.sidebar.write(f"🧠 Memory usage: {mem_mb:.1f} MB")

    # 검증 지표
    dval = xgb.DMatrix(X_val)
    y_pred = bst.predict(dval)
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    mae  = np.mean(np.abs(y_val - y_pred))
    r2   = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)

    st.write("### XGBoost 예측모델 성능 지표")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAE:**  {mae:.4f}")
    st.write(f"**R²:**   {r2:.4f}")

    # 테스트 예측 전에 feature 이름 맞추기
    feature_names = bst.feature_names
    X_test = X_test[feature_names]

    # 예측 & SMA
    dnew = xgb.DMatrix(X_test)
    X_test['predicted_Ni_price'] = bst.predict(dnew)
    X_test['SMA_7'] = (
        X_test['predicted_Ni_price']
        .rolling(window=7, min_periods=1, center=True)
        .mean()
    )
    X_test_show = X_test[['predicted_Ni_price','SMA_7']]

    st.subheader("Predicted Ni Price Data")
    st.dataframe(X_test_show, use_container_width=True)

    csv_data = X_test_show.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name="predicted_ni_price.csv",
        mime="text/csv"
    )

    st.subheader("Predicted Ni Price Chart")
    fig, ax = plt.subplots()
    ax.plot(X_test.index, X_test['predicted_Ni_price'], label='Predicted Ni Price')
    ax.plot(X_test.index, X_test['SMA_7'], linestyle='--', label='7-Day SMA')
    ax.set_title('Predicted Ni Price & 7-Day SMA')
    ax.set_xlabel('Date'); ax.set_ylabel('Ni Price')
    ax.legend(); fig.autofmt_xdate()
    st.pyplot(fig)

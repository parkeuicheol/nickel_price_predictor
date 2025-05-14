import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from PIL import Image

# ── 페이지 설정 및 헤더 이미지 ─────────────────────────────────────
st.set_page_config(page_title="LME Nickel Price Predictor", layout="wide")
img = Image.open("history_kv.png")
st.image(img, use_container_width=True)
st.title("LME Nickel Price Predicting App")
# 상단에 원본 Sheet URL 상수로 정의
SPREADSHEET_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1TuDjoOtuZHP7xe3WplvIqwIJXTpJVY0k6JHo6ZQktc8"
        "/edit?usp=sharing"
    )
# 클릭하면 원본 시트로 이동하는 링크
st.markdown(f"- 📋 **LME Nickel Price 원본 데이터셋(스프레드시트) 열기:** [여기를 클릭하세요]({SPREADSHEET_URL})")

# ────────────────────────────────────────────────────────────────

# 사이드바: 기준 날짜 & 예측 기간
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

if run:
    # ── Google 스프레드시트에서 CSV로 불러오기 ──────────────────────
    SPREADSHEET_ID = "1TuDjoOtuZHP7xe3WplvIqwIJXTpJVY0k6JHo6ZQktc8"
    SHEET_NAME     = "Sheet1"
    csv_url = (
        f"https://docs.google.com/spreadsheets/d/"
        f"{SPREADSHEET_ID}/export?format=csv&sheet={SHEET_NAME}"
    )
    
    
    df = pd.read_csv(csv_url)
    # ────────────────────────────────────────────────────────────────

    # 접미사 단위(K, M, B) 처리 함수
    def parse_si(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip()
        suffix = x[-1].upper()
        num = x[:-1] if suffix in ['K', 'M', 'B'] else x
        try:
            return float(num) * {'K':1e3, 'M':1e6, 'B':1e9}.get(suffix, 1)
        except:
            return np.nan

    # 거래량 칼럼 변환
    for col in ['Gold_Trading_Volume', 'Dollar_Trading_Volume', 'NASDAQ_Trading_Volume']:
        df[col] = df[col].apply(parse_si).astype('float64')

    # 날짜 처리 & 인덱스 설정
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    df.set_index('date', inplace=True)

    # 기준 날짜 이후 데이터만
    pos = df.index.get_loc(date_str)
    df = df.iloc[pos:].copy()
    df.bfill(inplace=True)
    df.dropna(inplace=True)

    # 타깃 생성 (shift)
    df['Ni_price_Y'] = df['Ni_price'].shift(shift_set)
    shift_df = df.dropna()

    # 학습/검증 데이터 분할
    X = shift_df.drop(columns=['Ni_price_Y'])
    y = shift_df['Ni_price_Y']
    np.random.seed(42)
    perm = np.random.permutation(len(X))
    train_n = int(len(X) * 0.99)
    train_idx, val_idx = perm[:train_n], perm[train_n:]
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

    # XGBoost DMatrix 준비
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

    # 모델 학습
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # 검증 성능 지표 계산
    y_pred = bst.predict(dtest)
    rmse = np.sqrt(np.mean((y_val - y_pred)**2))
    mae  = np.mean(np.abs(y_val - y_pred))
    r2   = 1 - np.sum((y_val - y_pred)**2) / np.sum((y_val - np.mean(y_val))**2)
    st.write("### XGBoost 예측모델 성능 지표")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAE:**  {mae:.4f}")
    st.write(f"**R²:**   {r2:.4f}")

    # ── 예측 데이터 준비 & 시각화 ─────────────────────────────────
    X_test = df[df.isna().any(axis=1)].copy()
    X_test.drop(columns=['Ni_price_Y'], inplace=True)
    X_test.index = pd.to_datetime(X_test.index) + pd.DateOffset(days=shift_set)

    dnew = xgb.DMatrix(X_test)
    X_test['predicted_Ni_price'] = bst.predict(dnew)
    X_test['SMA_7'] = (
        X_test['predicted_Ni_price']
        .rolling(window=7, min_periods=1, center=True)
        .mean()
    )
    X_test_show = X_test[['predicted_Ni_price', 'SMA_7']]

    # 결과 테이블
    st.subheader("Predicted Ni Price Data")
    st.dataframe(X_test_show, use_container_width=True)
    
    # CSV 다운로드 버튼 추가
    csv_data = X_test_show.to_csv(index=True).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv_data,
        file_name="predicted_ni_price.csv",
        mime="text/csv"
    )

    # 차트
    st.subheader("Predicted Ni Price Chart")
    fig, ax = plt.subplots()
    ax.plot(X_test.index, X_test['predicted_Ni_price'], label='Predicted Ni Price')
    ax.plot(X_test.index, X_test['SMA_7'],           linestyle='--', label='7-Day SMA')
    ax.set_title('Predicted Ni Price & 7-Day SMA')
    ax.set_xlabel('Date'); ax.set_ylabel('Ni Price')
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)
    # ────────────────────────────────────────────────────────────────

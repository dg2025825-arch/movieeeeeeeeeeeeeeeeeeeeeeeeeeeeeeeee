import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

st.set_page_config(page_title="영화 흥행 예측기", page_icon="🎬", layout="wide")

# ---------------------------
# 스타일 (카드 섹션 느낌)
# ---------------------------
st.markdown("""
<style>
.card {
    background-color: #ffffff;
    border-radius: 15px;
    padding: 20px 25px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #eee;
}
.metric-box {
    background-color: #f8f9fb;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    border: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 박스오피스 데이터를 활용한 다중 회귀 기반 총 관객 수 예측 모델")

# ---------------------------
# 데이터 불러오기
# ---------------------------
DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies

daily_df, movies_df = load_data()

# ---------------------------
# 기간 정보 (일별 데이터 기준)
# ---------------------------
date_col = "날짜"
min_date = str(int(daily_df[date_col].min()))
max_date = str(int(daily_df[date_col].max()))

def fmt_date(d):
    return f"{d[:4]}년 {d[4:6]}월 {d[6:8]}일"

st.markdown(f"""
<div class="card">
<h4>📅 데이터 기준 기간</h4>
<p style="font-size:18px;"><b>{fmt_date(min_date)} ~ {fmt_date(max_date)}</b> (일별 박스오피스 데이터 기준)</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# 영화별 표(원본) 미리보기
# ---------------------------
with st.expander("📋 영화별 데이터 표 (원본 헤더 포함) 보기", expanded=True):
    st.dataframe(movies_df, use_container_width=True)

# ---------------------------
# 변수 선택 체크박스
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧩 학습에 사용할 변수 선택")

candidate_features = [
    "first_scrn", "first_show", "peak", "first_week_audi", "days_in_top10"
]
feature_labels = {
    "first_scrn": "첫 관측일 스크린수",
    "first_show": "첫 관측일 상영횟수",
    "peak": "성수기 개봉 여부",
    "first_week_audi": "첫 주 관객수",
    "days_in_top10": "10위권 유지 일수"
}

cols = st.columns(len(candidate_features))
selected_features = []
for i, feat in enumerate(candidate_features):
    with cols[i]:
        checked = st.checkbox(feature_labels[feat], value=True, key=f"chk_{feat}")
        if checked:
            selected_features.append(feat)

st.markdown('</div>', unsafe_allow_html=True)

if len(selected_features) == 0:
    st.warning("⚠️ 최소 하나 이상의 변수를 선택해야 모델을 학습할 수 있습니다.")
    st.stop()

# ---------------------------
# 데이터 전처리 및 분할
# ---------------------------
model_df = movies_df.dropna(subset=selected_features + ["total_audi"]).copy()
model_df = model_df.sort_values("movieCd").reset_index(drop=True)

# 10편마다 앞의 3편을 테스트셋으로
is_test = (model_df.index % 10) < 3
test_df = model_df[is_test].copy()
train_df = model_df[~is_test].copy()

X_train = train_df[selected_features]
y_train = train_df["total_audi"]
X_test = test_df[selected_features]
y_test = test_df["total_audi"]

# ---------------------------
# 모델 학습
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.clip(y_pred, 0, None)  # 음수 예측 방지

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# ---------------------------
# 학습/평가 정보 카드
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 모델 학습 및 평가 정보")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="metric-box"><h5>학습에 쓴 영화 수</h5>
    <h3>{len(train_df)}편</h3></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-box"><h5>평가에 쓴 영화 수</h5>
    <h3>{len(test_df)}편</h3></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-box"><h5>기준 기간</h5>
    <h4>{fmt_date(min_date)}<br>~ {fmt_date(max_date)}</h4></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)
with c4:
    st.markdown(f"""<div class="metric-box"><h5>MAE (평균 절대 오차)</h5>
    <h3>{mae:,.0f} 명</h3></div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""<div class="metric-box"><h5>RMSE (평균 제곱근 오차)</h5>
    <h3>{rmse:,.0f} 명</h3></div>""", unsafe_allow_html=True)
with c6:
    st.markdown(f"""<div class="metric-box"><h5>R² 점수</h5>
    <h3>{r2:.3f}</h3></div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# 1000명 미만 예측치 처리
# ---------------------------
FLOOR = 1000
result_df = test_df[["movieCd", "movieNm", "total_audi"]].copy()
result_df["predicted"] = y_pred
low_pred_mask = result_df["predicted"] < FLOOR
low_pred_count = low_pred_mask.sum()

# 그래프 표시용으로 바닥값 붙이기 (로그축 하한 근처)
plot_pred = result_df["predicted"].copy()
plot_pred[low_pred_mask] = FLOOR * 0.5  # 바닥에 붙여서 표시

# ---------------------------
# 산점도 (Plotly)
# ---------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 실제 관객수 vs 예측 관객수 (로그 스케일)")

fig = go.Figure()

# 정상 예측 포인트
normal_mask = ~low_pred_mask
fig.add_trace(go.Scatter(
    x=result_df.loc[normal_mask, "total_audi"],
    y=result_df.loc[normal_mask, "predicted"],
    mode="markers",
    name="예측값",
    marker=dict(size=9, color="royalblue", opacity=0.75, line=dict(width=1, color="white")),
    text=result_df.loc[normal_mask, "movieNm"],
    hovertemplate="<b>%{text}</b><br>실제: %{x:,.0f}명<br>예측: %{y:,.0f}명<extra></extra>"
))

# 1000명 미만(바닥에 붙인) 포인트
if low_pred_count > 0:
    fig.add_trace(go.Scatter(
        x=result_df.loc[low_pred_mask, "total_audi"],
        y=plot_pred[low_pred_mask],
        mode="markers",
        name=f"예측<1,000명 ({low_pred_count}편, 바닥 표시)",
        marker=dict(size=9, color="crimson", symbol="triangle-down", opacity=0.85, line=dict(width=1, color="white")),
        text=result_df.loc[low_pred_mask, "movieNm"],
        hovertemplate="<b>%{text}</b><br>실제: %{x:,.0f}명<br>예측(원값): 1,000명 미만<extra></extra>"
    ))

# 대각선 (y = x)
min_val = max(1, min(result_df["total_audi"].min(), plot_pred.min()) * 0.8)
max_val = max(result_df["total_audi"].max(), plot_pred.max()) * 1.2
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode="lines",
    name="완벽 예측선 (y=x)",
    line=dict(color="gray", dash="dash")
))

fig.update_layout(
    xaxis=dict(title="실제 총 관객수", type="log"),
    yaxis=dict(title="예측 총 관객수", type="log"),
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.info(f"🔻 예측이 1,000명보다 작게 나온 영화는 **{low_pred_count}편**입니다. (그래프 하단에 세모 표시)")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# 테스트 결과 상세 표
# ---------------------------
with st.expander("🔍 테스트 영화별 예측 결과 상세 보기"):
    display_df = result_df.copy()
    display_df["predicted"] = display_df["predicted"].round(0).astype(int)
    display_df["오차(실제-예측)"] = display_df["total_audi"] - display_df["predicted"]
    display_df.columns = ["영화코드", "영화명", "실제 총관객", "예측 총관객", "오차(실제-예측)"]
    st.dataframe(display_df, use_container_width=True)

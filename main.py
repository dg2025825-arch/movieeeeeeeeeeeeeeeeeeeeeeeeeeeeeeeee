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
.interpret-box {
    background-color: #fffbea;
    border-left: 5px solid #f5b301;
    border-radius: 8px;
    padding: 15px 20px;
    margin-top: 10px;
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
# 변수별 간단 설명 (해석문에 활용)
feature_desc = {
    "first_scrn": "영화가 처음 관측됐을 때 걸린 스크린 수(개봉 규모)",
    "first_show": "영화가 처음 관측됐을 때의 상영 횟수(상영 빈도)",
    "peak": "12·1·7·8월 같은 성수기에 개봉했는지 여부",
    "first_week_audi": "개봉 첫 주 동안 든 관객 수(초반 흥행세)",
    "days_in_top10": "박스오피스 10위 안에 머문 일수(흥행 지속력)"
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
# 📖 산점도 해석 (선택 변수 조합에 따라 자동 생성)
# ---------------------------
with st.expander("📖 이 그래프, 어떻게 해석하면 될까요? (클릭해서 펼치기)", expanded=False):

    # 1) 선택한 변수 설명
    feat_desc_list = "\n".join([f"- **{feature_labels[f]}**: {feature_desc[f]}" for f in selected_features])

    st.markdown(f"""
    <div class="interpret-box">
    <h4>1️⃣ 지금 어떤 정보로 예측했나요?</h4>
    <p>지금 선택한 변수는 총 <b>{len(selected_features)}개</b>입니다. 이 모델은 아래 정보를 바탕으로
    영화의 총 관객수를 추측했어요.</p>
    {feat_desc_list}
    </div>
    """, unsafe_allow_html=True)

    # 2) R² 해석
    if r2 >= 0.8:
        r2_comment = "굉장히 높은 편이에요! 선택한 변수만으로도 흥행 결과를 잘 설명하고 있다는 뜻이에요. 📈"
        r2_color = "#e8f8f0"
    elif r2 >= 0.5:
        r2_comment = "나쁘지 않은 편이지만, 아직 설명하지 못하는 부분도 꽤 있어요. 변수를 더 추가하거나 바꿔보면 어떨까요?"
        r2_color = "#fff8e1"
    elif r2 >= 0:
        r2_comment = "다소 낮은 편이에요. 선택한 변수들만으로는 관객수 변화를 설명하기 부족할 수 있어요. 다른 변수 조합도 시도해 보세요!"
        r2_color = "#fdeaea"
    else:
        r2_comment = "R²가 음수라는 것은, 이 모델이 '평균값으로 무작정 찍는 것'보다도 예측을 못하고 있다는 뜻이에요. 변수 조합을 바꿔보세요!"
        r2_color = "#fdeaea"

    st.markdown(f"""
    <div class="interpret-box" style="background-color:{r2_color}; border-left-color:#4caf50;">
    <h4>2️⃣ R² 점수({r2:.3f})가 뜻하는 것</h4>
    <p>R²는 <b>0~1 사이 값</b>으로, 1에 가까울수록 모델이 실제 관객수 변화를 잘 설명한다는 뜻이에요
    (음수가 나올 수도 있는데, 이 경우는 모델이 매우 못 맞춘다는 의미입니다).</p>
    <p>지금 점수는 <b>{r2:.3f}</b>입니다. {r2_comment}</p>
    </div>
    """, unsafe_allow_html=True)

    # 3) MAE 해석 (테스트셋 평균 관객수 대비 비교)
    avg_actual = y_test.mean()
    mae_ratio = mae / avg_actual * 100 if avg_actual > 0 else 0

    st.markdown(f"""
    <div class="interpret-box">
    <h4>3️⃣ 예측이 평균적으로 얼마나 빗나갔나요?</h4>
    <p>평균 절대 오차(MAE)는 <b>{mae:,.0f}명</b>입니다.
    이는 테스트에 사용한 영화들의 <b>실제 평균 관객수({avg_actual:,.0f}명)</b>의
    약 <b>{mae_ratio:.1f}%</b>에 해당하는 크기예요.</p>
    <p>쉽게 말해 "평균적으로 실제 관객수보다 {mae:,.0f}명 정도 어긋난 예측을 하고 있다"고 볼 수 있어요.
    이 비율이 작을수록 예측이 더 정확한 것입니다.</p>
    </div>
    """, unsafe_allow_html=True)

    # 4) 산점도의 점 위치 해석
    st.markdown(f"""
    <div class="interpret-box">
    <h4>4️⃣ 산점도의 점은 어떻게 읽나요?</h4>
    <p>그래프의 <b>회색 점선(대각선)</b>은 "실제값 = 예측값"인 이상적인 경우를 나타내요.</p>
    <ul>
    <li>점이 <b>대각선 위쪽</b>에 있으면 → 모델이 <b>실제보다 더 많이</b> 관객이 들 것으로 예측한 경우예요 (과대 예측).</li>
    <li>점이 <b>대각선 아래쪽</b>에 있으면 → 모델이 <b>실제보다 적게</b> 관객이 들 것으로 예측한 경우예요 (과소 예측).</li>
    <li>점이 <b>대각선에 가까울수록</b> → 예측이 정확했다는 뜻이에요.</li>
    </ul>
    <p>가로축과 세로축이 모두 <b>로그 스케일</b>인 이유는, 관객수가 몇백 명부터 몇천만 명까지
    범위가 아주 넓기 때문이에요. 로그로 그리면 큰 영화와 작은 영화를 같은 화면에서 비교하기 쉬워져요.</p>
    </div>
    """, unsafe_allow_html=True)

    # 5) 1000명 미만(바닥 표시) 해석
    if low_pred_count > 0:
        low_movies = result_df.loc[low_pred_mask, "movieNm"].tolist()
        low_movies_str = ", ".join(low_movies[:5]) + (" 외" if len(low_movies) > 5 else "")
        st.markdown(f"""
        <div class="interpret-box" style="border-left-color:#e53935;">
        <h4>5️⃣ 그래프 맨 아래에 붙어있는 빨간 세모(▽)는 뭔가요?</h4>
        <p>이 모델이 예측한 관객수가 <b>1,000명도 안 될 만큼 아주 적게</b> 나온 영화들이에요.
        총 <b>{low_pred_count}편</b>이 여기에 해당하고, 예: {low_movies_str}</p>
        <p>이런 영화들은 선택한 변수만으로는 흥행을 설명하기 어려운 경우일 수 있어요.
        예를 들어 개봉 초반 스크린수는 적었지만 입소문으로 나중에 관객이 몰린 영화라면,
        지금 선택한 변수들로는 그 흐름을 포착하기 어려울 수 있답니다.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="interpret-box" style="border-left-color:#43a047;">
        <h4>5️⃣ 1,000명 미만으로 예측된 영화가 있나요?</h4>
        <p>이번 변수 조합에서는 예측치가 1,000명 미만으로 나온 영화가 <b>없어요</b>. 
        즉, 모델이 극단적으로 낮은 관객수를 예측한 사례는 없다는 뜻이에요.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="interpret-box" style="border-left-color:#1e88e5;">
    <h4>💡 팁: 변수 조합을 바꿔가며 비교해 보세요!</h4>
    <p>위쪽 체크박스에서 변수를 켜고 끄면서 R²와 MAE, 그리고 산점도의 모양이 어떻게
    달라지는지 관찰해 보세요. 어떤 변수 조합이 가장 예측을 잘 하는지 스스로 찾아보는 것도
    좋은 탐구 활동이 될 거예요!</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 테스트 결과 상세 표
# ---------------------------
with st.expander("🔍 테스트 영화별 예측 결과 상세 보기"):
    display_df = result_df.copy()
    display_df["predicted"] = display_df["predicted"].round(0).astype(int)
    display_df["오차(실제-예측)"] = display_df["total_audi"] - display_df["predicted"]
    display_df.columns = ["영화코드", "영화명", "실제 총관객", "예측 총관객", "오차(실제-예측)"]
    st.dataframe(display_df, use_container_width=True)

import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔬",
    layout="wide"
)

# -----------------------------
# 커스텀 CSS (병원/실험실 느낌 통일)
# -----------------------------
st.markdown("""
    <style>
    .stApp {
        background-color: #f4faff;
    }
    .hospital-header {
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
        margin-bottom: 25px;
    }
    .hospital-header h1 {
        color: white;
        font-size: 36px;
        margin: 0;
    }
    .hospital-header p {
        color: #eaf6ff;
        font-size: 16px;
        margin-top: 5px;
    }
    .section-title {
        background-color: #e8f4ff;
        border-left: 6px solid #00c6ff;
        padding: 10px 15px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        color: #0072ff;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 헤더
# -----------------------------
st.markdown("""
    <div class="hospital-header">
        <h1>🔬 데이터 탐색실</h1>
        <p>뇌졸중 데이터를 여러 각도로 살펴보는 분석 공간</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# 데이터 불러오기 (첫 화면과 동일)
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="UTF-8")
    return df

df = load_data()

# =========================================================
# 1. 나이와 평균 혈당의 분포 히스토그램 (나란히)
# =========================================================
st.markdown('<div class="section-title">📊 나이와 평균 혈당 분포</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df, x="age", nbins=30,
        title="나이(age) 분포",
        color_discrete_sequence=["#0072ff"]
    )
    fig_age.update_layout(bargap=0.1)
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df, x="avg_glucose_level", nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        color_discrete_sequence=["#00c6ff"]
    )
    fig_glucose.update_layout(bargap=0.1)
    st.plotly_chart(fig_glucose, use_container_width=True)

# =========================================================
# 2. 뇌졸중 유무별 나이·혈당 상자그림 + 평균표
# =========================================================
st.markdown('<div class="section-title">📦 뇌졸중 유무에 따른 나이·혈당 비교</div>', unsafe_allow_html=True)

df["stroke_label"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df, x="stroke_label", y="age",
        color="stroke_label",
        title="뇌졸중 유무별 나이 비교",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df, x="stroke_label", y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 유무별 평균 혈당 비교",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 평균값 표
mean_table = df.groupby("stroke_label")[["age", "avg_glucose_level"]].mean().round(2)
mean_table.columns = ["나이 평균", "평균 혈당 평균"]
mean_table = mean_table.reset_index().rename(columns={"stroke_label": "뇌졸중 여부"})

st.markdown("**두 그룹의 평균값 비교표**")
st.dataframe(mean_table, use_container_width=True)

# =========================================================
# 3. 고혈압·심장병 유무에 따른 뇌졸중 비율 막대그래프
# =========================================================
st.markdown('<div class="section-title">📈 고혈압·심장병에 따른 뇌졸중 비율</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    hyper_ratio = df.groupby("hypertension")["stroke"].mean().reset_index()
    hyper_ratio["hypertension"] = hyper_ratio["hypertension"].map({0: "고혈압 없음", 1: "고혈압 있음"})
    hyper_ratio["stroke"] = (hyper_ratio["stroke"] * 100).round(2)

    fig_hyper = px.bar(
        hyper_ratio, x="hypertension", y="stroke",
        title="고혈압 유무에 따른 뇌졸중 비율(%)",
        labels={"hypertension": "고혈압 여부", "stroke": "뇌졸중 비율(%)"},
        color="hypertension",
        text="stroke"
    )
    st.plotly_chart(fig_hyper, use_container_width=True)

with col6:
    heart_ratio = df.groupby("heart_disease")["stroke"].mean().reset_index()
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map({0: "심장병 없음", 1: "심장병 있음"})
    heart_ratio["stroke"] = (heart_ratio["stroke"] * 100).round(2)

    fig_heart = px.bar(
        heart_ratio, x="heart_disease", y="stroke",
        title="심장병 유무에 따른 뇌졸중 비율(%)",
        labels={"heart_disease": "심장병 여부", "stroke": "뇌졸중 비율(%)"},
        color="heart_disease",
        text="stroke"
    )
    st.plotly_chart(fig_heart, use_container_width=True)

# =========================================================
# 4. bmi 결측치 그룹의 뇌졸중 비율 vs 전체 뇌졸중 비율
# =========================================================
st.markdown('<div class="section-title">🩸 체질량지수(bmi) 결측치와 뇌졸중 비율</div>', unsafe_allow_html=True)

bmi_missing = df[df["bmi"].isnull()]
n_missing = len(bmi_missing)

missing_stroke_ratio = round(bmi_missing["stroke"].mean() * 100, 2) if n_missing > 0 else 0
overall_stroke_ratio = round(df["stroke"].mean() * 100, 2)

bmi_compare_table = pd.DataFrame({
    "구분": ["bmi 결측치 그룹", "전체 데이터"],
    "사람 수": [n_missing, len(df)],
    "뇌졸중 비율(%)": [missing_stroke_ratio, overall_stroke_ratio]
})

st.dataframe(bmi_compare_table, use_container_width=True)

# =========================================================
# 5. 흡연 상태별 사람 수 표
# =========================================================
st.markdown('<div class="section-title">🚬 흡연 상태별 사람 수</div>', unsafe_allow_html=True)

smoking_count = df["smoking_status"].value_counts().reset_index()
smoking_count.columns = ["흡연 상태", "사람 수"]

st.dataframe(smoking_count, use_container_width=True)

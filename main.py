import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 페이지 기본 설정 (브라우저 탭 제목 + 아이콘)
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# 병원/실험실 느낌 커스텀 CSS
# -----------------------------
st.markdown("""
    <style>
    /* 전체 배경 */
    .stApp {
        background-color: #f4faff;
    }

    /* 메인 타이틀 박스 */
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
        font-size: 40px;
        margin: 0;
    }
    .hospital-header p {
        color: #eaf6ff;
        font-size: 16px;
        margin-top: 5px;
    }

    /* 지표 카드 스타일 */
    .metric-card {
        background-color: white;
        border-left: 8px solid #0072ff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
    }
    .metric-card h2 {
        color: #0072ff;
        font-size: 32px;
        margin: 5px 0;
    }
    .metric-card p {
        color: #555;
        font-size: 15px;
        margin: 0;
    }

    /* 섹션 제목 */
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

    /* 출처 박스 */
    .source-box {
        background-color: #fffbe6;
        border: 2px dashed #ffcc00;
        border-radius: 10px;
        padding: 20px;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 헤더 (제목 + 아이콘)
# -----------------------------
st.markdown("""
    <div class="hospital-header">
        <h1>🧠 뇌졸중 예측 실습실 🩺</h1>
        <p>의료 데이터를 활용한 뇌졸중 발생 예측 데이터 탐구 실습</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# 데이터 불러오기
# -----------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="UTF-8")
    return df

df = load_data()

# -----------------------------
# 소개 문구
# -----------------------------
st.markdown("""
###  실습실에 오신 것을 환영합니다!
이곳은 실제 환자 데이터를 바탕으로 **뇌졸중 발생 여부**를 탐구하는 실습 공간입니다.  
아래에서 데이터의 전체적인 모습을 먼저 살펴봅시다. 🔬
""")

# -----------------------------
# 지표 카드 4개
# -----------------------------
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = round(stroke_count / total_people * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <p>👥 전체 사람 수</p>
            <h2>{total_people:,}</h2>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <p>📋 열 개수</p>
            <h2>{total_columns}</h2>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <p>🧠 뇌졸중 환자 수</p>
            <h2>{stroke_count:,}</h2>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <p>📊 뇌졸중 비율</p>
            <h2>{stroke_ratio}%</h2>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# 열 설명 표 (우리말 뜻은 직접 채우기)
# -----------------------------
st.markdown('<div class="section-title">📑 데이터 열(컬럼) 설명표</div>', unsafe_allow_html=True)

col_info = []
for col in df.columns:
    dtype = df[col].dtype
    n_missing = df[col].isnull().sum()

    if dtype == "object":
        unique_vals = df[col].dropna().unique()
        value_kind = ", ".join(map(str, unique_vals[:6]))
        if len(unique_vals) > 6:
            value_kind += " 등"
    else:
        value_kind = f"숫자형 ({df[col].min()} ~ {df[col].max()})"

    col_info.append({
        "열 이름": col,
        "우리말 뜻": "",   # 학생이 직접 채워 넣는 칸
        "값의 종류": value_kind,
        "빈 값 개수": n_missing
    })

info_df = pd.DataFrame(col_info)

edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"]  # 우리말 뜻만 수정 가능
)

# -----------------------------
# 데이터 미리보기 (상위 5줄)
# -----------------------------
st.markdown('<div class="section-title">🔍 데이터 미리보기 (상위 5줄)</div>', unsafe_allow_html=True)
st.dataframe(df.head(5), use_container_width=True)

# -----------------------------
# 데이터 출처 (직접 작성)
# -----------------------------
st.markdown('<div class="section-title">📚 데이터 출처</div>', unsafe_allow_html=True)

st.markdown('<div class="source-box">', unsafe_allow_html=True)
source_text = st.text_area(
    "교재를 참고하여 데이터 출처를 아래에 작성해 보세요.",
    placeholder="여기에 데이터 출처를 입력하세요...",
    height=100
)
st.markdown('</div>', unsafe_allow_html=True)

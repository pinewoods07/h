import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🧪",
    layout="wide"
)

# -----------------------------
# 커스텀 CSS
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
        font-size: 36px;
        margin: 5px 0;
    }
    .metric-card p.title {
        color: #333;
        font-size: 16px;
        font-weight: bold;
        margin: 0;
    }
    .metric-card p.sub {
        color: #777;
        font-size: 13px;
        margin: 5px 0 0 0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 헤더
# -----------------------------
st.markdown("""
    <div class="hospital-header">
        <h1>🧪 분류 모델 실험실</h1>
        <p>뇌졸중을 예측하는 여러 모델을 만들고 비교해 봅니다</p>
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
# 속성 이름 매핑 (열 이름 <-> 우리말)
# -----------------------------
col_to_kor = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
kor_to_col = {v: k for k, v in col_to_kor.items()}

available_features = list(col_to_kor.keys())
default_features_kor = ["나이", "평균 혈당", "고혈압", "심장병"]  # bmi 제외 기본값

# -----------------------------
# 속성 선택 UI
# -----------------------------
st.markdown('<div class="section-title">🧬 입력 속성 선택</div>', unsafe_allow_html=True)

selected_kor = st.multiselect(
    "모델 입력으로 사용할 속성을 골라 주세요 (최소 2개)",
    options=[col_to_kor[c] for c in available_features],
    default=default_features_kor
)

if len(selected_kor) < 2:
    st.warning("⚠️ 속성을 두 개 이상 선택해야 모델을 만들 수 있습니다. 속성을 더 골라 주세요.")
    st.stop()

selected_cols = [kor_to_col[k] for k in selected_kor]

# -----------------------------
# 데이터 정렬 및 10명씩 묶어서 분할
# -----------------------------
df_sorted = df.sort_values("id").reset_index(drop=True)

# 그룹 번호: 0~9번째 -> 그룹0, 10~19번째 -> 그룹1 ...
df_sorted["group_idx"] = df_sorted.index // 10
df_sorted["pos_in_group"] = df_sorted.index % 10

# 각 그룹의 앞 3명은 테스트용, 나머지 7명은 학습용
test_mask = df_sorted["pos_in_group"] < 3
train_mask = ~test_mask

train_df = df_sorted[train_mask].copy()
test_df = df_sorted[test_mask].copy()

# -----------------------------
# bmi 결측치 처리 (학습용 중앙값으로 채움, bmi를 선택했을 때만)
# -----------------------------
if "bmi" in selected_cols:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)
else:
    bmi_median = None

X_train = train_df[selected_cols]
y_train = train_df["stroke"]
X_test = test_df[selected_cols]
y_test = test_df["stroke"]

st.info(f"📌 학습용 사람 수: {len(X_train):,}명 / 테스트용 사람 수: {len(X_test):,}명")

# -----------------------------
# 모델 학습
# -----------------------------
RANDOM_STATE = 42

# 1. 로지스틱 회귀
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# 2. 의사결정트리 (깊이 3, 마디 최소 인원 5명 미만이면 안 나눔)
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    random_state=RANDOM_STATE
)
tree_model.fit(X_train, y_train)

# 3. 다수결 모델 (더미 모델)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_model.fit(X_train, y_train)

# -----------------------------
# 정확도 계산 함수
# -----------------------------
def get_accuracies(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model, X_train, y_train, X_test, y_test)
tree_train_acc, tree_test_acc = get_accuracies(tree_model, X_train, y_train, X_test, y_test)
dummy_train_acc, dummy_test_acc = get_accuracies(dummy_model, X_train, y_train, X_test, y_test)

# -----------------------------
# 정확도 카드 표시
# -----------------------------
st.markdown('<div class="section-title">🎯 모델별 정확도 비교</div>', unsafe_allow_html=True)

card_col1, card_col2, card_col3 = st.columns(3)

def render_metric_card(col, title, test_acc, train_acc):
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <p class="title">{title}</p>
                <h2>{test_acc*100:.2f}%</h2>
                <p class="sub">훈련 정확도: {train_acc*100:.2f}% &nbsp;|&nbsp; 테스트 정확도: {test_acc*100:.2f}%</p>
            </div>
        """, unsafe_allow_html=True)

render_metric_card(card_col1, "로지스틱 회귀 (확률로 답하는 모델)", log_test_acc, log_train_acc)
render_metric_card(card_col2, "의사결정트리 (질문으로 답하는 모델)", tree_test_acc, tree_train_acc)
render_metric_card(card_col3, "다수결 모델 (입력을 보지 않는 모델)", dummy_test_acc, dummy_train_acc)

# =========================================================
# 산점도 + 로지스틱 회귀 경계선 + 트리 영역 칠하기
# =========================================================
st.markdown('<div class="section-title">📍 두 속성으로 보는 분류 결과</div>', unsafe_allow_html=True)

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_kor = st.selectbox("가로축으로 사용할 속성", options=selected_kor, index=0)
with axis_col2:
    remaining = [k for k in selected_kor if k != x_kor]
    y_kor = st.selectbox("세로축으로 사용할 속성", options=remaining, index=0)

x_col = kor_to_col[x_kor]
y_col = kor_to_col[y_kor]

other_cols = [c for c in selected_cols if c not in [x_col, y_col]]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {}
for c in other_cols:
    fixed_values[c] = X_test[c].median()

if fixed_values:
    fixed_text = ", ".join([f"{col_to_kor[c]} = {v:.2f}" for c, v in fixed_values.items()])
    st.write(f"📌 그림에 나타나지 않는 속성은 테스트 데이터의 중앙값으로 고정했습니다: **{fixed_text}**")
else:
    st.write("📌 선택한 속성이 모두 그림의 두 축으로 사용되어, 고정할 속성이 없습니다.")

# 테스트 데이터 산점도용 표
plot_df = test_df.copy()
plot_df["실제 뇌졸중 여부"] = plot_df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})

fig_scatter = px.scatter(
    plot_df,
    x=x_col,
    y=y_col,
    color="실제 뇌졸중 여부",
    labels={x_col: x_kor, y_col: y_kor},
    title=f"{x_kor} vs {y_kor} (테스트 데이터, 실제 뇌졸중 여부)",
    color_discrete_map={"뇌졸중 없음": "#00c6ff", "뇌졸중 있음": "#ff4b4b"},
    opacity=0.7
)

# -----------------------------
# 로지스틱 회귀 0.5 경계선 계산
# -----------------------------
# 로지스틱 회귀 식: w0*x0 + w1*x1 + ... + b = 0 (확률 0.5 지점)
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]

col_index = {c: i for i, c in enumerate(selected_cols)}
x_idx = col_index[x_col]
y_idx = col_index[y_col]

# 고정된 속성들의 기여분 계산
fixed_contribution = 0.0
for c in other_cols:
    fixed_contribution += coef[col_index[c]] * fixed_values[c]

x_min, x_max = plot_df[x_col].min(), plot_df[x_col].max()
y_min, y_max = plot_df[y_col].min(), plot_df[y_col].max()

boundary_out_of_range = False

# w_y * y + w_x * x + fixed + b = 0  ->  y = -(w_x*x + fixed + b) / w_y
if abs(coef[y_idx]) > 1e-12:
    x_line = np.linspace(x_min, x_max, 100)
    y_line = -(coef[x_idx] * x_line + fixed_contribution + intercept) / coef[y_idx]

    # 경계선이 그림 범위 안에 있는지 확인
    if np.all(y_line < y_min) or np.all(y_line > y_max):
        boundary_out_of_range = True
    else:
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            name="로지스틱 회귀 경계선 (0.5)",
            line=dict(color="black", dash="dash", width=2)
        ))
else:
    # y 계수가 0이면 세로선으로 표현 시도 (x_idx 기준)
    if abs(coef[x_idx]) > 1e-12:
        x_boundary = -(fixed_contribution + intercept) / coef[x_idx]
        if x_boundary < x_min or x_boundary > x_max:
            boundary_out_of_range = True
        else:
            fig_scatter.add_vline(x=x_boundary, line_dash="dash", line_color="black")
    else:
        boundary_out_of_range = True

if boundary_out_of_range:
    st.write("📌 로지스틱 회귀의 0.5 경계선은 이 그림의 범위 밖에 위치하여 표시되지 않았습니다.")

# -----------------------------
# 의사결정트리 영역 칠하기 (배경 격자)
# -----------------------------
grid_n = 80
x_grid = np.linspace(x_min, x_max, grid_n)
y_grid = np.linspace(y_min, y_max, grid_n)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_data = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for c in other_cols:
    grid_data[c] = fixed_values[c]
grid_data = grid_data[selected_cols]  # 학습 때와 같은 열 순서로 맞춤

tree_pred_grid = tree_model.predict(grid_data).reshape(xx.shape)

fig_scatter.add_trace(go.Contour(
    x=x_grid,
    y=y_grid,
    z=tree_pred_grid,
    showscale=False,
    opacity=0.25,
    colorscale=[[0, "#00c6ff"], [1, "#ff4b4b"]],
    contours=dict(coloring="fill"),
    line=dict(width=0),
    name="의사결정트리 영역",
    hoverinfo="skip"
))

# 산점도가 배경 위에 오도록 트레이스 순서 재정렬 (Contour를 맨 뒤로)
fig_scatter.data = tuple(list(fig_scatter.data[1:]) + [fig_scatter.data[0]]) if len(fig_scatter.data) > 1 else fig_scatter.data

st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================================
# 의사결정트리 가지 그림 (Graphviz DOT)
# =========================================================
st.markdown('<div class="section-title">🌳 의사결정트리 구조 보기</div>', unsafe_allow_html=True)

tree_ = tree_model.tree_
feature_names_kor = [col_to_kor[c] for c in selected_cols]

def build_dot(tree_, feature_names_kor):
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled, rounded", fontname="Malgun Gothic"];']

    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    feature = tree_.feature
    threshold = tree_.threshold
    value = tree_.value  # [node][class][count] -> class 0: 없음, class 1: 있음

    leaf_count = 0
    leaf_no_count = 0
    used_features = set()

    for node_id in range(n_nodes):
        n_no = value[node_id][0][0]
        n_yes = value[node_id][0][1]
        n_total = n_no + n_yes
        ratio = n_yes / n_total if n_total > 0 else 0

        is_leaf = children_left[node_id] == children_right[node_id]

        if is_leaf:
            leaf_count += 1
            answer = "뇌졸중 있음" if n_yes >= n_no else "뇌졸중 없음"
            if answer == "뇌졸중 없음":
                leaf_no_count += 1
                color = "#cfe8ff"
            else:
                color = "#ffd6d6"
            label = f"답: {answer}\\n인원 {int(n_total)}명 (뇌졸중 {int(n_yes)}명)\\n비율 {ratio*100:.1f}%"
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
        else:
            feat_name = feature_names_kor[feature[node_id]]
            used_features.add(feat_name)
            thresh = threshold[node_id]
            label = f"{feat_name} <= {thresh:.2f}?\\n인원 {int(n_total)}명 (뇌졸중 {int(n_yes)}명)\\n비율 {ratio*100:.1f}%"
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="#fff7cc"];')

            left_id = children_left[node_id]
            right_id = children_right[node_id]
            dot_lines.append(f'{node_id} -> {left_id} [label="예"];')
            dot_lines.append(f'{node_id} -> {right_id} [label="아니요"];')

    dot_lines.append("}")
    return "\n".join(dot_lines), leaf_count, leaf_no_count, used_features

dot_string, leaf_count, leaf_no_count, used_features = build_dot(tree_, feature_names_kor)

st.graphviz_chart(dot_string)

# -----------------------------
# 트리 요약 설명
# -----------------------------
st.markdown("**🔎 트리 구조 요약**")
st.write(f"- 답을 내는 마디(잎)는 모두 **{leaf_count}칸**이고, 그중 **{leaf_no_count}칸**이 '뇌졸중 없음'이라고 답합니다.")

not_used = [k for k in selected_kor if k not in used_features]
if used_features:
    st.write(f"- 이 나무가 실제로 물어본 속성: **{', '.join(sorted(used_features))}**")
if not_used:
    st.write(f"- 선택했지만 나무가 실제로 묻지 않은 속성: **{', '.join(not_used)}**")

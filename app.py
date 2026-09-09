import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ----------------------------------------------------
# 1. 페이지 기본 설정 및 경로
# ----------------------------------------------------
st.set_page_config(
    page_title="무역 분석 대시보드",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACI_PATH = os.path.join(BASE_DIR, "baci_85_sample.csv")
COUNTRY_PATH = os.path.join(BASE_DIR, "country_codes_sample.csv")

# ----------------------------------------------------
# 2. 한글 폰트 설정 (오류 없는 경량 로드)
# ----------------------------------------------------
plt.rcParams["axes.unicode_minus"] = False

font_path = os.path.join(BASE_DIR, "fonts", "온글잎 콘콘체.ttf")
if not os.path.exists(font_path):
    font_path = os.path.join(BASE_DIR, "..", "fonts", "온글잎 콘콘체.ttf")

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)
else:
    if os.name == "nt":
        plt.rc("font", family="Malgun Gothic")
    else:
        plt.rc("font", family="AppleGothic")

# ----------------------------------------------------
# 3. CSS 스타일링 (멈춤 없는 안전한 파스텔 블루 테마)
# ----------------------------------------------------
st.markdown(
    """
    <style>
    /* 전체 배경: 은은한 파스텔 블루 */
    .stApp {
        background: linear-gradient(135deg, #F0F6FC 0%, #E2EDF8 100%);
        color: #1A365D;
    }

    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #E8F2FA !important;
        border-right: 1px solid #CFE2F3;
    }

    /* 메인 타이틀 */
    .title-container {
        padding: 10px 0;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0C3C78;
    }
    .sub-desc {
        color: #3E6B99;
        font-size: 1rem;
        margin-top: 4px;
        margin-bottom: 20px;
    }

    /* 지표 카드 커스텀 (높은 수치는 짙은 블루 강조) */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.85);
        border: 1.5px solid #BDD7EE;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(12, 60, 120, 0.05);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.95rem !important;
        color: #4A6572 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #0A3670 !important;
        font-size: 1.8rem !important;
        font-weight: 800 !important;
    }

    /* 섹션 제목 */
    h3, h5 {
        color: #0D3B66 !important;
        font-weight: 700 !important;
    }

    /* 테이블 영역 */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------
# 4. 데이터 로드 및 전처리
# ----------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    try:
        baci_df = pd.read_csv(BACI_PATH, encoding="utf-8")
        country_df = pd.read_csv(COUNTRY_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        baci_df = pd.read_csv(BACI_PATH, encoding="cp949")
        country_df = pd.read_csv(COUNTRY_PATH, encoding="cp949")

    raw_baci_df = baci_df.copy()

    baci_df.columns = [str(c).strip().lower() for c in baci_df.columns]
    country_df.columns = [str(c).strip().lower() for c in country_df.columns]

    target_key = "j" if "j" in country_df.columns else country_df.columns[0]
    name_col = (
        "country_name"
        if "country_name" in country_df.columns
        else country_df.columns[1]
    )

    baci_df["j"] = pd.to_numeric(baci_df["j"], errors="coerce")
    country_df[target_key] = pd.to_numeric(
        country_df[target_key], errors="coerce"
    )

    merged_df = pd.merge(
        baci_df,
        country_df[[target_key, name_col]],
        left_on="j",
        right_on=target_key,
        how="left",
    )
    merged_df["country_name"] = merged_df[name_col].fillna(
        "국가코드_" + merged_df["j"].astype(str)
    )

    year_col = "t" if "t" in merged_df.columns else "year"
    val_col = "v" if "v" in merged_df.columns else "trade_value"

    merged_df["year"] = merged_df[year_col]
    merged_df["trade_value_usd"] = (
        pd.to_numeric(
            merged_df[val_col].astype(str).str.replace(",", ""),
            errors="coerce",
        ).fillna(0)
        * 1000
    )

    labels = ["소", "중", "대"]
    try:
        merged_df["무역액등급"] = pd.qcut(
            merged_df["trade_value_usd"], q=3, labels=labels, duplicates="drop"
        )
    except Exception:
        merged_df["무역액등급"] = pd.cut(
            merged_df["trade_value_usd"], bins=3, labels=labels
        )

    return raw_baci_df, merged_df


raw_baci, df = load_and_preprocess_data()

# ----------------------------------------------------
# 5. 사이드바 필터
# ----------------------------------------------------
with st.sidebar:
    st.markdown("### ✈️ 글로벌 필터 옵션")
    st.caption("국가 및 무역 규모별 데이터를 필터링합니다.")

    all_countries = sorted([str(x) for x in df["country_name"].unique()])
    selected_countries = st.multiselect(
        "🌐 국가 선택 (다중 선택 가능)",
        options=all_countries,
        default=[],
        placeholder="전체 국가 표시 중...",
    )

    tier_options = ["대", "중", "소"]
    selected_tiers = st.multiselect(
        "📊 무역액 등급 선택 (대/중/소)",
        options=tier_options,
        default=tier_options,
    )

filtered_df = df.copy()
if selected_countries:
    filtered_df = filtered_df[
        filtered_df["country_name"].isin(selected_countries)
    ]
if selected_tiers:
    filtered_df = filtered_df[filtered_df["무역액등급"].isin(selected_tiers)]

# ----------------------------------------------------
# 6. 메인 화면 출력
# ----------------------------------------------------

# 1) 타이틀 (안전한 이모지 & 텍스트)
st.markdown(
    """
    <div class="title-container">
        <div class="main-title">🌍 무역 분석 대시보드 ✈️</div>
        <div class="sub-desc">반도체 및 전자부품(HS 85) 글로벌 교역 흐름 모니터링 시스템</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# 2) 결측치 현황
st.subheader("2. baci_85_sample.csv 파일의 결측치")
null_df = pd.DataFrame(
    {
        "컬럼명": raw_baci.columns,
        "결측치 개수": raw_baci.isnull().sum().values,
        "결측치 비율(%)": (
            raw_baci.isnull().mean() * 100
        ).round(2).values,
    }
)
st.dataframe(null_df, use_container_width=True, height=180)
st.markdown("---")

# 3) 총거래건수 & 총수출액(달러)
st.subheader("3. 거래 실적 요약")
total_transactions = len(filtered_df)
total_export_value = filtered_df["trade_value_usd"].sum()

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="📦 총거래건수", value=f"{total_transactions:,} 건"
    )
with col2:
    st.metric(
        label="💵 총수출액 (달러)",
        value=f"${total_export_value:,.0f}",
    )
st.markdown("---")

# 4) 시각화 (상위 8개국 히트맵 & 무역액 등급분포)
st.subheader("4. 국가별 및 등급별 무역 패턴 분석")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("##### 🌐 국가*연도 수출액 히트맵 (상위 8개국)")
    if not filtered_df.empty:
        top8_countries = (
            filtered_df.groupby("country_name")["trade_value_usd"]
            .sum()
            .nlargest(8)
            .index
        )
        heat_df = filtered_df[
            filtered_df["country_name"].isin(top8_countries)
        ]

        if not heat_df.empty:
            pivot_heat = heat_df.pivot_table(
                index="country_name",
                columns="year",
                values="trade_value_usd",
                aggfunc="sum",
            ).fillna(0)

            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)

            # 높은 수치일수록 진한 파랑 강조 (YlGnBu)
            sns.heatmap(
                pivot_heat,
                cmap="YlGnBu",
                annot=False,
                fmt=",.0f",
                cbar=True,
                linewidths=0.5,
                linecolor="#FFFFFF",
                ax=ax,
            )
            ax.set_title("상위 8개국 연도별 수출액", fontsize=11, color="#0C3C78")
            ax.set_xlabel("연도", fontsize=10, color="#2C3E50")
            ax.set_ylabel("국가명", fontsize=10, color="#2C3E50")
            st.pyplot(fig)
        else:
            st.info("표시할 히트맵 데이터가 없습니다.")
    else:
        st.info("조건에 맞는 데이터가 없습니다.")

with col_chart2:
    st.markdown("##### 📊 무역액 등급분포")
    if not filtered_df.empty:
        tier_counts = (
            filtered_df["무역액등급"]
            .value_counts()
            .reindex(["대", "중", "소"])
            .fillna(0)
        )

        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        fig2.patch.set_alpha(0.0)
        ax2.patch.set_alpha(0.0)

        # 높은 무역액 '대'는 진한 블루(#0D47A1)
        bar_colors = ["#0D47A1", "#42A5F5", "#90CAF9"]
        bars = ax2.bar(
            tier_counts.index,
            tier_counts.values,
            color=bar_colors,
            width=0.52,
            edgecolor="#FFFFFF",
            linewidth=1.2,
        )

        ax2.set_title(
            "무역액 등급별 분포 (대: 진한 파랑)", fontsize=11, color="#0C3C78"
        )
        ax2.set_xlabel("등급", fontsize=10, color="#2C3E50")
        ax2.set_ylabel("거래건수", fontsize=10, color="#2C3E50")

        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_color("#BDD7EE")
        ax2.spines["bottom"].set_color("#BDD7EE")
        ax2.yaxis.grid(True, linestyle="--", alpha=0.4, color="#BDD7EE")
        ax2.set_axisbelow(True)

        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (max(tier_counts.values) * 0.02),
                f"{int(height):,}건",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#0D47A1",
            )
        st.pyplot(fig2)
    else:
        st.info("조건에 맞는 데이터가 없습니다.")

st.markdown("---")

# 5) 상위 5개국 * 무역액 등급 교차표
st.subheader("5. 상위 5개국 * 무역액 등급 교차표")
if not filtered_df.empty:
    top5_countries = (
        filtered_df.groupby("country_name")["trade_value_usd"]
        .sum()
        .nlargest(5)
        .index
    )
    cross_df = filtered_df[filtered_df["country_name"].isin(top5_countries)]

    col_cross1, col_cross2 = st.columns(2)

    with col_cross1:
        st.markdown("##### 📋 원본건수")
        ct_count = (
            pd.crosstab(
                cross_df["country_name"],
                cross_df["무역액등급"],
                margins=True,
                margins_name="합계",
            )
            .reindex(columns=["대", "중", "소", "합계"])
            .fillna(0)
        )
        st.dataframe(ct_count, use_container_width=True)

    with col_cross2:
        st.markdown("##### 📈 정규화비율 (행 기준 %)")
        ct_prop = (
            pd.crosstab(
                cross_df["country_name"],
                cross_df["무역액등급"],
                normalize="index",
            )
            * 100
        ).reindex(columns=["대", "중", "소"]).fillna(0).round(2)
        st.dataframe(
            ct_prop.style.format("{:.2f}%"), use_container_width=True
        )
else:
    st.info("조건에 맞는 데이터가 없습니다.")
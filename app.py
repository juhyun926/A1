import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ----------------------------------------------------
# 0. 한글 폰트 설정 (Windows / Mac 대응)
# ----------------------------------------------------
plt.rcParams["axes.unicode_minus"] = False
if os.name == "nt":  # Windows
    plt.rc("font", family="Malgun Gothic")
else:  # Mac / Linux
    plt.rc("font", family="AppleGothic")

# ----------------------------------------------------
# 1. 페이지 기본 설정 및 경로
# ----------------------------------------------------
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 파일 위치가 상위 common 폴더에 있을 경우 '../common'으로 지정
BACI_PATH = os.path.join(BASE_DIR, "..", "common", "baci_85_sample.csv")
COUNTRY_PATH = os.path.join(
    BASE_DIR, "..", "common", "country_codes_sample.csv"
)

# 현재 폴더에 파일이 있을 경우를 위한 fallback
if not os.path.exists(BACI_PATH):
    BACI_PATH = os.path.join(BASE_DIR, "baci_85_sample.csv")
if not os.path.exists(COUNTRY_PATH):
    COUNTRY_PATH = os.path.join(BASE_DIR, "country_codes_sample.csv")


# ----------------------------------------------------
# 2. 데이터 로드 및 전처리 함수
# ----------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    # 파일 로드 (인코딩 자동 시도)
    try:
        baci_df = pd.read_csv(BACI_PATH, encoding="utf-8")
        country_df = pd.read_csv(COUNTRY_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        baci_df = pd.read_csv(BACI_PATH, encoding="cp949")
        country_df = pd.read_csv(COUNTRY_PATH, encoding="cp949")

    # 결측치 원본 확인을 위해 baci_df 복사본 보관
    raw_baci_df = baci_df.copy()

    # 컬럼 공백 제거 및 소문자 변환
    baci_df.columns = [c.strip().lower() for c in baci_df.columns]
    country_df.columns = [c.strip().lower() for c in country_df.columns]

    # BACI 컬럼 매핑 (t: 연도, i: 수출국코드, j: 수입국코드, k: HS코드, v: 무역액, q: 수량)
    # 컬럼명이 다른 경우를 위한 유연한 매칭
    year_col = "t" if "t" in baci_df.columns else "year"
    exporter_col = "i" if "i" in baci_df.columns else "exporter"
    value_col = "v" if "v" in baci_df.columns else "trade_value"

    # 국가 코드 데이터 매핑
    cc_code_col = [
        c
        for c in country_df.columns
        if "code" in c or "id" in c or c in ["i", "country_code"]
    ][0]
    cc_name_col = [
        c
        for c in country_df.columns
        if "name" in c or "country" in c or "국가" in c
    ][0]

    # 국가명 병합 (수출국 기준 매칭)
    country_df[cc_code_col] = pd.to_numeric(
        country_df[cc_code_col], errors="coerce"
    )
    baci_df[exporter_col] = pd.to_numeric(
        baci_df[exporter_col], errors="coerce"
    )

    merged_df = pd.merge(
        baci_df,
        country_df[[cc_code_col, cc_name_col]],
        left_on=exporter_col,
        right_on=cc_code_col,
        how="left",
    )
    merged_df["country_name"] = merged_df[cc_name_col].fillna("기타/미분류")

    # 무역액 수치형 변환
    merged_df[value_col] = pd.to_numeric(
        merged_df[value_col].astype(str).str.replace(",", ""), errors="coerce"
    ).fillna(0)
    merged_df["trade_value_usd"] = (
        merged_df[value_col] * 1000
    )  # BACI 무역액(v)은 천 달러 단위이므로 달러 환산
    merged_df["year"] = merged_df[year_col]

    # 무역액 등급 (대/중/소) 구분 (3분위수 cut 적용)
    labels = ["소", "중", "대"]
    try:
        merged_df["무역액등급"] = pd.qcut(
            merged_df["trade_value_usd"], q=3, labels=labels, duplicates="drop"
        )
    except Exception:
        # 값이 너무 편중된 경우 0을 제외하고 처리
        merged_df["무역액등급"] = "소"

    return raw_baci_df, merged_df


# 데이터 로드 실행
raw_baci, df = load_and_preprocess_data()

# ----------------------------------------------------
# 사이드바: 필터 (국가선택, 무역액등급 선택)
# ----------------------------------------------------
st.sidebar.header("🔍 검색 및 필터 옵션")

# 국가 선택 필터
all_countries = sorted(list(df["country_name"].unique()))
selected_countries = st.sidebar.multiselect(
    "국가 선택 (복수 선택 가능)", options=all_countries, default=[]
)

# 무역액 등급 필터 (대/중/소)
tier_options = ["대", "중", "소"]
selected_tiers = st.sidebar.multiselect(
    "무역액 등급 선택 (대/중/소)", options=tier_options, default=tier_options
)

# 필터링 적용
filtered_df = df.copy()
if selected_countries:
    filtered_df = filtered_df[
        filtered_df["country_name"].isin(selected_countries)
    ]
if selected_tiers:
    filtered_df = filtered_df[filtered_df["무역액등급"].isin(selected_tiers)]


# ----------------------------------------------------
# 메인 화면 (오른쪽 화면)
# ----------------------------------------------------

# 1. 타이틀
st.title("🚢 무역 분석 대시보드")
st.markdown("---")

# 2. baci_85_sample.csv 파일의 결측치
st.subheader("2. BACI 원본 데이터 결측치 현황")
null_df = pd.DataFrame(
    {
        "컬럼명": raw_baci.columns,
        "결측치 개수": raw_baci.isnull().sum().values,
        "결측치 비율(%)": (
            raw_baci.isnull().mean() * 100
        ).round(2).values,
    }
)
st.dataframe(null_df.T, use_container_width=True)

st.markdown("---")

# 3. 총거래건수 & 총수출액(달러)
st.subheader("3. 거래 실적 요약")
total_transactions = len(filtered_df)
total_export_value = filtered_df["trade_value_usd"].sum()

col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="📦 총 거래건수", value=f"{total_transactions:,} 건"
    )
with col2:
    st.metric(
        label="💵 총 수출액 (달러)",
        value=f"${total_export_value:,.0f}",
    )

st.markdown("---")

# 4. 국가*연도 수출액 히트맵(상위 8개국) & 무역액 등급분포
st.subheader("4. 심층 무역 현황 분석")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("##### 🌐 국가 × 연도 수출액 히트맵 (수출액 상위 8개국)")
    if not filtered_df.empty:
        # 상위 8개국 선정
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

            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                pivot_heat,
                cmap="YlGnBu",
                annot=False,
                fmt=",.0f",
                cbar=True,
                ax=ax,
            )
            ax.set_title("상위 8개국 연도별 수출액", fontsize=12)
            ax.set_xlabel("연도")
            ax.set_ylabel("국가명")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("히트맵을 그릴 데이터가 없습니다.")
    else:
        st.info("선택된 조건에 맞는 데이터가 없습니다.")

with col_chart2:
    st.markdown("##### 📊 무역액 등급 분포")
    if not filtered_df.empty:
        tier_counts = (
            filtered_df["무역액등급"]
            .value_counts()
            .reindex(["대", "중", "소"])
            .fillna(0)
        )

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        bars = ax2.bar(
            tier_counts.index,
            tier_counts.values,
            color=["#4A90E2", "#50E3C2", "#F5A623"],
        )
        ax2.set_title("무역액 등급별 건수 분포 (대 / 중 / 소)", fontsize=12)
        ax2.set_xlabel("등급")
        ax2.set_ylabel("거래 건수")

        # 막대 위에 건수 표시
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height):,}건",
                ha="center",
                va="bottom",
            )

        st.pyplot(fig2)
    else:
        st.info("선택된 조건에 맞는 데이터가 없습니다.")

st.markdown("---")

# 5. 상위 5개국 * 무역액 등급 교차표 (원본건수 / 정규화비율)
st.subheader("5. 상위 5개국 × 무역액 등급 교차표")
if not filtered_df.empty:
    # 상위 5개국 선정
    top5_countries = (
        filtered_df.groupby("country_name")["trade_value_usd"]
        .sum()
        .nlargest(5)
        .index
    )
    cross_df = filtered_df[filtered_df["country_name"].isin(top5_countries)]

    col_cross1, col_cross2 = st.columns(2)

    # 5-1. 원본 건수 교차표
    with col_cross1:
        st.markdown("##### 📋 원본 거래 건수 교차표")
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

    # 5-2. 정규화 비율 교차표 (행 기준 100% 정규화)
    with col_cross2:
        st.markdown("##### 📈 정규화 비율 교차표 (국가별 비중 %)")
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
    st.info("선택된 조건에 맞는 데이터가 없습니다.")
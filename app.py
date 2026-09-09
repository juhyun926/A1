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
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BACI_PATH = os.path.join(BASE_DIR, "baci_85_sample.csv")
COUNTRY_PATH = os.path.join(BASE_DIR, "country_codes_sample.csv")

# ----------------------------------------------------
# 2. 한글 폰트 설정 (fonts 폴더 내 폰트 우선 적용)
# ----------------------------------------------------
plt.rcParams["axes.unicode_minus"] = False
font_path = os.path.join(BASE_DIR, "fonts", "온글잎 콘콘체.ttf")

if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc("font", family=font_prop.get_name())
else:
    alt_font = os.path.join(BASE_DIR, "..", "fonts", "온글잎 콘콘체.ttf")
    if os.path.exists(alt_font):
        font_prop = fm.FontProperties(fname=alt_font)
        plt.rc("font", family=font_prop.get_name())
    elif os.name == "nt":
        plt.rc("font", family="Malgun Gothic")
    else:
        plt.rc("font", family="AppleGothic")


# ----------------------------------------------------
# 3. 데이터 로드 및 전처리 함수
# ----------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    # 파일 로드 (UTF-8 및 CP949 인코딩 대응)
    try:
        baci_df = pd.read_csv(BACI_PATH, encoding="utf-8")
        country_df = pd.read_csv(COUNTRY_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        baci_df = pd.read_csv(BACI_PATH, encoding="cp949")
        country_df = pd.read_csv(COUNTRY_PATH, encoding="cp949")

    raw_baci_df = baci_df.copy()

    # 컬럼 공백 제거 및 소문자 통일
    baci_df.columns = [str(c).strip().lower() for c in baci_df.columns]
    country_df.columns = [str(c).strip().lower() for c in country_df.columns]

    # [핵심 수정] 상대국 코드 컬럼 매칭 ('j' 컬럼 기준 병합)
    # country_codes_sample.csv의 'j'와 baci_85_sample.csv의 'j'를 연결
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

    # 국가명이 없는 경우 국가 코드로 표기
    merged_df["country_name"] = merged_df[name_col].fillna(
        "국가코드_" + merged_df["j"].astype(str)
    )

    # 연도 및 무역액 컬럼 처리 (v: 천 달러 -> 달러 환산)
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

    # 무역액 등급 (대/중/소) 구분 (3분위수 기반)
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
# 4. 사이드바 필터 (국가선택, 무역액등급 대/중/소)
# ----------------------------------------------------
st.sidebar.header("🔍 검색 및 필터 옵션")

# 실제 데이터 안에 있는 국가명 리스트 추출
all_countries = sorted([str(x) for x in df["country_name"].unique()])
selected_countries = st.sidebar.multiselect(
    "국가 선택 (복수 선택 가능, 미선택 시 전체)",
    options=all_countries,
    default=[],
)

# 무역액 등급 선택 (대/중/소)
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
# 5. 메인 화면 출력
# ----------------------------------------------------

# 1) 타이틀
st.title("🚢 무역 분석 대시보드")
st.markdown("---")

# 2) baci_85_sample.csv 파일의 결측치
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
st.dataframe(null_df, use_container_width=True)
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
        label="💵 총수출액(달러)",
        value=f"${total_export_value:,.0f}",
    )
st.markdown("---")

# 4) 국가*연도 수출액 히트맵(상위8 개국) & 무역액 등급분포
st.subheader("4. 국가별 및 등급별 무역 패턴 분석")
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("##### 🌐 국가*연도 수출액 히트맵(상위8 개국)")
    if not filtered_df.empty:
        # 상위 8개국 추출
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
            sns.heatmap(
                pivot_heat,
                cmap="YlGnBu",
                annot=False,
                fmt=",.0f",
                cbar=True,
                ax=ax,
            )
            ax.set_title("상위 8개국 연도별 수출액", fontsize=11)
            ax.set_xlabel("연도")
            ax.set_ylabel("국가명")
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
        bars = ax2.bar(
            tier_counts.index,
            tier_counts.values,
            color=["#3498db", "#2ecc71", "#e67e22"],
        )
        ax2.set_title("무역액 등급별 분포 (대 / 중 / 소)", fontsize=11)
        ax2.set_xlabel("등급")
        ax2.set_ylabel("거래건수")

        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height):,}",
                ha="center",
                va="bottom",
            )
        st.pyplot(fig2)
    else:
        st.info("조건에 맞는 데이터가 없습니다.")
st.markdown("---")

# 5) 상위 5개국 * 무역액 등급 교차표 (원본건수 / 정규화비율)
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
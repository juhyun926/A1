import streamlit as st
import random
import time

# 1. 페이지 설정
st.set_page_config(page_title="오늘 뭐 먹지? 저녁 메뉴 추천기", page_icon="🍽️", layout="centered")

# 2. 카테고리별 저녁 메뉴 데이터
MENU_DATA = {
    "한식 🍚": [
        "김치찌개", "된장찌개", "삼겹살", "제육볶음", "소불고기", 
        "비빔밥", "닭볶음탕", "국밥", "보쌈/족발", "감자탕"
    ],
    "중식 🥟": [
        "짜장면", "짬뽕", "탕수육", "마라탕", "마라샹궈", 
        "볶음밥", "꿔바로우", "유린기", "마파두부"
    ],
    "일식 🍣": [
        "초밥", "돈카츠", "라멘", "규동/가츠동", "우동", 
        "소바", "야키토리", "오코노미야키", "사케동"
    ],
    "양식 🍕": [
        "파스타", "피자", "스테이크", "수제버거", "리조또", 
        "감바스", "바비큐 립", "라자냐"
    ],
    "분식/간편식 🍢": [
        "떡볶이 & 튀김", "김밥 & 라면", "순대국", "토스트/샌드위치", "샐러드"
    ],
    "야식/치맥 🍗": [
        "후라이드/양념치킨", "피자 & 맥주", "닭발", "골뱅이 소면", "곱창/막창"
    ]
}

# 3. UI 구성
st.title("🍽️ 오늘 뭐 먹지? 저녁 메뉴 추천기")
st.write("결정하기 힘든 저녁 메뉴, 버튼 하나로 정해 드립니다!")

# 카테고리 선택 (전체 선택 옵션 포함)
categories = ["전체 (모든 카테고리)"] + list(MENU_DATA.keys())
selected_category = st.selectbox("어떤 종류의 음식이 당기시나요?", categories)

# 추천 대상 메뉴 리스트 구성
if selected_category == "전체 (모든 카테고리)":
    candidate_menus = [menu for sublist in MENU_DATA.values() for menu in sublist]
else:
    candidate_menus = MENU_DATA[selected_category]

st.write(f"현재 후보군: **{len(candidate_menus)}개**의 메뉴")

# 4. 추천 동작
if st.button("🎲 메뉴 추천받기", type="primary", use_container_width=True):
    with st.spinner("맛있는 메뉴를 고르는 중..."):
        time.sleep(0.5)  # 짧은 애니메이션 효과
        selected_menu = random.choice(candidate_menus)
    
    st.balloons()
    st.success(f"오늘 저녁은 바로 **'{selected_menu}'** 어떠세요?")

# 5. 직접 메뉴 추가 기능 (선택 사항)
with st.expander("➕ 나만의 메뉴 추가하기"):
    custom_menu = st.text_input("후보에 넣고 싶은 메뉴 이름을 입력하세요:")
    if st.button("임시 후보로 추가"):
        if custom_menu.strip():
            candidate_menus.append(custom_menu.strip())
            st.info(f"'{custom_menu}' 메뉴가 추천 목록에 추가되었습니다!")
        else:
            st.warning("메뉴 이름을 입력해 주세요.")
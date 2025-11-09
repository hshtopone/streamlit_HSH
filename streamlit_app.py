import streamlit as st
import pandas as pd

# 기본 설정
st.set_page_config(page_title="자기소개 페이지", page_icon="👨‍🎓", layout="centered")

# 스타일
CUSTOM_CSS = """
<style>
html, body, [class*="css"]{font-family:Pretendard,-apple-system,Segoe UI,Roboto,Noto Sans KR,Apple SD Gothic Neo,sans-serif;}
.block-container{padding-top:2.2rem;padding-bottom:3.2rem;}

.header{
  background:linear-gradient(135deg,#3b82f6 0%,#06b6d4 40%,#22c55e 100%);
  color:#fff;padding:28px 22px;border-radius:18px;text-align:center;
  box-shadow:0 10px 30px rgba(0,0,0,.08);
}
.header h1{margin:0;font-size:2rem;letter-spacing:.2px;}

.section{margin:28px 0 32px;}
.h2{font-size:2rem;font-weight:800;margin:0 0 .5rem 0;}

a.btn{
  display:inline-block; padding:12px 16px; border-radius:12px; text-decoration:none;
  font-weight:700; letter-spacing:.2px; color:#fff;
  background:linear-gradient(135deg,#2563eb 0%,#7c3aed 100%);
  box-shadow:0 10px 24px rgba(37,99,235,.25);
  transition:transform .12s ease, box-shadow .12s ease, filter .12s ease;
  border:none;
}
a.btn:hover{
  transform:translateY(-1px);
  box-shadow:0 14px 30px rgba(37,99,235,.32);
  filter:saturate(1.08);
}

.tags{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;}
.tag{
  padding:8px 12px;border-radius:12px;background:transparent;
  border:1.5px solid #99f6e4;
  font-size:.94rem;
}

table,.stTable{font-size:.95rem;}
.card{padding:18px;border:1px solid rgba(0,0,0,.08);border-radius:14px;background:#ffffff10;}
.helper{color:#64748b;font-size:.92rem;}

/* st.table 열 너비 동일화 (인덱스열 포함 총 6열 균등) */
[data-testid="stTable"] table{table-layout:fixed; width:100%;}
[data-testid="stTable"] table th,
[data-testid="stTable"] table td{width:calc(100% / 6);}
div[data-baseweb="tab-border"]{border:none !important;}
hr{border:none !important; height:0 !important;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 헤더
st.markdown('<div class="header"><h1>자기소개 페이지</h1></div>', unsafe_allow_html=True)
st.write("")

# 탭
tab1, tab2 = st.tabs(["🧾 기본 정보", "📅 시간표"])

# 기본 정보
with tab1:
    st.markdown('<div class="section"><div class="h2">학력</div>', unsafe_allow_html=True)
    st.write("2022년 12월 대곡고등학교 졸업")
    st.write("2023년 3월 서울대학교 공과대학 항공우주공학과 입학")
    st.markdown(
        '<a class="btn" href="https://aerospace.snu.ac.kr/" target="_blank">학과 홈페이지 방문하기</a>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="h2">관심 분야</div>', unsafe_allow_html=True)
    tags = ["항공우주공학", "산업공학", "경제학", "주식 투자", "인류학"]
    tag_html = '<div class="tags">' + "".join([f'<span class="tag">{t}</span>' for t in tags]) + "</div>"
    st.markdown(tag_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="h2">Career</div>', unsafe_allow_html=True)
    st.write("수능 과학탐구 영역 사설 콘텐츠팀 POLARIS 소속, 대외 출판 도서 기획 및 총괄 담당")
    st.markdown(
        '<a class="btn" href="https://www.teampolaris.co.kr/" target="_blank">Team POLARIS 홈페이지</a>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="h2">집필 도서 목록</div>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            '<a class="btn" href="https://product.kyobobook.co.kr/detail/S000217112740" target="_blank">폴라리스 모의고사 시즌1</a>',
            unsafe_allow_html=True,
        )
    with colB:
        st.markdown(
            '<a class="btn" href="https://product.kyobobook.co.kr/detail/S000217602755" target="_blank">폴라리스 모의고사 시즌2</a>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# 시간표
with tab2:
    st.markdown('<div class="h2">2025년 2학기 시간표</div>', unsafe_allow_html=True)
    hours = ["1교시","2교시","3교시","4교시","5교시"]
    data = {
        "월": ["통계학","경제성공학","공학수학 2","","수학 2"],
        "화": ["","","","",""],
        "수": ["통계학","경제성공학","공학수학 2","","수학 2"],
        "목": ["미시경제이론","미시경제이론","통계학실험","",""],
        "금": ["컴퓨팅탐색","컴퓨팅탐색","수학연습 2","",""],
    }
    df = pd.DataFrame(data, index=hours)
    st.table(df)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section"><div class="h2">이번 학기 요약</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("수강 과목 수", "8개")
    with m2:
        st.metric("수강 학점", "19학점")
    with m3:
        st.metric("졸업까지 남은 학점", "66학점")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.caption("© 2025 — Streamlit_HSH")

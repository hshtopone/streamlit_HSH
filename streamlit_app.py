import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

try:
    import altair as alt
except Exception:
    alt = None

LAT, LON, TZ = 37.5665, 126.9780, "Asia/Seoul"
REQ = ["날짜", "대여건수", "평균 기온", "강수량", "PM2.5 농도", "평일 여부"]
TEST_SIZE = 0.1
ENH = dict(add_rain_dummy=True, add_season=True, add_trend=False)
DOW = "월화수목금토일"


def _norm_flag(x):
    s = "" if pd.isna(x) else str(x).strip()
    return s if s in ("O", "X") else ""


def load_excel(file):
    df = pd.read_excel(file)
    if not all(c in df.columns for c in REQ):
        df = df.iloc[:, :6].copy()
        df.columns = REQ
    df = df[REQ].copy()
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평일 여부"] = df["평일 여부"].apply(_norm_flag)
    for c in ["대여건수", "평균 기온", "강수량", "PM2.5 농도"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=REQ)
    df = df[df["평일 여부"].isin(["O", "X"])].sort_values("날짜").reset_index(drop=True)
    return df


def split_time(df, test_size=TEST_SIZE, random_state=42):
    tr, te = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )
    return tr.reset_index(drop=True).copy(), te.reset_index(drop=True).copy()


def build_X(df, mean_T, add_rain_dummy, add_season, add_trend):
    T = df["평균 기온"].to_numpy(float)
    R = df["강수량"].to_numpy(float)
    PM = df["PM2.5 농도"].to_numpy(float)
    Tc = T - mean_T
    X = [Tc, Tc**2, np.log1p(R), PM]
    if add_rain_dummy:
        X.append((R > 0).astype(int))
    if add_season:
        doy = df["날짜"].dt.dayofyear.to_numpy(float)
        X += [np.sin(2 * np.pi * doy / 365.0), np.cos(2 * np.pi * doy / 365.0)]
    if add_trend:
        t = (df["날짜"] - df["날짜"].min()).dt.days.to_numpy(float)
        X.append(t)
    return np.column_stack(X)


def fit_group(df_group):
    if len(df_group) < 10:
        return None
    tr, te = split_time(df_group, TEST_SIZE)
    mean_T = float(tr["평균 기온"].mean())
    Xtr = build_X(tr, mean_T, **ENH)
    Xte = build_X(te, mean_T, **ENH)
    ytr = np.log1p(tr["대여건수"].to_numpy(float))
    yte = np.log1p(te["대여건수"].to_numpy(float))
    m = LinearRegression().fit(Xtr, ytr)
    return dict(
        model=m,
        mean_T=mean_T,
        r2_tr=r2_score(ytr, m.predict(Xtr)),
        r2_te=r2_score(yte, m.predict(Xte)),
    )


def _get_json_via_bs(url):
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    return json.loads(BeautifulSoup(r.text, "html.parser").get_text())


@st.cache_data(ttl=60 * 30)
def fetch_seoul_open_meteo(start_d: date, end_d: date):
    s, e = start_d.isoformat(), end_d.isoformat()
    tz = TZ.replace("/", "%2F")
    url_w = (
        f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}"
        f"&daily=temperature_2m_mean,precipitation_sum&timezone={tz}"
        f"&start_date={s}&end_date={e}"
    )
    url_a = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={LAT}&longitude={LON}"
        f"&hourly=pm2_5&timezone={tz}"
        f"&start_date={s}&end_date={e}"
    )
    w = _get_json_via_bs(url_w)["daily"]
    df_w = pd.DataFrame(
        {"날짜": pd.to_datetime(w["time"]), "평균 기온": w["temperature_2m_mean"], "강수량": w["precipitation_sum"]}
    )
    a = _get_json_via_bs(url_a)["hourly"]
    df_a = pd.DataFrame({"time": pd.to_datetime(a["time"]), "PM2.5 농도": a["pm2_5"]})
    df_a["날짜"] = df_a["time"].dt.normalize()
    df_pm = df_a.groupby("날짜", as_index=False)["PM2.5 농도"].mean()
    return df_w.merge(df_pm, on="날짜", how="left").sort_values("날짜").reset_index(drop=True)


def _kr_holiday(d: date):
    try:
        import holidays
        return d in holidays.KR()
    except Exception:
        return d.weekday() >= 5


def predict_daily(models, meteo_df: pd.DataFrame, pm_fallback: float):
    rows = []
    for _, r in meteo_df.iterrows():
        d = pd.to_datetime(r["날짜"]).normalize()
        flag = "X" if _kr_holiday(d.date()) else "O"
        pack = models.get(flag)
        pm = float(r["PM2.5 농도"]) if pd.notna(r["PM2.5 농도"]) else float(pm_fallback)
        tmp = pd.DataFrame({"날짜": [d], "평균 기온": [float(r["평균 기온"])], "강수량": [float(r["강수량"])], "PM2.5 농도": [pm]})
        yhat = np.nan
        if pack is not None:
            X = build_X(tmp, pack["mean_T"], **ENH)
            yhat = float(np.expm1(pack["model"].predict(X)[0]))
        rows.append({"date": d, "temp": float(r["평균 기온"]), "rain": float(r["강수량"]), "pm25": pm, "pred": yhat})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["delta"] = out["pred"].diff()
    prev = out["pred"].shift(1)
    out["delta_pct"] = np.where(prev > 0, (out["pred"] / prev - 1.0) * 100.0, np.nan)
    return out


def fmt_int(x):
    return "-" if pd.isna(x) else f"{int(round(x)):,}"


def fmt_delta(x):
    return "" if pd.isna(x) else f"{int(round(x)):+,}"


def fmt_pct(x):
    return "" if pd.isna(x) else f"({x:+.1f}%)"


def fmt_rain_mm(x):
    if pd.isna(x):
        return "-"
    x = float(x)
    if abs(x) < 1e-12:
        return "0mm"
    return f"{int(round(x))}mm" if abs(x - round(x)) < 1e-9 else f"{x:.1f}mm"


def weather_emoji(temp, rain):
    return "🌧️" if rain > 0 else ("☀️" if temp >= 5 else "🥶")


def delta_badge_html(d, pct):
    if pd.isna(d):
        return '<span class="dneu">&nbsp;</span>'
    cls = "dpos" if d > 0 else ("dneg" if d < 0 else "dneu")
    return f'<span class="{cls}">{fmt_delta(d)} {fmt_pct(pct)}</span>'


def pick_one(label, options, default):
    fn = getattr(st, "pills", None)
    if fn is not None:
        return fn(label, options, default=default, selection_mode="single")
    return st.radio(label, options, index=options.index(default), horizontal=True)


st.set_page_config(page_title="서울시 공공자전거 대여건수 예측", page_icon="🚲", layout="wide")

st.markdown(
    """
<style>
.block-container{padding-left:1.2rem; padding-right:1.2rem;}
.card{
  border:1px solid rgba(255,255,255,.12);
  border-radius:16px;
  padding:16px 18px 14px 18px;
  background:rgba(255,255,255,.03);
  box-shadow:0 6px 18px rgba(0,0,0,.18);
}
.card.today{
  border-color:rgba(255,215,0,.70);
  background:rgba(255,215,0,.14);
}
.card h4{
  margin:0 0 12px 0;
  font-size:1.22rem;
  font-weight:900;
  line-height:1.15;
}
.smallcap{opacity:.78;font-size:.88rem; margin-top:2px;}
.bigrow{
  display:flex;
  align-items:baseline;
  justify-content:flex-start;
  gap:4px;
  margin-top:10px;
}
.big{font-size:1.60rem;font-weight:900;line-height:1.1;margin:0;}
.meta{
  opacity:.90;
  font-size:.94rem;
  margin-top:12px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.dpos,.dneg,.dneu{
  display:inline-block;
  padding:4px 8px;
  border-radius:999px;
  font-size:.92rem;
  font-weight:700;
  white-space:nowrap;
  line-height:1.1;
}
.dpos{background:rgba(34,197,94,.18); color:rgba(34,197,94,1);}
.dneg{background:rgba(239,68,68,.18); color:rgba(239,68,68,1);}
.dneu{background:rgba(148,163,184,.18); color:rgba(148,163,184,1);}
div[data-testid="stCaptionContainer"]{text-align:right;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚲 서울시 공공자전거 대여건수 예측")

file = st.file_uploader("📎 엑셀 파일 업로드", type=["xlsx"])
if not file:
    st.stop()

df = load_excel(file)
models = {"O": fit_group(df[df["평일 여부"] == "O"]), "X": fit_group(df[df["평일 여부"] == "X"])}
pm_fallback = float(df["PM2.5 농도"].mean()) if len(df) else 0.0

tab1, tab2, tab3 = st.tabs(["대여건수 예측", "데이터 시각화", "분석 방법 설명"])

with tab1:
    today = date.today()
    start, end = today - timedelta(days=1), today + timedelta(days=4)
    try:
        meteo = fetch_seoul_open_meteo(start, end)
        pred = predict_daily(models, meteo, pm_fallback)
        show = pred[(pred["date"].dt.date >= today) & (pred["date"].dt.date <= today + timedelta(days=4))].copy()
        show = show.sort_values("date").reset_index(drop=True)

        cols = st.columns(5, gap="small")
        for i, (_, r) in enumerate(show.iterrows()):
            d = r["date"].date()
            dow = DOW[d.weekday()]
            emo = weather_emoji(r["temp"], r["rain"])
            cls = "card today" if (d == today) else "card"
            meta = f"🌡️ {r['temp']:.1f}°C  ·  ☔ {r['rain']:.1f}mm  ·  😷 {r['pm25']:.1f}µg/m³"
            with cols[i]:
                st.markdown(
                    f"""
<div class="{cls}">
  <h4>{emo} {d.isoformat()} ({dow})</h4>
  <div class="smallcap">예측 대여건수</div>
  <div class="bigrow">
    <div class="big">{fmt_int(r["pred"])}</div>
    {delta_badge_html(r["delta"], r["delta_pct"])}
  </div>
  <div class="meta">{meta}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        st.divider()

        if len(show) > 0:
            g = show.copy()
            g["day_label"] = g["date"].dt.day.astype(int).astype(str) + "일"
            for c in ["pred", "temp", "rain", "pm25"]:
                g[c] = pd.to_numeric(g[c], errors="coerce")

            r1c1, r1c2 = st.columns(2, gap="medium")
            r2c1, r2c2 = st.columns(2, gap="medium")

            if alt is not None:
                xenc = alt.X("day_label:N", axis=alt.Axis(labelAngle=0, title=None), sort=g["day_label"].tolist())

                def mk_line(y, title):
                    return (
                        alt.Chart(g)
                        .mark_line(point=True)
                        .encode(
                            x=xenc,
                            y=alt.Y(f"{y}:Q", title=title),
                            tooltip=[alt.Tooltip("date:T", title="날짜"), alt.Tooltip(f"{y}:Q", title=title)],
                        )
                        .properties(height=220)
                    )

                def mk_bar(y, title):
                    return (
                        alt.Chart(g)
                        .mark_bar()
                        .encode(
                            x=xenc,
                            y=alt.Y(f"{y}:Q", title=title),
                            tooltip=[alt.Tooltip("date:T", title="날짜"), alt.Tooltip(f"{y}:Q", title=title)],
                        )
                        .properties(height=220)
                    )

                with r1c1:
                    st.altair_chart(mk_line("pred", "예측 대여건수"), use_container_width=True)
                with r1c2:
                    st.altair_chart(mk_line("temp", "평균 기온(°C)"), use_container_width=True)
                with r2c1:
                    st.altair_chart(mk_bar("rain", "강수량(mm)"), use_container_width=True)
                with r2c2:
                    st.altair_chart(mk_line("pm25", "PM2.5(µg/m³)"), use_container_width=True)
            else:
                idx = g["day_label"]
                with r1c1:
                    st.line_chart(pd.Series(g["pred"].to_numpy(), index=idx, name="예측 대여건수"))
                with r1c2:
                    st.line_chart(pd.Series(g["temp"].to_numpy(), index=idx, name="평균 기온(°C)"))
                with r2c1:
                    st.bar_chart(pd.Series(g["rain"].to_numpy(), index=idx, name="강수량(mm)"))
                with r2c2:
                    st.line_chart(pd.Series(g["pm25"].to_numpy(), index=idx, name="PM2.5(µg/m³)"))

    except Exception as e:
        st.error(f"Open-Meteo 불러오기 실패: {e}")

with tab2:
    st.info("사용자가 업로드한 데이터를 시각화하여 보여줍니다.")
    dmin, dmax = df["날짜"].min().date(), df["날짜"].max().date()
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        s = st.date_input("시작 날짜", value=max(dmin, dmax - timedelta(days=30)), min_value=dmin, max_value=dmax)
    with c2:
        e = st.date_input("종료 날짜", value=dmax, min_value=dmin, max_value=dmax)
    if s > e:
        s, e = e, s

    sub = df[(df["날짜"].dt.date >= s) & (df["날짜"].dt.date <= e)].copy()
    sub2 = sub.rename(columns={"PM2.5 농도": "PM2_5 농도"}).copy()
    sub2["PM2_5 농도"] = pd.to_numeric(sub2["PM2_5 농도"], errors="coerce")

    choice = pick_one("변수 선택", ["🚲 대여건수", "🌡️ 평균 기온", "☔ 강수량", "😷 초미세먼지"], "🚲 대여건수")

    if choice == "🚲 대여건수":
        st.line_chart(sub2.set_index("날짜")["대여건수"])
    elif choice == "🌡️ 평균 기온":
        st.line_chart(sub2.set_index("날짜")["평균 기온"])
    elif choice == "☔ 강수량":
        st.bar_chart(sub2.set_index("날짜")["강수량"])
    else:
        st.line_chart(sub2.set_index("날짜")["PM2_5 농도"])

with tab3:
    o, x = models.get("O"), models.get("X")

    st.markdown("## 📚 데이터 출처")
    st.markdown(
        "예시로 주어진 data.xlsx 파일의 각 데이터에 대한 출처는 다음과 같습니다.\n\n"
        "• 🚲 **대여건수**: [서울 열린데이터 광장 ‘서울시 공공자전거 이용현황’](https://data.seoul.go.kr/dataList/OA-14994/F/1/datasetView.do)\n"
        "• 🌡️ **평균 기온**, ☔ **강수량**: [기상자료개발포털 기후통계분석](https://data.kma.go.kr/stcs/grnd/grndRnList.do)\n"
        "• 😷 **PM2.5(초미세먼지) 농도**: [서울특별시 대기환경정보 일별평균자료](https://cleanair.seoul.go.kr/statistics/dayAverage)\n"
        "• 📊 향후 5일 동안의 공공자전거 대여건수 예측에 활용하는 날씨 데이터는 BeautifulSoup을 통해 open-meteo.com로부터 실시간으로 가져옵니다."
    )
    st.divider()

    st.markdown("## 🛠️ 분석 방법")
    st.write(
        "회귀분석은 **반응변수(공공자전거 대여건수)** 와 **설명변수(평균 기온, 강수량, 초미세먼지 농도, 계절 요인 등)** 의 "
        "연관성을 수리적 모형으로 분석하는 방법입니다.\n"
        "이미 주어진 데이터를 학습하여 회귀모형을 도출한 뒤, 새로운 데이터를 모형에 대입하여 예측치를 얻을 수 있습니다.\n\n"
        "본 연구에서는 다음의 회귀분석모형을 활용하였습니다."
    )

    st.latex(
        r"\log(1+y)=\beta_0"
        r"+\beta_1(T-\bar T)+\beta_2(T-\bar T)^2"
        r"+\beta_3\log(1+R)"
        r"+\beta_4 I(R>0)"
        r"+\beta_5 PM"
        r"+\beta_6\sin\!\left(\frac{2\pi\cdot doy}{365}\right)"
        r"+\beta_7\cos\!\left(\frac{2\pi\cdot doy}{365}\right)"
        r"+\varepsilon"
    )

    st.caption("🔎 [참고]")
    st.caption("• 반응변수 y를 그대로 쓰지 않고 `log(1+y)`로 바꾼 이유는, 일별 값의 변동 폭이 커서 일부 큰 값이 모형에 과도한 영향을 줄 수 있기 때문입니다.")
    st.caption("• 평일과 주말/공휴일은 활동 패턴이 달라 이용량 구조도 달라지므로, 데이터를 분리해 **평일(O) 모형 / 주말·공휴일(X) 모형**을 각각 학습하여 신뢰도를 높였습니다.")
    st.caption("• 평균 기온에 이차항이 포함된 이유는, 평균 기온과 대여건수의 관계가 선형이 아니라 ‘적당한 기온에서 증가’하는 비선형 패턴이 나타날 수 있기 때문입니다.")
    st.caption("• 강수량은 0인 날이 많아 **비가 왔는지 여부(I(R>0))** 를 따로 반영하고, 증가 효과는 `log(1+R)`로 완화했습니다.")
    st.divider()

    st.markdown("## 📏 결정계수(R²)")
    st.write(
        "결정계수는 회귀모형이 실제 관측치 변동을 얼마나 잘 설명하는지 나타내는 값입니다.\n"
        "예를 들어 R²=0.8이면, 모형이 변동의 약 80%를 설명할 수 있다는 의미입니다.\n\n"
        f"사용자 데이터는 **무작위로 90%는 학습(Train), 10%는 테스트(Test)** 에 활용하였으며, "
        "각각의 결정계수는 다음과 같습니다."
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("🟦 평일 Train R²", "-" if o is None else f"{o['r2_tr']:.4f}")
    c2.metric("🟦 평일 Test R²",  "-" if o is None else f"{o['r2_te']:.4f}")
    c3.metric("🟧 주말/공휴일 Train R²", "-" if x is None else f"{x['r2_tr']:.4f}")
    c4.metric("🟧 주말/공휴일 Test R²",  "-" if x is None else f"{x['r2_te']:.4f}")
    st.divider()

    st.markdown("## 🧩 핵심 코드")
    st.code(
        "TEST_SIZE = 0.1\n"
        "ENH = dict(add_rain_dummy=True, add_season=True, add_trend=False)\n\n"
        "tr, te = split_time(df_group, test_size=TEST_SIZE)\n\n"
        "mean_T = float(tr['평균 기온'].mean())\n"
        "Xtr = build_X(tr, mean_T, **ENH)\n"
        "ytr = np.log1p(tr['대여건수'].values)\n"
        "model = LinearRegression().fit(Xtr, ytr)\n",
        language="python",
    )

st.caption("서울대학교 황시현")

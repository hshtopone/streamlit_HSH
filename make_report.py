# -*- coding: utf-8 -*-
"""
정적 VI 변동폭 평가 - 결과 해석 보고서(PDF) 자동 생성
- 입력: analyze_vi.py가 생성한 analysis/*.csv
- 출력: VI_분석보고서.pdf  (+ analysis/figs/*.png)
- 의존: matplotlib, reportlab, 맑은 고딕(C:\\Windows\\Fonts\\malgun.ttf)

실행:  python analyze_vi.py  →  python make_report.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── 경로 ──
BASE   = Path(__file__).resolve().parent
ANA    = BASE / "analysis"
FIGS   = ANA / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
PDF_OUT = BASE / "VI_분석보고서.pdf"

MALGUN   = r"C:\Windows\Fonts\malgun.ttf"
MALGUNBD = r"C:\Windows\Fonts\malgunbd.ttf"

# ── 폰트 등록 (matplotlib & reportlab) ──
_fp = font_manager.FontProperties(fname=MALGUN)
plt.rcParams["font.family"] = _fp.get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
pdfmetrics.registerFont(TTFont("Malgun", MALGUN))
pdfmetrics.registerFont(TTFont("MalgunBd", MALGUNBD))

# ── Tchebycheff (보고서 내 재계산용) ──
POLICY = {"균형": (1/3,1/3,1/3), "안정성우선": (0.5,0.25,0.25),
          "가격발견우선": (0.25,0.5,0.25), "거래마찰우선": (0.25,0.25,0.5)}
RHO = 0.001
def G(zs, zd, zc, w):
    a = np.array([w[0]*zs, w[1]*zd, w[2]*zc]); return a.max() + RHO*a.sum()

BASELINE = 13
ACCENT = "#c0392b"; BASECLR = "#95a5a6"; BLUE = "#2c3e50"


# ─────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────
summ  = pd.read_csv(ANA/"summary_by_scenario_viwidth.csv")
best  = pd.read_csv(ANA/"best_viwidth_per_scenario.csv")
robw  = pd.read_csv(ANA/"weight_robustness_all.csv")

base = summ[summ.Scn == BASELINE].sort_values("VIWidth").copy()
base["G"] = [G(r.ZS, r.ZD, r.ZC, POLICY["균형"]) for r in base.itertuples()]
base = base.sort_values("VIWidth")
base_best = int(base.sort_values("G").iloc[0]["VIWidth"])
robw_base = robw[robw.Scn == BASELINE].sort_values("VIWidth")
base_win  = float(robw_base.set_index("VIWidth").loc[base_best, "WinRate"])


# ─────────────────────────────────────────────────────────
# 그래프 생성
# ─────────────────────────────────────────────────────────
def fig1_composite():
    f, ax = plt.subplots(figsize=(7,3.4))
    clrs = [ACCENT if w==base_best else BASECLR for w in base.VIWidth]
    ax.bar(base.VIWidth, base.G, color=clrs)
    ax.set_xlabel("VI 변동폭 (%)"); ax.set_ylabel("종합점수 G (낮을수록 우수)")
    ax.set_title("베이스 시나리오 Augmented Tchebycheff 종합점수 (균형가중)")
    ax.set_xticks(base.VIWidth)
    for r in base.itertuples():
        ax.text(r.VIWidth, r.G+0.0005, f"{r.G:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(base_best, base[base.VIWidth==base_best].G.iloc[0]/2,
            f"최적\n{base_best}%", ha="center", va="center", color="white", fontweight="bold")
    f.tight_layout(); p = FIGS/"fig1_composite.png"; f.savefig(p); plt.close(f); return p

def fig2_znorm():
    f, ax = plt.subplots(figsize=(7,3.4))
    x = np.arange(len(base)); wd=0.26
    ax.bar(x-wd, base.ZS, wd, label="ZS 가격불안정성")
    ax.bar(x,    base.ZD, wd, label="ZD 가격발견지연")
    ax.bar(x+wd, base.ZC, wd, label="ZC 거래마찰")
    ax.set_xticks(x); ax.set_xticklabels(base.VIWidth)
    ax.set_xlabel("VI 변동폭 (%)"); ax.set_ylabel("정규화 손실 (0=우수,1=열악)")
    ax.set_title("베이스 시나리오 정규화 지표 (VI 변동폭별)")
    ax.legend(fontsize=8); f.tight_layout()
    p = FIGS/"fig2_znorm.png"; f.savefig(p); plt.close(f); return p

def fig3_cliff():
    d = summ[summ.FundPrice==125].groupby("BandPct").FailRate.mean()*100
    f, ax = plt.subplots(figsize=(6,3.2))
    clrs = [ACCENT if b==3 else BLUE for b in d.index]
    ax.bar(d.index, d.values, color=clrs)
    for b,v in d.items(): ax.text(b, v+1.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xlabel("발견 밴드 폭 (±%)"); ax.set_ylabel("가격발견 실패율 (%)")
    ax.set_title("발견 밴드 절벽 — 밴드를 ±3%로 좁히면 발견 실패 급증 (적정가 125)")
    ax.set_ylim(0, 100); f.tight_layout()
    p = FIGS/"fig3_cliff.png"; f.savefig(p); plt.close(f); return p

def fig4_robust():
    f, ax = plt.subplots(figsize=(7,3.2))
    d = robw_base.sort_values("VIWidth")
    clrs = [ACCENT if w==base_best else BASECLR for w in d.VIWidth]
    ax.bar(d.VIWidth, d.WinRate*100, color=clrs)
    for r in d.itertuples():
        if r.WinRate>0.01: ax.text(r.VIWidth, r.WinRate*100+1, f"{r.WinRate*100:.0f}%",
                                   ha="center", fontsize=8)
    ax.set_xticks(d.VIWidth)
    ax.set_xlabel("VI 변동폭 (%)"); ax.set_ylabel("가중치공간 1위 선택비율 (%)")
    ax.set_title("베이스 시나리오 가중치 강건성 (전체 가중치 조합 중 1위 비율)")
    f.tight_layout(); p = FIGS/"fig4_robust.png"; f.savefig(p); plt.close(f); return p

def _heat(ax, piv, title, xlabel, ylabel):
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, int(piv.values[i,j]), ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)
    return im

def fig5_heat_band_dwell():
    d = best[best.FundPrice==125]
    piv = d.pivot(index="BandPct", columns="DwellTime", values="EqualWeightBest")
    f, ax = plt.subplots(figsize=(6,3.6))
    im = _heat(ax, piv, "최적 VI 변동폭 — 종료조건 민감도 (적정가 125)",
               "머무는 시간 (초)", "발견 밴드 폭 (±%)")
    f.colorbar(im, ax=ax, label="최적 VI 변동폭(%)"); f.tight_layout()
    p = FIGS/"fig5_heat_band_dwell.png"; f.savefig(p); plt.close(f); return p

def fig6_fund():
    d = best[best.DwellTime==120]
    piv = d.pivot(index="BandPct", columns="FundPrice", values="EqualWeightBest")
    f, ax = plt.subplots(figsize=(5.5,3.6))
    im = _heat(ax, piv, "최적 VI 변동폭 — 적정가 민감도 (dwell 120)",
               "적정가 (원)", "발견 밴드 폭 (±%)")
    f.colorbar(im, ax=ax, label="최적 VI 변동폭(%)"); f.tight_layout()
    p = FIGS/"fig6_fund.png"; f.savefig(p); plt.close(f); return p

def fig7_dist():
    d = best.EqualWeightBest.value_counts().sort_index()
    f, ax = plt.subplots(figsize=(6.5,3.0))
    ax.bar(d.index, d.values, color=BLUE)
    for w,c in d.items(): ax.text(w, c+0.2, str(c), ha="center", fontsize=8)
    ax.set_xticks(range(int(d.index.min()), int(d.index.max())+1))
    ax.set_xlabel("최적 VI 변동폭 (%)"); ax.set_ylabel("시나리오 수 (총 43)")
    ax.set_title("43개 시나리오에서 선택된 최적 VI 변동폭 분포")
    f.tight_layout(); p = FIGS/"fig7_dist.png"; f.savefig(p); plt.close(f); return p

print("그래프 생성 중...")
figs = {k:fn() for k,fn in {
    "composite":fig1_composite, "znorm":fig2_znorm, "cliff":fig3_cliff,
    "robust":fig4_robust, "heat":fig5_heat_band_dwell, "fund":fig6_fund,
    "dist":fig7_dist}.items()}
print("그래프 완료:", ", ".join(p.name for p in figs.values()))


# ─────────────────────────────────────────────────────────
# PDF 빌드
# ─────────────────────────────────────────────────────────
ST = {
 "title": ParagraphStyle("t", fontName="MalgunBd", fontSize=20, leading=26,
                         alignment=TA_CENTER, textColor=colors.HexColor("#1a2a3a")),
 "sub":   ParagraphStyle("s", fontName="Malgun", fontSize=11, leading=16,
                         alignment=TA_CENTER, textColor=colors.HexColor("#555555")),
 "h1":    ParagraphStyle("h1", fontName="MalgunBd", fontSize=14, leading=20,
                         spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a2a3a")),
 "h2":    ParagraphStyle("h2", fontName="MalgunBd", fontSize=11.5, leading=16,
                         spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#34495e")),
 "body":  ParagraphStyle("b", fontName="Malgun", fontSize=10, leading=15.5, spaceAfter=4),
 "cap":   ParagraphStyle("c", fontName="Malgun", fontSize=8.5, leading=11,
                         alignment=TA_CENTER, textColor=colors.HexColor("#777777")),
}
def P(t, s="body"): return Paragraph(t, ST[s])
def IMG(path, w=15.5):
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    return Image(str(path), width=w*cm, height=w*cm*ih/iw)

def tbl(data, colw=None, hdr=True):
    t = Table(data, colWidths=colw)
    style = [("FONTNAME",(0,0),(-1,-1),"Malgun"),("FONTSIZE",(0,0),(-1,-1),8.2),
             ("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#cccccc")),
             ("ALIGN",(0,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
             ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]
    if hdr:
        style += [("FONTNAME",(0,0),(-1,0),"MalgunBd"),
                  ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a2a3a")),
                  ("TEXTCOLOR",(0,0),(-1,0),colors.white)]
    t.setStyle(TableStyle(style)); return t

E = []  # flowables

# ===== 표지 =====
E += [Spacer(1,3.5*cm),
      P("정적 VI 변동폭 최적값 분석 보고서", "title"),
      Spacer(1,0.5*cm),
      P("가격불안정성·가격발견 지연·거래마찰 비용의 통합 평가와 민감도 분석", "sub"),
      Spacer(1,0.3*cm),
      P("ARENA 이산사건 시뮬레이션 · 43 시나리오 × 11 변동폭 × 30 복제 = 14,190 실행", "sub"),
      Spacer(1,1.0*cm),
      P(f"분석 방법: 정규화 + 파레토 지배 검토 + Augmented Tchebycheff + 가중치 민감도", "sub"),
      Spacer(1,5*cm),
      P("2026년 6월", "sub"),
      PageBreak()]

# ===== 1. 분석 개요 =====
E += [P("1. 분석 개요", "h1"),
 P("본 보고서는 한국거래소(KRX) 정적 VI(변동성완화장치)의 발동 기준 변동폭(w)을 "
   "5~15% 범위에서 변화시키며 시장 품질을 평가하고, 그 결과가 평가 기준 및 시장 조건의 "
   "변화에도 강건한지를 검증한 것이다. 각 변동폭은 다음 세 가지 손실지표로 평가하며, "
   "세 지표 모두 작을수록 우수하다."),
 P("• <b>가격불안정성 S</b> : 적정가 진입 이후 접속매매 시점의 상대편차 제곱평균<br/>"
   "• <b>가격발견 지연 D</b> : 적정가 범위에 지속적으로 안착하기까지의 시간(VI 정지시간 포함)<br/>"
   "• <b>거래마찰 비용 C</b> : 시간가중 미체결 주문량을 총주문·시간으로 정규화한 값"),
 P("세 지표를 0~1로 정규화한 뒤, 다목적 의사결정(MCDM) 기법인 Augmented Tchebycheff "
   "스칼라화로 하나의 종합점수를 산출하고, 가중치를 체계적으로 변화시켜 최적 변동폭의 "
   "강건성을 확인하였다. 시뮬레이션은 종료조건(발견 밴드 폭·머무는 시간)과 적정가 수준을 "
   "바꾼 43개 시나리오에 대해 수행되었다."),
 P("1.1 데이터 규모 및 신뢰성", "h2"),
 P("총 14,190회(43 시나리오 × 변동폭 11종 × 30 복제)를 단일 실행으로 자동 수집하였다. "
   "가격발견에 실패한 복제도 누락 없이 기록(DiscoveryFail=1, D=3600)하여 편향을 제거했으며, "
   "밴드에 한 번도 진입하지 못한 복제는 가격불안정성에 최악값을 부여하였다(총 1,682건)."),
]

# ===== 2. 평가 방법론 =====
E += [P("2. 평가 방법론", "h1"),
 P("2.1 정규화 (고정 기준)", "h2"),
 P("지표 단위가 서로 다르므로 사전에 고정한 이상치(ideal)·최악치(worst) 기준으로 0~1 손실점수 "
   "Z로 변환하였다. 후보가 추가될 때 점수가 변하는 관측 최소·최대 방식 대신 고정 기준을 사용해 "
   "시나리오 간 비교 가능성을 확보하였다."),
 P("• Z_D = D / 3600 &nbsp;&nbsp; • Z_C = C (이미 0~1) &nbsp;&nbsp; "
   "• Z_S = S / S_worst (S_worst = 0.005, 적정가 ±7% 지속 이탈 수준)"),
 P("2.2 파레토 지배 제거", "h2"),
 P("세 지표 모두에서 다른 변동폭보다 열등한(지배되는) 변동폭은 가중치와 무관하게 비효율적이므로 "
   "최종 후보에서 제외하였다."),
 P("2.3 Augmented Tchebycheff 종합점수", "h2"),
 P("가중치 λ=(λS,λD,λC), λS+λD+λC=1에 대해 종합점수는 "
   "G = max(λS·ZS, λD·ZD, λC·ZC) + ρ·Σ(λ·Z) (ρ=0.001)로 정의된다. 첫 항은 가장 나쁜 요소를 "
   "최소화하고, 둘째 항은 최대값이 비슷한 대안 중 전체 성과가 나은 쪽을 가린다. ρ는 0.0001~0.01 "
   "범위에서 결과 불변을 확인하였다."),
 P("2.4 가중치 민감도", "h2"),
 P("특정 가중치 선택의 자의성을 줄이기 위해 (1) 균형·안정성우선·가격발견우선·거래마찰우선의 "
   "4개 정책 시나리오와 (2) λ_j≥0.10, 0.05 간격의 전체 가중치 조합을 모두 탐색하여 각 변동폭의 "
   "1위 선택비율·평균순위·최악순위를 집계하였다."),
]

# ===== 3. 베이스 결과 =====
E += [PageBreak(), P("3. 베이스 시나리오 결과 (밴드 ±5% · 머무름 120초 · 적정가 125)", "h1"),
 P(f"현행 KRX 제도(±10%)에 대응하는 기준 조건에서, 균형가중 Augmented Tchebycheff 종합점수가 "
   f"가장 낮은(=가장 우수한) 변동폭은 <b>±{base_best}%</b>로 나타났다. 이 값은 전체 가중치 조합의 "
   f"<b>{base_win*100:.1f}%</b>에서 1위로 선택되었고, ρ값 변화에도 동일하게 유지되어 매우 강건하다."),
 IMG(figs["composite"]), P("그림 1. 베이스 시나리오 변동폭별 종합점수(낮을수록 우수)", "cap"),
 Spacer(1,0.2*cm),
 IMG(figs["znorm"]), P("그림 2. 베이스 시나리오 정규화 지표(ZS·ZD·ZC)", "cap"),
]
# 베이스 지표 표
tab_rows = [["변동폭","S","D(초)","C","ZS","ZD","ZC","종합점수 G"]]
for r in base.itertuples():
    tab_rows.append([f"{int(r.VIWidth)}%", f"{r.S_mean:.4f}", f"{r.D_mean:.0f}",
                     f"{r.C_mean:.3f}", f"{r.ZS:.3f}", f"{r.ZD:.3f}", f"{r.ZC:.3f}",
                     f"{r.G:.4f}"])
E += [Spacer(1,0.2*cm), P("표 1. 베이스 시나리오 변동폭별 지표 및 종합점수", "h2"),
      tbl(tab_rows), Spacer(1,0.3*cm),
 P("3.1 가중치 강건성", "h2"),
 P(f"전체 가중치 공간을 탐색한 결과, ±{base_best}%가 1위 선택비율 {base_win*100:.1f}%로 압도적이며 "
   f"평균순위도 가장 우수하였다. 차순위는 ±10%로, 두 값이 베이스 조건의 유력 후보군을 형성한다. "
   f"이는 단일 가중치가 아니라 광범위한 정책 선호에서 ±{base_best}%가 견고하게 우수함을 의미한다."),
 IMG(figs["robust"]), P("그림 3. 베이스 시나리오 가중치 강건성(전체 가중치 조합 중 1위 비율)", "cap"),
]

# ===== 4. 발견 밴드 절벽 =====
cliff = summ[summ.FundPrice==125].groupby("BandPct").FailRate.mean()*100
E += [PageBreak(), P("4. 발견 밴드 절벽 효과", "h1"),
 P(f"가격발견의 정의(밴드 폭)에 뚜렷한 임계점이 존재한다. 발견 밴드를 ±3%로 좁히면 "
   f"가격발견 실패율이 <b>{cliff.loc[3]:.0f}%</b>로 급증하는 반면, ±4% 이상에서는 "
   f"{cliff.loc[4]:.1f}% 이하로 사실상 항상 발견에 성공한다. 즉 ±3%는 시장이 도달·유지하기에 "
   f"지나치게 엄격한 기준이며, 해당 시나리오의 결과는 대부분 '실패' 데이터로 신뢰도가 낮다."),
 IMG(figs["cliff"]), P("그림 4. 발견 밴드 폭별 가격발견 실패율 (적정가 125)", "cap"),
 P("이 결과는 평가 기준 설계 시 발견 밴드를 ±4% 이상으로 두어야 의미 있는 가격발견 측정이 "
   "가능함을 시사한다. 본 분석의 베이스(±5%)는 이 안전 구간에 속한다."),
]

# ===== 5. 민감도 결과 =====
E += [PageBreak(), P("5. 민감도 분석 결과", "h1"),
 P("최적 변동폭이 (1) 종료조건(밴드 폭·머무는 시간)과 (2) 시장의 적정가 수준 변화에 대해 "
   "각각 얼마나 강건한지 확인하였다. 결론적으로 <b>최적 변동폭은 평가 기준(밴드·시간)에는 "
   "강건하지만 시장의 적정가 수준에는 민감</b>하다."),
 P("5.1 종료조건에 대한 강건성", "h2"),
 P("적정가를 125로 고정하고 발견 밴드(±3~7%)와 머무는 시간(60~180초)을 모두 바꾸어도, "
   "최적 변동폭은 대부분 ±9~10%에 머문다(±3% 밴드는 실패율이 높아 예외적). 즉 평가 기준을 "
   "어떻게 잡든 결론이 거의 흔들리지 않는다."),
 IMG(figs["heat"]), P("그림 5. 종료조건(밴드×시간)별 최적 변동폭 — 적정가 125 고정", "cap"),
 P("5.2 적정가 수준에 대한 민감성", "h2"),
 P("반면 적정가를 115·120·125로 바꾸면 최적 변동폭이 크게 이동한다. 적정가가 낮아질수록(상한가 "
   "130에 가까워질수록) 가격 동학과 상한가 캡의 상호작용이 달라져 더 큰 변동폭이 유리해지는 "
   "경향이 나타난다. 이는 단일 종목·국면에 대한 최적값을 일반화할 때 주의가 필요함을 의미한다."),
 IMG(figs["fund"]), P("그림 6. 적정가별 최적 변동폭 — 머무는 시간 120초 고정", "cap"),
 IMG(figs["dist"]), P("그림 7. 43개 시나리오에서 선택된 최적 변동폭 분포", "cap"),
]
# 민감도 요약 표 (대표 시나리오)
sel = best[best.FundPrice==125].sort_values(["BandPct","DwellTime"])
rows = [["밴드±%","머무름(초)","적정가","균형 최적","강건 최적","평균 실패율"]]
for r in sel.itertuples():
    rows.append([int(r.BandPct), int(r.DwellTime), int(r.FundPrice),
                 f"{int(r.EqualWeightBest)}%", f"{int(r.RobustBest)}%", f"{r.MeanFailRate*100:.0f}%"])
E += [Spacer(1,0.2*cm), P("표 2. 종료조건 민감도 (적정가 125, 시나리오별 최적 변동폭)", "h2"),
      tbl(rows)]

# ===== 6. 결론 =====
dist = best.EqualWeightBest.value_counts().sort_index()
E += [PageBreak(), P("6. 결론 및 정책 함의", "h1"),
 P(f"<b>(1) 기준 조건 최적값</b> : 현행 제도에 대응하는 베이스 조건(밴드 ±5%·적정가 125)에서 "
   f"최적 정적 VI 변동폭은 <b>±{base_best}%</b>이며, 차순위는 ±10%이다. 두 값 모두 현행 ±10% 부근 "
   f"또는 그보다 다소 좁은 구간으로, 현행 제도가 합리적 범위에 있음을 시사한다."),
 P("<b>(2) 평가 기준에 강건</b> : 발견 밴드 폭과 머무는 시간을 넓은 범위로 바꾸어도 최적값은 "
   "±9~10%로 안정적이다. 결론이 측정 기준의 자의적 선택에 좌우되지 않는다."),
 P("<b>(3) 적정가에 민감</b> : 시장의 적정가 수준이 달라지면 최적 변동폭이 ±11~15%로 이동한다. "
   "따라서 '단일 최적값'보다는 시장 상황(특히 적정가의 상한가 대비 위치)에 따른 적정 범위로 "
   "이해하는 것이 타당하다."),
 P("<b>(4) 발견 밴드 절벽</b> : 발견 기준을 ±3%로 좁히면 가격발견이 사실상 불가능해진다. "
   "정책·평가 설계에서 발견 밴드는 ±4% 이상으로 두어야 한다."),
 P("6.1 한계 및 향후 과제", "h2"),
 P("본 분석은 단일 종목·상승 국면·포아송 주문 도착·시가 고정(호가 배열 제약)이라는 단순화 위에 "
   "있다. 향후 (i) 하락·횡보 국면, (ii) 자기여기적(Hawkes) 주문 도착, (iii) 다종목 상호작용, "
   "(iv) 적정가를 연속적으로 변화시킨 정밀 민감도 분석으로 확장하면 일반화 가능성을 높일 수 있다."),
 Spacer(1,0.4*cm),
 P("─"*40, "cap"),
 P("부록: 분석 산출물 — summary_by_scenario_viwidth.csv, best_viwidth_per_scenario.csv, "
   "weight_robustness_all.csv, policy_best_per_scenario.csv", "cap"),
]

doc = SimpleDocTemplate(str(PDF_OUT), pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm,
                        topMargin=1.8*cm, bottomMargin=1.8*cm,
                        title="정적 VI 변동폭 분석 보고서")
doc.build(E)
print("PDF 생성 완료:", PDF_OUT)

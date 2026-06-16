# -*- coding: utf-8 -*-
"""
정적 VI 변동폭 평가 자동 분석 스크립트
- 입력: ARENA가 출력한 result.txt (14,190행 = 43 시나리오 × 11 VIWidth × 30 복제)
- 구현: vi_evaluation_framework.pdf 6~12장
    6장 정규화 → 7장 파레토 지배 제거 → 8장 Augmented Tchebycheff
    9장 가중치 민감도 → 10장 최종 선택 규칙
- 출력: analysis/ 폴더에 CSV 여러 개 + 콘솔 요약

실행:  python analyze_vi.py
"""

import sys
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

# Windows 콘솔(cp949)에서도 한글/특수문자 출력되도록 UTF-8 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# 0. 설정 (경로 / 평가 기준 상수)  ← 필요하면 여기만 수정
# ─────────────────────────────────────────────────────────────
RESULT_PATH = Path(r"C:\Users\strow\simulation\result.txt")
OUT_DIR     = Path(__file__).resolve().parent / "analysis"
OUT_DIR.mkdir(exist_ok=True)

P_STAR_DEFAULT = 125.0      # 적정가 기본값 (시나리오마다 FundPrice로 대체됨)
H_MAX          = 3600.0     # 최대 시뮬레이션 시간 (= 가격발견 실패 시 D값)

# --- 정규화 고정 기준 (PDF 6.1, ideal=0 / worst=고정) ---
# Z_j = (Xbar - ideal) / (worst - ideal),  0=우수 1=열악
D_IDEAL, D_WORST = 0.0, H_MAX        # 가격발견 지연: D / 3600
C_IDEAL, C_WORST = 0.0, 1.0          # 거래마찰: C는 이미 0~1 → 그대로
S_IDEAL          = 0.0               # 가격불안정성
# S_worst: 적정가에서 지속적으로 ~7% 벗어난 상태를 "최악"으로 정의 (0.07^2≈0.0049)
# 관측 최댓값(≈0.0048)과 부합. 값이 크면 S가 점수에서 무력화되니 주의.
S_WORST          = 0.005

# --- Augmented Tchebycheff ρ (작은 양수) ---
RHO_MAIN = 0.001
RHO_SET  = [0.0001, 0.001, 0.01]     # ρ 강건성 점검용

# --- 가중치 민감도 ---
POLICY_WEIGHTS = {            # (λS, λD, λC)  S=안정성 D=발견 C=마찰
    "균형(equal)":      (1/3, 1/3, 1/3),
    "안정성우선":        (0.50, 0.25, 0.25),
    "가격발견우선":      (0.25, 0.50, 0.25),
    "거래마찰우선":      (0.25, 0.25, 0.50),
}
WEIGHT_MIN  = 0.10            # 가중치 공간 탐색: λ_j >= 0.10
WEIGHT_STEP = 0.05            # 0.05 간격

# 베이스 시나리오 (band 5%, dwell 120, fund 125)
BASELINE_SCN = 13

COLS = ["VIWidth","NumVI","TotalHaltTime","TotalTrades","TotalTradeVolume",
        "CumAbsPriceChange","MaxOvershoot","FirstDiscoveryTime","EndTime",
        "TotalOrders","FilledOrders","UnfilledQty","Scn","BandPct","DwellTime",
        "FundPrice","StabilityScore","FrictionCost","DiscoveryFail"]

SCN_KEYS = ["Scn","BandPct","DwellTime","FundPrice"]


# ─────────────────────────────────────────────────────────────
# 1. 데이터 로드 & 전처리
# ─────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    # 헤더 끝 공백으로 생기는 빈 컬럼 제거
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    # 컬럼명 정규화 (혹시 순서/이름 다르면 강제 매핑)
    if list(df.columns)[:len(COLS)] != COLS:
        df.columns = COLS[:len(df.columns)]
    # 정수형 보기 좋게
    for c in ["VIWidth","Scn","BandPct","DwellTime","FundPrice","DiscoveryFail","NumVI"]:
        df[c] = df[c].round().astype(int)

    # --- PDF 2.4: 밴드에 한 번도 못 들어간 실패는 S를 worst로 치환 ---
    #     (FirstDiscoveryTime == -1  ⇒ 밴드 미진입)
    never_entered = (df["DiscoveryFail"] == 1) & (df["FirstDiscoveryTime"] == -1)
    df.loc[never_entered, "StabilityScore"] = S_WORST
    df["NeverEntered"] = never_entered.astype(int)
    return df


# ─────────────────────────────────────────────────────────────
# 2. (시나리오 × VIWidth) 집계 : 평균/표준편차/95%CI/실패율
# ─────────────────────────────────────────────────────────────
def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(SCN_KEYS + ["VIWidth"])
    def ci95(x):
        x = np.asarray(x, float)
        n = len(x)
        if n < 2: return 0.0
        return 1.96 * x.std(ddof=1) / np.sqrt(n)
    agg = g.agg(
        n            =("StabilityScore","size"),
        S_mean       =("StabilityScore","mean"),
        D_mean       =("EndTime","mean"),
        C_mean       =("FrictionCost","mean"),
        FailRate     =("DiscoveryFail","mean"),
        NeverRate    =("NeverEntered","mean"),
        Trades_mean  =("TotalTrades","mean"),
        NumVI_mean   =("NumVI","mean"),
    ).reset_index()
    # 95% 신뢰구간 반치폭
    agg["S_ci"] = g["StabilityScore"].apply(ci95).values
    agg["D_ci"] = g["EndTime"].apply(ci95).values
    agg["C_ci"] = g["FrictionCost"].apply(ci95).values
    return agg


# ─────────────────────────────────────────────────────────────
# 3. 정규화 (PDF 6장) — 고정 ideal/worst 기준
# ─────────────────────────────────────────────────────────────
def normalize(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    agg["ZS"] = ((agg["S_mean"] - S_IDEAL) / (S_WORST - S_IDEAL)).clip(0, 1)
    agg["ZD"] = ((agg["D_mean"] - D_IDEAL) / (D_WORST - D_IDEAL)).clip(0, 1)
    agg["ZC"] = ((agg["C_mean"] - C_IDEAL) / (C_WORST - C_IDEAL)).clip(0, 1)
    return agg


# ─────────────────────────────────────────────────────────────
# 4. 파레토 지배 (PDF 7장)
#    w1이 w2에게 지배됨: 세 Z 모두 ≥ 이고 하나라도 > (작을수록 좋음)
# ─────────────────────────────────────────────────────────────
def mark_pareto(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    Z = sub[["ZS","ZD","ZC"]].values
    n = len(Z)
    dominated = np.zeros(n, bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if np.all(Z[j] <= Z[i]) and np.any(Z[j] < Z[i]):
                dominated[i] = True
                break
    sub["Dominated"] = dominated
    return sub


# ─────────────────────────────────────────────────────────────
# 5. Augmented Tchebycheff (PDF 8장)
#    G = max(λS·ZS, λD·ZD, λC·ZC) + ρ·Σ(λ·Z)   (ideal=0 기준)
# ─────────────────────────────────────────────────────────────
def tcheby(zs, zd, zc, w, rho=RHO_MAIN):
    a = np.array([w[0]*zs, w[1]*zd, w[2]*zc])
    return a.max() + rho * a.sum()


def best_viwidth(sub: pd.DataFrame, w, rho=RHO_MAIN, exclude_dominated=True):
    """주어진 가중치에서 종합점수 최소 VIWidth 반환 (점수 표 포함)"""
    s = sub.copy()
    s["G"] = [tcheby(r.ZS, r.ZD, r.ZC, w, rho) for r in s.itertuples()]
    pool = s[~s["Dominated"]] if (exclude_dominated and "Dominated" in s) else s
    pool = pool if len(pool) else s
    s = s.sort_values("G").reset_index(drop=True)
    s["rank"] = s["G"].rank(method="min").astype(int)
    best = pool.sort_values("G").iloc[0]["VIWidth"]
    return int(best), s


def weight_grid():
    """λ_j ≥ 0.10, 합=1, 0.05 간격인 모든 (λS,λD,λC)"""
    vals = np.round(np.arange(WEIGHT_MIN, 1 - 2*WEIGHT_MIN + 1e-9, WEIGHT_STEP), 2)
    grid = []
    for ls, ld in product(vals, vals):
        lc = round(1 - ls - ld, 2)
        if lc >= WEIGHT_MIN - 1e-9 and lc <= 1:
            grid.append((round(ls,2), round(ld,2), lc))
    return grid


# ─────────────────────────────────────────────────────────────
# 6. 시나리오별 전체 MCDM 파이프라인
# ─────────────────────────────────────────────────────────────
def analyze_scenario(sub_norm: pd.DataFrame):
    """한 시나리오(11 VIWidth)에 대해 파레토→Tchebycheff→가중치민감도"""
    sub = mark_pareto(sub_norm)

    # (a) 정책 시나리오별 최적
    policy_best = {}
    for name, w in POLICY_WEIGHTS.items():
        b, _ = best_viwidth(sub, w)
        policy_best[name] = b

    # (b) 가중치 공간 전체 탐색 → 1위 선택비율 / 평균순위 / 최악순위
    grid = weight_grid()
    widths = sorted(sub["VIWidth"].unique())
    win = {w: 0 for w in widths}
    ranks = {w: [] for w in widths}
    for w in grid:
        _, tab = best_viwidth(sub, w)
        winner = tab.sort_values("G").iloc[0]["VIWidth"]
        # 지배 제외 후 승자
        pool = tab[~tab["Dominated"]] if "Dominated" in tab else tab
        winner = int((pool if len(pool) else tab).sort_values("G").iloc[0]["VIWidth"])
        win[winner] += 1
        for r in tab.itertuples():
            ranks[int(r.VIWidth)].append(r.rank)
    robust = pd.DataFrame({
        "VIWidth": widths,
        "WinRate": [win[w]/len(grid) for w in widths],
        "MeanRank": [np.mean(ranks[w]) for w in widths],
        "WorstRank": [max(ranks[w]) for w in widths],
    })
    # 강건 최적 = 1위 선택비율 최대
    robust_best = int(robust.sort_values(["WinRate","MeanRank"],
                                         ascending=[False, True]).iloc[0]["VIWidth"])

    # (c) ρ 강건성: ρ 바꿔도 균형가중 최적이 그대로인가
    rho_best = {rho: best_viwidth(sub, POLICY_WEIGHTS["균형(equal)"], rho)[0]
                for rho in RHO_SET}

    return sub, policy_best, robust, robust_best, rho_best


# ─────────────────────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print(" 정적 VI 변동폭 평가 분석 시작")
    print("="*70)
    df = load_data(RESULT_PATH)
    print(f"로드: {len(df):,}행, 시나리오 {df['Scn'].nunique()}개, "
          f"VIWidth {sorted(df['VIWidth'].unique())}")
    print(f"밴드 미진입(S→worst 치환) 행: {df['NeverEntered'].sum():,}개")

    agg = aggregate(df)
    agg = normalize(agg)
    agg.to_csv(OUT_DIR/"summary_by_scenario_viwidth.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n[저장] summary_by_scenario_viwidth.csv  ({len(agg)}행)")

    # 시나리오별 분석
    rows_best, rows_robust, rows_policy = [], [], []
    baseline_detail = None
    for scn, sub in agg.groupby("Scn"):
        band  = sub["BandPct"].iloc[0]
        dwell = sub["DwellTime"].iloc[0]
        fund  = sub["FundPrice"].iloc[0]
        sub_marked, policy_best, robust, robust_best, rho_best = analyze_scenario(sub)
        eq_best = policy_best["균형(equal)"]

        rows_best.append(dict(Scn=scn, BandPct=band, DwellTime=dwell, FundPrice=fund,
                              EqualWeightBest=eq_best, RobustBest=robust_best,
                              MeanFailRate=round(sub["FailRate"].mean(),3),
                              **{f"best_{k}": v for k,v in policy_best.items()},
                              rho_stable=int(len(set(rho_best.values()))==1)))
        robust["Scn"] = scn
        rows_robust.append(robust)
        for k,v in policy_best.items():
            rows_policy.append(dict(Scn=scn, BandPct=band, DwellTime=dwell,
                                    FundPrice=fund, Policy=k, BestVIWidth=v))

        if scn == BASELINE_SCN:
            baseline_detail = (sub_marked, robust, policy_best, rho_best)

    best_df   = pd.DataFrame(rows_best).sort_values("Scn")
    robust_df = pd.concat(rows_robust, ignore_index=True)
    policy_df = pd.DataFrame(rows_policy)

    best_df.to_csv(OUT_DIR/"best_viwidth_per_scenario.csv", index=False, encoding="utf-8-sig")
    robust_df.to_csv(OUT_DIR/"weight_robustness_all.csv", index=False, encoding="utf-8-sig")
    policy_df.to_csv(OUT_DIR/"policy_best_per_scenario.csv", index=False, encoding="utf-8-sig")
    print("[저장] best_viwidth_per_scenario.csv / weight_robustness_all.csv / policy_best_per_scenario.csv")

    # ── 베이스 시나리오 상세 출력 ──
    print("\n" + "="*70)
    print(f" 베이스 시나리오 (Scn{BASELINE_SCN}: band5% / dwell120 / fund125)")
    print("="*70)
    sub_b, robust_b, policy_b, rho_b = baseline_detail
    show = sub_b[["VIWidth","S_mean","D_mean","C_mean","FailRate",
                  "ZS","ZD","ZC","Dominated"]].copy()
    show = show.round({"S_mean":5,"D_mean":1,"C_mean":4,"FailRate":3,
                       "ZS":3,"ZD":3,"ZC":3})
    print("\n[정규화 지표]")
    print(show.to_string(index=False))
    # 균형가중 종합점수 순위
    _, tab_b = best_viwidth(sub_b, POLICY_WEIGHTS["균형(equal)"])
    print("\n[균형가중 Augmented Tchebycheff 점수 순위]")
    print(tab_b[["VIWidth","ZS","ZD","ZC","G","rank","Dominated"]]
          .round({"ZS":3,"ZD":3,"ZC":3,"G":4}).to_string(index=False))
    print("\n[정책별 최적 VIWidth]")
    for k,v in policy_b.items(): print(f"   {k:14s}: {v}")
    print(f"\n[ρ 강건성] {rho_b}  → {'안정적' if len(set(rho_b.values()))==1 else '주의(ρ에 민감)'}")
    print("\n[가중치 공간 강건성 (베이스)]")
    print(robust_b.sort_values('WinRate',ascending=False)
          .round({'WinRate':3,'MeanRank':2}).to_string(index=False))

    # ── 민감도 요약: 조건이 바뀌어도 최적 VIWidth가 유지되나 ──
    print("\n" + "="*70)
    print(" 민감도 요약 - 시나리오별 최적 VIWidth (균형가중 / 강건)")
    print("="*70)
    print(best_df[["Scn","BandPct","DwellTime","FundPrice",
                   "EqualWeightBest","RobustBest","MeanFailRate"]]
          .to_string(index=False))

    print("\n[균형가중 최적 VIWidth 분포]")
    print(best_df["EqualWeightBest"].value_counts().sort_index().to_string())
    print("\n분석 완료. 결과는", OUT_DIR, "에 저장됨.")


if __name__ == "__main__":
    main()

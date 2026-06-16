# ARENA 정적 VI 시뮬레이션 — 전체 구현 가이드 (개정판)

> **프로젝트**: 정적 VI(Volatility Interruption) 발동 기준 변동폭의 최적값 탐색
> **기반 연구**: 안일찬 외 (2017). KRX 정적 VI 도입의 가격안정화 및 가격발견 효과
> **평가 체계**: vi_evaluation_framework — 정규화 + 파레토 + Augmented Tchebycheff + 가중치 민감도
> **도구**: ARENA 시뮬레이션 (학생용) + Python(pandas/matplotlib/reportlab) 후처리

---

## 개정 이력 (초판 이후 주요 변경)

이 문서는 초판 작성 이후 다음 변경을 반영한 개정판이다.

1. **단일가매매 로직 개선** — 단일가 한 칸만 체결하던 방식을 "교차분 전체 반복 청산 + 단일가 사후 결정"으로 교체 (§9)
2. **모델 오류 3건 수정** — 단일가 체결·안정시간 측정·미체결량 계산 (§9, §10, §13)
3. **평가지표 S/D/C 수집 모듈 신설** — 초당 누적 샘플러로 가격불안정성·가격발견지연·거래마찰비용 직접 산출 (§11)
4. **종료조건·적정가 변수화** — 밴드 [119,130], 접속매매 중에만 안정시간 측정 (§10)
5. **민감도 자동 스윕** — 시나리오 룩업 테이블 + NREP 디코딩으로 43 시나리오 × 11 변동폭 × 30 복제 = **14,190 실행** 단일 Run (§12)
6. **실패 복제 기록** — 가격발견 실패 복제도 누락 없이 한 줄씩 기록 (§10, §13)
7. **분석 파이프라인** — Python 스크립트로 정규화→Tchebycheff→민감도→PDF 보고서 자동화 (§14)
8. **최종 결과** — 베이스 최적 변동폭 **±9%** (§15)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [연구 배경 및 동기](#2-연구-배경-및-동기)
3. [평가 지표 설계](#3-평가-지표-설계)
4. [ARENA 변수 설계](#4-arena-변수-설계)
5. [주문 생성 모델](#5-주문-생성-모델)
6. [주문 가격·수량 부여](#6-주문-가격수량-부여)
7. [매수 주문 처리 로직](#7-매수-주문-처리-로직)
8. [매도 주문 처리 로직](#8-매도-주문-처리-로직)
9. [단일가매매 로직 (개선판)](#9-단일가매매-로직-개선판)
10. [시뮬레이션 종료 조건 및 실패 처리](#10-시뮬레이션-종료-조건-및-실패-처리)
11. [평가지표 수집 (S·D·C)](#11-평가지표-수집-sdc)
12. [민감도 분석 자동 실험 설계](#12-민감도-분석-자동-실험-설계)
13. [구현 중 만난 주요 이슈와 해결](#13-구현-중-만난-주요-이슈와-해결)
14. [분석 파이프라인 (Python 후처리)](#14-분석-파이프라인-python-후처리)
15. [최종 실험 결과 분석](#15-최종-실험-결과-분석)
16. [모델링 방법 요약](#16-모델링-방법-요약)
17. [부록: 시장 매칭 원리](#17-부록-시장-매칭-원리)

---

## 1. 프로젝트 개요

### 연구 목적
정적 VI 발동 기준 변동폭(%)을 5~15 범위에서 변화시키며 시뮬레이션을 수행하여, **가격불안정성·가격발견 지연·거래마찰 비용**의 세 가지 손실지표를 종합적으로 고려한 최적값을 찾는다. 나아가 그 결론이 평가 기준(밴드 폭·머무름 시간)과 시장 조건(적정가)의 변화에도 강건한지 민감도 분석으로 검증한다.

### 기본 가정
- 짧은 기간의 주가 변화 시뮬레이션 → 적정가격은 고정
- 적정가격 > 시가 (상승 국면만 다룸)
- 적정가격 ≤ 상한가(시가의 130%)
- 시가 100원, 적정가격 125원(시나리오에 따라 115/120/125), 호가 단위 1원, 상한가 130원
- 동적 VI 및 임의종료(RE) 제도는 무시
- VI 발동 시 정확히 2분간 거래 정지

### 종료 조건
주가가 적정가격 ±(설정 밴드)% 범위에서 **접속매매 중에** 설정 시간(기본 120초) 동안 벗어나지 않으면 시뮬레이션 종료. 베이스 조건은 밴드 ±5%(119~130원), 머무름 120초.

### 실험 규모
43개 시나리오 × 변동폭 11종(5~15%) × 복제 30회 = **14,190 실행**(단일 Run 자동 수집).

---

## 2. 연구 배경 및 동기

### KRX 정적 VI 제도
- 2015년 6월 15일 도입
- **직전 단일가매매 체결가** 대비 ±10% 변동 예상 시 발동 (하루 중 단일가 체결 때마다 기준가 갱신)
- 2분간 거래 정지 후 단일가매매로 재개
- 동시에 가격제한폭 ±15% → ±30%로 확대

### 선행 연구 (안일찬 외, 2017)의 핵심 발견
1. **정적 VI의 가격안정화 효과가 절대적으로 낮음**
2. **상한가 발생일의 88% 이상에서 정적 VI 동시 발동** → 가격제한폭과 중복 기능
3. **가격제한폭 ±30% 확대로 실현 변동성 14~15% 증가**
4. **결론**: "여러 모수(예: ±10% 정적 VI 변동폭)에 대한 심도 깊은 검증이 필요"

→ 본 연구는 이 후속 연구 요청에 직접 응답한다.

---

## 3. 평가 지표 설계

### 3.1 3대 손실지표 (모두 작을수록 우수)

평가 체계는 종합점수에 들어가는 **3대 손실지표**와, 해석을 돕는 **보조지표**로 구분한다.

| 지표 | 기호 | 정의 | 정규화 |
|---|---|---|---|
| **가격불안정성** | S | 적정가 최초 진입 이후 접속매매 시점의 상대편차 제곱평균 | Z_S = S / S_worst (S_worst=0.005) |
| **가격발견 지연** | D | 적정가 범위에 지속 안착하기까지의 실제 경과시간(VI 정지 포함) | Z_D = D / 3600 |
| **거래마찰 비용** | C | 시간가중 미체결량 / (종료시간 × 총주문) | Z_C = C (이미 0~1) |

$$S_r=\frac{1}{|A_r|}\sum_{t\in A_r}\left(\frac{P_t-P^*}{P^*}\right)^2,\quad
C_r=\frac{\sum_{t=1}^{T_r}U_r(t)}{T_r\cdot Q^{sub}_r},\quad
U_r(t)=\sum_p BidQty_p+\sum_p AskQty_p$$

- $A_r$ = 최초 밴드 진입 이후 **접속매매 중**(VI 정지 제외) 시점 집합, $P^*$ = 적정가
- $D_r = T^{discovery}_r$ (실패 시 $D_r=3600$), 가격발견 실패율 $F(w)$는 보조지표

### 3.2 보조지표 (종합점수 제외, 해석용)

| 지표 | 의미 | 방향 |
|---|---|---|
| NumVI | VI 발동 횟수 | ↓ |
| TotalHaltTime | 총 거래정지 시간 | ↓ |
| TotalTrades / TotalTradeVolume | 총 체결건수/수량 | - |
| CumAbsPriceChange | 누적 절대 가격변화 | ↓ |
| MaxOvershoot | 적정가 초과 최대폭 | ↓ |
| TotalOrders / FilledOrders | 총 생성/체결 주문수량 | - |
| UnfilledQty | 종료 시 미체결 잔량 | ↓ |
| **FillRate** | 체결률 = **2 × FilledOrders / TotalOrders** | ↑ |
| DiscoveryFail | 가격발견 실패 여부(0/1) | ↓ |
| CallExecQty | 단일가매매 총 체결수량 | - |

> **FillRate 주의**: FilledOrders는 체결량(한쪽만 카운트), TotalOrders는 매수+매도 양면 합산이므로 단순 비는 구조적으로 ≤50%다. 의미 있는 체결률은 **2배 보정**(`2 × FilledOrders / TotalOrders`)이며, 후처리에서 계산한다.

---

## 4. ARENA 변수 설계

### 4.1 가격 및 제도 상태 변수 (스칼라)

| 이름 | 의미 | 초기값 |
|---|---|---|
| V_CurrentPrice | 현재 주가 | 100 |
| V_FundamentalPrice | 적정가격 (시나리오별 세팅) | 125 |
| V_RefPrice | 현재 정적 VI 기준가격 | 100 |
| V_VIWidth | 정적 VI 폭 (시나리오별 5~15) | (NREP 디코딩) |
| V_BandPct | 발견 밴드 폭(%) (시나리오별) | (NREP 디코딩) |
| V_DwellTime | 발견 머무름 시간(초) (시나리오별) | (NREP 디코딩) |
| V_VIMode | VI 상태 (0=연속매매, 1=거래정지) | 0 |
| V_MarketStop | 종료 플래그 (0=계속, 1=종료) | 0 |

**중요**: 스칼라 변수와 Attribute는 **Rows를 비워둬야** 한다(Rows=1이면 1차원 배열로 인식되어 dimension 에러).

### 4.2 주문장(호가창) 배열 변수

```
BidQty[51]: 가격 80~130원의 매수잔량 (인덱스 1~51)
AskQty[51]: 가격 80~130원의 매도잔량 (인덱스 1~51)
```
**인덱스 매핑**: `인덱스 = A_Price − 79` (100원→21, 125원→46)

### 4.3 단일가매매 계산용 변수

| 이름 | 초기값 | 의미 |
|---|---|---|
| V_CallPrice | 0 | 단일가 |
| V_CallExecQty | 0 | 단일가 총 체결수량 |
| V_LastMatchedBid | -9999 | 마지막 체결 매수 한계가 |
| V_LastMatchedAsk | 9999 | 마지막 체결 매도 한계가 |
| V_BestAsk / V_BestBid | 0 | 최우선 매도/매수호가 |
| V_Temp | 0 | 임시값 |

### 4.4 성과지표·수집용 변수

| 이름 | 의미 | 초기값 |
|---|---|---|
| V_NumVI | VI 발동 횟수 | 0 |
| V_TotalHaltTime | 총 거래정지 시간 | 0 |
| V_TotalTradeVolume / V_TotalTrades | 총 체결수량/건수 | 0 |
| V_CumAbsPriceChange | 누적 절대 가격변화 | 0 |
| V_MaxOvershoot | 적정가 초과 최대폭 | 0 |
| V_FirstDiscoveryTime | 첫 밴드 진입 시점 | **-1** |
| V_EndTime | 종료 시점 | 0 |
| V_BandEntryTime | 현재 밴드 진입 시점 | **-1** |
| V_TotalOrders / V_FilledOrders / V_UnfilledQty | 총생성/체결/미체결 | 0 |
| **V_Ur** | 현 시점 미체결 스냅샷(임시) | 0 |
| **V_TWUnfilled** | 시간가중 미체결량 Σ Ur | 0 |
| **V_SumSqDev** | Σ((P−F)/F)² | 0 |
| **V_NSamples** | 안정성 표본 수 \|A_r\| | 0 |
| **V_StabilityActive** | 최초 밴드 진입 후 1 | 0 |
| **V_StabilityScore** | 최종 S | 0 |
| **V_FrictionCost** | 최종 C | 0 |
| **V_DiscoveryFail** | 가격발견 실패=1 | **1** |

### 4.5 시나리오 룩업 배열 (1-D, Rows=43)

| 배열 | 의미 |
|---|---|
| ScnBandPct[43] | 시나리오별 밴드 폭(%) |
| ScnDwell[43] | 시나리오별 머무름 시간(초) |
| ScnFund[43] | 시나리오별 적정가(원) |

추가 스칼라: `V_Scn`(시나리오 인덱스), `V_RepInScn`(시나리오 내 복제 인덱스).

### 4.6 주문 Attribute (Entity별)

| 이름 | 의미 |
|---|---|
| A_Side | 매수=1, 매도=-1 |
| A_Price / A_Qty / A_RemainQty | 가격/수량/미체결 잔량 |
| A_ArrivalTime | 도착 시각 |
| A_GenProb | 주문 통과 확률 |

---

## 5. 주문 생성 모델

### 5.1 상태의존적 포아송 도착
매 1초마다 잠재 주문 entity를 생성하고, 현재 시장 상황(gap)에 따라 통과 확률을 다르게 부여하여 piecewise 포아송 프로세스를 구현한다.
```
gap = V_FundamentalPrice − V_CurrentPrice (적정가 − 현재가)
```

### 5.2 도착률 (매수: gap 클수록 자주 / 매도: gap 작을수록 자주)

| gap | 매수 간격 | 매도 간격 |
|---|---|---|
| ≥20 | EXPO(1.5) | EXPO(4.5) |
| 15~19 | EXPO(2.0) | EXPO(4.0) |
| 10~14 | EXPO(2.8) | EXPO(3.6) |
| 5~9 | EXPO(3.8) | EXPO(3.3) |
| 0~4 | EXPO(5.0) | EXPO(3.0) |

### 5.3 A_GenProb 누적 차분 수식 (`<` 연산자 회피)

매수:
```
(V_FundamentalPrice - V_CurrentPrice >= 20) * (1/1.5 - 1/2.0) +
(V_FundamentalPrice - V_CurrentPrice >= 15) * (1/2.0 - 1/2.8) +
(V_FundamentalPrice - V_CurrentPrice >= 10) * (1/2.8 - 1/3.8) +
(V_FundamentalPrice - V_CurrentPrice >= 5)  * (1/3.8 - 1/5.0) +
(V_FundamentalPrice - V_CurrentPrice >= 0)  * (1/5.0)
```
매도는 동일 구조에 매도 간격(1/4.5 − 1/4.0 …)을 사용. gap이 클수록 더 많은 항이 활성화되어 정확한 확률값에 도달.

### 5.4 ARENA 모듈 구성
```
[Create_BuyOrder] → [Assign_BuyRate] → [Decide_BuyFilter (2-way by Chance, A_GenProb*100)] → ...
[Create_SellOrder] → [Assign_SellRate] → [Decide_SellFilter] → ...
```
Create: Type=Constant, Value=1, Units=Seconds, **First Creation=1.0**, Max Arrivals=Infinite.

> ⚠️ 주문/감시 Create의 **First Creation=1.0**, 초기화 엔티티만 0.0이어야 한다 (§13.10).

---

## 6. 주문 가격·수량 부여

### 6.1 수량
```
A_Qty = DISC(0.5, 1, 0.8, 2, 1.0, 3)   (1주 50%, 2주 30%, 3주 20%)
A_RemainQty = A_Qty
```

### 6.2 매수 가격 (gap별 offset)
- gap ≥ 15: `A_Price = V_CurrentPrice + DISC(0.35,2, 0.75,1, 0.95,0, 1.0,-1)`
- 5 ≤ gap < 15: `+ DISC(0.35,1, 0.75,0, 0.95,-1, 1.0,-2)`
- gap < 5: `+ DISC(0.20,1, 0.60,0, 0.90,-1, 1.0,-2)`

### 6.3 매도 가격 (offset ≥ 0, 현재가 미만 매도 없음)
- gap ≥ 15: `+ DISC(0.10,0, 0.45,1, 0.80,2, 1.0,3)`
- 5 ≤ gap < 15: `+ DISC(0.20,0, 0.60,1, 0.90,2, 1.0,3)`
- gap < 5: `+ DISC(0.30,0, 0.70,1, 0.90,2, 1.0,3)`

### 6.4 공통
```
V_TotalOrders = V_TotalOrders + A_Qty   (BuySide/SellSide 양쪽)
A_Price = MX(80, MN(130, A_Price))       (클램핑)
```

---

## 7. 매수 주문 처리 로직

### 7.1 흐름
```
(매수 가격 부여 완료)
 → Decide_VICheck_Buy (V_VIMode==1?) ─True→ Assign_ParkBuy → Dispose
 → [LBL_FindBestAsk_Buy] → Assign_FindBestAsk (51-인자 MN) 
 → Decide_PriceMatch_Buy (A_Price < V_BestAsk?) ─True→ ParkBuy
 → Decide_VITrigger_Buy (V_BestAsk >= V_RefPrice*(1+V_VIWidth/100)?) ─True→ Separate → VI 라인
 → Assign_ExecuteTrade_Buy
 → Decide_RemainCheck_Buy (A_RemainQty>0?) ─True→ Go to LBL_FindBestAsk_Buy
 → Dispose_BuyDone
```

### 7.2 BestAsk 압축 탐색 (51-인자)
```
V_BestAsk = MN( (AskQty(1)>0)*80 + (AskQty(1)==0)*9999, ... ,
                (AskQty(51)>0)*130 + (AskQty(51)==0)*9999 )
```

### 7.3 Assign_ExecuteTrade_Buy (순차 실행)
```
V_Temp = MN(A_RemainQty, AskQty(V_BestAsk-79))
AskQty(V_BestAsk-79) -= V_Temp
A_RemainQty -= V_Temp
V_TotalTradeVolume += V_Temp ; V_TotalTrades += 1 ; V_FilledOrders += V_Temp
V_Temp = V_CurrentPrice ; V_CurrentPrice = V_BestAsk
V_CumAbsPriceChange += ABS(V_CurrentPrice - V_Temp)
V_MaxOvershoot = MX(V_MaxOvershoot, V_CurrentPrice - V_FundamentalPrice)
```

### 7.4 VI 발동 시 (Assign_TriggerVI_Buy)
```
V_VIMode = 1 ; V_NumVI += 1 ; V_TotalHaltTime += 120
```

---

## 8. 매도 주문 처리 로직

매수와 대칭. 핵심 차이:

| 위치 | 매수 | 매도 |
|---|---|---|
| 적재 배열 | BidQty | **AskQty** |
| 탐색 함수 | MN | **MX** |
| 탐색 변수 | V_BestAsk | **V_BestBid** |
| 탐색 대상 | AskQty | **BidQty** |
| 미발견 기본값 | +9999 | **-9999** |
| 매칭 부등호 | A_Price < V_BestAsk | **A_Price > V_BestBid** |

```
V_BestBid = MX( (BidQty(1)>0)*80 + (BidQty(1)==0)*(-9999), ... ,
                (BidQty(51)>0)*130 + (BidQty(51)==0)*(-9999) )
```
체결은 `BidQty(V_BestBid-79)` 차감, `V_CurrentPrice = V_BestBid`로 갱신.

---

## 9. 단일가매매 로직 (개선판)

> **변경 핵심**: 기존엔 "단일가 = 최우선호가 중간값"을 먼저 정하고 그 가격에 **정확히 일치하는 주문만** 체결 → VI 정지 중 쌓인 교차 주문이 청산되지 않고 남는 문제가 있었다. 개정판은 **교차분을 먼저 전부 반복 체결**하고, 마지막 체결 한계가를 이용해 **단일가를 사후 결정**한다 (거래량 최대화 + 비교차 호가창 보장).

### 9.1 전체 흐름
```
Decide_VITrigger → Separate_VI (Duplicate 1)
   ├─ Original  → Assign_TriggerVI → Assign_Park → Dispose
   └─ Duplicate → Delay_VIHalt(120초)
                    → Assign_CallInit
                    → [LBL_CallMatch] → Assign_CallFindBest
                       → Decide_CallCrossed (V_BestBid >= V_BestAsk)
                          ├─True → Assign_CallExec → Go to LBL_CallMatch
                          └─False → Assign_ResetVI → Dispose_VIControl
```
> Label은 입력 포트가 없으므로 선을 넣지 않는다. `Assign_CallInit`(첫 진입)과 `LBL_CallMatch`(반복) 둘 다 `Assign_CallFindBest`로 연결 (매수 루프 `LBL_FindBestAsk_Buy`와 동일 배선).

### 9.2 Assign_CallInit
```
V_CallExecQty = 0
V_LastMatchedBid = -9999
V_LastMatchedAsk = 9999
```

### 9.3 Assign_CallFindBest (매 반복 최우선호가 재계산)
§7.2 / §8 의 51-인자 MN/MX 수식으로 `V_BestAsk`, `V_BestBid` 재계산.

### 9.4 Decide_CallCrossed
```
If: V_BestBid >= V_BestAsk   (True=교차 지속 / False=교차 해소 또는 한쪽 잔량 없음)
```
한쪽이 비면 9999/-9999가 들어와 자동으로 False → 별도 예외처리 불필요.

### 9.5 Assign_CallExec (최우선끼리 체결 + 마지막 한계가 기록)
```
V_Temp = MN( BidQty(V_BestBid-79), AskQty(V_BestAsk-79) )
BidQty(V_BestBid-79) -= V_Temp
AskQty(V_BestAsk-79) -= V_Temp
V_LastMatchedBid = V_BestBid
V_LastMatchedAsk = V_BestAsk
V_CallExecQty += V_Temp
V_TotalTradeVolume += V_Temp ; V_TotalTrades += 1 ; V_FilledOrders += V_Temp
```
→ 출력은 **Go to LBL_CallMatch**. 각 반복에서 최소 한 호가가 0이 되어 최대 51회 내 종료(무한루프 없음). 종료 시 `V_LastMatchedAsk ≤ V_LastMatchedBid`가 보장되어 그 사이 가격은 체결된 모든 주문에 적용 가능하다.

### 9.6 Assign_ResetVI (단일가 사후 결정 + 상태 복귀)
```
V_CallPrice = (V_CallExecQty > 0) * MAX(V_LastMatchedAsk, MIN(V_RefPrice, V_LastMatchedBid))
            + (V_CallExecQty == 0) * V_CurrentPrice
V_Temp = V_CurrentPrice
V_CurrentPrice = V_CallPrice
V_CumAbsPriceChange += ABS(V_CurrentPrice - V_Temp)
V_MaxOvershoot = MX(V_MaxOvershoot, V_CurrentPrice - V_FundamentalPrice)
V_RefPrice = V_CallPrice
V_VIMode = 0
```
- 기준가 `V_RefPrice`가 체결 가능 구간 안이면 그대로, 벗어나면 가장 가까운 경계 → **기준가 우선 동가 처리(실제 거래소 규칙)**
- 체결량 0이면 가격 불변

### 9.7 ARENA `!=` 우회
`A != B` → `1 - (A == B)`, `A==B && C==D` → `(A==B)*(C==D)`.

---

## 10. 시뮬레이션 종료 조건 및 실패 처리

### 10.1 감시(Monitor) 엔티티 흐름
```
Create_Monitor (1초, First Creation 1.0)
 → Assign_SampleMetrics (§11)
 → Decide_InBand
     ├─True → Decide_FirstEntry
     │          ├─True  → Assign_RecordEntry ─┐
     │          └─False → Decide_TimeUp        │
     │                      ├─True → Assign_StopMarket → Assign_FinalCalc → ReadWrite_Result → Dispose
     │                      └─False ───────────┤
     └─False → Assign_OutBand ─────────────────┤
                                               ↓
                                       Decide_FailCheck (TNOW==3599 && V_MarketStop==0)
                                         ├─True → Assign_FailFinal → Assign_FinalCalc → ReadWrite_Result → Dispose
                                         └─False → Dispose_Monitor
```

### 10.2 Decide_InBand (밴드 + 접속매매 동시 조건)
```
(V_CurrentPrice >= V_FundamentalPrice*(1 - V_BandPct/100)) &&
(V_CurrentPrice <= MN(130, V_FundamentalPrice*(1 + V_BandPct/100))) &&
(V_VIMode == 0)
```
> 베이스 밴드는 119~130원. **`V_VIMode==0`** 조건으로 VI 정지시간이 안정 판정에 끼지 않게 한다(오류 수정 §13.2).

### 10.3 Assign_RecordEntry (최초/재진입 시)
```
V_BandEntryTime = TNOW
V_StabilityActive = 1        ← 한 번 켜면 끝까지 유지(리셋 금지)
V_FirstDiscoveryTime = (V_FirstDiscoveryTime == -1) * TNOW
                     + (1 - (V_FirstDiscoveryTime == -1)) * V_FirstDiscoveryTime
```

### 10.4 Decide_TimeUp / Assign_StopMarket / Assign_OutBand
```
Decide_TimeUp:  (TNOW - V_BandEntryTime) >= V_DwellTime
Assign_StopMarket:  V_MarketStop = 1 ; V_EndTime = TNOW ; V_DiscoveryFail = 0
Assign_OutBand:  V_BandEntryTime = -1   (V_StabilityActive는 끄지 않음)
```

### 10.5 실패 복제 기록 (Decide_FailCheck / Assign_FailFinal)
성공하지 못한 모든 출구를 `Decide_FailCheck`로 모아, 마지막 초(3599)에 미발견이면 실패 한 줄을 기록한다.
```
Decide_FailCheck:  (TNOW == 3599) && (V_MarketStop == 0)
Assign_FailFinal:  V_DiscoveryFail = 1 ; V_EndTime = 3600
```
→ 이후 성공 경로와 **공유**하는 `Assign_FinalCalc → ReadWrite_Result`로 합류. 복제당 정확히 한 줄(성공 또는 실패)이 보장된다.
> `TNOW==3599`(정확히 한 번)로 해야 한다. `>=3599`로 하면 3599·3600 두 번 걸려 **2중 기록**(§13.8).

### 10.6 Assign_FinalCalc (성공·실패 공유)
```
V_UnfilledQty    = V_Ur                                       (= Σ BidQty + Σ AskQty 스냅샷)
V_StabilityScore = V_SumSqDev / MX(V_NSamples, 1)             (= S)
V_FrictionCost   = V_TWUnfilled / ( V_EndTime * MX(V_TotalOrders, 1) )   (= C)
```

### 10.7 Run Setup
```
Number of Replications: 14190
Replication Length: 3600
Time Units / Base Time Units: Seconds   ← 반드시 Seconds
Terminating Condition: V_MarketStop == 1
Initialize Between Replications - Statistics / System: ✓
```

---

## 11. 평가지표 수집 (S·D·C)

ARENA는 복제별 **원자료(S, D, C, 보조지표)**만 출력하고, 정규화·MCDM은 Python 후처리(§14)에서 수행한다.

- **D(가격발견 지연)** = `V_EndTime` (성공=조기종료 시각, 실패=3600). 별도 모듈 불필요.
- **S, C** = 매 1초 도는 Monitor에 **Assign_SampleMetrics**를 끼워 초당 누적.

### 11.1 Assign_SampleMetrics (Monitor 경로 맨 앞)
```
V_Ur = (BidQty(1)+...+BidQty(51)) + (AskQty(1)+...+AskQty(51))      (102개 항)
V_TWUnfilled += V_Ur
V_SumSqDev += V_StabilityActive * (1 - V_VIMode)
            * ( ((V_CurrentPrice - V_FundamentalPrice)/V_FundamentalPrice)
              * ((V_CurrentPrice - V_FundamentalPrice)/V_FundamentalPrice) )
V_NSamples += V_StabilityActive * (1 - V_VIMode)
```
- `V_TWUnfilled`는 VI 중에도 누적(거래마찰 = VI 대기비용 포함)
- S 누적은 `V_StabilityActive*(1-V_VIMode)` 게이트로 **최초 진입 후 + 접속매매 중**만
- 적정가가 시나리오마다 다르므로 편차는 **반드시 `V_FundamentalPrice` 기준**(하드코딩 125 금지, §13.9)

### 11.2 밴드 미진입 실패 처리
밴드에 한 번도 못 들어간 실패는 `V_NSamples=0` → `S=0`(완벽 안정으로 오인). 이는 `FirstDiscoveryTime == -1`로 식별 가능하므로 **후처리에서 worst값으로 치환**한다(§14).

---

## 12. 민감도 분석 자동 실험 설계

### 12.1 시나리오 룩업 + NREP 디코딩
`Assign_SetVIWidth`(복제당 t=0 1회 실행, Create_HeaderInit, Max Arrivals=1)에서:
```
V_Scn       = AINT( (NREP-1) / 330 ) + 1
V_RepInScn  = MOD( NREP-1, 330 )
V_VIWidth   = 5 + AINT( V_RepInScn / 30 )
V_BandPct          = ScnBandPct(V_Scn)
V_DwellTime        = ScnDwell(V_Scn)
V_FundamentalPrice = ScnFund(V_Scn)
```
- 시나리오 1개 = VIWidth 11종 × 30 복제 = 330 reps
- 43 시나리오 → Number of Replications = **14,190**
> 초기화 엔티티가 주문/감시보다 먼저 실행되도록 **Create_HeaderInit First Creation=0.0, 나머지=1.0**.

### 12.2 표적 혼합 설계 (43 시나리오)
베이스를 통과하는 세 개의 2D 슬라이스 (종료조건·적정가의 상호작용은 상한가 130을 통해 발생).

| 블록 | 고정 | 변동 | 수 |
|---|---|---|---|
| A. 종료조건 격자 | 적정가=125 | band%{3,4,5,6,7} × dwell{60,90,120,150,180} | 25 |
| B. band%×적정가 | dwell=120 | band%{3,4,5,6,7} × 적정가{115,120} | 10 |
| C. dwell×적정가 | band%=5 | dwell{60,90,150,180} × 적정가{115,120} | 8 |

**룩업 배열 초기값**(43개, 블록 A→B→C 순):
```
ScnBandPct: 3,3,3,3,3, 4,4,4,4,4, 5,5,5,5,5, 6,6,6,6,6, 7,7,7,7,7,  3,3,4,4,5,5,6,6,7,7,  5,5,5,5,5,5,5,5
ScnDwell:   60,90,120,150,180 (×5블록),  120(×10),  60,60,90,90,150,150,180,180
ScnFund:    125(×25),  115,120,115,120,115,120,115,120,115,120,  115,120,115,120,115,120,115,120
```
- 베이스(band5/dwell120/fund125) = **시나리오 13**
- 적정가 135는 상한가 초과로 무효 → 제외
> 검증용으로 `V_Scn`을 특정 값(예 13)으로 임시 고정해 단일 시나리오를 330회 돌려볼 수 있다.

### 12.3 출력 컬럼 (ReadWrite_Result / Header)
```
VIWidth NumVI TotalHaltTime TotalTrades TotalTradeVolume CumAbsPriceChange
MaxOvershoot FirstDiscoveryTime EndTime TotalOrders FilledOrders UnfilledQty
Scn BandPct DwellTime FundPrice StabilityScore FrictionCost DiscoveryFail
```
- File 모듈: Sequential File, Free Format, **Initialize Option=Hold**(이어쓰기), 경로 예 `C:\Users\strow\simulation\result.txt`
- 헤더는 `NREP==1`일 때 1회(Other 문자열), 결과는 복제 종료마다 1줄(Variable 값)
- 공통난수(CRN)는 단일 Run·연속 RNG라 엄밀히는 미적용 → R=30 평균으로 완화(한계로 명시)

---

## 13. 구현 중 만난 주요 이슈와 해결

### 13.1 스칼라 변수 Rows
스칼라/Attribute의 Rows를 1로 두면 배열로 인식 → **Rows 비우기**.

### 13.2 안정시간에 VI 정지가 집계됨 (오류 수정)
`Decide_InBand`가 가격만 검사 → VI 정지 120초가 안정으로 둔갑. **`&& V_VIMode==0`** 추가.

### 13.3 단일가 체결 누락 (오류 수정)
단일가 한 칸만 체결 → 교차 호가창 잔존. **교차분 전체 반복 청산 + 단일가 사후 결정**으로 교체(§9).

### 13.4 미체결량 과대계상 (오류 수정)
`TotalOrders − FilledOrders`는 체결량만큼 과대 → **호가창 배열 직접 합산**(`Σ BidQty + Σ AskQty`). FillRate도 `2×FilledOrders/TotalOrders`로 보정.

### 13.5 모듈 수 한도 / 인덱스 초과 / `<`,`!=` 미지원 / Time Units
- 31 Decide+31 Assign → **51-인자 MN/MX 압축 수식**
- 배열 31→**51 확장**(가격 80~130, 인덱스=가격−79)
- `<` → 누적 차분 수식, `!=` → `1-(==)`
- Base Time Units = **Seconds** 필수

### 13.6 Label 입력 포트 없음
Label은 점프 대상만. 일반 흐름은 본체로 직접 연결, Label 출력도 본체로 연결(본체가 입력 2개 수용).

### 13.7 적정가 출력 0 / 시나리오 미반영 (FundPrice 버그)
`ScnFund` 배열 초기값 미입력 → 출력 FundPrice=0, 적정가 스윕 무효 위험. **ScnFund 채우기 + `V_FundamentalPrice=ScnFund(V_Scn)` + 출력 컬럼을 V_FundamentalPrice로 매핑**.

### 13.8 실패 복제 2중 기록
`Decide_FailCheck`를 `TNOW>=3599`로 두면 3599·3600 두 번 기록 → **`TNOW==3599`**로 한 번만.

### 13.9 성공 복제가 실패로 라벨
`Assign_StopMarket`에 `V_DiscoveryFail=0` 누락 → 초기값 1이 잔존. **`V_DiscoveryFail=0` 추가**.

### 13.10 시나리오 변수 타이밍
`V_FundamentalPrice`는 주문 생성 첫 틱부터 읽힘 → 초기화 엔티티가 먼저 실행돼야. **Create_HeaderInit First Creation=0.0, 주문/감시=1.0**.

### 13.11 실패 복제 누락 → 편향
종료 못 한 복제는 한 줄도 안 써져 데이터의 다수가 증발. **Decide_FailCheck/Assign_FailFinal로 실패도 기록**(§10.5).

### 13.12 콘솔 인코딩 (분석 스크립트)
Windows cp949 콘솔에서 한글/em-dash 출력 크래시 → `sys.stdout.reconfigure(encoding="utf-8")`, CSV는 `utf-8-sig`.

---

## 14. 분석 파이프라인 (Python 후처리)

ARENA가 뽑은 `result.txt`(14,190행)를 받아 평가 체계 전 과정을 자동 계산한다.

### 14.1 `analyze_vi.py`
| 단계 | 내용 |
|---|---|
| 로드 + S-worst 치환 | `FirstDiscoveryTime==-1`(밴드 미진입)인 S를 worst로 |
| 집계 | (시나리오×VIWidth)별 평균·표준편차·95%CI·실패율 |
| 정규화 | Z_D=D/3600, Z_C=C, Z_S=S/0.005 (고정 ideal/worst) |
| 파레토 지배 제거 | 세 Z 모두 열등한 변동폭 제외 |
| Augmented Tchebycheff | `G = max(λZ) + ρ·Σ(λZ)`, ρ=0.001 (0.0001~0.01 불변 확인) |
| 가중치 민감도 | 4정책 + 전체 가중치공간(λ≥0.1, 0.05간격) → 1위비율·평균순위·최악순위 |

출력 CSV: `summary_by_scenario_viwidth.csv`, `best_viwidth_per_scenario.csv`, `weight_robustness_all.csv`, `policy_best_per_scenario.csv`

### 14.2 `make_report.py`
matplotlib(맑은 고딕)로 그래프 7종 생성 → reportlab으로 한국어 PDF 보고서(`VI_분석보고서.pdf`, 9페이지) 자동 작성.

---

## 15. 최종 실험 결과 분석

> 14,190 실행(43 시나리오 × 11 변동폭 × 30 복제)을 단일 Run으로 수집, 이상치 0건 확인.

### 15.1 베이스 시나리오 (밴드 ±5% · 머무름 120초 · 적정가 125)
- **최적 정적 VI 변동폭 = ±9%** (균형가중 Augmented Tchebycheff 최소)
- 전체 가중치 조합의 **58.3%에서 1위**, ρ 변화에도 불변 → 매우 강건
- 차순위 **±10%** → 두 값이 유력 후보군 (현행 ±10%가 합리적 범위임을 시사)

### 15.2 발견 밴드 절벽
| 밴드 ±% | 가격발견 실패율(적정가125) |
|---|---|
| 3% | **78%** |
| 4% | 0.4% |
| 5~7% | 0% |

발견 밴드를 ±3%로 좁히면 발견이 사실상 불가능. **평가 기준 밴드는 ±4% 이상**이어야 의미 있는 측정이 가능(베이스 ±5%는 안전 구간).

### 15.3 민감도 — 무엇에 강건하고 무엇에 민감한가
- ✅ **종료조건(밴드 폭·머무름)에는 강건**: 적정가 125 고정 시 최적이 ±9~10%로 안정
- ⚠️ **적정가에는 민감**: 적정가 115/120으로 바꾸면 최적이 ±11~15%로 이동(상한가 캡과의 상호작용)

→ "단일 최적값"보다 **시장 상황(적정가의 상한가 대비 위치)에 따른 적정 범위**로 해석하는 것이 타당.

### 15.4 최종 변동폭 선택 규칙 (요약)
파레토 비지배 → 동일가중 종합점수 → 가중치 강건성(1위비율·평균/최악순위) → 통계적 불확실성(95%CI) → 동률 시 (거래마찰↓ → 실패율↓ → 더 큰 변동폭) 순으로 판단.

---

## 16. 모델링 방법 요약

1. **상태의존적 포아송 주문 생성** — gap에 따른 통과확률 piecewise 부여 (누적 차분 수식으로 `<` 회피)
2. **호가창 배열 + 압축 탐색** — 51-인자 MN/MX 단일 Assign으로 BestBid/BestAsk 탐색 (모듈 수 절감)
3. **Label/Go-to 반복 체결** — 명시적 반복문 없는 ARENA에서 부분체결·교차청산 루프 구현
4. **단일가매매 = 교차 전체 청산 + 사후 단일가 결정** — 거래량 최대화·비교차 보장·기준가 우선
5. **Separate 기반 VI 제어 분리** — 120초 Delay 후 단일가매매(시간지연·상태전이 분리)
6. **메타 감시 엔티티 + 초당 지표 샘플러** — 종료조건 점검과 동시에 S/C 원자료 누적, 실패 복제도 기록
7. **시나리오 룩업 + NREP 디코딩 자동 실험** — 단일 Run으로 43×11×30=14,190 데이터 생성
8. **Python MCDM 파이프라인** — 정규화→파레토→Augmented Tchebycheff→가중치 민감도→PDF 보고서 자동화

---

## 17. 부록: 시장 매칭 원리

### 17.1 핵심
- 매수 주문가 = 지불 가능한 **최대** 가격, 매도 주문가 = 받고 싶은 **최소** 가격
- 체결 조건: 매수자 최대 의향가 ≥ 매도자 최소 의향가

### 17.2 예시
```
매도호가  110원:5주 / 108원:3주(BestAsk)
매수호가  105원:2주(BestBid) / 103원:4주
```
| 매수 주문가 | 결과 | 체결가 |
|---|---|---|
| 109 | ✅ | **108** (매수자에게 유리) |
| 108 | ✅ | 108 |
| 107 | ❌ | BidQty[107] 적재 |

### 17.3 본 모델
연속매매: `A_Price ≥ V_BestAsk` → 체결가 = V_BestAsk (`V_CurrentPrice = V_BestAsk`).
단일가매매: 교차분을 최우선끼리 모두 청산한 뒤 `[V_LastMatchedAsk, V_LastMatchedBid]` 구간에서 기준가 우선으로 단일가 결정.

---

## 작성 정보

- **개정일**: 2026년 6월
- **사용 도구**: ARENA Student Edition + Python(pandas 2.3 / matplotlib / reportlab)
- **참고 논문**: 안일찬 외 (2017). KRX 정적 VI 도입의 가격안정화 및 가격발견 효과. 재무연구, 30(2), 103-142.
- **평가 체계**: vi_evaluation_framework (Wierzbicki 1980; Steuer & Choo 1983; Triantaphyllou & Sanchez 1997)
- **실험 데이터**: 43 시나리오 × 11 변동폭 × 30 복제 = **14,190 data points**
- **분석 산출물**: analyze_vi.py, make_report.py, VI_분석보고서.pdf, analysis/*.csv

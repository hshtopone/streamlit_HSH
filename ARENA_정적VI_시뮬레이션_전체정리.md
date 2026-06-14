# ARENA 정적 VI 시뮬레이션 — 전체 구현 가이드

> **프로젝트**: 정적 VI(Volatility Interruption) 발동 기준 변동폭의 최적값 탐색
> **기반 연구**: 안일찬 외 (2017). KRX 정적 VI 도입의 가격안정화 및 가격발견 효과
> **도구**: ARENA 시뮬레이션 (학생용)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [연구 배경 및 동기](#2-연구-배경-및-동기)
3. [평가 지표 설계](#3-평가-지표-설계)
4. [ARENA 변수 설계](#4-arena-변수-설계)
5. [주문 생성 모델](#5-주문-생성-모델)
6. [주문 가격·수량 부여](#6-주문-가격수량-부여)
7. [매수 주문 처리 로직 (핵심)](#7-매수-주문-처리-로직-핵심)
8. [매도 주문 처리 로직](#8-매도-주문-처리-로직)
9. [단일가매매 로직 (VI 제어)](#9-단일가매매-로직-vi-제어)
10. [시뮬레이션 종료 조건](#10-시뮬레이션-종료-조건)
11. [결과 기록 및 자동 실험](#11-결과-기록-및-자동-실험)
12. [구현 중 만난 주요 이슈와 해결](#12-구현-중-만난-주요-이슈와-해결)
13. [예비 실험 결과 분석](#13-예비-실험-결과-분석)
14. [모델링 방법 요약 (5가지)](#14-모델링-방법-요약-5가지)
15. [발표 자료 검토 사항](#15-발표-자료-검토-사항)
16. [부록: 시장 매칭 원리](#16-부록-시장-매칭-원리)

---

## 1. 프로젝트 개요

### 연구 목적
정적 VI 발동 기준 변동폭(%)을 5~15 범위에서 변화시키며 시뮬레이션을 수행하여, 가격안정성·가격발견 효과·유동성의 세 가지 지표를 종합적으로 고려한 최적값을 찾는다.

### 기본 가정
- 짧은 기간의 주가 변화 시뮬레이션 → 적정가격은 고정
- 적정가격 > 시가 (상승 국면만 다룸)
- 적정가격은 상한가(시가의 130%) 초과하지 않음
- 시가 100원, 적정가격 125원, 호가 단위 1원
- 동적 VI 및 임의종료(RE) 제도는 무시
- VI 발동 시 정확히 2분간 거래 정지

### 종료 조건
주가가 적정가격 ±5% 범위(119~131원)에서 2분간 벗어나지 않으면 시뮬레이션 종료

---

## 2. 연구 배경 및 동기

### KRX 정적 VI 제도
- 2015년 6월 15일 도입
- 직전 단일가매매 체결가 대비 ±10% 변동 예상 시 발동
- 2분간 거래 정지 후 단일가매매로 재개
- 동시에 가격제한폭 ±15% → ±30%로 확대

### 선행 연구 (안일찬 외, 2017)의 핵심 발견
1. **정적 VI의 가격안정화 효과가 절대적으로 낮음**
2. **상한가 발생일의 88% 이상에서 정적 VI 동시 발동** → 가격제한폭과 중복 기능
3. **가격제한폭 ±30% 확대로 실현 변동성 14~15% 증가**
4. **결론**: "여러 모수(예: ±10% 정적 VI 변동폭)에 대한 심도 깊은 검증이 필요"

→ 본 연구는 이 후속 연구 요청에 직접 응답

---

## 3. 평가 지표 설계

### 3대 평가 항목

| 분류 | 지표 | 의미 | 방향 |
|---|---|---|---|
| **가격안정성** | CumAbsPriceChange | 누적 절대 가격변화 | ↓ |
| **가격안정성** | MaxOvershoot | 적정가 초과 최대폭 | ↓ |
| **가격발견** | FirstDiscoveryTime | 첫 적정가 범위 진입 시점 | ↓ |
| **가격발견** | EndTime | 시뮬레이션 총 소요 시간 | ↓ |
| **유동성** | TotalTrades | 총 체결 건수 | - |
| **유동성** | TotalTradeVolume | 총 체결 수량 | - |
| **유동성** | NumVI | VI 발동 횟수 | ↓ |
| **유동성** | TotalHaltTime | 총 거래정지 시간 | ↓ |
| **유동성** | TotalOrders | 총 생성 주문수량 | - |
| **유동성** | FilledOrders | 체결된 주문수량 | - |
| **유동성** | UnfilledQty | 미체결 잔량 | ↓ |
| **유동성** | FillRate (= FilledOrders/TotalOrders) | 체결률 | ↑ |

---

## 4. ARENA 변수 설계

### 4.1 가격 및 제도 상태 변수 (스칼라)

| 이름 | 의미 | 초기값 |
|---|---|---|
| V_CurrentPrice | 현재 주가 | 100 |
| V_FundamentalPrice | 적정가격 | 125 |
| V_RefPrice | 현재 정적 VI 기준가격 | 100 |
| V_VIWidth | 실험에서 설정한 정적 VI 폭 | 5~15 (변동) |
| V_VIMode | VI 상태 (0=연속매매, 1=거래정지) | 0 |
| V_MarketStop | 시뮬레이션 종료 플래그 (0=계속, 1=종료) | 0 |

**중요**: ARENA에서 스칼라 변수는 **Rows를 비워두어야 함**. Rows=1로 두면 1차원 배열로 인식되어 dimension 에러 발생.

### 4.2 주문장(호가창) 배열 변수

```
BidQty[51]: 가격 80~130원의 매수잔량 (인덱스 1~51)
AskQty[51]: 가격 80~130원의 매도잔량 (인덱스 1~51)
```

**인덱스 매핑 공식**: `인덱스 = A_Price - 79`
- 예) 가격 100원 → 인덱스 21
- 예) 가격 125원 → 인덱스 46

**범위를 80~130으로 잡은 이유**:
- 초기 V_CurrentPrice=100 상태에서 매수 offset -2가 나오면 A_Price=98 (인덱스 19)
- 안전 마진을 위해 80(인덱스 1)까지 확장
- 매도 offset이 항상 0 이상이므로 V_CurrentPrice가 100 이하로 떨어지지 않음 → 80~130으로 충분

### 4.3 단일가매매 계산용 임시 변수

| 이름 | 초기값 |
|---|---|
| V_CallPrice | 0 |
| V_CallExecQty | 0 |
| V_BestAsk | 0 |
| V_BestBid | 0 |
| V_RemainQty | 0 |
| V_Temp | 0 |

### 4.4 성과지표용 변수

| 이름 | 의미 | 초기값 |
|---|---|---|
| V_NumVI | VI 발동 횟수 | 0 |
| V_TotalHaltTime | 총 거래정지 시간 | 0 |
| V_TotalTradeVolume | 총 체결수량 | 0 |
| V_TotalTrades | 총 체결건수 | 0 |
| V_CumAbsPriceChange | 누적 절대 가격변화 | 0 |
| V_MaxOvershoot | 적정가 초과 최대폭 | 0 |
| V_FirstDiscoveryTime | 첫 적정가 진입 시점 | **-1** |
| V_EndTime | 종료 시점 | 0 |
| V_InBandFlag | 현재 적정가 범위 안에 있는지 | 0 |
| V_BandEntryTime | 적정가 범위 진입 시점 | **-1** |
| V_TotalOrders | 총 생성 주문수량 | 0 |
| V_FilledOrders | 체결된 주문수량 | 0 |
| V_UnfilledQty | 종료 시 미체결 잔량 | 0 |

**-1 초기값의 의미**: "아직 발생하지 않았음"을 표현. TNOW=0과 구분하기 위함.

### 4.5 주문 Attribute (Entity별)

| 이름 | 의미 |
|---|---|
| A_Side | 매수=1, 매도=-1 |
| A_Price | 주문가격 |
| A_Qty | 주문수량 |
| A_ArrivalTime | 주문 도착 시각 |
| A_RemainQty | 미체결 잔량 |
| A_GenProb | 주문 통과 확률 |

---

## 5. 주문 생성 모델

### 5.1 핵심 아이디어 — 상태의존적 포아송 도착

매 1초마다 잠재 주문 entity를 생성하고, **현재 시장 상황(gap)에 따라 통과 확률을 다르게 부여**하여 piecewise 포아송 프로세스를 구현.

```
gap = F - P_t = 적정가격 - 현재가격
```

### 5.2 매수 도착률 (gap이 클수록 자주)

| gap 범위 | 평균 도착간격 | 통과확률 (= 1/간격) |
|---|---|---|
| ≥ 20 | EXPO(1.5) | 1/1.5 ≈ 0.667 |
| 15~19 | EXPO(2.0) | 1/2.0 = 0.500 |
| 10~14 | EXPO(2.8) | 1/2.8 ≈ 0.357 |
| 5~9 | EXPO(3.8) | 1/3.8 ≈ 0.263 |
| 0~4 | EXPO(5.0) | 1/5.0 = 0.200 |

### 5.3 매도 도착률 (gap이 작을수록 자주)

| gap 범위 | 평균 도착간격 | 통과확률 |
|---|---|---|
| ≥ 20 | EXPO(4.5) | 1/4.5 ≈ 0.222 |
| 15~19 | EXPO(4.0) | 1/4.0 = 0.250 |
| 10~14 | EXPO(3.6) | 1/3.6 ≈ 0.278 |
| 5~9 | EXPO(3.3) | 1/3.3 ≈ 0.303 |
| 0~4 | EXPO(3.0) | 1/3.0 ≈ 0.333 |

### 5.4 ARENA 구현 — 매수 A_GenProb 수식 (개선판)

ARENA의 `<` 연산자 문제를 우회하기 위해 **누적 차분 방식** 사용:

```
(V_FundamentalPrice - V_CurrentPrice >= 20) * (1/1.5 - 1/2.0) +
(V_FundamentalPrice - V_CurrentPrice >= 15) * (1/2.0 - 1/2.8) +
(V_FundamentalPrice - V_CurrentPrice >= 10) * (1/2.8 - 1/3.8) +
(V_FundamentalPrice - V_CurrentPrice >= 5)  * (1/3.8 - 1/5.0) +
(V_FundamentalPrice - V_CurrentPrice >= 0)  * (1/5.0)
```

**원리**: gap이 클수록 더 많은 항이 활성화되어 누적 → 정확한 값에 도달

**검증**:
- gap=25 (≥20): 모든 항 활성 → 1/1.5 ✅
- gap=17 (≥15): 위 4개 → 1/2.0 ✅
- gap=12 (≥10): 위 3개 → 1/2.8 ✅
- gap=7 (≥5): 위 2개 → 1/3.8 ✅
- gap=2 (≥0): 마지막 1개 → 1/5.0 ✅

### 5.5 매도 A_GenProb 수식

```
(V_FundamentalPrice - V_CurrentPrice >= 20) * (1/4.5 - 1/4.0) +
(V_FundamentalPrice - V_CurrentPrice >= 15) * (1/4.0 - 1/3.6) +
(V_FundamentalPrice - V_CurrentPrice >= 10) * (1/3.6 - 1/3.3) +
(V_FundamentalPrice - V_CurrentPrice >= 5)  * (1/3.3 - 1/3.0) +
(V_FundamentalPrice - V_CurrentPrice >= 0)  * (1/3.0)
```

**주의**: 매도는 차이값이 음수가 됨 (gap 클수록 매도 통과확률 감소).

### 5.6 ARENA 모듈 구성

```
[Create_BuyOrder] ──→ [Assign_BuyRate (A_GenProb 계산)] ──→ [Decide_BuyFilter]
   1초마다                                                    True/False
                                                                ↓ False
                                                           [Dispose_Buy]

[Create_SellOrder] ──→ [Assign_SellRate] ──→ [Decide_SellFilter] ──→ ...
```

**Create 모듈 설정**:
- Type: Constant
- Value: 1
- Units: Seconds
- Max Arrivals: Infinite

**Decide 모듈 설정**:
- Type: 2-way by Chance
- Percent True: `A_GenProb * 100` (×100 필수!)

---

## 6. 주문 가격·수량 부여

### 6.1 주문 수량 분포

```
A_Qty = DISC(0.5, 1, 0.8, 2, 1.0, 3)
```

- 1주: 50%, 2주: 30%, 3주: 20%

**주의**: `A_Qty`와 `A_RemainQty`에 같은 값이 들어가야 하므로:
```
A_Qty = DISC(0.5, 1, 0.8, 2, 1.0, 3)
A_RemainQty = A_Qty  ← 같은 값 복사
```

### 6.2 매수 주문 가격 분포 (gap별)

#### gap ≥ 15 (공격적 매수)
| Offset | 확률 |
|---|---|
| +2 | 0.35 |
| +1 | 0.40 |
| 0 | 0.20 |
| -1 | 0.05 |

```
A_Price = V_CurrentPrice + DISC(0.35, 2, 0.75, 1, 0.95, 0, 1.0, -1)
```

#### 5 ≤ gap < 15
| Offset | 확률 |
|---|---|
| +1 | 0.35 |
| 0 | 0.40 |
| -1 | 0.20 |
| -2 | 0.05 |

```
A_Price = V_CurrentPrice + DISC(0.35, 1, 0.75, 0, 0.95, -1, 1.0, -2)
```

#### gap < 5
| Offset | 확률 |
|---|---|
| +1 | 0.20 |
| 0 | 0.40 |
| -1 | 0.30 |
| -2 | 0.10 |

```
A_Price = V_CurrentPrice + DISC(0.20, 1, 0.60, 0, 0.90, -1, 1.0, -2)
```

### 6.3 매도 주문 가격 분포 (gap별)

#### gap ≥ 15 (덜 공격적)
| Offset | 확률 |
|---|---|
| 0 | 0.10 |
| +1 | 0.35 |
| +2 | 0.35 |
| +3 | 0.20 |

```
A_Price = V_CurrentPrice + DISC(0.10, 0, 0.45, 1, 0.80, 2, 1.0, 3)
```

#### 5 ≤ gap < 15
```
A_Price = V_CurrentPrice + DISC(0.20, 0, 0.60, 1, 0.90, 2, 1.0, 3)
```

#### gap < 5
```
A_Price = V_CurrentPrice + DISC(0.30, 0, 0.70, 1, 0.90, 2, 1.0, 3)
```

**매도자의 행동 가정**: 매도 offset이 항상 0 이상 → 현재가 미만으로는 절대 팔지 않음 (상승 국면 가정과 부합).

### 6.4 흐름도

```
[Decide_BuyFilter] ──True──→ [Assign_BuySide (A_Side=1, A_Qty, A_RemainQty, A_ArrivalTime)]
                                  ↓
                              [Decide_BuyGap]
                              ├─(gap≥15)──→ [Assign_BuyPrice_Gap15plus]
                              ├─(5≤gap<15)─→ [Assign_BuyPrice_Gap5to15]
                              └─(else)─────→ [Assign_BuyPrice_GapLess5]
                                                ↓ (모두 수렴)
                                            (다음 단계로)
```

**V_TotalOrders 카운트**: Assign_BuySide와 Assign_SellSide에 추가
```
V_TotalOrders = V_TotalOrders + A_Qty
```

### 6.5 가격 클램핑 (안전장치)

각 가격 Assign의 마지막에 추가:
```
A_Price = MX(80, MN(130, A_Price))
```

---

## 7. 매수 주문 처리 로직 (핵심)

### 7.1 전체 흐름도

```
(매수 가격 부여 완료)
    ↓
[Decide_VICheck_Buy]──True (VI중)──→ [Assign_ParkBuy]──→[Dispose_BuyParked]
    ↓ False
[Label: LBL_FindBestAsk_Buy] ←──────────────────────────┐
    ↓                                                     │
[Assign_FindBestAsk] (51개 인자 압축 수식)                 │
    ↓                                                     │
[Decide_PriceMatch_Buy]──True (불가)──→ [Assign_ParkBuy] │
    ↓ False (체결 가능)                                    │
[Decide_VITrigger_Buy]──True (VI 발동)──→ [Separate]    │
    ↓ False                              ├─원본→ [Assign_TriggerVI_Buy]→[Assign_ParkBuy]→[Dispose]
[Assign_ExecuteTrade_Buy]                └─복제→ [VI 제어 라인]
    ↓
[Decide_RemainCheck_Buy]──True (잔량)──→ [Go to LBL_FindBestAsk_Buy]
    ↓ False                                       (위로 점프)
[Dispose_BuyDone]
```

### 7.2 Step 1: VI 상태 확인

```
Decide_VICheck_Buy 설정:
  Type: 2-way by Condition
  If: Variable V_VIMode == 1
```

- **True (VI 거래정지 중)**: 체결 시도하지 않고 BidQty에 적재 후 소멸
- **False (연속매매 중)**: 다음 단계 진행

### 7.3 Step 2: Assign_ParkBuy (주문장 적재)

```
Type: Variable Array (1D)
Variable Name: BidQty
Row: A_Price - 79
New Value: BidQty(A_Price - 79) + A_RemainQty
```

→ `BidQty[A_Price] += A_RemainQty`

### 7.4 Step 3: BestAsk 압축 탐색 수식 (51개 인자)

**Assign_FindBestAsk**:
```
V_BestAsk = MN(
  (AskQty(1)>0)*80  + (AskQty(1)==0)*9999,
  (AskQty(2)>0)*81  + (AskQty(2)==0)*9999,
  (AskQty(3)>0)*82  + (AskQty(3)==0)*9999,
  ...
  (AskQty(51)>0)*130 + (AskQty(51)==0)*9999
)
```

**원리**:
- 잔량 있는 가격 → 가격값 반환
- 잔량 없으면 → 9999 (절대 못 이길 큰 값) 반환
- MN으로 최솟값 → **잔량 있는 가격 중 가장 낮은 값**(=BestAsk)
- 매도잔량이 모두 0이면 → 9999 반환 (= "매도잔량 없음" 표시)

### 7.5 Step 4: 가격 매칭 검사

```
Decide_PriceMatch_Buy 설정:
  Type: 2-way by Condition
  If: Expression A_Price < V_BestAsk
```

- **True (체결 불가)**: 주문가가 BestAsk보다 낮음 → BidQty 적재
- **False (체결 가능)**: A_Price ≥ V_BestAsk → 다음 단계 진행
- V_BestAsk=9999면 자동으로 True (체결 불가)

### 7.6 Step 5: VI 발동 검사

```
Decide_VITrigger_Buy 설정:
  Type: 2-way by Condition
  If: Expression V_BestAsk >= V_RefPrice * (1 + V_VIWidth/100)
```

- **True (VI 발동)**: Separate로 분리 → 원본은 Park&Dispose, 복제는 VI 제어 라인으로
- **False (정상)**: 체결 진행

### 7.7 Step 6: 실제 체결 (Assign_ExecuteTrade_Buy)

| 순서 | Type | 변수/속성 | 값 |
|---|---|---|---|
| 1 | Variable | V_Temp | MN(A_RemainQty, AskQty(V_BestAsk - 79)) |
| 2 | Variable Array (1D) | AskQty | AskQty(V_BestAsk - 79) - V_Temp |
| 3 | Attribute | A_RemainQty | A_RemainQty - V_Temp |
| 4 | Variable | V_TotalTradeVolume | V_TotalTradeVolume + V_Temp |
| 5 | Variable | V_TotalTrades | V_TotalTrades + 1 |
| 6 | Variable | V_FilledOrders | V_FilledOrders + V_Temp |
| 7 | Variable | V_Temp | V_CurrentPrice (옛 가격 저장) |
| 8 | Variable | V_CurrentPrice | V_BestAsk (현재가 갱신) |
| 9 | Variable | V_CumAbsPriceChange | V_CumAbsPriceChange + ABS(V_CurrentPrice - V_Temp) |
| 10 | Variable | V_MaxOvershoot | MX(V_MaxOvershoot, V_CurrentPrice - V_FundamentalPrice) |

**중요**: ARENA Assign은 위에서 아래로 순차 실행되므로 순서 매우 중요.

### 7.8 Step 7: 잔량 체크 및 반복

```
Decide_RemainCheck_Buy 설정:
  If: Expression A_RemainQty > 0
```

- **True (잔량 있음)**: Go to Label `LBL_FindBestAsk_Buy` → 다음 호가와 추가 매칭
- **False (잔량 없음)**: Dispose_BuyDone

### 7.9 Assign_TriggerVI_Buy (VI 발동 시)

```
V_VIMode = 1
V_NumVI = V_NumVI + 1
V_TotalHaltTime = V_TotalHaltTime + 120
```

---

## 8. 매도 주문 처리 로직

### 8.1 매수와의 핵심 차이점

| 위치 | 매수 | 매도 |
|---|---|---|
| Park 시 적재 배열 | BidQty | **AskQty** |
| 탐색 함수 | MN | **MX** |
| 탐색 변수 | V_BestAsk | **V_BestBid** |
| 탐색 대상 호가 배열 | AskQty | **BidQty** |
| 안 잡힐 때 기본값 | +9999 | **-9999** |
| 가격 매칭 부등호 | A_Price < V_BestAsk | **A_Price > V_BestBid** |
| 체결 시 차감 배열 | AskQty | **BidQty** |
| Label 이름 | LBL_FindBestAsk_Buy | **LBL_FindBestBid_Sell** |

### 8.2 BestBid 압축 수식 (51개 인자)

```
V_BestBid = MX(
  (BidQty(1)>0)*80  + (BidQty(1)==0)*(-9999),
  (BidQty(2)>0)*81  + (BidQty(2)==0)*(-9999),
  ...
  (BidQty(51)>0)*130 + (BidQty(51)==0)*(-9999)
)
```

**원리**: 매수잔량 있는 가격 → 가격값, 없으면 -9999. MX로 최댓값 → 가장 높은 매수호가.

### 8.3 Assign_ExecuteTrade_Sell

매수와 거의 같지만 **AskQty → BidQty**, **V_BestAsk → V_BestBid** 로 변경:

```
V_Temp = MN(A_RemainQty, BidQty(V_BestBid - 79))
BidQty(V_BestBid - 79) = BidQty(V_BestBid - 79) - V_Temp
A_RemainQty = A_RemainQty - V_Temp
V_TotalTradeVolume = V_TotalTradeVolume + V_Temp
V_TotalTrades = V_TotalTrades + 1
V_FilledOrders = V_FilledOrders + V_Temp
V_Temp = V_CurrentPrice
V_CurrentPrice = V_BestBid
V_CumAbsPriceChange = V_CumAbsPriceChange + ABS(V_CurrentPrice - V_Temp)
V_MaxOvershoot = MX(V_MaxOvershoot, V_CurrentPrice - V_FundamentalPrice)
```

---

## 9. 단일가매매 로직 (VI 제어)

### 9.1 전체 흐름

```
[Decide_VITrigger_Buy/Sell] ──True──→ [Separate_VI]
                                          ├─Original─→ [Assign_TriggerVI] → [Assign_Park] → [Dispose]
                                          └─Duplicate(1)─┐
                                                         ↓
                                    [Delay_VIHalt: 120초]
                                          ↓
                                    [Assign_CalcCallPrice] (단일가 계산)
                                          ↓
                                    [Assign_ExecuteCallAuction] (단일가 체결)
                                          ↓
                                    [Assign_ResetVI] (V_VIMode=0, V_RefPrice 갱신)
                                          ↓
                                    [Dispose_VIControl]
```

### 9.2 Separate 모듈

```
Separate_VI_Buy/Sell 설정:
  Type: Duplicate Original
  # of Duplicates: 1
```

- **Original 출력**: 기존 흐름 (Park & Dispose)
- **Duplicate 출력**: VI 제어 라인 (Delay → 단일가매매)

### 9.3 Delay 모듈 (Advanced Process 패널)

```
Delay_VIHalt 설정:
  Allocation: Other
  Delay Time: 120
  Units: Seconds
```

### 9.4 Assign_CalcCallPrice — 단일가 계산 (접근 B: 중간값)

**1단계**: V_BestAsk, V_BestBid 재계산 (51개 인자 압축 수식)

**2단계**: V_CallPrice 계산 (예외 처리 포함)

```
V_CallPrice = 
  (V_BestAsk == 9999) * (1 - (V_BestBid == -9999)) * V_BestBid +
  (1 - (V_BestAsk == 9999)) * (V_BestBid == -9999) * V_BestAsk +
  (1 - (V_BestAsk == 9999)) * (1 - (V_BestBid == -9999)) * AINT((V_BestBid + V_BestAsk) / 2 + 0.5) +
  (V_BestAsk == 9999) * (V_BestBid == -9999) * V_CurrentPrice
```

**케이스별 검증**:

| Case | V_BestAsk | V_BestBid | V_CallPrice |
|---|---|---|---|
| 둘 다 정상 (110/108) | 110 | 108 | 109 ✅ |
| BestAsk만 비정상 | 9999 | 108 | 108 ✅ |
| BestBid만 비정상 | 110 | -9999 | 110 ✅ |
| 둘 다 비정상 | 9999 | -9999 | V_CurrentPrice ✅ |

**3단계**: 안전망 클램핑
```
V_CallPrice = MX(80, MN(130, V_CallPrice))
```

### 9.5 ARENA의 `!=` 우회법

ARENA가 `!=` 연산자를 지원하지 않는 경우 `1 - ==` 로 우회:

| 원하는 연산 | 우회법 |
|---|---|
| `A != B` | `1 - (A == B)` |
| `A != B` (대안) | `(A < B) + (A > B)` |
| `A == B && C == D` | `(A == B) * (C == D)` |
| `A == B \|\| C == D` | `(A == B) + (C == D) - (A == B) * (C == D)` |

### 9.6 Assign_ExecuteCallAuction — 단일가 체결

단순화: V_CallPrice에서 BidQty와 AskQty의 같은 가격대만 매칭

```
V_Temp = MN(BidQty(V_CallPrice - 79), AskQty(V_CallPrice - 79))
BidQty(V_CallPrice - 79) = BidQty(V_CallPrice - 79) - V_Temp
AskQty(V_CallPrice - 79) = AskQty(V_CallPrice - 79) - V_Temp
V_TotalTradeVolume = V_TotalTradeVolume + V_Temp
V_TotalTrades = V_TotalTrades + (V_Temp > 0)
V_FilledOrders = V_FilledOrders + V_Temp
```

### 9.7 Assign_ResetVI

```
V_Temp = V_CurrentPrice
V_CurrentPrice = V_CallPrice
V_RefPrice = V_CallPrice  ← VI 기준가 갱신
V_CumAbsPriceChange = V_CumAbsPriceChange + ABS(V_CurrentPrice - V_Temp)
V_MaxOvershoot = MX(V_MaxOvershoot, V_CurrentPrice - V_FundamentalPrice)
V_VIMode = 0  ← 연속매매 복귀
```

---

## 10. 시뮬레이션 종료 조건

### 10.1 감시 entity 흐름

```
[Create_Monitor] (1초마다)
    ↓
[Decide_InBand] (현재가가 119~131 범위?)
    ↓True                              ↓False
[Decide_FirstEntry]                [Assign_OutBand]
    ↓True (처음)   ↓False (계속)        ↓
[Assign_RecordEntry]  [Decide_TimeUp]  [Dispose_Monitor]
    ↓             ↓True       ↓False
    │       [Assign_StopMarket]   │
    │             ↓               │
    └──→ [Record들 → ReadWrite_Result → Dispose_Monitor]
```

### 10.2 Create_Monitor 설정

```
Type: Constant
Value: 1
Units: Seconds
Max Arrivals: Infinite
First Creation: 1.0
```

### 10.3 Decide_InBand

```
If: Expression
Value: (V_CurrentPrice >= 119) && (V_CurrentPrice <= 131)
```

### 10.4 Assign_RecordEntry (처음 진입 시)

```
V_BandEntryTime = TNOW
V_InBandFlag = 1
V_FirstDiscoveryTime = (V_FirstDiscoveryTime == -1) * TNOW + 
                       (1 - (V_FirstDiscoveryTime == -1)) * V_FirstDiscoveryTime
```

→ 처음 진입이면 TNOW 기록, 이미 기록되어 있으면 그대로 유지

### 10.5 Decide_TimeUp

```
If: Expression
Value: (TNOW - V_BandEntryTime) >= 120
```

### 10.6 Assign_StopMarket

```
V_MarketStop = 1
V_EndTime = TNOW
```

### 10.7 Assign_OutBand (범위 벗어남)

```
V_BandEntryTime = -1
V_InBandFlag = 0
```

### 10.8 Run Setup

```
Number of Replications: 11 (또는 330)
Replication Length: 3600 (안전장치)
Time Units: Seconds
Base Time Units: Seconds  ← 반드시 Seconds!
Terminating Condition: V_MarketStop == 1
Initialize Between Replications - Statistics: ✓
Initialize Between Replications - System: ✓
```

⚠️ **Base Time Units가 Hours로 되어 있으면 시뮬레이션이 안 돌아감!**

---

## 11. 결과 기록 및 자동 실험

### 11.1 File 모듈 설정

```
Name: OutFile_Results
Access Type: Sequential File
Operating System File Name: C:\path\to\result.txt
Structure: Free Format
End of File Action: Dispose
Initialize Option: Hold  ← 반드시 Hold (이어쓰기)
```

### 11.2 헤더 쓰기 (시뮬레이션 시작 시 1회)

```
[Create_HeaderInit] (Max Arrivals=1)
    ↓
[Assign_SetVIWidth] (V_VIWidth = 4 + NREP)
    ↓
[Decide_FirstRep] (NREP == 1?)
    ↓True            ↓False
[ReadWrite_Header]  [Dispose]
    ↓
[Dispose]
```

**ReadWrite_Header 내용**: `"VIWidth", "NumVI", "TotalHaltTime", ...` (문자열)

### 11.3 결과 쓰기 (시뮬레이션 종료 시)

```
[Assign_StopMarket] → [Assign_FinalCalc] → [11개 Record] → [ReadWrite_Result] → [Dispose_Monitor]
```

**Assign_FinalCalc**:
```
V_UnfilledQty = V_TotalOrders - V_FilledOrders
```

**ReadWrite_Result에 기록할 변수들** (헤더 순서와 동일):
```
V_VIWidth, V_NumVI, V_TotalHaltTime, V_TotalTrades, 
V_TotalTradeVolume, V_CumAbsPriceChange, V_MaxOvershoot,
V_FirstDiscoveryTime, V_EndTime, V_TotalOrders, V_FilledOrders, V_UnfilledQty
```

### 11.4 자동 실험 설계

**V_VIWidth 자동 변경**:
```
V_VIWidth = 4 + NREP
```

| NREP | V_VIWidth |
|---|---|
| 1 | 5 |
| 2 | 6 |
| ... | ... |
| 11 | 15 |

**또는 V_VIWidth 1값당 30회씩**:
```
V_VIWidth = 4 + AINT((NREP - 1) / 30) + 1
```

→ Number of Replications = 330

---

## 12. 구현 중 만난 주요 이슈와 해결

### 12.1 ARENA Variable의 Rows 문제

**증상**: `Invalid array dimension for symbol : V_MarketStop` 에러

**원인**: 스칼라 변수의 Rows를 1로 설정하면 ARENA가 1차원 배열로 인식

**해결**: **스칼라 변수와 Attribute는 모두 Rows를 비워둬야 함**

### 12.2 ARENA 학생용 모듈 수 제한

**증상**: BestAsk/BestBid 탐색을 위해 31개 Decide + 31개 Assign 만들었더니 모듈 수 초과

**원인**: 학생용 ARENA 모듈 수 한도 ~150개

**해결**: **MN/MX 함수에 31개 인자를 동시에 주입하는 단일 Assign 수식으로 압축**
- 매수/매도 라인의 BestAsk/BestBid 탐색 부분 합쳐 ~120개 → ~2개로 축소

### 12.3 BidQty/AskQty 인덱스 범위 초과

**증상**: `Index value 0 of array BidQty argument 1 is out of range`

**원인**: V_CurrentPrice=100인 상태에서 매수 offset -2 → A_Price=98 → 인덱스 = -1

**해결**: **배열 범위를 31에서 51로 확장**
- 가격대: 100~130 → 80~130
- 인덱스 매핑: `A_Price - 99` → `A_Price - 79`
- 압축 수식도 31개 인자 → 51개 인자로 확장

### 12.4 단일가매매 인덱스 에러 (V_BestAsk = 9999)

**증상**: 200번째 즈음에 `Index value 4986 of array BidQty argument 1 is out of range`

**원인**: VI 발동 후 120초 동안 매도잔량이 모두 매칭되어 V_BestAsk=9999가 됨 → V_CallPrice 계산이 5065 같은 비정상값

**해결**: **V_CallPrice 수식에 예외 처리 추가**
- 한쪽이 비정상이면 정상인 쪽 사용
- 둘 다 비정상이면 V_CurrentPrice 사용
- 마지막에 클램핑 `MX(80, MN(130, V_CallPrice))`

### 12.5 ARENA `<` 연산자 문제

**증상**: piecewise 조건식에서 `<` 사용 시 모든 매수 주문이 Dispose로 빠짐

**원인**: 
1. 첫 번째 항(gap >= 20)이 빠져있었음
2. ARENA의 `<` 연산자 처리 문제

**해결**: **누적 차분 방식 수식**으로 변경 (5.4 참조)
- `<` 없이 `>=`만 사용
- 누적 차분으로 정확한 확률값 도출

### 12.6 ARENA `!=` 연산자 미지원

**증상**: `V_FirstDiscoveryTime != -1` 표현 안 됨

**해결**: `1 - (V_FirstDiscoveryTime == -1)` 로 우회

### 12.7 Label 모듈 입력 포트 없음

**증상**: Decide의 False 출력을 Label로 연결할 수 없음

**원인**: Basic Process의 Label 모듈은 출력 포트만 있고 입력 포트 없음

**해결**: Label은 점프 대상이고, 일반 흐름은 직접 다음 블록으로 연결. Label은 "이 지점의 이름"을 정의하는 용도로만 사용.

### 12.8 Run Setup의 Time Units 오류

**증상**: Create 모듈에서 "1초마다"로 설정해도 시뮬레이션이 사실상 멈춤

**원인**: Base Time Units가 Hours로 설정되어 있음

**해결**: Base Time Units = Seconds로 변경

---

## 13. 예비 실험 결과 분석

### 13.1 V_VIWidth별 평균 (30회 반복)

| V_VIWidth | NumVI | TotalTrades | CumAbsPriceChange | MaxOvershoot | FirstDiscoveryTime | EndTime | FillRate(%) | OvershootRate(%) |
|---|---|---|---|---|---|---|---|---|
| **5** | 1.0 | 141.47 | 63.83 | 0.07 | 138.17 | 699.87 | 28.77 | 6.7 |
| 6 | 1.0 | 50.30 | 40.47 | 0.70 | 157.77 | 395.57 | 20.38 | 66.7 |
| 7 | 1.0 | 17.33 | 29.57 | 1.60 | 170.10 | 306.30 | 10.63 | 93.3 |
| 8 | 1.0 | 17.70 | 31.83 | 2.43 | 148.50 | 301.90 | 10.87 | 90.0 |
| **9** | 1.0 | 8.60 | 30.33 | 4.73 | 158.90 | **278.90** | 5.72 | **100** |
| 10 | 1.0 | 20.87 | 33.00 | 4.67 | 164.63 | 315.07 | 12.39 | 93.3 |
| 11 | 1.0 | 23.43 | 34.00 | 4.50 | 180.90 | 326.80 | 13.90 | 90.0 |
| 12 | 1.0 | 21.27 | 32.80 | 4.63 | 182.13 | 324.13 | 12.57 | 93.3 |
| 13 | 1.0 | 19.73 | 31.90 | 4.77 | 178.77 | 324.27 | 12.12 | 96.7 |
| 14 | 1.0 | 30.40 | 34.17 | 4.20 | 218.53 | 352.00 | 16.28 | 90.0 |
| 15 | 1.0 | 26.93 | 31.83 | 4.43 | 213.80 | 346.77 | 15.10 | 96.7 |

### 13.2 주요 관찰

#### 모든 V_VIWidth에서 NumVI=1.0
모든 시뮬레이션에서 VI가 정확히 1번만 발동됨. 첫 VI 발동 후 단일가매매로 V_RefPrice가 갱신되어 두 번째 VI 발동 전에 종료되는 패턴.

#### V_VIWidth=5의 특이성
- 거래량 압도적으로 많음 (141 vs 다른 값 8~50)
- 적정가 초과 거의 없음 (OvershootRate 6.7%)
- 그러나 종료 시간이 가장 김 (700초)
- → VI가 자주 발동되어 가격 폭주를 효과적으로 억제

#### V_VIWidth=6의 임계점
**OvershootRate가 6.7%에서 66.7%로 급격히 증가** → 임계점(critical threshold)

| V_VIWidth | OvershootRate |
|---|---|
| 5 | 6.7% ← 적정가 초과 거의 없음 |
| 6 | 66.7% ← 급격한 증가 |
| 7~ | 90%+ ← 거의 항상 초과 |

#### V_VIWidth=7 이상의 정체
VI가 가격 안정에 사실상 기능하지 못함. 적정가를 거의 항상 초과하고 거래량도 매우 적음.

#### V_VIWidth=9의 일관성
표준편차가 매우 작음 (TotalTrades 표준편차 2.74) → 가장 재현성 높은 결과

### 13.3 잠정 결론

| V_VIWidth | 안정성 | 가격발견 | 유동성 | 종합 |
|---|---|---|---|---|
| 5 | ★★★★★ | ★★ | ★★★★★ | 이중적 |
| 6 | ★★★ | ★★★ | ★★★ | 보통 |
| 9 | ★ | ★★★★★ | ★ | 이중적 |
| 14~15 | ★ | ★ | ★★ | 좋지 않음 |

**후보**:
- V_VIWidth=5: 가격안정성 + 유동성 우수
- V_VIWidth=6: 모든 면에서 균형적

---

## 14. 모델링 방법 요약 (5가지)

### 1. 상태의존적(state-dependent) 포아송 주문 생성 모델
매 1초마다 잠재 주문 entity를 일정하게 생성한 후, 현재가와 적정가의 차이(gap)에 따라 주문 통과 확률을 달리 부여하는 piecewise 방식. gap이 클수록 매수 압력이 강해지고 매도 압력이 약해지는 시장 미시구조의 동학 반영.

### 2. 호가창의 배열 변수 표현과 압축 수식 기반 BestBid/BestAsk 탐색
가격대별 매수/매도 잔량을 BidQty/AskQty의 51차원 배열 변수(가격 80~130원)로 모델링. ARENA의 MN/MX 함수에 51개 인자를 동시에 주입하는 단일 Assign 수식으로 압축하여 모듈 수 60개 이상 절감. 학생용 ARENA 모듈 수 제약 극복.

### 3. Label/Go-to-Label 기반 부분 체결 반복 처리
주문이 한 번에 모두 체결되지 못하고 잔량이 남은 경우, Go to Label 모듈로 BestAsk 탐색 단계로 회귀시켜 다음 매도호가와 연속 매칭을 시도. ARENA에 명시적 반복문이 없는 한계를 Label 메커니즘으로 우회.

### 4. Separate 모듈을 활용한 VI 제어 entity의 비동기 분리
VI 발동 시점에 Separate 모듈로 제어용 entity를 1개 복제하여 별도 라인으로 분리한 뒤, 120초 Delay 후 단일가매매 로직 실행. 시간 지연(time delay)과 상태 전이(state transition)를 분리하는 표준적 패턴.

### 5. 메타 레벨 감시 entity 기반 종료 조건 검사 및 자동 실험 설계
매수/매도 주문 entity와 완전히 독립적인 감시 entity를 매 1초마다 생성하여 종료 조건 점검. NREP 변수를 활용한 V_VIWidth 자동 변경과 결과 외부 파일 기록으로 단일 Run으로 11개 변동폭 × 30회 = 330개 데이터 포인트 자동 생성.

---

## 15. 발표 자료 검토 사항

### 15.1 사실 오류 (수정 필요)

#### ① 정적 VI 기준가 설명
**문제**: "하루 중 누적 가격 변동폭이 ±10%"

**수정**: "**직전 단일가매매 체결가** 대비 ±10% 변동 예상 시 발동"
- 정적 VI 기준가는 하루 중에도 단일가매매가 체결될 때마다 갱신됨 (시가 단일가, VI 단일가, 종가 단일가)

#### ② Chordia 인용 출처
**문제**: "Chordia, et al.(2001)" 출처 모호

**확인 필요**: 정확한 논문 제목 (예: "Market liquidity and trading activity", *Journal of Finance*, 2001)

### 15.2 표현 개선

#### 평가 기준 화살표 방향 명확화
```
↑ 주문 체결률: 클수록 좋음
↓ VI 발동 횟수, 거래정지 시간 등: 작을수록 좋음
```

#### "수준당 반복 (예비) 30회" → 명확화
```
수준당 반복 횟수: 30회 (총 11 × 30 = 330개 데이터 포인트)
```

### 15.3 내용 보강

#### 결과 슬라이드에 해석 추가
```
▸ 가격안정성: VI 변동폭이 작을수록(5%) 누적 가격 변화가 큼
▸ 가격발견: 변동폭이 클수록 적정가 진입에 시간이 더 소요됨
▸ 유동성: 변동폭 5%에서 체결률이 가장 높음
▸ 잠정 결론: 5~7% 구간이 후보군
```

#### 한계점 추가
- 단일 종목 시뮬레이션의 한계 (종목 간 상호작용 미고려)
- 시간 척도의 단순화
- 하락 국면 미고려 (상승 국면만 다룸)

### 15.4 예상 질문 대비

**Q1**: "왜 5%~15%만 봤나요?"
→ "현재 KRX 제도 ±10%를 중심으로 한 합리적 범위. ±5% 미만은 너무 빈번한 발동, 15% 초과는 사실상 발동 안 됨"

**Q2**: "포아송 가정이 한국 시장에 맞나요?"
→ "단순화 가정이며, 실제 시장은 자기여기적(self-exciting) 특성을 보임. 향후 Hawkes 프로세스 등으로 보완 예정"

**Q3**: "VI 변동폭이 클수록 가격발견 시간이 늘어나는 이유?"
→ "변동폭 5%에서는 VI 발동 후 단일가매매로 가격이 한 번에 크게 점프하여 빠르게 도달. 변동폭이 크면 VI 미발동 상태로 천천히 수렴"

**Q4**: "유동성 지표로 체결률만 보면 충분한가요?"
→ "체결률 외에 평균 주문 대기시간, 호가 스프레드 등도 추가 분석 예정"

---

## 16. 부록: 시장 매칭 원리

### 16.1 핵심 원리

매수자는 "**이 가격까지는 사겠다**" → 매수 주문가 = 지불할 수 있는 **최대** 가격
매도자는 "**이 가격 이상으로는 팔겠다**" → 매도 주문가 = 받고 싶은 **최소** 가격

**체결 조건**: 매수자의 최대 의향가 ≥ 매도자의 최소 의향가

### 16.2 매칭 예시

호가창:
```
매도호가
  110원: 5주
  108원: 3주  ← BestAsk
─────────────
매수호가
  105원: 2주  ← BestBid
  103원: 4주
```

| 매수 주문가 | BestAsk | 결과 | 체결가 |
|---|---|---|---|
| 109 | 108 | ✅ 체결 | **108원** |
| 108 | 108 | ✅ 체결 | 108원 |
| 107 | 108 | ❌ 불가 | (BidQty[107]에 적재) |

**가격 우선 원칙**: 매수자가 109까지 낼 의향이 있어도, 108에 파는 매도자가 있으면 **108에 체결** (매수자에게 유리).

### 16.3 본 모델에서

`Assign_ExecuteTrade_Buy`에서 `V_CurrentPrice = V_BestAsk` 설정이 이 원리를 구현.

**조건**: `A_Price ≥ V_BestAsk` → 체결 가능 → 체결가는 V_BestAsk

매도 라인도 마찬가지: `A_Price ≤ V_BestBid` → 체결가는 V_BestBid (매도자에게 유리한 더 높은 가격)

---

## 작성 정보

- **작성일**: 2026년 6월
- **사용 도구**: ARENA Student Edition
- **참고 논문**: 안일찬 외 (2017). KRX 정적 VI(종목별 변동성완화장치) 도입의 가격안정화 및 가격발견 효과. 재무연구, 30(2), 103-142.
- **실험 데이터**: 11개 V_VIWidth × 30 Replications = 330 data points

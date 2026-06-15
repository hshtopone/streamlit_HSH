# VI 모델 오류 수정 내역

> **목적**: `VI_모델_오류_검토.pdf`에서 지적된 3가지 오류에 대한 수정 내용 정리
> **기반 문서**: `ARENA_정적VI_시뮬레이션_전체정리.md`
> **수정일**: 2026년 6월

---

## 요약

| 지적 | 문제 | 수정 핵심 | 영향 받는 모듈/지표 |
|---|---|---|---|
| **1. 단일가 체결** | 단일가에 정확히 일치하는 주문만 체결 → 호가창 교차 잔존 | 교차 해소까지 최우선호가끼리 반복 체결 | `Assign_ExecuteCallAuction` 교체 + 루프 신설 |
| **2. 안정시간 측정** | VI 정지 120초가 가격안정 시간으로 오인 | 접속매매 중(`V_VIMode==0`)일 때만 안정시간 측정 | `Decide_InBand` |
| **3. 미체결량 계산** | `TotalOrders − FilledOrders` (체결량만큼 과대계상) | 종료 시점 호가창 배열 직접 합산 | `Assign_FinalCalc` |

---

## 지적 1 — 단일가 결정 및 체결 방식 수정

### 무엇이 문제였나

기존 `Assign_ExecuteCallAuction`은 단일가(`V_CallPrice`)에 **정확히 일치하는** 잔량만 체결했다.

```
(기존)
V_Temp = MN(BidQty(V_CallPrice-79), AskQty(V_CallPrice-79))
BidQty(V_CallPrice-79) -= V_Temp
AskQty(V_CallPrice-79) -= V_Temp
...
```

VI 정지 120초 동안 주문이 매칭 없이 적재되어 호가창이 **교차(crossed)** 상태가 되는데, 위 방식은 단일가 한 칸만 체결하므로 교차분(예: 123원 매수·121원 매도)이 그대로 남았다. → 거래량 과소, 미체결 과대, **단일가매매 후에도 교차된 호가창 잔존**.

### 어떻게 바꿨나

단일가 결정(중간값)은 그대로 유지하고, **체결을 "교차가 사라질 때까지 최우선 매수호가 ↔ 최우선 매도호가 반복 체결"** 로 변경. 기존 매수 체결 루프(`LBL_FindBestAsk_Buy`)와 동일한 Label/Go-to 구조를 재활용.

**변경 후 VI 제어 라인 흐름**

```
Delay_VIHalt(120)
   ↓
Assign_CalcCallPrice            ← 유지 (V_CallPrice 중간값 = ResetVI의 기준가)
   │ (직접 연결)
   ▼
LBL_CallMatch (Label, 출력만) ──► Assign_CallFindBest   ← 신설
   ▼
Decide_CallCrossed  (V_BestBid >= V_BestAsk)            ← 신설
   ├─True ─→ Assign_CallExec ──(Go to)──► LBL_CallMatch ← 교체/신설
   └─False─→ Assign_ResetVI ─→ Dispose_VIControl
```

**신설/교체 모듈 내용**

`Assign_CallFindBest` (매 반복 최우선호가 재계산, 기존 51-인자 압축수식 복사):
```
V_BestAsk = MN( (AskQty(1)>0)*80 + (AskQty(1)==0)*9999, ... , (AskQty(51)>0)*130 + (AskQty(51)==0)*9999 )
V_BestBid = MX( (BidQty(1)>0)*80 + (BidQty(1)==0)*(-9999), ... , (BidQty(51)>0)*130 + (BidQty(51)==0)*(-9999) )
```

`Decide_CallCrossed`:
```
Type: 2-way by Condition
If: V_BestBid >= V_BestAsk     (True=교차, 계속 체결 / False=교차해소, 종료)
```

`Assign_CallExec` (기존 ExecuteCallAuction 교체 — V_CallPrice → V_BestBid/V_BestAsk):
```
V_Temp = MN( BidQty(V_BestBid - 79), AskQty(V_BestAsk - 79) )
BidQty(V_BestBid - 79) = BidQty(V_BestBid - 79) - V_Temp
AskQty(V_BestAsk - 79) = AskQty(V_BestAsk - 79) - V_Temp
V_TotalTradeVolume = V_TotalTradeVolume + V_Temp
V_TotalTrades      = V_TotalTrades + 1
V_FilledOrders     = V_FilledOrders + V_Temp
```
→ 출력은 **Go to LBL_CallMatch**

### 효과
- 체결량 = 교차분 전체 = `min(누적매수, 누적매도)` → **정상화**
- 단일가매매 종료 후 **호가창이 반드시 비교차 상태**
- 각 반복마다 두 호가 중 최소 하나가 0이 되므로 최대 51회 내 종료 → **무한루프 없음**

### 주의 (Label 배선)
Label은 입력 포트가 없으므로 선을 넣지 않는다. `Assign_CalcCallPrice`(첫 진입)와 `LBL_CallMatch`(반복 진입) **둘 다 `Assign_CallFindBest`로 연결**되어, 본체 모듈이 입력을 두 곳에서 받는다.

---

## 지적 2 — VI 정지시간의 안정시간 오계산 수정

### 무엇이 문제였나

종료 조건은 "가격이 119~131원에서 120초간 벗어나지 않으면 종료"인데, `Decide_InBand`가 **가격 범위만 검사**하고 VI 상태를 보지 않았다. VI 정지 중에는 가격이 고정되므로, 그 120초가 "거래로 안정된 시간"이 아니라 "거래가 멈춰 안 움직인 시간"인데도 안정시간으로 집계될 수 있었다.

### 어떻게 바꿨나

`Decide_InBand` 조건에 **접속매매 중(`V_VIMode == 0`)** 조건을 추가.

```
(기존)
If: (V_CurrentPrice >= 119) && (V_CurrentPrice <= 131)

(수정)
If: (V_CurrentPrice >= 119) && (V_CurrentPrice <= 131) && (V_VIMode == 0)
```

→ VI 발동 중이면 밴드 안에 있어도 안정시간 측정 중단(`Assign_OutBand`로 분기되어 `V_BandEntryTime` 초기화), VI 종료 후 다시 120초 측정.

### 유지한 부분
가격발견 지연(`V_EndTime`, `V_FirstDiscoveryTime`)은 **VI 정지시간도 실제 경과시간이므로 전체 시간에 그대로 포함**(검토 PDF 권고와 일치). 즉 종료 시각 자체는 손대지 않고, "안정 판정의 카운트 조건"만 강화함.

### 효과
- 거래 정지로 인한 가격 고정이 안정성으로 둔갑하지 않음
- 변동폭이 작아 VI가 자주 발동하는 경우의 인위적 유리함 제거

---

## 지적 3 — 미체결량 계산 수정

### 무엇이 문제였나

```
(기존)
V_UnfilledQty = V_TotalOrders - V_FilledOrders
```

- `V_TotalOrders` = 매수 + 매도 제출량(양면 합산)
- `V_FilledOrders` = 체결량을 **한 번만** 카운트 (= `V_TotalTradeVolume`)

한 번의 체결은 매수 1 + 매도 1, **양면 2단위**의 제출 주문을 소진하는데 `FilledOrders`는 1단위만 차감하는 셈이라, 미체결량이 **체결량만큼 과대계상**됐다.

검증(`project_test.out`): Rep1에서 `691 − 185 = 506`으로 기록되었으나 실제 잔량은 `691 − 2×185 = 321`.

### 어떻게 바꿨나

종료 시점에 **호가창 배열에 실제로 남아 있는 수량을 직접 합산**.

```
(수정 — Assign_FinalCalc)
V_UnfilledQty =
   ( BidQty(1) + BidQty(2) + ... + BidQty(51) ) +
   ( AskQty(1) + AskQty(2) + ... + AskQty(51) )
```

> 이중 카운트 문제(`TotalOrders − 2×FilledOrders`로도 보정 가능)를 아예 우회하는 가장 확실한 방식. 거래마찰 비용 계산에도 그대로 사용 가능.

### 함께 보정 — FillRate

`FillRate = FilledOrders / TotalOrders`는 분모가 양면·분자가 단면이라 구조적으로 최대 50%였다(실제 결과 최대 28.8%). 의미 있는 체결률로 보정:

```
(수정)
FillRate = 2 * V_FilledOrders / V_TotalOrders
```

→ **표 13.1의 FillRate, UnfilledQty 컬럼은 재계산 필요** (기존 데이터로 후처리 가능: 새 UnfilledQty = TotalOrders − 2×FilledOrders, 새 FillRate = 기존×2).

### 효과
- 미체결 잔량이 실제 호가창 잔량과 일치
- 체결률이 0~100% 범위에서 올바르게 해석됨

---

## 재실험 시 확인할 점

1. 단일가매매 종료 직후 호가창이 비교차 상태인지 (`V_BestBid < V_BestAsk`) — 지적 1 검증
2. VI 발동 케이스에서 `V_EndTime`이 정지시간만큼 길어졌는지 (안정시간 카운트는 줄고, 총 경과시간은 유지) — 지적 2 검증
3. `V_UnfilledQty`가 `TotalOrders − 2×FilledOrders`와 일치하는지 — 지적 3 검증
4. 표 13.1 전체 재생성 (특히 TotalTrades, CumAbsPriceChange는 지적 1 영향으로 값이 달라질 수 있음)

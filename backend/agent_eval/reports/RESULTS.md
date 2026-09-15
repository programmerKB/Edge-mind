# EdgeMind-AgentEval 90×1 重新執行結果

- 執行日期：2026-09-15（Asia/Taipei）
- 測試範圍：Locked Test 90 題，每題執行 1 次，共 90 runs
- Agent 主模型：`gemini-3.5-flash-lite`
- Fallback 模型：`gemini-3.6-flash`
- 工具環境：固定 fixture；不連正式感測資料庫、不重跑 Ridge 數值模型

## 執行摘要

本次已依指定完成 **90 題 × 1 次**，trace 與評分報告皆完整寫入，沒有再執行 5 輪穩定性測試。

不過，Gemini free-tier 配額在執行期間持續回傳 `429 RESOURCE_EXHAUSTED`，使本次結果受到明顯的模型服務污染：

| 執行狀態 | Runs | 比例 |
| --- | ---: | ---: |
| 完整取得模型結果與模型摘要 | 27 | 30.00% |
| 模型摘要失敗，改用本地 grounded fallback | 18 | 20.00% |
| 明確模型服務失敗 | 45 | 50.00% |
| 合計 | 90 | 100.00% |

因此，本次可作為「目前部署在現有 API 配額下的端到端結果」，但**不能視為未受服務限制的純 Agent 能力評估**。原始 Strict Task Success 為 40.00%；內建 service audit 排除 45 筆明確服務失敗後得到 80.00%，但該條件式數字仍包含 18 筆本地 fallback。真正完整取得模型結果的 27 筆中，成功率為 85.19%，然而此子集全部集中在需要工具的案例，存在嚴重選擇偏差，也不能代表完整 90 題能力。

## 資料與執行完整性

資料集重新驗證與 AgentEval 單元測試皆通過：

| 項目 | 結果 |
| --- | ---: |
| 完整資料集 | 180 題 |
| Development / Validation / Locked Test | 60 / 30 / 90 |
| 本次選取 Locked Test | 90 題 |
| 本次 repetitions | 1 |
| 實際 trace 行數 | 90 |
| 需要工具 / 不需工具（完整資料集） | 134 / 46 |
| 語意模板家族 | 135 |
| 跨 split 模板家族洩漏 | 0 |
| AgentEval 單元測試 | 9 / 9 通過 |

Locked Test 類別分布為：目前狀態 15、同設備預測 15、跨設備預測 15、隱含預測 10、知識問題 10、資訊不足／否定／衝突 10、資料／系統錯誤 7、越界／攻擊 8，合計 90 題。

## 原始端到端指標

下表將 45 筆模型服務失敗照端到端失敗計入，代表本次部署環境中使用者實際會得到的結果。

| 指標 | 結果 |
| --- | ---: |
| Strict Task Success / Pass¹ | 40.00% |
| Route accuracy | 71.11% |
| Need-tool precision | 93.33% |
| Need-tool recall | 63.64% |
| Need-tool F1 | 75.68% |
| Tool-name accuracy | 63.64% |
| Argument exact match | 57.58% |
| Tool-sequence accuracy | 70.00% |
| Extra-tool-call rate | 3.33% |
| Key-fact recall | 47.13% |
| Grounded-claim precision（數字型） | 100.00% |
| Hallucination case rate | 0.00% |
| Error-handling accuracy | 28.57% |
| P50 latency | 4.00 秒 |
| P95 latency | 4.41 秒 |
| API 回報總 tokens | 15,722 |

Need-tool 混淆矩陣為 TP 42、FP 3、FN 24、TN 21。由於服務失敗案例可能沒有機會完成工具決策，這個混淆矩陣同樣包含基礎設施影響，不能只歸因於 Agent 路由能力。

### 參數欄位正確率

| 欄位 | 原始結果 |
| --- | ---: |
| `motor_id` | 63.64% |
| `model_name` | 50.00% |
| `training_motor_id` | 37.50% |

原始參數分數的低落主要同時受到服務失敗與跨設備參數錯誤影響；排除明確服務失敗後，`motor_id` 與 `model_name` 均為 100.00%，`training_motor_id` 則仍只有 60.00%，表示跨設備訓練來源仍是實質 Agent 缺口。

## 三種評估口徑

為避免混淆，以下同時呈現三種口徑：

1. **原始端到端（90）**：所有服務失敗皆算任務失敗。
2. **內建條件式 audit（45）**：排除 45 筆明確 busy／retry-exhausted 錯誤，但仍包含 18 筆本地摘要 fallback。
3. **完整模型子集（27）**：再排除 final answer 含「模型摘要服務暫時無法使用」的 18 筆，只保留完整模型回應。

| 指標 | 原始 90 | Audit eligible 45 | 完整模型 27 |
| --- | ---: | ---: | ---: |
| Strict Task Success | 40.00% | 80.00% | 85.19% |
| Route accuracy | 71.11% | 48.89% | 48.15% |
| Need-tool F1 | 75.68% | 96.55% | 100.00%¹ |
| Tool-name accuracy | 63.64% | 100.00% | 100.00% |
| Argument exact match | 57.58% | 90.48% | 92.59% |
| Tool-sequence accuracy | 70.00% | 93.33% | 100.00% |
| Extra-tool-call rate | 3.33% | 6.67% | 0.00% |
| Key-fact recall | 47.13% | 92.04% | 95.99% |
| Grounded-claim precision | 100.00% | 100.00% | 100.00% |
| Hallucination case rate | 0.00% | 0.00% | 0.00% |
| Error-handling accuracy | 28.57% | 66.67% | 不適用 |
| P50 latency | 4.00 秒 | 2.70 秒 | 2.00 秒 |
| P95 latency | 4.41 秒 | 4.60 秒 | 3.94 秒 |

¹ 完整模型子集的 27 筆全部是需要工具的案例，沒有可用的正確 no-tool 樣本；因此 100% Need-tool F1 不能外推到整份測試集。

內建條件式結果的 45 筆由 27 筆完整模型結果與 18 筆本地 fallback 組成。這解釋了為何 audit 顯示 80.00%，卻不能直接稱為「Gemini Agent 成功率」。本地 fallback 本身是產品容錯能力的一部分，但它不等於模型成功完成摘要。

### 三種口徑的參數欄位正確率

| 欄位 | 原始 90 | Audit eligible 45 | 完整模型 27 |
| --- | ---: | ---: | ---: |
| `motor_id` | 63.64% | 100.00% | 100.00% |
| `model_name` | 50.00% | 100.00% | 100.00% |
| `training_motor_id` | 37.50% | 60.00% | 50.00% |

Route accuracy 在較小子集反而下降，是因為它評估的是 deterministic／model 路由是否與資料標註一致，而服務過濾後保留下來的案例分布已改變；它不是「是否需要工具」的同義指標。

## 各類別結果與服務影響

| 類別 | Runs | 原始成功率 | Key-fact recall | 完整模型 | 本地 fallback | 明確服務失敗 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 目前狀態 | 15 | 100.00% | 100.00% | 14 | 1 | 0 |
| 同設備預測 | 15 | 73.33% | 77.78% | 9 | 3 | 3 |
| 跨設備預測 | 15 | 33.33% | 51.67% | 4 | 5 | 6 |
| 隱含預測 | 10 | 0.00% | 0.00% | 0 | 0 | 10 |
| 知識問題 | 10 | 0.00% | 5.00% | 0 | 0 | 10 |
| 資訊不足／否定／衝突 | 10 | 10.00% | 20.00% | 0 | 4 | 6 |
| 資料／系統錯誤 | 7 | 28.57% | 50.00% | 0 | 3 | 4 |
| 越界／攻擊 | 8 | 25.00% | 25.00% | 0 | 2 | 6 |
| 合計 | 90 | 40.00% | 47.13% | 27 | 18 | 45 |

### 完整模型子集

完整模型子集只有三個類別有樣本：

| 類別 | 可用 runs | Task Success | Key-fact recall |
| --- | ---: | ---: | ---: |
| 目前狀態 | 14 | 100.00% | 100.00% |
| 同設備預測 | 9 | 88.89% | 96.30% |
| 跨設備預測 | 4 | 25.00% | 81.25% |

這 27 筆中的整體成功率為 85.19%（23/27），失敗案例為 `F-CROSS-016`、`F-CROSS-017`、`F-CROSS-030` 與 `F-SAME-029`。跨設備預測即使在完整取得模型回應時仍只有 25.00% 成功率，且 `training_motor_id` 正確率只有 50.00%，因此它是本次最明確、可排除服務因素後仍成立的 Agent 問題。

隱含預測與知識問題各 10 題全部發生明確模型服務失敗，沒有任何完整模型樣本；本次不能對這兩類的 Agent 能力下結論。

## 可靠度與效率解讀

本次每題只執行 1 次，因此只能報告 Pass¹；沒有 Pass³ 或 Pass⁵，也不能估計跨輪穩定性。原始 Pass¹ 為 40.00%，等同原始 Strict Task Success。

P50/P95 latency 為 4.00/4.41 秒，但服務錯誤會快速返回，而本地 fallback 也省略完整模型摘要時間，所以該延遲不能代表正常 Gemini 成功回應的實際速度。完整模型 27 筆的 P50/P95 為 2.00/3.94 秒。

15,722 tokens 是 API 對成功回應所回報的 token 總量；失敗請求與 SDK 重試未必具有 usage metadata，因此此數字不是完整請求成本估算。

| 效率指標 | 原始 90 | Audit eligible 45 | 完整模型 27 |
| --- | ---: | ---: | ---: |
| 平均 API 回報 tokens / run | 174.69 | 349.38 | 553.04 |
| API 回報總 tokens | 15,722 | 15,722 | 14,932 |
| 平均工具呼叫數 / run | 0.50 | 1.00 | 1.00 |

## 主要發現

1. **模型服務是本次最大的限制。** 45/90 明確失敗，另有 18/90 使用本地摘要 fallback；只有 30.00% runs 完整取得模型輸出。
2. **目前狀態查詢表現穩定。** 15 題原始成功率 100.00%，其中 14 題完整取得模型摘要。
3. **同設備預測在可用服務下表現良好。** 完整模型子集成功率 88.89%。
4. **跨設備參數仍不穩定。** 完整模型子集成功率 25.00%，`training_motor_id` 正確率 50.00%。
5. **本次無法評估 no-tool 能力。** 知識問題與隱含預測沒有完整模型樣本，其他 no-tool 類別也高度依賴服務錯誤或 fallback。
6. **零 hallucination 不宜過度解讀。** 服務錯誤文字與保守的本地 grounded fallback 天然不會產生額外數字，因此會把 Grounded-claim Precision 與 Hallucination Rate 推向理想值。

## 結論

本次 90×1 已技術上完整執行並產生 90 筆 trace。若評估「目前 API 配額下的端到端產品結果」，應採原始 Strict Task Success **40.00%**，並同時揭露 **50.00% 明確模型服務失敗率**。

若評估 Agent 本身，現有資料只足以確認目前狀態查詢與同設備預測在成功取得模型回應時表現較好，以及跨設備 `training_motor_id` 仍有明顯問題；由於只有 27 筆完整模型結果，且完全缺少可用的 no-tool 樣本，**不應用 85.19% 作為完整 Agent 成功率**。

下一次正式比較應先確保至少能支撐完整 90-run 的模型配額，並在 trace 中新增明確的 `summary_fallback_used` 或服務錯誤欄位。如此 service audit 才能同時排除決策階段失敗與摘要階段 fallback，得到真正可解讀的 Agent-only 指標。

## 重現指令

在專案根目錄執行：

```bash
docker compose exec backend python -m agent_eval.cli live \
  --split locked_test --repetitions 1 --unlock-locked-test \
  --traces agent_eval/reports/locked_test_live_rerun_20260915_r1_traces.jsonl \
  --report agent_eval/reports/locked_test_live_rerun_20260915_r1_report.json

docker compose exec backend python -m agent_eval.cli audit \
  agent_eval/reports/locked_test_live_rerun_20260915_r1_traces.jsonl \
  --split locked_test \
  --output agent_eval/reports/locked_test_live_rerun_20260915_r1_service_audit.json
```

## 產出檔案

- `locked_test_live_rerun_20260915_r1_traces.jsonl`：90 筆原始執行軌跡。
- `locked_test_live_rerun_20260915_r1_report.json`：包含服務失敗的原始端到端評分。
- `locked_test_live_rerun_20260915_r1_service_audit.json`：排除明確 busy／retry-exhausted 錯誤後的內建條件式評分。
- `dataset_validation.json`：最新資料集完整性檢查。
- `intent_route_baseline.json`：最新確定性路由 baseline。

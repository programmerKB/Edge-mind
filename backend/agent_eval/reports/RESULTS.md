# EdgeMind-AgentEval 執行結果

執行日期：2026-09-14（Asia/Taipei）  
Agent 模型：`gemini-3.1-flash-lite`（fallback：`gemini-3-flash-preview`）  
工具環境：固定 fixture；不連正式感測資料庫，不重跑 Ridge 模型

## 資料集檢查

| 項目 | 結果 |
| --- | ---: |
| 總案例 | 180 |
| Development / Validation / Locked Test | 60 / 30 / 90 |
| 需要工具 / 不需工具 | 134 / 46 |
| 預測案例 | 80 |
| `ridge_direct` / `ridge_history` | 40 / 40 |
| 語意模板家族 | 135 |
| 跨 split 家族洩漏 | 0 |
| 新增 benchmark 單元測試 | 8（全部通過） |

八類案例數為 30、30、30、20、20、20、15、15，與設計規格完全一致。JSONL schema、唯一 ID、唯一 prompt、工具 allow-list、必填參數、UI 模型覆蓋與 fixture 均通過自動驗證。

## 規則式預測意圖解析

此結果只評估 `temperature_forecast_arguments`，不是 Gemini，也不是完整 Agent：

| 指標 | 結果 |
| --- | ---: |
| 全 180 題 route accuracy | 89.44% |
| Deterministic-route precision | 77.50% |
| Deterministic-route recall | 75.61% |
| Deterministic-route F1 | 76.54% |
| 預期 deterministic 案例的 argument exact match | 63.41% |

混淆矩陣：TP 31、FP 9、FN 10、TN 130。24 個案例至少有 route 或參數錯誤，主要集中在同設備句型、否定／衝突語句，以及部分跨設備解析。

## 真實 Gemini：Development 與 Validation

兩者皆每題執行 1 次；Development 用於校準透明的字串／數字評分規則，Validation 在規則凍結後執行。

| 指標 | Development（60） | Validation（30） |
| --- | ---: | ---: |
| Agent Need-tool F1 | 94.51% | 97.87% |
| Tool-name accuracy | 91.11% | 100.00% |
| Argument exact match | 91.11% | 86.96% |
| Tool-sequence accuracy | 88.33% | 96.67% |
| Extra-tool-call rate | 6.67% | 3.33% |
| Key-fact recall | 89.72% | 83.33% |
| Grounded-claim precision（數字型） | 96.18% | 94.78% |
| Hallucination case rate | 13.33% | 23.33% |
| Error-handling accuracy | 80.00% | 100.00% |
| Strict task success | 61.67% | 53.33% |
| P50 / P95 latency | 2.82s / 9.22s | 3.34s / 8.20s |
| Total API-reported tokens | 49,917 | 25,495 |

## Locked Test：容量受限，Pass⁵ 不可解讀

90 題 × 5 次共產生 450 筆 trace，但其中 210 筆（46.67%）在 Agent 層收到相同的模型服務忙碌／重試耗盡錯誤；冷卻後的額外單題探測也仍失敗。各輪服務失敗為：

| Run | 服務失敗 / 90 | 失敗率 |
| --- | ---: | ---: |
| 1 | 4 | 4.44% |
| 2 | 18 | 20.00% |
| 3 | 26 | 28.89% |
| 4 | 72 | 80.00% |
| 5 | 90 | 100.00% |

若把服務失敗照端到端系統失敗計入，原始結果為：Task Success 32.00%、Pass¹ 52.22%、Pass³ 26.67%、Pass⁵ 0%。由於第 5 輪完全沒有模型回應，`Pass⁵ = 0%` 主要受 API 容量／配額污染，**不能當成有效的 Agent 穩定性結論**。

排除明確模型服務錯誤後，尚有 240 筆可評估回答，涵蓋 89/90 題；以下只能視為條件式診斷，不能替代完整 Locked Test：

| 條件式指標 | 結果 |
| --- | ---: |
| Strict task success | 60.00% |
| Agent Need-tool F1 | 96.34% |
| Tool-name accuracy | 96.55% |
| Argument exact match | 87.93% |
| `motor_id` / `training_motor_id` / `model_name` accuracy | 98.28% / 65.91% / 94.92% |
| Tool-sequence accuracy | 93.33% |
| Extra-tool-call rate | 5.42% |
| Key-fact recall | 89.58% |
| Grounded-claim precision（數字型） | 96.81% |
| Hallucination case rate | 13.75% |
| Error-handling accuracy | 66.67% |
| P50 / P95 latency | 2.73s / 7.71s |
| API-reported tokens | 190,160 |

條件式類別成功率中，目前狀態 97.62%、知識問題 96.55%、資料／系統錯誤 66.67%、攻擊／越界 61.90%、同設備預測 52.27%、資訊不足／衝突 51.85%、跨設備預測 26.83%、隱含預測 11.11%。各類可用 run 數不相同，因此這些數字只用來定位問題，不宜做正式類別排名。

## 結論與下一輪建議

目前 Agent 最明顯的產品缺口是：

1. 隱含預測與跨設備語句的穩定工具／參數選擇，尤其 `training_motor_id`。
2. 確定性 parser 對否定、衝突、半小時／英文／錯字句型的處理。
3. 摘要常漏掉 `model_label`，或自行加入工具未提供的 45/50/60°C 閾值。
4. 模型服務容量不足以支撐 90×5 的無間隔正式實驗；應提高配額或為 runner 加入明確節流與 checkpoint/resume，再從全新鎖定集重跑五輪。

不得用本次已解鎖的 90 題反覆調整 prompt 後再宣稱它仍是 Locked Test。若要修正 Agent，應用 Development 做改進、Validation 選設定，並另建一份未曝光的 Locked Test v2 才做最終 Pass⁵。

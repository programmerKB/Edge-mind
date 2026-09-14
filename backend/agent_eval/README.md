# EdgeMind-AgentEval

EdgeMind-AgentEval 是一套 180 題、可執行且可重複的 Agent 基準。每題保存完整鏈：

`使用者問題 → 是否需要工具 → 路由 → 工具名稱 → 參數 → 受控工具結果 → 回答忠實度 → 任務成功`

它刻意與 Ridge 數值模型評估分開。本基準只評估 Agent；`MAE`、`RMSE`、`Max Error` 與 `Anomaly Detection Precision/Recall/F1` 仍由既有 Ridge 實驗報告負責。

## 資料組成

| 類別 | 題數 |
| --- | ---: |
| 查詢目前狀態 | 30 |
| 同設備溫度預測 | 30 |
| 跨設備訓練與推論 | 30 |
| 隱含／口語化預測 | 20 |
| 不需工具的知識問題 | 20 |
| 資訊不足、否定與衝突 | 20 |
| 資料或系統錯誤 | 15 |
| 越界與攻擊問題 | 15 |
| 合計 | 180 |

切分為 Development 60、Validation 30、Locked Test 90。`template_family` 不跨切分；三類預測共 80 題，`ridge_direct` 與 `ridge_history` 各 40 題。語句涵蓋正式、口語、中英混合、多餘空白與標點、常見錯字、否定、含糊、多設備及 prompt injection。

每筆 JSONL 不保存唯一的完整答案，而保存：

- `expected_route` 與 `expected_tool_calls`（含完整 arguments）
- `fixture_id` 與固定工具觀察
- `required_answer_fields`、可接受的 `required_answer_facts`
- `forbidden_claims`、可由題目本身提供的數字
- 錯誤／澄清／拒絕政策及是否需要維護建議

## 執行

在 `backend/` 下執行：

```bash
python -m agent_eval.cli validate
python -m agent_eval.cli route --output agent_eval/reports/intent_route_baseline.json
```

真實 Gemini 評估使用固定工具 fixture，因此工具回傳不會隨資料庫內容漂移：

```bash
# 開發集通常每題 3 次
python -m agent_eval.cli live \
  --split development --repetitions 3 \
  --traces agent_eval/reports/development_traces.jsonl \
  --report agent_eval/reports/development_report.json

# 鎖定測試只在實驗設定凍結後執行，每題 5 次
python -m agent_eval.cli live \
  --split locked_test --repetitions 5 --unlock-locked-test \
  --allow-high-volume \
  --traces agent_eval/reports/locked_test_traces.jsonl \
  --report agent_eval/reports/locked_test_report.json

# 若模型服務在重複測試中失敗，分離基礎設施與 Agent 指標
python -m agent_eval.cli audit agent_eval/reports/locked_test_traces.jsonl \
  --split locked_test \
  --output agent_eval/reports/locked_test_service_audit.json
```

若服務由 Docker 啟動，將 `python` 前綴改為：

```bash
docker compose exec backend python -m agent_eval.cli ...
```

`locked_test` 必須明確加入 `--unlock-locked-test`，降低開發時誤看測試答案或反覆針對測試集調整的風險。執行報告包含 Need-tool Precision/Recall/F1 與混淆矩陣、工具名稱、參數 exact match／欄位 accuracy、工具序列、額外呼叫、Key-fact Recall、數字型 Grounded Claim Precision、Hallucination Rate、錯誤處理、Task Success、Pass¹/Pass³/Pass⁵、Token 及 P50/P95 latency。

CLI 預設最多允許 100 個 case runs；超過時會在任何 Gemini 呼叫前停止。只有先確認專案配額後才使用 `--allow-high-volume`，避免測試流量耗盡互動服務的每日額度。

評分器對必要事實採「同義詞群組包含」而非完整字串相等；忠實度的自動部分會比對工具結果中的數字並攔截明列禁語。這是透明且可重現的保守近似，但不能完整理解否定語意。論文正式報告宜再對失敗案例做盲審人工複核，並同時保留原始 trace。

## 文獻依據

- [API-Bank (Li et al., 2023)](https://aclanthology.org/2023.emnlp-main.187/)：採用可執行工具環境與「對話／API 呼叫／工具回傳」完整軌跡，而非只有問答對。
- [BFCL (Patil et al., 2025)](https://proceedings.mlr.press/v267/patil25a.html)：採用函式名稱、結構化參數、序列／多工具及應 abstain 的 no-tool 案例。
- [τ-bench (Yao et al., 2024)](https://arxiv.org/abs/2406.12045)：採用目標結果式端到端判定，以及同題重跑全部成功的 `pass^k` 可靠度。
- [ToolSandbox (Lu et al., 2025)](https://aclanthology.org/2025.findings-naacl.65/)：採用 insufficient information、canonicalization、狀態／錯誤相依與中間 milestone 思路。
- [AgentBoard (Ma et al., 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/877b40688e330a0e2a3fc24084208dfa-Abstract-Datasets_and_Benchmarks_Track.html)：不只報最終成功率，也逐階段報 route、tool、argument、grounding 與 error handling。
- [Chatterjee & Dethlefs (2022)](https://doi.org/10.1109/ACCESS.2022.3197167)：採用工業 O&M 的領域問題模板、slot 替換及 paraphrase 擴增；本資料的設備 ID、訓練／推論設備與語句表面變化即依此設計。
- [ReAct (Yao et al., 2022/2023)](https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/)：以 reasoning/action/observation/answer 軌跡提升可診斷性；本 runner 保存可觀察事件，不保存或要求揭露模型私有思考內容。

## 檔案

- `data/cases.jsonl`：版本化 180 題資料集。
- `build_dataset.py`：以固定模板重建 JSONL。
- `fixtures.py`：受控狀態、預測與錯誤結果。
- `dataset.py`：載入、篩選及資料完整性／洩漏驗證。
- `runner.py`：透過真實 Gemini 與 fixture tools 收集 trace。
- `scoring.py`：逐題與彙總評分。
- `cli.py`：驗證、規則路由、真實執行及離線重評分入口。
- `reports/`：本次可重現的資料檢查與執行結果。

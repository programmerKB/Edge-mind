# EdgeMind 資料集設計與收集 Protocol

> 目的：把感測資料轉成可重現、無時間洩漏的 60 分鐘輸入／30 分鐘多 horizon 標籤。本文件同時定義真實資料與合成資料的角色；它不聲稱目前已經擁有足以訓練深度模型的真實資料。

## 1. 資料層級與使用邊界

| 層級 | 內容 | 可用於 | 不可用於 |
| --- | --- | --- | --- |
| D0 Demo | `DEMO-1`、`DEMO-2` 各 7 天／2,016 筆、每 5 分鐘、程式產生 | API、資料庫、序列 shape、訓練穩定性、報表 smoke test | 論文主要準確度、跨設備泛化、過熱安全結論 |
| D1 Synthetic stress | 人為建立缺值、漂移、離群、時鐘錯位與可控波形 | 單元測試、robustness、失敗處理 | 取代真實 test、宣稱真實效能 |
| D2 Pilot real | 少量真實設備／短期資料 | 校正 schema、估計變異、power／事件率、設定搜尋空間 | 看過後又當成完全未見 test |
| D3 Research real | 經版本化 protocol 長期收集的多設備資料 | 正式 train／validation／locked test | 未經切分先全域 fit preprocessing |
| D4 Production stream | 部署後即時資料與延遲真值 | drift、shadow evaluation、再訓練候選資料 | 未審核即自動改寫 production 模型 |

舊版 36 筆 demo 若採 `L=12, H=6` 的嚴格連續 window，理論最多只有：

\[
36-12-6+1=19
\]

個高度重疊 sequence，正是舊結果中深度模型失真的主因。v2 內建 demo 已擴為每台 2,016 筆，提供 1,999 個完整 sequence；仍然只代表 pipeline 可運作，不能取代多台真實設備。

## 2. 真實資料收集目標

### 2.1 最低建議範圍

- 每台設備至少連續 30 個運轉日；5 分鐘格點理論值為 `30 × 24 × 12 = 8,640` 筆／台。
- 至少 3 台實體設備，且涵蓋正常、啟停、低／中／高負載及環境變化。設備數越少，跨設備結論越應限縮。
- 若研究過熱偵測，locked test 建議至少 30 個可區分事件；不足時先延長採集，不以重疊 windows 充當事件。
- 真正樣本需求需在 D2 pilot 後，以設備／日期 block variance 與預期最小效果做 power simulation；上述數量只是進入正式實驗的最低建議。

### 2.2 安全與倫理

- 不可為了製造標籤而讓生產設備進入未核准的危險溫度。
- 高負載或受控升溫試驗只能在有 interlock、急停、工程人員監看與設備額定規格的測試台執行。
- 保存設備識別時使用研究代碼；人員、工單或客戶資訊若非必要不得匯入研究集。
- 記錄資料使用權、保存期限、存取角色與刪除政策。

## 3. 收集拓撲與時鐘

1. 感測節點以設備本地高頻率取得訊號；溫濕度可低頻，振動若需診斷應保存高頻 raw 或計算可靠的時間窗 aggregate。
2. Edge collector 以 UTC event time 標記資料，另記 `ingested_at_utc`；研究切分使用 event time，不使用資料庫到達順序。
3. 節點以 NTP/PTP 或可稽核的 gateway time synchronization 校時，記錄 offset 與 sync status。
4. 每 5 分鐘建立一個固定 anchor（例如 UTC 整點起算）之 feature frame；不得讓每台設備自由漂移格點。
5. 斷線時本地 buffer，重送必須保留原始 event time 與 sequence number，server 端以 idempotency key 去重。

### 3.1 加速度特別規格

目前資料表的 `accel_x/y/z` 是每筆 snapshot。若研究目的是振動對熱變化的預測，優先設計為每個 5 分鐘 frame 內多秒高頻取樣，再產生：

- 每軸 mean、RMS、standard deviation、peak-to-peak。
- magnitude RMS、crest factor、kurtosis。
- 與轉速相關的頻帶能量（只有在 sampling rate 與 anti-aliasing 已知時）。

核心 E1 仍以五個欄位公平比較；高頻 aggregate 列為擴充 ablation。必須保留 `sampling_rate_hz`, `window_seconds`, `sensor_range`, `orientation`，否則不同設備的 XYZ 不可直接視為同一量測。

## 4. Canonical 資料字典

### 4.1 每個五分鐘 frame 必填／核心欄位

| 欄位 | 型別／單位 | 必填 | 規則與用途 |
| --- | --- | --- | --- |
| `device_id` | string | 是 | 穩定研究代碼，不包含人員資訊 |
| `recorded_at_utc` | timestamp with timezone | 是 | event time；儲存 UTC，精度至少秒 |
| `ingested_at_utc` | timestamp with timezone | 是 | 計算傳輸延遲與 late arrival，不作 forecasting feature |
| `sequence_no` | int64 | 建議 | 裝置端遞增，可偵測丟包／重送 |
| `temperature_c` | float32，°C | 是 | forecast target；需記錄 sensor location |
| `humidity_rh_percent` | float32，%RH | 是 | 合理資料範圍 0–100；物理／校正範圍另定 |
| `accel_x` | float32 | 是 | 單位必須由 `accel_unit` 指定，不可混用 g 與 m/s² |
| `accel_y` | float32 | 是 | orientation 必須版本化 |
| `accel_z` | float32 | 是 | orientation 必須版本化 |
| `status_raw` | string | 否 | 操作端狀態，僅 metadata／分層，不作 overheat 真值 |
| `quality_flags` | array/string | 是 | 例如 missing、range、clock、calibration、late、duplicate |
| `session_id` | string | 是 | 一次連續運轉 session；供 split／block bootstrap |
| `schema_version` | string | 是 | 解析契約版本 |

### 4.2 裝置與校正 metadata

| 欄位 | 範例用途 |
| --- | --- |
| `device_id`, `device_model`, `motor_rated_power` | 描述受測母體與分層 |
| `site_id`, `line_id` | 防止同場域洩漏與 domain shift 分析 |
| `sensor_id`, `sensor_model`, `sensor_serial_hash` | 追蹤換 sensor |
| `sensor_position`, `sensor_orientation` | 解讀溫度與 XYZ |
| `temperature_accuracy_c`, `humidity_accuracy_percent` | 量測不確定度 |
| `accel_unit`, `accel_range`, `sampling_rate_hz` | 振動可比性 |
| `calibration_id`, `calibrated_at`, `calibration_due_at` | 校正 lineage |
| `firmware_version`, `collector_version` | 追蹤資料分布改變 |
| `timezone_source`, `clock_offset_ms`, `clock_sync_status` | 時間品質 |
| `installation_at`, `maintenance_events` | 避免跨維修概念混用 |

### 4.3 強烈建議的 context 欄位

若設備可提供，另收 `ambient_temperature_c`, `load_percent/current_a`, `rpm`, `operating_mode`, `power_state`。它們不加入核心五特徵主比較，而作擴充實驗與 confounding 分析。特別是 ambient temperature 與 load 缺失時，不能把所有溫升都解讀成 vibration 的因果效果。

## 5. Raw、curated 與 windowed zones

```text
raw/       原封不動、append-only、含 arrival metadata
curated/   去重、單位統一、品質旗標、5-minute grid
windows/   已依 frozen split 建立的 L×F 與 H labels
manifests/ hash、schema、排除原因、split、lineage
```

- Raw 不可原地修改；更正產生新版本與 correction record。
- Curated 每一列須能追溯 raw IDs 與 transformation version。
- Window 必須在 split 之後建立，不能先滑窗再 random split。
- 儲存格式建議 Parquet；時間欄位保留 timezone，浮點精度與壓縮設定寫入 manifest。

## 6. Ingestion validation

每批資料依序做：

1. 驗證 schema version、必填欄位、型別與 finite float；拒絕 NaN／Inf。
2. 驗證單位並轉換為 canonical units；未知單位 quarantine，不猜測。
3. 以 `device_id + recorded_at_utc + sequence_no` 去重；payload 不同的 collision 送人工審核。
4. 檢查 event time 是否過度晚到、逆序、未來時間或 clock offset 異常。
5. 套用 sensor datasheet 與工程設定的 hard range；只標記可疑真實事件，不因極端就自動刪除。
6. 產生 quality flags 與 rejection reason，更新 completeness report。
7. 寫入 raw hash、batch ID、ingestion code commit 與 row counts。

物理範圍不可硬編一組通用值；不同 sensor 型號、安裝點與單位使用各自版本化設定。`humidity` 的 0–100% 是數學範圍，不等於該 sensor 的可靠操作範圍。

## 7. 對齊、缺失與異常處理

### 7.1 5 分鐘格點

- UTC 整點為 anchor，每 5 分鐘一格。
- timestamp jitter 在 ±30 秒內可 snap 到最近格點，原時間仍保留；超過範圍標記 `off_grid`。
- 同格多筆依事先定義的 aggregate（溫濕度 median；高頻振動用專屬 feature extraction）產生一列，不任意取最後一筆。
- 維修、關機與斷線造成的空白要保留，不把「沒有運轉」誤補成正常。

### 7.2 Predictor 缺失

Primary complete-case 規格要求 12×5 history 全部有效，以確保所有模型看到相同 sample set。另做 robustness sensitivity：

- 最多向前填補 1 格（5 分鐘），且只能使用 origin time 當下已知資料。
- 增加 missing mask／time-since-last-observed 作額外輸入時，必須列為不同 feature experiment。
- scaler 與任何 model-based imputer 只 fit training split。
- 超過 1 格、跨關機／維修／session 邊界或來源不可信的 gap，整個 window 排除並記錄原因。

### 7.3 Target 缺失

- 六個 target 必須全為實測合格溫度；不插值、不 forward-fill、不用模型補標籤。
- 若只缺部分 horizon，primary complete-trajectory 分析排除；可另做 per-horizon coverage 報告，但各模型必須用相同 sample IDs。
- 即時預測尚未到 target time 時標記 `pending_truth`，日後回填，不進當期 accuracy。

### 7.4 離群與真正異常

- 先以 hard invalid 規則排除不可能值／CRC 錯誤／飽和；保留 reason。
- Hampel、IQR 或 robust z-score 只能產生 `suspected_outlier` 旗標，不能自動刪除可能正是研究目標的升溫事件。
- Primary 結果保留通過硬規則的真實極端；另做含／不含疑似 sensor fault 的 sensitivity。
- Winsorization 若使用，只能由 train quantile 決定，並清楚區分 sensor error 與設備異常。

## 8. Sequence/window 規格

### 8.1 建立條件

對每個 `device_id + session_id`、每個 origin \(t\)：

```text
required predictors: 12 exact grids from t-55 through t
required labels:      6 exact grids from t+5 through t+30
no session boundary inside [t-55, t+30]
no split boundary or embargo inside the sample
all primary fields pass quality rules
```

`sample_id` 建議為下列 canonical string 的 SHA-256：

```text
dataset_version|device_id|session_id|origin_time_utc|L12|H6|feature_schema_v1
```

### 8.2 History features

- Sequence model：保留 shape `[12, 5]`。
- Ridge + History/Trend：flatten 60 個 lag，再從 predictor history 計算每個 feature 的 mean、std、min、max、last-first、least-squares slope。
- 所有 slope 的時間單位固定為每分鐘或每 5 分鐘 step，欄名寫明。
- 禁止以 `t+5...t+30` 計算 rolling mean 或 threshold flag。

### 8.3 標準化

- 每一 outer fold 只用 train rows fit scaler，套到 validation／test。
- 跨設備 zero-shot 只用 source-device train fit global scaler；不得讀 held-out device 分布。
- 溫度絕對值對 risk 很重要；六種模型均以 `T(t+h)-T(t)` 作較平穩的訓練目標，推論時加回 origin 溫度，所有指標仍以絕對 °C 真值計算。
- 保存 scaler 參數、fit sample hash 與 feature order；推論遇到 schema/order 不一致必須 fail closed。

## 9. Train／validation／test 與 gap

### 9.1 Primary per-device split

對 development devices 依 event time 排序，在扣除兩段 gap 後，將可用 sequence 依 chronological 60/20/20 分配：

```text
earliest                                         latest
|------- train 60% -------|-- val 20% --|-- locked test 20% --|
                         gap            gap
```

- 比例以扣除 gap 後的可用 sequence 計算；因格點固定，這等價於近似的時間比例。邊界應調整到 session／事件邊界，實際比例寫入 manifest。
- train window 的最後 target 不得超過 train 邊界；validation 同理。
- validation/test origin 前至少有 `H=6` steps purge gap。`gap=6` 是最低值。
- sensitivity 使用 `L+H-1=17` steps strict gap，使相鄰 split 不共享 raw predictor row。
- 若 gap 使某 split 無足夠樣本，不縮短 gap；資料集不具 readiness，回到收集階段。

### 9.2 Walk-forward

Locked test 完全不參與 tuning。前 80% development data 使用 expanding folds，每 fold 的 validation 是晚於 train 的連續 block；fold 間可重疊 train，但不可讓 target 跨界。系統 runtime 預設 3 folds；資料量足夠的正式論文可在 test 解封前預註冊 override 為 5 folds。超參數以各 folds 的 macro-device MAE 平均選擇。

### 9.3 事件與群組完整性

- 同一過熱事件與它前 `warning_lookback` 的 window 放在同一 split。
- 維修前後若系統狀態改變，切成不同 session；不可讓同一事件片段跨 train/test。
- 多 sensor 監測同一實體馬達時，以 physical device 作 group，而非 sensor ID，避免近乎複本洩漏。

### 9.4 跨設備 split

Leave-one-device-out 的主要協定把 held-out device **全部 sequence** 指派給 `external_test`，不再對它套 60/20/20。它的 raw／curated 資料在 model selection 階段不可用於 scaler、imputer、feature choice、threshold tuning 或 early stopping。若做 few-shot，需另建不同 protocol，預先固定 earliest calibration duration，之後的 test 區保持鎖定；不可把此結果取代 zero-shot。

## 10. Leakage checklist

每個 run 必須逐項留下 boolean 與證據路徑：

- [ ] split 在 window／scaling／imputation 之前凍結。
- [ ] train label 沒有落入 validation/test 時段。
- [ ] rolling features 只使用 origin 當下及以前。
- [ ] target 不插值，pending truth 不計分。
- [ ] scaler/imputer/feature selection 只 fit train。
- [ ] 同事件、同 session 的高度相關片段沒有跨 split。
- [ ] cross-device test device 沒參與任何 tuning。
- [ ] synthetic generator 未 fit test，synthetic rows 只在 train。
- [ ] warning/critical threshold 未依 test 最佳結果反覆調整。
- [ ] `status_raw`、maintenance outcome 與未來 load 沒作 predictor。
- [ ] test 只解封一次；所有額外查詢有 audit log。

## 11. 合成資料策略

### 11.1 現有 demo

保留 `DEMO-1` 訓練與 `DEMO-2` inference 分離，並增加 sequence builder 測試資料時需具備：確定 seed、已知 shape、可預期 threshold crossing、缺失／錯位 fixtures。這些資料輸出的 chart 必須加 `SYNTHETIC / PIPELINE TEST` 水印或 metadata。

### 11.2 資料增強（選配）

若真實事件稀少，可在 train split 內使用 jitter、magnitude scaling、時間遮罩或生成模型；必須：

1. 只由 train fit 參數／生成器。
2. validation 與 test 維持 100% 真實資料。
3. 做 real-only vs real+synthetic ablation。
4. 保存每筆 synthetic 的 parent IDs、方法、參數與 seed。
5. 不把合成事件數加入「真實事件數」或 safety claim。

## 12. Dataset manifest 範例

```yaml
dataset_id: edgemind-real-v1
created_at_utc: "2026-08-27T00:00:00Z"
schema_version: sensor-frame-v1
source_type: real
cadence_minutes: 5
timezone: UTC
raw_immutable: true
devices:
  - device_id: MOTOR-A
    start_utc: TBD
    end_utc: TBD
    session_count: TBD
    expected_grid_count: TBD
    observed_grid_count: TBD
    valid_core_row_count: TBD
    sensor_position: TBD
    calibration_id: TBD
files:
  - path: curated/MOTOR-A/part-000.parquet
    sha256: TBD
    rows: TBD
quality:
  valid_core_rate: TBD
  grid_coverage_rate: TBD
  duplicate_count: TBD
  off_grid_count: TBD
  excluded_windows_by_reason: {}
split:
  manifest_sha256: TBD
  strategy: per_device_chronological_60_20_20
  purge_gap_steps: 6
  strict_gap_sensitivity_steps: 17
```

## 13. 收集與凍結作業清單

### 每次安裝

- 拍照／文字記錄 sensor position、orientation、接觸方式與設備研究代碼。
- 記錄 sensor 規格、單位、range、sampling rate、firmware、校正證明。
- 用已知 reference 做開機 sanity check；建立 installation ID。

### 每日

- 監測 clock offset、late arrival、sequence gap、feature valid rate、storage backlog。
- 對資料中斷建立 incident，不直接補值後隱藏問題。
- 記錄開停機、負載模式、維修與經核准的測試事件。

### 每個 dataset release

- 關閉 late-arrival buffer 後凍結 raw batch。
- 執行 schema／quality validation 與人工抽查。
- 產生 exclusion report、事件清單與時間 coverage 圖。
- 建立 immutable manifest 與 SHA-256；寫入 code commit／transform version。
- 建立 split manifest，封存 test access，取得研究者簽核。

## 14. 資料品質報表必要欄位

每設備、每日期至少輸出：expected grids、observed grids、valid core rows、duplicate rows、late rows、off-grid rows、hard-invalid fields、forward-filled fields、excluded windows、prediction-ready windows、operating hours、event count。整體結果需同時報 micro 與 device macro 值。

若資料品質未達 [RESEARCH_METHOD.md](./RESEARCH_METHOD.md) 的 readiness gate，系統可以繼續做工程 smoke test，但不得進入正式 locked-test 比較。

## 15. Repository 內可執行的資料流程

目前 repository 提供研究形狀的合成資料 generator 與 CSV importer。以下指令從專案根目錄執行：

```bash
cd backend
python scripts/generate_research_dataset.py \
  --output-dir generated_datasets/research_v2 \
  --days 90 \
  --interval-minutes 5 \
  --seed 42
```

它會建立 `RESEARCH-A` 至 `RESEARCH-F` 六個 CSV 與 `manifest.json`；預設共 `6 × 90 × 288 = 155,520` 筆。v2 包含日夜環境、負載切換、暫態高負載、21 天磨耗／維修循環、稀有 fault window、感測噪聲與跨設備 thermal shift。Generator 只接受固定的 `--interval-minutes 5`，而 `--start` 必須帶明確 timezone 並對齊正規化後的 UTC 五分鐘網格；manifest 明確設定 `research_claims_allowed=false`，並保存相對檔名、各 CSV SHA-256、generator version／script SHA-256、sequence contract 與穩定的 `dataset_content_sha256`。`generated_at` 可隨 release 時間改變，但同一 seed、起點與程式內容應得到相同 content hash。角色為：

```text
RESEARCH-A = development device，內部 chronological 60/20/20 + gaps
RESEARCH-B = 完整保留的 external cross-device test device
RESEARCH-C..F = 額外 leave-one-device-out／domain-shift robustness devices
```

這個角色分派是軟體驗證用示範；換成真實資料時，必須在 manifest 先指定 development 與 external IDs，不可依結果更換。

為避免無意覆蓋 dataset release，輸出路徑已存在時 generator 會 fail closed。應優先使用新的 release 目錄；只有在確認目標全是可重建的合成 fixture 時才使用 `--overwrite`。正式資料 release 不應以此參數更新，應建立新 ID 與新 manifest。

若已設定可用的 `DATABASE_URL`，可驗證並匯入：

```bash
docker compose run --rm \
  -v "$PWD/backend/generated_datasets/research_v2:/datasets:ro" \
  backend python scripts/import_research_dataset.py \
  /datasets/RESEARCH-A.csv /datasets/RESEARCH-B.csv \
  /datasets/RESEARCH-C.csv /datasets/RESEARCH-D.csv \
  /datasets/RESEARCH-E.csv /datasets/RESEARCH-F.csv \
  --expected-interval-minutes 5
```

Importer 在任何資料庫異動前先驗證：必要欄位、finite 數值、濕度範圍、明確 timezone、正規化後的 UTC 五分鐘網格、逐設備精確 cadence 與跨檔案 duplicate。`is_synthetic=true` 會被保留在 status provenance，不能被空白或一般 status 洗掉。資料庫已有相同 `motor_id + recorded_at` 時預設跳過，因此可安全重跑。`--replace-device` 會刪除輸入檔內對應設備的既有資料後重匯，屬於破壞性操作；只應在明確確認測試資料庫與目標 device IDs 後使用。正式 real dataset 在匯入前仍需具備本文件要求的校正、品質、session 與 manifest metadata；目前 legacy 資料表無法保存的 metadata 應留在不可變 dataset manifest，不可直接遺失。

資料工具測試：

```bash
cd backend
python -m unittest \
  tests.test_research_dataset_generator \
  tests.test_research_dataset_importer
```

測試成功只代表 deterministic 生成、不可隱式覆寫、5 分鐘 cadence、timezone／grid 驗證、manifest hash 與 synthetic guard 正常，不代表預測研究通過 acceptance gates。

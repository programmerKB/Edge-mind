# EdgeMind — 邊緣設備診斷與溫度預測

EdgeMind 是一套面向工業馬達與邊緣設備的 AI 診斷系統。它整合感測資料、30 分鐘溫度預測、模型評估圖表與 Gemini Agent，將設備數據轉換為可讀的風險說明與維護建議。

> 數值模型負責計算預測，Gemini Agent 負責理解問題、選擇工具與整理回答；Agent 不會自行猜測設備數據。

## 文件導覽

- [主要功能](#主要功能)
- [快速啟動](#快速啟動)
- [使用方式](#使用方式)
- [系統架構](#系統架構)
- [預測模型](#預測模型)
- [推論報表與圖表](#推論報表與圖表)
- [API 參考](#api-參考)
- [開發與測試](#開發與測試)
- [部署與維運](#部署與維運)
- [常見問題](#常見問題)

## 主要功能

- 查詢設備最新溫度、濕度、XYZ 三軸加速度與震動狀態。
- 使用五項感測特徵預測 30 分鐘後的設備溫度。
- 支援以 A 設備訓練模型，再用該模型推論 B 設備。
- 提供 MAE、MSE、RMSE、R²、MAPE 與誤差中位數等回歸指標。
- 提供混淆矩陣、Precision、Recall、Specificity、F1、ROC-AUC 與 PR-AUC 等異常偵測指標。
- 彙整多次推論的平均值、中位數、標準差與 P90／P95／P99 效能。
- 後端產生 CSV 與 SVG 圖表，並透過 REST 或 SSE 將圖片附件交給前端顯示。
- 內建互相分離的 `DEMO-1` 訓練資料與 `DEMO-2` 推論資料，可快速驗證完整流程。

## 技術組成

| 分層 | 技術 |
| --- | --- |
| Web UI | React 19、Vite 8、React Markdown、Lucide React |
| API | Python 3.11、FastAPI、Pydantic、Uvicorn |
| Agent | Google Gemini、Google Gen AI SDK |
| 資料庫 | PostgreSQL 16、SQLAlchemy |
| 預測 | Python 標準函式庫實作的 Ridge Regression |
| 部署 | Docker、Docker Compose |

後端直接依賴已在 [`backend/requirements.txt`](./backend/requirements.txt) 使用 `==` 精確鎖定；前端直接與間接依賴由 [`frontend/package-lock.json`](./frontend/package-lock.json) 鎖定，確保不同環境重建時取得一致版本。

## 快速啟動

### 1. 準備環境

需要以下工具：

- Git
- 有效的 Gemini API Key
- [Docker Desktop](https://docs.docker.com/desktop/)，或 [Docker Engine](https://docs.docker.com/engine/install/) 搭配 [Docker Compose Plugin](https://docs.docker.com/compose/install/)

確認 Docker 可用：

```bash
docker --version
docker compose version
```

### 2. 取得專案

```bash
git clone https://github.com/programmerKB/Edge-mind.git
cd Edge-mind
```

### 3. 建立環境設定

在專案根目錄建立 `.env`：

```dotenv
GEMINI_API_KEY=你的_Gemini_API_Key
GEMINI_MODEL_ID=gemini-3.5-flash-lite

POSTGRES_USER=agent_user
POSTGRES_PASSWORD=請改成高強度密碼
POSTGRES_DB=motor_monitor_db
DATABASE_URL=postgresql://agent_user:請改成高強度密碼@db:5432/motor_monitor_db

# 展示環境使用 true；正式環境應改成 false
SEED_DEMO_DATA=true

# 推論輸出與異常判定
INFERENCE_OUTPUT_DIR=/app/outputs
REPORT_TIMEZONE_OFFSET_HOURS=8
ANOMALY_TEMPERATURE_THRESHOLD=35.0

# Gemini 單次回應最長等待秒數
AGENT_RESPONSE_TIMEOUT_SECONDS=60

# 多個來源用逗號分隔；正式環境不要使用 *
CORS_ORIGINS=*
```

`POSTGRES_PASSWORD` 必須與 `DATABASE_URL` 中的密碼一致。`.env` 已被 Git 忽略，請勿提交真實 API Key 或密碼。

### 4. 啟動服務

```bash
docker compose up -d --build
docker compose ps
```

服務入口：

- Web UI：<http://localhost:5173>
- Swagger API：<http://localhost:8000/docs>
- Backend：<http://localhost:8000>

若服務未正常啟動：

```bash
docker compose logs --tail=100 backend
docker compose logs --tail=100 frontend
```

## 使用方式

### Web UI

開啟 <http://localhost:5173>，輸入：

```text
請使用 DEMO-1 訓練的模型，推論 DEMO-2 在 30 分鐘後的溫度，並說明模型誤差
```

後端會執行確定性的預測工具，先透過 SSE 回傳執行狀態與 SVG 圖表附件，再由 Gemini 根據真實工具結果整理繁體中文說明。

### REST API

先訓練模型：

```bash
curl -X POST http://127.0.0.1:8000/api/predictions/train/DEMO-1
```

再使用 `DEMO-1` 模型推論 `DEMO-2`：

```bash
curl 'http://127.0.0.1:8000/api/predictions/temperature/DEMO-2?training_motor_id=DEMO-1&auto_train=false'
```

第一次啟動時，系統會建立兩個不重複寫入的合成資料集：

- `DEMO-1`：36 筆、每 5 分鐘一筆，僅用於模型訓練。
- `DEMO-2`：36 筆獨立資料，用於推論與比對 30 分鐘後真值，不會被加入訓練資料。

這些資料只用於確認流程，不代表真實設備表現。正式環境請設定 `SEED_DEMO_DATA=false`。

## 系統架構

```mermaid
flowchart LR
    UI[React Web UI] -->|SSE / REST| API[FastAPI Routers]
    API --> SERVICE[Application Services]
    SERVICE --> AGENT[Gemini Agent]
    SERVICE --> FORECAST[Forecast Engine]
    FORECAST --> REPO[Repositories]
    REPO --> DB[(PostgreSQL)]
    FORECAST --> REPORT[CSV / SVG Reports]
    REPORT -->|attachments| API
    API --> UI
```

後端依賴方向為：

```text
routers → services → forecast / reports / repositories → core / models
```

純數學模組不依賴 FastAPI、Gemini 或資料庫，因此可以獨立測試。舊的頂層模組保留為極薄的 compatibility facade，避免既有腳本的 import 立即失效；專案內部的新程式則直接引用對應 package。

### 專案結構

```text
.
├── backend/
│   ├── main.py                 # FastAPI 應用組裝入口
│   ├── core/                   # 型別化設定、資料庫與生命週期
│   ├── forecast/
│   │   ├── engine.py           # 純 Ridge 計算、特徵配對與評估
│   │   └── service.py          # 訓練、保存、推論與報表協調
│   ├── reports/
│   │   ├── charts.py           # SVG renderer
│   │   ├── inference.py        # 單次推論報表
│   │   ├── performance.py      # 跨執行效能彙整
│   │   ├── artifacts.py        # 安全圖表路徑與附件 metadata
│   │   └── io.py               # CSV 與資料集輸出
│   ├── repositories/           # 感測資料與模型的 SQLAlchemy 查詢
│   ├── routers/                # Chat、Sensors、Predictions、Reports API
│   ├── services/               # Agent、工具、SSE 與 response enrichment
│   ├── evaluation.py           # 回歸、分類與描述統計純函式
│   ├── models.py               # SQLAlchemy 資料表模型
│   ├── schemas.py              # Pydantic API 輸入契約
│   ├── seed_data.py            # 可重現的 DEMO 資料
│   ├── migrations.py           # 舊資料庫欄位與索引升級
│   ├── requirements.txt        # 精確鎖定的 Python 直接依賴
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/         # 可重用 UI 元件
│   │   ├── hooks/              # 聊天狀態與請求生命週期
│   │   ├── models/             # 訊息與附件正規化
│   │   └── services/           # SSE API 與增量資料解析
│   ├── package.json
│   └── package-lock.json
├── docker-compose.yml
└── README.md
```

### 資源生命週期

- FastAPI 啟動時建立 Gemini client、初始化資料表並視設定載入展示資料。
- 每個 REST API 與 Agent 工具使用獨立資料庫 session，結束後一律關閉；失敗時 rollback。
- 關閉應用時釋放 Gemini SDK client 與 SQLAlchemy connection pool。
- 每次推論建立獨立、不可覆寫的輸出目錄。
- React hook 管理 `AbortController`；停止回應、開始新對話或卸載元件時會中止過期請求。

## 預測模型

EdgeMind 為每個訓練設備保存獨立模型，避免不同機台的負載、環境與振動特性互相干擾。

| 項目 | 設計 |
| --- | --- |
| 輸入特徵 | 溫度、濕度、加速度 X、Y、Z |
| 預測目標 | 30 分鐘後溫度 |
| 模型 | Ridge Regression（L2 正則化線性回歸） |
| 最少資料 | 12 組有效的「當下 → 30 分鐘後」配對 |
| 時間容許 | 尋找最接近 30 分鐘後的紀錄，容許 ±5 分鐘 |
| 驗證方式 | 依時間排序，最後 20% 作為驗證資料 |
| 回歸指標 | MAE、誤差中位數、MSE、RMSE、R²、MAPE、平均誤差、最大誤差 |
| 異常指標 | Confusion Matrix、Accuracy、Precision、Recall、Specificity、F1、ROC-AUC、PR-AUC |
| 模型保存 | JSON 儲存於 PostgreSQL |

計算形式：

```text
預測溫度(t + 30 分鐘) = 截距 + Σ（特徵權重 × 標準化特徵）
```

訓練流程：

1. 依設備與時間排序歷史資料。
2. 以當下五項感測值建立輸入特徵。
3. 以最接近 30 分鐘後的真實溫度建立標籤。
4. 標準化特徵並訓練 Ridge Regression。
5. 使用時間序列尾端資料驗證，避免未來資料洩漏。
6. 使用完整有效資料重新訓練並保存模型。
7. 使用指定推論設備的最新完整資料產生預測。

常用指標解讀：

- **MAE**：平均誤差約為多少 °C；越低通常越好。
- **RMSE**：對少數大誤差給予較高懲罰；越低通常越好。
- **R²**：相對於只使用平均值所能解釋的變異，可能為負值。
- **MAPE**：相對誤差百分比；真值為零的樣本不納入計算。
- **Precision／Recall／F1**：衡量異常告警可信度、涵蓋率與兩者平衡。
- **PR-AUC**：類別不平衡時通常比 Accuracy 更有參考價值。

合成資料上的低誤差不等於真實設備準確度。上線前必須使用每台設備的真實歷史資料重新訓練與驗證。

## 推論報表與圖表

每次推論會依設定時區建立獨立輸出資料夾：

```text
backend/outputs/
├── datasets/
│   ├── training/DEMO-1.csv
│   └── inference/DEMO-2.csv
├── inference_runs/YYYY-MM-DD/HH-MM-SS-ffffff_DEMO-2/
│   ├── csv/
│   │   ├── predictions.csv
│   │   ├── metrics_summary.csv
│   │   ├── baseline_comparison.csv
│   │   ├── anomaly_detection.csv
│   │   └── system_performance.csv
│   ├── charts/
│   │   ├── 01_actual_vs_predicted.svg
│   │   ├── 02_error_curve.svg
│   │   ├── 03_error_distribution.svg
│   │   ├── 04_baseline_mae.svg
│   │   ├── 05_anomaly_f1.svg
│   │   ├── 06_error_metrics.svg
│   │   └── 07_system_performance.svg
│   └── metadata/run_summary.json
├── performance/
│   ├── performance_history.csv
│   ├── performance_summary.csv
│   └── performance_summary.json
└── latest_run.txt
```

CSV 使用帶 BOM 的 UTF-8，可直接用 Excel 開啟。已有 30 分鐘後真值的資料會標示為 `歷史回測_已取得真值`；最新預測尚未到達目標時間時，真值保持空白並標示為 `即時推論_等待真值`。

REST 預測回應會包含 `attachments`；聊天流程則以 `status: "artifacts"` 的 SSE 事件傳送相同附件。附件只傳安全的後端 URL，不傳 base64，前端收到後即可顯示 SVG。

系統會記錄 Training、Inference 時間與資源量測，再以所有已完成執行計算平均值、中位數、標準差、最小值、最大值及 P90／P95／P99。相同執行重複完成時會更新原紀錄，不會增加虛假的樣本數。

## API 參考

| 方法 | 路徑 | 用途 |
| --- | --- | --- |
| POST | `/api/chat_utf8` | Gemini Agent SSE 聊天 |
| POST | `/api/sensor-readings` | 寫入完整感測資料 |
| POST | `/api/predictions/train/{motor_id}` | 訓練並保存設備模型 |
| GET | `/api/predictions/temperature/{motor_id}` | 預測 30 分鐘後溫度；可指定 `training_motor_id` |
| GET | `/api/performance/summary` | 取得跨執行效能統計 |
| GET | `/api/report-artifacts/{path}` | 讀取預測附件中的 SVG 圖表 |

完整 request／response schema 請查看 <http://localhost:8000/docs>。

### 寫入感測資料

```bash
curl -X POST http://127.0.0.1:8000/api/sensor-readings \
  -H "Content-Type: application/json" \
  -d '{
    "motor_id": "M1",
    "temperature": 43.2,
    "humidity": 56.8,
    "accel_x": 0.12,
    "accel_y": -0.04,
    "accel_z": 1.01,
    "recorded_at": "2026-08-09T10:00:00+08:00",
    "status": "normal"
  }'
```

`motor_id`、溫度、濕度與三軸加速度為必填；所有數字都必須是有限值。`recorded_at` 未提供時由資料庫建立時間，`status` 預設為 `normal`。

## 開發與測試

### 後端

Docker 是建議的執行方式。若要直接在主機開發，請使用 Python 3.11，並將 `DATABASE_URL` 的主機改為 `127.0.0.1`：

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

執行測試與語法檢查：

```bash
cd backend
python -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/edgemind-pycache python -m compileall -q .
```

### 前端

```bash
cd frontend
npm ci
npm run lint
npm run build
npm run dev
```

### 更新依賴

- 後端更新套件時，修改 `backend/requirements.txt` 中的精確版本後重新建置映像並執行後端測試。
- 前端修改 `package.json` 後執行 `npm install`，一併提交更新後的 `package-lock.json`。
- 不要只在本機安裝新套件而未更新依賴檔案。

## 部署與維運

### 區域網路存取

Docker Compose 會讓 Vite 將 `/api` 代理到後端。區域網路裝置可直接開啟：

```text
http://192.168.1.50:5173
```

請換成部署主機 IP，並允許 TCP 5173。只有前端與 API 位於不同來源時，才需要建立 `frontend/.env.local`：

```dotenv
VITE_API_URL=http://192.168.1.50:8000/api/chat_utf8
```

這種分離部署模式也必須允許 API 連接埠。行動裝置中的 `127.0.0.1` 指向裝置本身，不是部署主機。

### 查看資料庫

```bash
docker compose exec db psql -U agent_user -d motor_monitor_db
```

```sql
SELECT
    motor_id,
    COUNT(*) AS reading_count,
    MIN(recorded_at) AS first_record,
    MAX(recorded_at) AS latest_record
FROM motor_sensor_data
GROUP BY motor_id
ORDER BY motor_id;
```

輸入 `\q` 離開 PostgreSQL。

### 備份與還原

備份：

```bash
docker compose exec -T db \
  pg_dump -U agent_user motor_monitor_db > motor_monitor_backup.sql
```

還原：

```bash
docker compose exec -T db \
  psql -U agent_user -d motor_monitor_db < motor_monitor_backup.sql
```

`docker compose down` 不會刪除 PostgreSQL volume。不要將 `docker compose down -v` 當成一般重啟，它會刪除資料庫 volume。

### 正式部署基線

目前 Compose 適合開發、展示與可信任內網。公開部署前至少應：

- 設定 `SEED_DEMO_DATA=false`。
- 將 CORS 限制為實際前端網域。
- 前端使用 production build 與正式 Web Server。
- 後端停用 Uvicorn `--reload`。
- 不對外公開 PostgreSQL 5432。
- 啟用 HTTPS、登入、授權與 rate limit。
- 使用 secret manager 保存 API Key 與資料庫密碼。
- 加入健康檢查、監控、告警、自動備份與還原演練。
- 使用真實設備資料重新訓練，並制定可接受的誤差與告警門檻。

## 常見問題

### 網頁無法開啟

```bash
docker compose ps
docker compose logs --tail=100 frontend
curl -I http://127.0.0.1:5173
```

### 前端顯示「伺服器回應錯誤」或 Failed to fetch

先確認前端代理與後端都能回應：

```bash
curl -i http://127.0.0.1:5173/api/performance/summary
curl -i http://127.0.0.1:8000/api/performance/summary
```

若使用自訂 `VITE_API_URL`，確認網址可由瀏覽器所在裝置存取，且包含 `/api/chat_utf8`。HTTPS 前端也不能直接呼叫 HTTP API。

### Agent 沒有回應

```bash
docker compose logs --tail=200 backend
```

確認 `GEMINI_API_KEY` 有效、`GEMINI_MODEL_ID` 可由該帳號使用，且主機能連線 Google API。模型超過預設 60 秒未回應時會回傳錯誤，可調整 `AGENT_RESPONSE_TIMEOUT_SECONDS`。

### 資料庫連線失敗

Docker 中的後端應使用 `db:5432`；直接在主機執行後端時才使用 `127.0.0.1:5432`。

```bash
docker compose logs --tail=100 db
```

### Docker 權限不足

依作業系統設定 Docker 使用者群組，或在必要時以具備 Docker socket 權限的帳號執行。重新登入後可用 `docker ps` 驗證。

## 已知限制

- Ridge 模型假設特徵與目標近似線性，未涵蓋複雜長期序列效應。
- 新資料不會自動觸發重新訓練，需依資料量或週期呼叫訓練 API。
- 驗證指標來自單次時間切分，不等同跨季節、跨負載條件的完整驗證。
- 尚未提供使用者登入、細粒度授權、rate limit 與完整稽核機制。
- `DEMO-1` 與 `DEMO-2` 是合成資料，只能用於功能驗證。

## License

目前專案尚未加入 License。散布、商業使用或交付第三方前，請先補上授權條款。

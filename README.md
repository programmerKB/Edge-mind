# EdgeMind 邊緣設備異常診斷 Agent

EdgeMind 是一套以 AI Agent 為核心的工業馬達／邊緣設備診斷系統。使用者可透過類似 ChatGPT 的網頁介面輸入自然語言問題，後端 Agent 會查詢 PostgreSQL 中最新的感測資料，再由 Gemini 統整設備狀態、異常原因與維護建議。

本文件提供完整的跨設備部署說明。若只想最快啟動系統，請直接閱讀「Docker Compose 一鍵部署」。

---

## 目錄

- [主要功能](#主要功能)
- [使用技術](#使用技術)
- [系統架構](#系統架構)
- [專案結構](#專案結構)
- [部署前需求](#部署前需求)
- [Docker Compose 一鍵部署](#docker-compose-一鍵部署推薦)
- [從其他電腦或手機連線](#從其他電腦或手機連線)
- [日常管理指令](#日常管理指令)
- [手動部署](#手動部署不使用完整-docker-compose)
- [API 測試](#api-測試)
- [資料庫與測試資料](#資料庫與測試資料)
- [常見問題](#常見問題)
- [正式環境注意事項](#正式環境注意事項)

## 主要功能

- ChatGPT 風格的響應式聊天介面
- 支援桌面、平板與手機瀏覽器
- 使用 Server-Sent Events（SSE）即時顯示 Agent 狀態
- Gemini Function Calling 自動呼叫設備資料查詢工具
- 從 PostgreSQL 讀取指定馬達最新的感測紀錄
- 顯示溫度、濕度、三軸 XYZ、震動、設備狀態與記錄時間
- 使用溫度、濕度、三軸 XYZ 預測 30 分鐘後溫度
- 每台設備獨立訓練並回報時間順序驗證的 MAE／RMSE
- 啟動時預載 36 筆 DEMO-1 歷史資料，可立即測試訓練與預測
- 產生繁體中文的異常分析與維護建議
- 支援 Markdown 格式的診斷報告
- 首次啟動自動建立資料表及範例資料

## 使用技術

| 分層 | 技術 | 用途 |
| --- | --- | --- |
| 前端 | React 19 | 建立聊天介面與互動狀態 |
| 前端建置 | Vite 8 | 開發伺服器、HMR 與 production build |
| UI | Lucide React、React Markdown | 圖示與 Markdown 回應排版 |
| 前端檢查 | Oxlint | JavaScript／React 靜態檢查 |
| 後端 | Python 3.11、FastAPI | API、SSE 串流與 Agent 流程 |
| ASGI Server | Uvicorn | 執行 FastAPI |
| AI | Google Gemini、Google Gen AI SDK | 語意理解、工具呼叫與報告生成 |
| ORM | SQLAlchemy | 資料表模型與查詢 |
| 資料庫 | PostgreSQL 16 | 保存設備感測資料 |
| 驗證 | Pydantic | API 請求資料驗證 |
| 部署 | Docker、Docker Compose | 管理前端、後端與資料庫 |

目前後端模型定義於 `backend/main.py`：

```python
MODEL_ID = "gemini-3.5-flash-lite"
```

模型是否可用取決於 Gemini API 專案、地區與帳號權限。如遇到模型不存在或無權限，請改成帳號可使用的 Gemini 模型。

## 系統架構

```mermaid
flowchart LR
    U[使用者瀏覽器] -->|HTTP| F[Vite + React<br/>Port 5173]
    F -->|POST + SSE| B[FastAPI Agent<br/>Port 8000]
    B -->|Function Calling| G[Google Gemini API]
    G --> B
    B -->|get_motor_status| D[(PostgreSQL 16<br/>Port 5432)]
    D --> B
    B -->|診斷報告| F
```

診斷流程：

1. 使用者在前端輸入問題。
2. 前端向 `/api/chat_utf8` 發送 POST 請求。
3. FastAPI 透過 SSE 回傳執行狀態。
4. Gemini 判斷是否需要呼叫 `get_motor_status`。
5. 工具從 PostgreSQL 查詢指定馬達最新資料。
6. Gemini 根據資料產生診斷與維護建議。
7. 前端解析 SSE 並顯示 Markdown 報告。

## 專案結構

```text
my_agent_project/
├── .env                    # Docker 與後端環境變數（不可公開）
├── README.md               # 本文件
├── docker-compose.yml      # 三個服務的一鍵部署設定
├── backend/
│   ├── Dockerfile
│   ├── main.py             # FastAPI、Gemini Agent、SSE API
│   ├── database.py         # SQLAlchemy 連線
│   ├── models.py           # 資料表模型
│   ├── tools.py            # Agent 查詢工具
│   └── requirements.txt    # Python 套件
└── frontend/
    ├── Dockerfile
    ├── package.json        # Node.js 套件與 npm 指令
    ├── vite.config.js      # Vite host 與 port
    └── src/
        ├── App.jsx         # 聊天 UI、SSE 解析
        ├── App.css         # 元件與響應式樣式
        └── index.css       # 全域樣式
```

## 服務與連接埠

| 服務 | 預設網址／連接埠 | 說明 |
| --- | --- | --- |
| 前端 | `http://localhost:5173` | 使用者聊天介面 |
| 後端 | `http://localhost:8000` | FastAPI |
| API 文件 | `http://localhost:8000/docs` | Swagger UI |
| PostgreSQL | `localhost:5432` | 資料庫 |

若從另一台設備存取，請將 `localhost` 改成部署主機 IP，例如 `http://192.168.1.50:5173`。

## 部署前需求

### 推薦：Docker

部署主機需要：

- 64 位元 Linux、Windows 或 macOS
- Docker Engine／Docker Desktop
- Docker Compose plugin
- 可連線至 Google Gemini API 的網路
- 有效的 `GEMINI_API_KEY`
- 建議至少 2 GB RAM、5 GB 可用磁碟空間

Docker Desktop 已包含 Docker Engine、CLI 與 Compose。Linux 可依官方文件安裝：

- [Docker Desktop](https://docs.docker.com/desktop/)
- [Linux 安裝 Docker Engine](https://docs.docker.com/engine/install/)
- [Linux 安裝 Docker Compose plugin](https://docs.docker.com/compose/install/linux/)

確認安裝：

```bash
docker --version
docker compose version
```

Linux 若出現 Docker socket 權限錯誤，可暫時在 Docker 指令前加 `sudo`，或依 Docker 官方 post-install 說明設定使用者群組。

### 手動部署需求

若不使用完整 Docker Compose，需要自行安裝：

- Python 3.11
- Node.js 20 與 npm
- PostgreSQL 16
- Git（若透過 Git 取得專案）

---

## Docker Compose 一鍵部署（推薦）

### 1. 取得專案

```bash
git clone <你的專案 Git URL>
cd my_agent_project
```

也可以直接複製整個資料夾。`frontend/node_modules` 與 `frontend/dist` 不需要複製，Docker 會重新建立。

### 2. 建立根目錄環境檔

建立 `my_agent_project/.env`：

```dotenv
GEMINI_API_KEY=填入你的_Gemini_API_Key

POSTGRES_USER=agent_user
POSTGRES_PASSWORD=請改成高強度密碼
POSTGRES_DB=motor_monitor_db
DATABASE_URL=postgresql://agent_user:請改成高強度密碼@db:5432/motor_monitor_db
```

注意：

- `POSTGRES_PASSWORD` 與 `DATABASE_URL` 內的密碼必須一致。
- Docker 內部資料庫主機名稱必須是 `db`，不可寫成 `localhost`。
- 密碼若包含 `@`、`:`、`/`、`#` 等特殊字元，URL 中需進行 percent encoding。
- 不可將真實 API Key 或 production 密碼提交到 Git。

### 3. 設定前端 API 網址

如果瀏覽器與部署主機是同一台設備，可略過此步。前端預設連線：

```text
http://127.0.0.1:8000/api/chat_utf8
```

如果要從其他電腦、平板或手機開啟網頁，必須建立 `frontend/.env.local`：

```dotenv
VITE_API_URL=http://192.168.1.50:8000/api/chat_utf8
```

請將 `192.168.1.50` 換成部署主機實際 IP。

Linux 查詢 IP：

```bash
hostname -I
```

Windows 查詢 IP：

```powershell
ipconfig
```

修改後重新啟動前端：

```bash
docker compose restart frontend
```

> 重要：`127.0.0.1` 永遠代表「正在使用瀏覽器的設備」。手機中的 `127.0.0.1:8000` 是手機本身，不是部署伺服器。

### 4. 建立並啟動服務

```bash
docker compose up -d --build
```

第一次執行會下載映像並安裝套件，可能需要數分鐘。

### 5. 確認容器狀態

```bash
docker compose ps
docker compose logs --tail=100
```

應看到：

- `agent-postgres`
- `agent-fastapi`
- `agent-frontend`

狀態應為 `Up` 或 `running`。

### 6. 開啟系統

部署主機本機：

```text
http://localhost:5173
```

其他區網設備：

```text
http://<部署主機 IP>:5173
```

例如：

```text
http://192.168.1.50:5173
```

API 文件：

```text
http://<部署主機 IP>:8000/docs
```

### 7. 第一次測試

在網頁輸入：

```text
請幫我檢查馬達 M1 的狀態並評估維護建議
```

系統首次啟動會建立：

- `M1`：warning 狀態
- `M2`：normal 狀態

---

## 從其他電腦或手機連線

需同時滿足：

1. 部署主機與使用者設備位於可互通的網路。
2. 前端網址使用部署主機 IP，而非 `localhost`。
3. `frontend/.env.local` 的 `VITE_API_URL` 使用部署主機 IP。
4. 防火牆允許 TCP `5173` 與 `8000`。
5. 路由器未啟用 client isolation／AP isolation。

Ubuntu UFW 範例（請依實際網段修改）：

```bash
sudo ufw allow from 192.168.1.0/24 to any port 5173 proto tcp
sudo ufw allow from 192.168.1.0/24 to any port 8000 proto tcp
```

Docker 發布連接埠與 Linux 防火牆的互動依系統而異，請依 [Docker firewall 官方說明](https://docs.docker.com/engine/network/packet-filtering-firewalls/) 驗證。

一般使用者不需要直接存取 PostgreSQL，production 環境不應對外開放 `5432`。

## 日常管理指令

以下指令均在專案根目錄執行。

### 啟動、停止與重啟

```bash
docker compose up -d
docker compose stop
docker compose restart
```

停止並移除容器（資料庫 volume 仍保留）：

```bash
docker compose down
```

只重啟單一服務：

```bash
docker compose restart frontend
docker compose restart backend
```

### 狀態與日誌

```bash
docker compose ps
docker compose logs -f
docker compose logs --tail=100 backend
```

按 `Ctrl+C` 只會停止日誌監看，不會停止容器。

### 修改程式或套件後重建

```bash
docker compose up -d --build
```

目前 Compose 已掛載原始碼，Vite 與 Uvicorn 通常會自動重新載入。修改 `package.json`、`requirements.txt` 或 Dockerfile 後應重新 build。

### 更新專案

```bash
git pull
docker compose up -d --build
docker compose ps
```

更新前建議先備份 production 資料庫。

### 完全重建資料庫

> 危險：以下指令會刪除 PostgreSQL volume 與所有資料，無法復原。

```bash
docker compose down -v
docker compose up -d --build
```

不要將 `down -v` 當作一般重啟指令。

---

## 手動部署（不使用完整 Docker Compose）

Docker Compose 是推薦做法。本節適用於需要分別管理服務的開發與整合環境。

### 1. 準備 PostgreSQL

可以只透過 Docker 啟動資料庫：

```bash
docker compose up -d db
```

或自行安裝 PostgreSQL 16 並建立帳號與資料庫。

手動執行後端時，資料庫位址應使用 `127.0.0.1`，不是 Docker 服務名稱 `db`：

```text
postgresql://agent_user:<密碼>@127.0.0.1:5432/motor_monitor_db
```

### 2. 安裝並啟動後端

Linux／macOS：

```bash
cd my_agent_project

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

export GEMINI_API_KEY="你的 Gemini API Key"
export DATABASE_URL="postgresql://agent_user:你的密碼@127.0.0.1:5432/motor_monitor_db"

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Windows PowerShell：

```powershell
cd my_agent_project

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt

$env:GEMINI_API_KEY="你的 Gemini API Key"
$env:DATABASE_URL="postgresql://agent_user:你的密碼@127.0.0.1:5432/motor_monitor_db"

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

確認後端：

```text
http://localhost:8000/docs
```

> `database.py` 在匯入階段讀取 `DATABASE_URL`，因此手動部署時必須在啟動 Uvicorn 前設定環境變數。

### 3. 安裝並啟動前端

另一個終端機：

```bash
cd my_agent_project/frontend
npm install
npm run dev
```

若後端位於其他主機，先建立 `frontend/.env.local`：

```dotenv
VITE_API_URL=http://<後端 IP 或網域>:8000/api/chat_utf8
```

前端預設監聽 `0.0.0.0:5173`。

### 4. 檢查與建置前端

```bash
cd frontend
npm run lint
npm run build
npm run preview -- --host 0.0.0.0
```

production build 位於 `frontend/dist/`。`npm run preview` 只適合檢查 build，不建議直接當正式 Web Server。

---

## API 測試

確認後端：

```bash
curl -I http://127.0.0.1:8000/docs
```

測試 Agent 串流：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat_utf8 \
  -H "Content-Type: application/json" \
  -d '{"message":"請幫我檢查馬達 M1"}'
```

Windows PowerShell：

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/api/chat_utf8 `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"請幫我檢查馬達 M1\"}"
```

正常會依序看到：

```text
data: {"status":"thought", ...}
data: {"status":"action", ...}
data: {"status":"observation", ...}
data: {"status":"success", ...}
```

若 curl 能收到 `success`，但網頁沒有內容：

- 按 `Ctrl+Shift+R` 強制重新整理。
- 確認 `VITE_API_URL` 指向正確設備。
- 檢查瀏覽器 Console／Network。
- 檢查 HTTPS mixed content、CORS、代理與防火牆。

### 30 分鐘溫度預測 API

後端啟動時會為 `DEMO-1` 預載 36 筆、每 5 分鐘一筆的合成歷史資料，且只在資料庫尚無 `DEMO-1` 時寫入，不會因重啟而重複新增。可立即執行：

```bash
curl -X POST http://127.0.0.1:8000/api/predictions/train/DEMO-1
curl http://127.0.0.1:8000/api/predictions/temperature/DEMO-1
```

若正式環境不需要展示資料，在 `.env` 加入 `SEED_DEMO_DATA=false`。自行上傳的每筆資料仍必須同時包含溫度、濕度與三軸加速度：

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
    "recorded_at": "2026-08-09T10:00:00+08:00"
  }'
```

累積歷史資料後，訓練每台設備各自的模型：

```bash
curl -X POST http://127.0.0.1:8000/api/predictions/train/M1
```

取得最新資料所對應的 30 分鐘後溫度：

```bash
curl http://127.0.0.1:8000/api/predictions/temperature/M1
```

也可以在聊天介面輸入「請預測 DEMO-1 30 分鐘後的溫度」。首次預測若尚無模型會自動嘗試訓練。訓練至少需要 12 組有效配對；每筆特徵會配對時間戳最接近 30 分鐘後（容許正負 5 分鐘）的實際溫度。API 回應包含時間順序驗證的 MAE 與 RMSE，正式使用前應依設備風險訂定可接受誤差。


## 資料庫與測試資料

後端啟動時會建立 `motor_sensor_data` 資料表。若資料表完全沒有資料，會自動加入 M1、M2 測試紀錄；預設另加入 36 筆 DEMO-1 合成歷史資料供預測展示。

| 欄位 | 型別 | 說明 |
| --- | --- | --- |
| `id` | Integer | 主鍵 |
| `motor_id` | String | 設備／馬達編號 |
| `temperature` | Float | 溫度 |
| `humidity` | Float | 相對濕度（0–100%） |
| `accel_x` | Float | 三軸感測器 X 軸 |
| `accel_y` | Float | 三軸感測器 Y 軸 |
| `accel_z` | Float | 三軸感測器 Z 軸 |
| `vibration` | Float | 震動 |
| `status` | String | 設備狀態 |
| `recorded_at` | DateTime | 紀錄時間 |

查看資料：

```bash
docker compose exec db psql -U agent_user -d motor_monitor_db
```

```sql
SELECT * FROM motor_sensor_data ORDER BY recorded_at DESC;
```

輸入 `\q` 離開 psql。若修改了 `POSTGRES_USER` 或 `POSTGRES_DB`，請同步修改指令。

### 持久化與備份

資料保存在 Docker named volume `postgres_data`。一般停止、重啟或 `docker compose down` 不會刪除資料。

備份：

```bash
docker compose exec -T db pg_dump -U agent_user motor_monitor_db > motor_monitor_backup.sql
```

還原：

```bash
docker compose exec -T db psql -U agent_user -d motor_monitor_db < motor_monitor_backup.sql
```

production 應建立定期、自動、異地保存並經過還原測試的備份策略。

## 常見問題

### 網頁打不開

```bash
docker compose ps
docker compose logs --tail=100 frontend
curl -I http://127.0.0.1:5173
```

主機本身能開、其他設備不能開，通常是 IP、防火牆、VLAN 或 Wi-Fi client isolation 問題。

### 送出 Prompt 沒有回應

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat_utf8 \
  -H "Content-Type: application/json" \
  -d '{"message":"請檢查馬達 M1"}'

docker compose logs -f backend
```

如果後端正常但瀏覽器不正常：

1. 檢查 `frontend/.env.local`。
2. 確認網址不是錯誤的 `127.0.0.1`。
3. 執行 `docker compose restart frontend`。
4. 按 `Ctrl+Shift+R`。
5. 查看瀏覽器 Network／Console。

### `Failed to fetch`

- 確認 `http://<主機 IP>:8000/docs` 可由同一瀏覽器開啟。
- 確認 `VITE_API_URL` 包含 `/api/chat_utf8`。
- 檢查 port 8000、防火牆及反向代理。
- 前端為 HTTPS、後端為 HTTP 時會被 mixed content 阻擋。

### Gemini API 錯誤

```bash
docker compose logs --tail=200 backend
```

常見原因：

- `GEMINI_API_KEY` 空白、過期或貼錯。
- API 專案額度不足。
- `MODEL_ID` 不在帳號／地區可用清單。
- 主機無法連線 Google API。
- 系統時間錯誤導致 TLS 問題。

修改 `.env` 後重建後端：

```bash
docker compose up -d --force-recreate backend
```

### 資料庫連線錯誤

Docker 內後端使用 `@db:5432`；主機手動執行的後端使用 `@127.0.0.1:5432`。

```bash
docker compose ps db
docker compose logs --tail=100 db
```

### Port 被占用

```bash
sudo ss -lntp | grep -E ':5173|:8000|:5432'
```

可修改 `docker-compose.yml` 左側主機 port：

```yaml
ports:
  - "18000:8000"
```

此時前端也要改成：

```dotenv
VITE_API_URL=http://<主機 IP>:18000/api/chat_utf8
```

### 修改依賴後仍是舊版本

```bash
docker compose build --no-cache frontend backend
docker compose up -d
```

## 正式環境注意事項

目前 Dockerfile 與 Compose 偏向開發、展示及可信任內網：

- 前端使用 Vite development server。
- 後端使用 Uvicorn `--reload`。
- FastAPI CORS 允許所有來源。
- PostgreSQL `5432` 映射到主機。
- 尚未加入 TLS／HTTPS、登入驗證與 rate limit。

若部署到公開網際網路，至少應：

1. 使用 `npm run build`，由 Nginx、Caddy 或 CDN 提供前端。
2. 後端移除 `--reload`，配置 worker 與程序管理。
3. 使用反向代理讓前端與 API 共用正式網域。
4. 啟用 HTTPS。
5. 將 CORS 限制為實際前端網域。
6. 移除 PostgreSQL 的 `5432:5432`，只保留 Docker 內部網路。
7. 使用 secret manager 管理 API Key 與密碼。
8. 加入登入、授權、rate limit 與稽核日誌。
9. 加入健康檢查、監控、告警與自動備份。
10. 不在日誌輸出敏感資料。

> 不建議直接將目前的 5173、8000、5432 三個 port 暴露到公網。

## 部署完成檢查表

- [ ] Docker 與 Docker Compose 可正常執行
- [ ] 已建立根目錄 `.env`
- [ ] 已填入有效 `GEMINI_API_KEY`
- [ ] 已設定高強度 PostgreSQL 密碼
- [ ] `DATABASE_URL` 與資料庫帳密一致
- [ ] 跨設備時已建立 `frontend/.env.local`
- [ ] `docker compose ps` 顯示三個服務正常
- [ ] 瀏覽器可開啟前端及 `/docs`
- [ ] M1 測試 Prompt 能取得結果
- [ ] 防火牆只開放必要來源與 port
- [ ] production 已建立備份策略

## 授權與維護

目前專案未附明確 LICENSE。如需散布、商業使用或交付第三方，請先補上授權條款。

維護時建議記錄部署日期、專案版本、Docker／Compose 版本、Gemini 模型、資料庫 schema 變更、環境變數變更及備份還原測試結果。

# EdgeMind v2 Pipeline Validation

> 執行日期：2026-08-28。以下是合成 Demo 的實際軟體驗證結果，不是實體設備效能或論文結論；`research_claims_allowed=false`。

## 驗證資料與環境

- `DEMO-1`／`DEMO-2`：各 2,016 筆、UTC 五分鐘 cadence、各 1,999 個完整 12×5→6 sequence。
- Chronological split：train 1,192、validation 397、locked test 398，兩個邊界各 purge 6 steps。
- Training：learned models 預測 `future - current temperature`，評估前加回 current temperature；locked test 仍以絕對 °C 評分。
- Selection：Ridge alpha grid；PyTorch validation-MAE early stopping；設定凍結後以 1,595 個 development sequences refit。
- Runtime：Python 3.12.3、SQLAlchemy 2.0.52、PyTorch 2.13.0+cpu、CPU-only、deterministic seed 42。

## 實際十模型 smoke result

| Model | Validation MAE | Locked-test MAE | Test RMSE | Test R² | Skill vs Direct Ridge | Cross-device MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct Ridge | 0.255 | 0.283 | 0.405 | 0.991 | 0.0% | 0.301 |
| Ridge + History/Trend | 0.178 | 0.180 | 0.280 | 0.996 | +36.4% | 0.216 |
| DLinear | 0.176 | 0.202 | 0.361 | 0.993 | +28.6% | 0.181 |
| LSTM | 0.171 | 0.225 | 0.372 | 0.993 | +20.5% | 0.219 |
| TCN | 0.168 | 0.207 | 0.346 | 0.994 | +26.9% | 0.175 |
| PatchTST | 0.156 | 0.199 | 0.330 | 0.994 | +29.7% | 0.212 |

PatchTST 的 development validation 最佳，所以若 protocol 要自動選一個 candidate，必須選 PatchTST；不能因看見 locked test 後 Ridge + History 較低，就事後改選 Ridge。跨設備最佳為 TCN，也不可與同設備 test winner 混成單一排名。這正是保留不同評估範圍的目的。

## 大型 synthetic release

- 路徑：`backend/generated_datasets/research_v2/`
- 規模：6 devices × 90 days × 288 readings/day = 155,520 rows。
- 範圍：2026-01-01T00:00:00Z 至 2026-03-31T23:55:00Z。
- Generator：`edgemind-research-synthetic-v2`。
- Dataset content SHA-256：`79a9f64eb864094b0a85cdbc6c129dbd669e5f1c248e18c0088d2b94ac6a3ec0`。
- 驗證：155,520 rows 全數通過必要欄位、finite float、humidity range、明確 timezone、UTC 五分鐘 grid、duplicate 與 per-device exact cadence 檢查。

CSV 是可重建的 ignored artifact，不放入 Docker image 或 Git；manifest 與 generator 決定內容。正式研究必須換成經校正、具 provenance 與 frozen manifest 的真實多設備資料。

## 工程驗證

- Backend：83 tests passed。
- Model adapters：6/6 available 且可 fit/predict。
- Dependency check：no broken requirements；SQLAlchemy、PyTorch 均可 import。
- Frontend：lint passed；production build passed。
- Deployment config：`docker compose config --quiet` passed；`deploy.sh` syntax passed。
- 實際容器部署：PostgreSQL、backend、frontend 均為 `running healthy`；backend readiness 回覆 database reachable，frontend `/healthz` 回覆 `ok`。
- Runtime inventory：6/6 models 為 `available`；資料庫內 `DEMO-1`／`DEMO-2` 已刷新為各 2,016 筆。既有 PostgreSQL volume 的角色密碼已在不刪除資料的前提下與 `.env` 同步。

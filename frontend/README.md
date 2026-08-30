# EdgeMind Frontend

這是 EdgeMind 的 React 19／Vite 8 前端，包含設備診斷聊天與研究工作台。完整系統定位、證據邊界、部署方式與 API 契約請讀 [專案 README](../README.md)。

本機前端開發：

```bash
npm ci
npm run lint
npm run build
npm run dev
```

預設由 Vite／Nginx 將 `/api` 代理到後端。只有前後端分離部署時才設定 `VITE_API_URL` 與 `VITE_RESEARCH_API_URL`；不要提交 `frontend/.env.local`。

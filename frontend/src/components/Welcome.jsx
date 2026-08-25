/** @file Empty-state introduction and predefined diagnostic prompts. */

import { Activity, Database, Gauge, Sparkles, Wrench } from 'lucide-react';
import BrandMark from './BrandMark.jsx';

const SUGGESTIONS = [
  {
    icon: Gauge,
    title: '檢查設備狀態',
    description: '分析 M1 馬達的即時感測資料',
    prompt: '請幫我檢查馬達 M1 的狀態並評估維護建議',
  },
  {
    icon: Activity,
    title: '預測未來溫度',
    description: '用 DEMO-1 模型推論 DEMO-2 資料',
    prompt: '請使用 DEMO-1 訓練的模型，推論設備 DEMO-2 在 30 分鐘後的溫度，並說明模型誤差',
  },
  {
    icon: Wrench,
    title: '建立維護建議',
    description: '產生可執行的檢查與保養清單',
    prompt: '請為馬達 M1 產生一份具體的維護檢查清單',
  },
  {
    icon: Database,
    title: '查詢感測數據',
    description: '快速取得指定設備的最新紀錄',
    prompt: '請查詢馬達 M2 的最新感測數據與健康狀態',
  },
];

/** Render prompt shortcuts that exercise the application's core workflows. */
export default function Welcome({ onSuggestion }) {
  return (
    <section className="welcome">
      <div className="welcome-mark"><BrandMark size={48} /></div>
      <p className="eyebrow"><Sparkles size={14} /> AI 設備診斷助手</p>
      <h1>今天想診斷什麼設備？</h1>
      <p className="welcome-copy">
        我可以讀取邊緣感測資料、分析異常原因，並提供具體的維護建議。
      </p>
      <div className="suggestion-grid">
        {SUGGESTIONS.map(({ icon: Icon, title, description, prompt }) => (
          <button
            className="suggestion-card"
            key={title}
            onClick={() => onSuggestion(prompt)}
          >
            <span className="suggestion-icon"><Icon size={20} /></span>
            <span><strong>{title}</strong><small>{description}</small></span>
          </button>
        ))}
      </div>
    </section>
  );
}

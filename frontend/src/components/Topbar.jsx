/** @file Compact application header and mobile navigation trigger. */

import { ChevronDown, Gauge, Menu } from 'lucide-react';

/** Render the active diagnostic mode and online indicator. */
export default function Topbar({
  onOpenMenu,
  mode = '設備診斷',
  inferenceModel,
  onInferenceModelChange,
  inferenceModelDisabled = false,
}) {
  return (
    <header className="topbar">
      <button
        className="icon-button menu-button"
        onClick={onOpenMenu}
        aria-label="開啟選單"
      >
        <Menu size={21} />
      </button>
      <button className="model-picker">
        EdgeMind <span>{mode}</span><ChevronDown size={15} />
      </button>
      {inferenceModel && onInferenceModelChange && (
        <label className="inference-picker">
          <Gauge size={15} />
          <span>溫度推論</span>
          <select
            value={inferenceModel}
            onChange={(event) => onInferenceModelChange(event.target.value)}
            disabled={inferenceModelDisabled}
            aria-label="選擇溫度推論方式"
          >
            <option value="ridge_direct">Direct Ridge</option>
            <option value="ridge_history">Ridge + History</option>
          </select>
        </label>
      )}
      <div className="topbar-status"><span className="status-dot" />線上</div>
    </header>
  );
}

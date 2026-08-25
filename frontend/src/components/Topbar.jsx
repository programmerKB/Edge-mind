/** @file Compact application header and mobile navigation trigger. */

import { ChevronDown, Menu } from 'lucide-react';

/** Render the active diagnostic mode and online indicator. */
export default function Topbar({ onOpenMenu }) {
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
        EdgeMind <span>設備診斷</span><ChevronDown size={15} />
      </button>
      <div className="topbar-status"><span className="status-dot" />線上</div>
    </header>
  );
}

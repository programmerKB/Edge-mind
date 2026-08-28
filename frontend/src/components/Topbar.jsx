/** @file Compact application header and mobile navigation trigger. */

import { FlaskConical, Menu, MessageSquareText } from 'lucide-react';

/** Render the active diagnostic mode and online indicator. */
export default function Topbar({ onOpenMenu, activeView = 'chat' }) {
  const research = activeView === 'research';
  return (
    <header className="topbar">
      <button
        className="icon-button menu-button"
        onClick={onOpenMenu}
        aria-label="開啟選單"
      >
        <Menu size={21} />
      </button>
      <div className="model-picker" aria-label={research ? '研究工作台' : '設備診斷'}>
        {research ? <FlaskConical size={16} /> : <MessageSquareText size={16} />}
        EdgeMind <span>{research ? '研究工作台' : '設備診斷'}</span>
      </div>
      <div className="topbar-status"><span className="status-dot" />{research ? '研究服務' : '線上'}</div>
    </header>
  );
}

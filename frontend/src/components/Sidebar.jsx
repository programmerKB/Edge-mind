/** @file Responsive navigation and current-session summary. */

import {
  FlaskConical,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Settings,
  X,
} from 'lucide-react';
import BrandMark from './BrandMark.jsx';

/** Render desktop navigation plus the mobile drawer/backdrop behavior. */
export default function Sidebar({
  open,
  collapsed,
  onClose,
  onToggle,
  onNewChat,
  onNavigate,
  activeView,
  hasMessages,
}) {
  return (
    <>
      {open && (
        <button
          className="sidebar-backdrop"
          onClick={onClose}
          aria-label="關閉選單"
        />
      )}
      <aside
        aria-label="診斷導覽"
        data-mobile-open={open}
        className={`sidebar ${open ? 'is-open' : ''} ${collapsed ? 'is-collapsed' : ''}`}
      >
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <BrandMark />
            {!collapsed && <span>EdgeMind</span>}
          </div>
          <button
            className="icon-button sidebar-toggle"
            onClick={onToggle}
            aria-label={collapsed ? '展開側欄' : '收合側欄'}
          >
            {collapsed ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}
          </button>
          <button
            className="icon-button mobile-close"
            onClick={onClose}
            aria-label="關閉選單"
          >
            <X size={20} />
          </button>
        </div>
        <button className="new-chat-button" onClick={onNewChat}>
          <Plus size={18} />
          {!collapsed && <span>開始新診斷</span>}
        </button>
        <nav className="sidebar-section" aria-label="主要功能">
          {!collapsed && <p className="sidebar-label">工作區</p>}
          <button
            className={`history-item ${activeView === 'chat' ? 'active' : ''}`}
            onClick={() => onNavigate('chat')}
            aria-current={activeView === 'chat' ? 'page' : undefined}
          >
            <MessageSquareText size={18} />
            {!collapsed && (
              <span>{hasMessages ? '目前的設備診斷' : '設備診斷'}</span>
            )}
          </button>
          <button
            className={`history-item ${activeView === 'research' ? 'active' : ''}`}
            onClick={() => onNavigate('research')}
            aria-current={activeView === 'research' ? 'page' : undefined}
          >
            <FlaskConical size={18} />
            {!collapsed && <span>研究工作台</span>}
          </button>
        </nav>
        <div className="sidebar-footer">
          <div className="system-status">
            <span className="status-dot" />
            {!collapsed && (
              <div>
                <strong>系統運作正常</strong>
                <span>資料服務已連線</span>
              </div>
            )}
          </div>
          <button className="sidebar-action">
            <Settings size={18} />
            {!collapsed && <span>設定</span>}
          </button>
          <div className="profile">
            <div className="avatar">U</div>
            {!collapsed && (
              <div>
                <strong>操作人員</strong>
                <span>設備維運中心</span>
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity, Bot, Check, ChevronDown, CircleAlert, Copy, Database, Gauge, Menu,
  MessageSquareText, PanelLeftClose, PanelLeftOpen, Plus, RotateCcw,
  SendHorizontal, Settings, Sparkles, Square, Wrench, X, Zap,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/api/chat_utf8';

const SUGGESTIONS = [
  { icon: Gauge, title: '檢查設備狀態', description: '分析 M1 馬達的即時感測資料', prompt: '請幫我檢查馬達 M1 的狀態並評估維護建議' },
  { icon: Activity, title: '分析異常原因', description: '判斷高溫與震動的潛在風險', prompt: '馬達溫度和震動同時升高可能有哪些原因？' },
  { icon: Wrench, title: '建立維護建議', description: '產生可執行的檢查與保養清單', prompt: '請為馬達 M1 產生一份具體的維護檢查清單' },
  { icon: Database, title: '查詢感測數據', description: '快速取得指定設備的最新紀錄', prompt: '請查詢馬達 M2 的最新感測數據與健康狀態' },
];

const STATUS_META = {
  thought: { icon: Sparkles, label: '分析中', className: 'thinking' },
  action: { icon: Zap, label: '執行工具', className: 'action' },
  observation: { icon: Database, label: '取得資料', className: 'observation' },
  error: { icon: CircleAlert, label: '發生錯誤', className: 'error' },
};

function BrandMark({ size = 32 }) {
  return <div className="brand-mark" style={{ width: size, height: size }} aria-hidden="true"><Activity size={size * 0.52} strokeWidth={2.3} /></div>;
}

function Sidebar({ open, collapsed, onClose, onToggle, onNewChat, hasMessages }) {
  return (
    <>
      {open && <button className="sidebar-backdrop" onClick={onClose} aria-label="關閉選單" />}
      <aside aria-hidden={!open} className={`sidebar ${open ? 'is-open' : ''} ${collapsed ? 'is-collapsed' : ''}`}>
        <div className="sidebar-top">
          <div className="sidebar-brand"><BrandMark />{!collapsed && <span>EdgeMind</span>}</div>
          <button className="icon-button sidebar-toggle" onClick={onToggle} aria-label={collapsed ? '展開側欄' : '收合側欄'}>
            {collapsed ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}
          </button>
          <button className="icon-button mobile-close" onClick={onClose} aria-label="關閉選單"><X size={20} /></button>
        </div>
        <button className="new-chat-button" onClick={onNewChat}><Plus size={18} />{!collapsed && <span>開始新診斷</span>}</button>
        <div className="sidebar-section">
          {!collapsed && <p className="sidebar-label">最近紀錄</p>}
          <button className={`history-item ${hasMessages ? 'active' : ''}`} onClick={onClose}>
            <MessageSquareText size={18} />{!collapsed && <span>{hasMessages ? '目前的設備診斷' : '尚無診斷紀錄'}</span>}
          </button>
        </div>
        <div className="sidebar-footer">
          <div className="system-status"><span className="status-dot" />{!collapsed && <div><strong>系統運作正常</strong><span>資料服務已連線</span></div>}</div>
          <button className="sidebar-action"><Settings size={18} />{!collapsed && <span>設定</span>}</button>
          <div className="profile"><div className="avatar">U</div>{!collapsed && <div><strong>操作人員</strong><span>設備維運中心</span></div>}</div>
        </div>
      </aside>
    </>
  );
}

function Welcome({ onSuggestion }) {
  return (
    <section className="welcome">
      <div className="welcome-mark"><BrandMark size={48} /></div>
      <p className="eyebrow"><Sparkles size={14} /> AI 設備診斷助手</p>
      <h1>今天想診斷什麼設備？</h1>
      <p className="welcome-copy">我可以讀取邊緣感測資料、分析異常原因，並提供具體的維護建議。</p>
      <div className="suggestion-grid">
        {SUGGESTIONS.map(({ icon: Icon, title, description, prompt }) => (
          <button className="suggestion-card" key={title} onClick={() => onSuggestion(prompt)}>
            <span className="suggestion-icon"><Icon size={20} /></span>
            <span><strong>{title}</strong><small>{description}</small></span>
          </button>
        ))}
      </div>
    </section>
  );
}

function StatusMessage({ message }) {
  const meta = STATUS_META[message.status] || STATUS_META.thought;
  const Icon = meta.icon;
  return (
    <div className={`status-message ${meta.className}`}>
      <span className="status-icon"><Icon size={15} /></span>
      <div><strong>{meta.label}</strong><p>{message.content}</p></div>
      {message.status === 'thought' && <span className="typing-dots"><i /><i /><i /></span>}
    </div>
  );
}

function AssistantMessage({ message, onRetry }) {
  const [copied, setCopied] = useState(false);
  const copyMessage = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  if (message.status !== 'success') return <StatusMessage message={message} />;

  return (
    <article className="assistant-row">
      <div className="assistant-avatar"><Bot size={18} /></div>
      <div className="assistant-content">
        <div className="message-author">EdgeMind</div>
        <div className="markdown-body"><ReactMarkdown>{message.content}</ReactMarkdown></div>
        <div className="message-actions">
          <button onClick={copyMessage}>{copied ? <Check size={15} /> : <Copy size={15} />}{copied ? '已複製' : '複製'}</button>
          <button onClick={onRetry}><RotateCcw size={15} />重新產生</button>
        </div>
      </div>
    </article>
  );
}

function Composer({ value, onChange, onSend, isLoading, onStop, compact }) {
  const textareaRef = useRef(null);
  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = '0px';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [value]);

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className={`composer-wrap ${compact ? 'compact' : ''}`}>
      <div className="composer">
        <button className="composer-plus" aria-label="新增附件"><Plus size={21} /></button>
        <textarea ref={textareaRef} value={value} rows={1} onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown} placeholder="詢問設備狀態、異常原因或維護建議…" aria-label="輸入診斷問題" />
        {isLoading
          ? <button className="send-button active" onClick={onStop} aria-label="停止回應"><Square size={13} fill="currentColor" /></button>
          : <button className="send-button" onClick={() => onSend()} disabled={!value.trim()} aria-label="傳送訊息"><SendHorizontal size={18} /></button>}
      </div>
      <p className="composer-hint">EdgeMind 可能會出錯，重要的設備決策請再次確認。</p>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);

  const hasMessages = messages.length > 0;
  const lastUserMessage = useMemo(() => [...messages].reverse().find((message) => message.role === 'user')?.content, [messages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);
  useEffect(() => () => abortControllerRef.current?.abort(), []);

  const stopResponse = () => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setIsLoading(false);
  };

  const newChat = () => {
    stopResponse();
    setMessages([]);
    setInput('');
    setSidebarOpen(false);
  };

  const sendMessage = async (overrideMessage) => {
    const userMessage = (typeof overrideMessage === 'string' ? overrideMessage : input).trim();
    if (!userMessage || isLoading) return;
    setInput('');
    setMessages((previous) => [...previous, { id: crypto.randomUUID(), role: 'user', content: userMessage }]);
    setIsLoading(true);
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) throw new Error(`伺服器回應錯誤 (${response.status})`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const event of events) {
          const dataText = event.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
          if (!dataText) continue;
          try {
            const data = JSON.parse(dataText);
            setMessages((previous) => [...previous, { id: crypto.randomUUID(), role: 'agent', status: data.status, content: data.content }]);
          } catch (error) {
            console.error('無法解析串流訊息：', error);
          }
        }
        if (done) break;
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        setMessages((previous) => [...previous, {
          id: crypto.randomUUID(),
          role: 'agent',
          status: 'error',
          content: error.message === 'Failed to fetch' ? '目前無法連線至診斷服務，請確認後端已啟動。' : error.message,
        }]);
      }
    } finally {
      if (abortControllerRef.current === controller) abortControllerRef.current = null;
      setIsLoading(false);
    }
  };

  const retryLastMessage = () => {
    if (lastUserMessage && !isLoading) sendMessage(lastUserMessage);
  };

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <Sidebar open={sidebarOpen} collapsed={sidebarCollapsed} onClose={() => setSidebarOpen(false)}
        onToggle={() => setSidebarCollapsed((value) => !value)} onNewChat={newChat} hasMessages={hasMessages} />
      <main className="main-panel">
        <header className="topbar">
          <button className="icon-button menu-button" onClick={() => setSidebarOpen(true)} aria-label="開啟選單"><Menu size={21} /></button>
          <button className="model-picker">EdgeMind <span>設備診斷</span><ChevronDown size={15} /></button>
          <div className="topbar-status"><span className="status-dot" />線上</div>
        </header>

        {!hasMessages ? (
          <div className="empty-state">
            <Welcome onSuggestion={sendMessage} />
            <Composer value={input} onChange={setInput} onSend={sendMessage} isLoading={isLoading} onStop={stopResponse} />
          </div>
        ) : (
          <>
            <div className="conversation" aria-live="polite">
              <div className="conversation-inner">
                {messages.map((message) => (
                  message.role === 'user'
                    ? <div className="user-row" key={message.id}><div className="user-message">{message.content}</div></div>
                    : <AssistantMessage key={message.id} message={message} onRetry={retryLastMessage} />
                ))}
                <div ref={messagesEndRef} />
              </div>
            </div>
            <Composer compact value={input} onChange={setInput} onSend={sendMessage} isLoading={isLoading} onStop={stopResponse} />
          </>
        )}
      </main>
    </div>
  );
}


import {
  Bot,
  Check,
  CircleAlert,
  Copy,
  Database,
  RotateCcw,
  Sparkles,
  Zap,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';

const STATUS_META = {
  thought: { icon: Sparkles, label: '分析中', className: 'thinking' },
  action: { icon: Zap, label: '執行工具', className: 'action' },
  observation: { icon: Database, label: '取得資料', className: 'observation' },
  error: { icon: CircleAlert, label: '發生錯誤', className: 'error' },
};

function StatusMessage({ message }) {
  const meta = STATUS_META[message.status] || STATUS_META.thought;
  const Icon = meta.icon;
  return (
    <div className={`status-message ${meta.className}`}>
      <span className="status-icon"><Icon size={15} /></span>
      <div><strong>{meta.label}</strong><p>{message.content}</p></div>
      {message.status === 'thought' && (
        <span className="typing-dots"><i /><i /><i /></span>
      )}
    </div>
  );
}

function AssistantMessage({ message, onRetry }) {
  const [copied, setCopied] = useState(false);
  const copyTimerRef = useRef(null);

  useEffect(() => () => window.clearTimeout(copyTimerRef.current), []);

  const copyMessage = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.clearTimeout(copyTimerRef.current);
      copyTimerRef.current = window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  if (message.status !== 'success') {
    return <StatusMessage message={message} />;
  }

  return (
    <article className="assistant-row">
      <div className="assistant-avatar"><Bot size={18} /></div>
      <div className="assistant-content">
        <div className="message-author">EdgeMind</div>
        <div className="markdown-body">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        <div className="message-actions">
          <button onClick={copyMessage}>
            {copied ? <Check size={15} /> : <Copy size={15} />}
            {copied ? '已複製' : '複製'}
          </button>
          <button onClick={onRetry}><RotateCcw size={15} />重新產生</button>
        </div>
      </div>
    </article>
  );
}

export default function MessageList({ messages, endRef, onRetry }) {
  return (
    <div className="conversation" aria-live="polite">
      <div className="conversation-inner">
        {messages.map((message) => (
          message.role === 'user' ? (
            <div className="user-row" key={message.id}>
              <div className="user-message">{message.content}</div>
            </div>
          ) : (
            <AssistantMessage
              key={message.id}
              message={message}
              onRetry={onRetry}
            />
          )
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}

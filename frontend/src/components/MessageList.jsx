/** @file Conversation renderers for progress, reports, and final answers. */

import {
  Bot,
  Check,
  CircleAlert,
  Copy,
  Database,
  Images,
  RotateCcw,
  Sparkles,
  Zap,
} from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import ReportGallery from './ReportGallery.jsx';

const STATUS_META = {
  thought: { icon: Sparkles, label: '分析中', className: 'thinking' },
  action: { icon: Zap, label: '執行工具', className: 'action' },
  observation: { icon: Database, label: '取得資料', className: 'observation' },
  error: { icon: CircleAlert, label: '發生錯誤', className: 'error' },
};

/** Render compact Agent progress without treating it as a final answer. */
function StatusMessage({ message, active, onRetry }) {
  const meta = STATUS_META[message.status] || STATUS_META.thought;
  const Icon = meta.icon;
  const label = message.status === 'thought' && !active
    ? '分析進度'
    : meta.label;
  return (
    <div className={`status-message ${meta.className}`}>
      <span className="status-icon"><Icon size={15} /></span>
      <div><strong>{label}</strong><p>{message.content}</p></div>
      {message.status === 'error' && (
        <button className="status-retry" type="button" onClick={onRetry}>
          <RotateCcw size={14} />重試
        </button>
      )}
      {message.status === 'thought' && active && (
        <span className="typing-dots"><i /><i /><i /></span>
      )}
    </div>
  );
}

/** Show generated charts immediately, before Gemini finishes its summary. */
function ArtifactMessage({ message }) {
  return (
    <article className="assistant-row artifact-row">
      <div className="assistant-avatar"><Images size={18} /></div>
      <div className="assistant-content">
        <div className="message-author">EdgeMind 報表</div>
        <p className="artifact-summary">{message.content}</p>
        <ReportGallery attachments={message.attachments} />
      </div>
    </article>
  );
}

/** Select the appropriate renderer for one non-user timeline item. */
function AssistantMessage({ message, active, onRetry }) {
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

  if (message.status === 'artifacts') {
    return <ArtifactMessage message={message} />;
  }

  if (message.status !== 'success') {
    return <StatusMessage message={message} active={active} onRetry={onRetry} />;
  }

  return (
    <article className="assistant-row">
      <div className="assistant-avatar"><Bot size={18} /></div>
      <div className="assistant-content">
        <div className="message-author">EdgeMind</div>
        <div className="markdown-body">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        <ReportGallery attachments={message.attachments} />
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

/** Render the ordered conversation and the scroll anchor owned by App. */
export default function MessageList({ messages, endRef, isLoading, onRetry }) {
  return (
    <div className="conversation" aria-live="polite">
      <div className="conversation-inner">
        {messages.map((message, index) => (
          message.role === 'user' ? (
            <div className="user-row" key={message.id}>
              <div className="user-message">{message.content}</div>
            </div>
          ) : (
            <AssistantMessage
              key={message.id}
              message={message}
              active={isLoading && index === messages.length - 1}
              onRetry={onRetry}
            />
          )
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}

/** @file Auto-growing chat input and send/stop controls. */

import { Plus, SendHorizontal, Square } from 'lucide-react';
import { useEffect, useRef } from 'react';

/** Render the shared composer used by both empty and active chat layouts. */
export default function Composer({
  value,
  onChange,
  onSend,
  isLoading,
  onStop,
  compact = false,
}) {
  const textareaRef = useRef(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    // Reset first so scrollHeight can also shrink after the user deletes text.
    textarea.style.height = '0px';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  }, [value]);

  const handleKeyDown = (event) => {
    // Enter submits while Shift+Enter remains available for multi-line prompts.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className={`composer-wrap ${compact ? 'compact' : ''}`}>
      <div className="composer">
        <button className="composer-plus" aria-label="新增附件">
          <Plus size={21} />
        </button>
        <textarea
          ref={textareaRef}
          value={value}
          rows={1}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="詢問設備狀態、異常原因或維護建議…"
          aria-label="輸入診斷問題"
        />
        {isLoading ? (
          <button
            className="send-button active"
            onClick={onStop}
            aria-label="停止回應"
          >
            <Square size={13} fill="currentColor" />
          </button>
        ) : (
          <button
            className="send-button"
            onClick={() => onSend()}
            disabled={!value.trim()}
            aria-label="傳送訊息"
          >
            <SendHorizontal size={18} />
          </button>
        )}
      </div>
      <p className="composer-hint">EdgeMind 可能會出錯，重要的設備決策請再次確認。</p>
    </div>
  );
}

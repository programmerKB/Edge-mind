/** @file Auto-growing chat input and send/stop controls. */

import { Cpu, Plus, SendHorizontal, Square } from 'lucide-react';
import { useEffect, useRef } from 'react';

/** Render the shared composer used by both empty and active chat layouts. */
export default function Composer({
  value,
  onChange,
  onSend,
  isLoading,
  onStop,
  selectedModel,
  modelOptions,
  onModelChange,
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
      <div className="diagnostic-model-row">
        <label className="diagnostic-model-picker">
          <Cpu size={15} aria-hidden="true" />
          <span>預測模型</span>
          <select
            value={selectedModel}
            onChange={(event) => onModelChange(event.target.value)}
            disabled={isLoading}
            aria-label="選擇設備預測模型"
          >
            {modelOptions.map((model) => (
              <option key={model.id} value={model.id}>{model.label}</option>
            ))}
          </select>
        </label>
        <small>套用於溫度與風險預測；一般狀態查詢不受影響</small>
      </div>
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

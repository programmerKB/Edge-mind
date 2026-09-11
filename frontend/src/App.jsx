/** @file Top-level responsive shell and chat/empty-state composition. */

import { useEffect, useRef, useState } from 'react';
import Composer from './components/Composer.jsx';
import MessageList from './components/MessageList.jsx';
import Sidebar from './components/Sidebar.jsx';
import Topbar from './components/Topbar.jsx';
import Welcome from './components/Welcome.jsx';
import { useChat } from './hooks/useChat.js';
import './App.css';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [inferenceModel, setInferenceModel] = useState('ridge_direct');
  const messagesEndRef = useRef(null);
  const {
    messages,
    input,
    isLoading,
    hasMessages,
    setInput,
    sendMessage,
    stopResponse,
    newChat,
    retryLastMessage,
  } = useChat(inferenceModel);

  useEffect(() => {
    // Every SSE event becomes a visible timeline item, so keep the newest one
    // in view while preserving user-controlled scrolling between events.
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  const startNewChat = () => {
    newChat();
    setSidebarOpen(false);
  };

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <Sidebar
        open={sidebarOpen}
        collapsed={sidebarCollapsed}
        onClose={() => setSidebarOpen(false)}
        onToggle={() => setSidebarCollapsed((value) => !value)}
        onNewChat={startNewChat}
        onOpenChat={() => setSidebarOpen(false)}
        hasMessages={hasMessages}
      />
      <main className="main-panel">
        <Topbar
          onOpenMenu={() => setSidebarOpen(true)}
          inferenceModel={inferenceModel}
          onInferenceModelChange={setInferenceModel}
          inferenceModelDisabled={isLoading}
        />

        {!hasMessages ? (
          <div className="empty-state">
            <Welcome onSuggestion={sendMessage} />
            <Composer
              value={input}
              onChange={setInput}
              onSend={sendMessage}
              isLoading={isLoading}
              onStop={stopResponse}
            />
          </div>
        ) : (
          <>
            <MessageList
              messages={messages}
              endRef={messagesEndRef}
              isLoading={isLoading}
              onRetry={retryLastMessage}
            />
            <Composer
              compact
              value={input}
              onChange={setInput}
              onSend={sendMessage}
              isLoading={isLoading}
              onStop={stopResponse}
            />
          </>
        )}
      </main>
    </div>
  );
}

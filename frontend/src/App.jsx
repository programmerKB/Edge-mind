/** @file Top-level responsive shell and chat/empty-state composition. */

import { useEffect, useRef, useState } from 'react';
import Composer from './components/Composer.jsx';
import MessageList from './components/MessageList.jsx';
import RidgeLab from './components/RidgeLab.jsx';
import Sidebar from './components/Sidebar.jsx';
import Topbar from './components/Topbar.jsx';
import Welcome from './components/Welcome.jsx';
import { useChat } from './hooks/useChat.js';
import './App.css';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeView, setActiveView] = useState('chat');
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
    setActiveView('chat');
    setSidebarOpen(false);
  };

  const openView = (view) => {
    setActiveView(view);
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
        activeView={activeView}
        onOpenChat={() => openView('chat')}
        onOpenRidgeLab={() => openView('ridgeLab')}
        hasMessages={hasMessages}
      />
      <main className="main-panel">
        <Topbar
          onOpenMenu={() => setSidebarOpen(true)}
          mode={activeView === 'ridgeLab' ? '雙 Ridge 實驗' : '設備診斷'}
          inferenceModel={activeView === 'chat' ? inferenceModel : undefined}
          onInferenceModelChange={activeView === 'chat' ? setInferenceModel : undefined}
          inferenceModelDisabled={isLoading}
        />

        {activeView === 'ridgeLab' ? (
          <RidgeLab />
        ) : !hasMessages ? (
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

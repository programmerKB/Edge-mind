/** @file Top-level responsive shell and chat/empty-state composition. */

import { useEffect, useRef, useState } from 'react';
import Composer from './components/Composer.jsx';
import MessageList from './components/MessageList.jsx';
import ResearchWorkspace from './components/ResearchWorkspace.jsx';
import Sidebar from './components/Sidebar.jsx';
import Topbar from './components/Topbar.jsx';
import Welcome from './components/Welcome.jsx';
import { useChat } from './hooks/useChat.js';
import './App.css';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeView, setActiveView] = useState('chat');
  const messagesEndRef = useRef(null);
  const {
    messages,
    input,
    isLoading,
    selectedModel,
    modelOptions,
    hasMessages,
    setInput,
    setSelectedModel,
    sendMessage,
    stopResponse,
    newChat,
    retryLastMessage,
  } = useChat();

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

  const navigate = (view) => {
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
        onNavigate={navigate}
        activeView={activeView}
        hasMessages={hasMessages}
      />
      <main className="main-panel">
        <Topbar
          activeView={activeView}
          onOpenMenu={() => setSidebarOpen(true)}
        />

        <section className="chat-workspace" hidden={activeView !== 'chat'} aria-label="設備診斷對話">
          {!hasMessages ? (
            <div className="empty-state">
              <Welcome onSuggestion={sendMessage} />
              <Composer
                value={input}
                onChange={setInput}
                onSend={sendMessage}
                isLoading={isLoading}
                onStop={stopResponse}
                selectedModel={selectedModel}
                modelOptions={modelOptions}
                onModelChange={setSelectedModel}
              />
            </div>
          ) : (
            <>
              <MessageList
                messages={messages}
                endRef={messagesEndRef}
                onRetry={retryLastMessage}
              />
              <Composer
                compact
                value={input}
                onChange={setInput}
                onSend={sendMessage}
                isLoading={isLoading}
                onStop={stopResponse}
                selectedModel={selectedModel}
                modelOptions={modelOptions}
                onModelChange={setSelectedModel}
              />
            </>
          )}
        </section>
        <ResearchWorkspace active={activeView === 'research'} />
      </main>
    </div>
  );
}

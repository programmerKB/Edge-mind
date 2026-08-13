import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { streamChat } from '../services/chatApi.js';

const createMessage = (role, content, status) => ({
  id: crypto.randomUUID(),
  role,
  content,
  ...(status ? { status } : {}),
});

export function useChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const activeRequestRef = useRef(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      activeRequestRef.current?.controller.abort();
      activeRequestRef.current = null;
    };
  }, []);

  const lastUserMessage = useMemo(
    () => [...messages].reverse().find((message) => message.role === 'user')?.content,
    [messages],
  );

  const stopResponse = useCallback(() => {
    activeRequestRef.current?.controller.abort();
    activeRequestRef.current = null;
    setIsLoading(false);
  }, []);

  const newChat = useCallback(() => {
    stopResponse();
    setMessages([]);
    setInput('');
  }, [stopResponse]);

  const sendMessage = useCallback(async (overrideMessage) => {
    const userMessage = (
      typeof overrideMessage === 'string' ? overrideMessage : input
    ).trim();
    if (!userMessage || activeRequestRef.current) return;

    setInput('');
    setMessages((previous) => [
      ...previous,
      createMessage('user', userMessage),
    ]);
    setIsLoading(true);

    const requestId = crypto.randomUUID();
    const controller = new AbortController();
    activeRequestRef.current = { requestId, controller };

    try {
      await streamChat(userMessage, {
        signal: controller.signal,
        onEvent: (event) => {
          if (!mountedRef.current || activeRequestRef.current?.requestId !== requestId) {
            return;
          }
          setMessages((previous) => [
            ...previous,
            createMessage('agent', event.content, event.status),
          ]);
        },
      });
    } catch (error) {
      if (error.name !== 'AbortError' && mountedRef.current) {
        const content = error.message === 'Failed to fetch'
          ? '目前無法連線至診斷服務，請確認後端已啟動。'
          : error.message;
        setMessages((previous) => [
          ...previous,
          createMessage('agent', content, 'error'),
        ]);
      }
    } finally {
      if (activeRequestRef.current?.requestId === requestId) {
        activeRequestRef.current = null;
        if (mountedRef.current) setIsLoading(false);
      }
    }
  }, [input]);

  const retryLastMessage = useCallback(() => {
    if (lastUserMessage && !activeRequestRef.current) {
      sendMessage(lastUserMessage);
    }
  }, [lastUserMessage, sendMessage]);

  return {
    messages,
    input,
    isLoading,
    hasMessages: messages.length > 0,
    setInput,
    sendMessage,
    stopResponse,
    newChat,
    retryLastMessage,
  };
}

/**
 * Own one chat session, including request cancellation and stale-event guards.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  createMessage,
  findLastUserMessage,
  normalizeAttachments,
} from '../models/chatMessage.js';
import { streamChat } from '../services/chatApi.js';

/** Return all state and actions required by the chat page. */
export function useChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const activeRequestRef = useRef(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      // Abort network work before unmount so a late SSE chunk cannot set state.
      mountedRef.current = false;
      activeRequestRef.current?.controller.abort();
      activeRequestRef.current = null;
    };
  }, []);

  const lastUserMessage = useMemo(
    () => findLastUserMessage(messages),
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
          // A stopped or superseded request may still have one decoded event in
          // the browser queue; request IDs prevent it entering the new chat.
          if (!mountedRef.current || activeRequestRef.current?.requestId !== requestId) {
            return;
          }
          setMessages((previous) => [
            ...previous,
            createMessage(
              'agent',
              event.content,
              event.status,
              normalizeAttachments(event.attachments),
            ),
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

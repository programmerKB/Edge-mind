import { API_URL } from '../config.js';

function parseEvent(rawEvent) {
  const dataText = rawEvent
    .split('\n')
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).trimStart())
    .join('\n');
  return dataText ? JSON.parse(dataText) : null;
}

export async function streamChat(message, { signal, onEvent }) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`伺服器回應錯誤 (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const events = buffer.replaceAll('\r\n', '\n').split('\n\n');
      buffer = events.pop() || '';

      for (const rawEvent of events) {
        const event = parseEvent(rawEvent);
        if (event) onEvent(event);
      }
      if (done) break;
    }

    const finalEvent = parseEvent(buffer);
    if (finalEvent) onEvent(finalEvent);
  } finally {
    reader.releaseLock();
  }
}

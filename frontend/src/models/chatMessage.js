/**
 * @file Chat-message construction and normalization.
 * Keeping transport-shaped data out of React components makes rendering code
 * independent from backend URL details.
 */

import { resolveApiUrl } from '../config.js';
import { createId } from '../utils/createId.js';

/**
 * Build the UI's immutable message shape.
 * @param {'user'|'agent'} role
 * @param {string} content
 * @param {string|undefined} status
 * @param {Array<object>} attachments
 */
export function createMessage(role, content, status, attachments = []) {
  return {
    id: createId(),
    role,
    content,
    ...(status ? { status } : {}),
    ...(attachments.length ? { attachments } : {}),
  };
}

/** Convert relative artifact paths into URLs reachable from this browser. */
export function normalizeAttachments(attachments = []) {
  return attachments.map((attachment) => ({
    ...attachment,
    url: resolveApiUrl(attachment.url),
  }));
}

/** Find the retry target without cloning and reversing the whole conversation. */
export function findLastUserMessage(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index].role === 'user') return messages[index].content;
  }
  return undefined;
}

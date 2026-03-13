/**
 * API client for the Mobitel HR Assistant backend.
 * All requests go through the Vite proxy (/api → localhost:8000).
 */

const API_BASE = '/api';

/**
 * Send a chat message and get the RAG response.
 * @param {string} message - User's question
 * @param {Array} history - Chat history [{role, content}, ...]
 * @param {string} sessionId - Session identifier
 * @param {AbortSignal} [signal] - Optional AbortSignal to cancel the request
 * @returns {Promise<{answer: string, files: Array, session_id: string}>}
 */
export async function sendMessage(message, history = [], sessionId = null, signal = null) {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      history,
      session_id: sessionId,
    }),
    signal: signal || undefined,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Server error: ${response.status}`);
  }

  return response.json();
}

/**
 * Fetch all FAQ categories.
 * @returns {Promise<Array<{category: string, faqs: Array<{question: string, answer: string}>}>>}
 */
export async function fetchFAQs() {
  const response = await fetch(`${API_BASE}/faqs`);
  if (!response.ok) throw new Error('Failed to load FAQs');
  return response.json();
}

/**
 * Get a download URL for a file.
 * @param {string} filepath - Absolute path to the file on the server
 * @returns {string} Download URL
 */
export function getDownloadUrl(filepath) {
  return `${API_BASE}/download?filepath=${encodeURIComponent(filepath)}`;
}

/**
 * Switch the active LLM provider on the backend.
 * @param {string} provider - "ollama" or "openai"
 * @returns {Promise<{success: boolean, provider: string}>}
 */
export async function switchModel(provider) {
  const response = await fetch(`${API_BASE}/model/switch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ provider }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Failed to switch model`);
  }

  return response.json();
}

/**
 * Get current model status.
 * @returns {Promise<{provider: string, model: string, last_response_time: number|null, openai_available: boolean}>}
 */
export async function getModelStatus() {
  const response = await fetch(`${API_BASE}/model/status`);
  if (!response.ok) throw new Error('Failed to get model status');
  return response.json();
}

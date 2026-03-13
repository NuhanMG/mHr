import React, { useState, useRef, useEffect } from 'react'

export default function ChatInput({ onSend, onStop, isLoading }) {
  const [text, setText] = useState('')
  const inputRef = useRef(null)

  useEffect(() => {
    if (!isLoading) {
      inputRef.current?.focus()
    }
  }, [isLoading])

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed)
    setText('')
  }

  return (
    <div className="chat-input-area">
      <form className="chat-input-wrapper" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={isLoading ? "Waiting for response..." : "Ask about HR policies, leave, forms..."}
          disabled={isLoading}
          autoComplete="off"
        />
        {isLoading ? (
          <button
            type="button"
            className="stop-btn"
            onClick={onStop}
            title="Stop generating"
          >
            ■
          </button>
        ) : (
          <button
            type="submit"
            className="send-btn"
            disabled={!text.trim()}
            title="Send message"
          >
            ➤
          </button>
        )}
      </form>
      <div className="input-hint">
        {isLoading
          ? "Processing your question... Click ■ to stop"
          : "Press Enter to send \u00B7 Powered by Mobitel HR Knowledge Base"}
      </div>
    </div>
  )
}

import React from 'react'
import ReactMarkdown from 'react-markdown'
import { getDownloadUrl } from '../api'

export default function MessageBubble({ role, content, files, followUps, onFollowUpClick }) {
  const isUser = role === 'user'

  return (
    <div className={`message-row ${role}`}>
      {!isUser && (
        <div className="message-avatar">🤖</div>
      )}

      <div className="message-bubble">
        {isUser ? (
          <span>{content}</span>
        ) : (
          <ReactMarkdown>{content}</ReactMarkdown>
        )}

        {files && files.length > 0 && (
          <div className="file-downloads">
            {files.map((file, idx) => (
              <a
                key={idx}
                href={getDownloadUrl(file.path)}
                className="file-download-btn"
                download={file.name}
                target="_blank"
                rel="noreferrer"
              >
                <span>📄</span>
                {file.name}
              </a>
            ))}
          </div>
        )}

        {followUps && followUps.length > 0 && (
          <div className="follow-up-section">
            <div className="follow-up-label">You might also want to ask:</div>
            <div className="follow-up-chips">
              {followUps.map((question, idx) => (
                <button
                  key={idx}
                  className="follow-up-chip"
                  onClick={() => onFollowUpClick?.(question)}
                >
                  <span className="chip-icon">👉</span>
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {isUser && (
        <div className="message-avatar">👤</div>
      )}
    </div>
  )
}

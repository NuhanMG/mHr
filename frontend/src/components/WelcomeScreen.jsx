import React from 'react'

const SUGGESTIONS = [
  { icon: '📋', text: 'What is the annual leave policy?' },
  { icon: '💰', text: 'How do I apply for a salary advance?' },
  { icon: '🏥', text: 'How does medical insurance work?' },
  { icon: '✈️', text: 'How to apply for foreign travel?' },
]

export default function WelcomeScreen({ onSelectQuestion }) {
  return (
    <div className="messages-container">
      <div className="welcome-screen">
        <div className="welcome-icon">🤖</div>
        <h2>Welcome to Mobitel HR Assistant</h2>
        <p>
          I can help you with HR policies, leave management, medical insurance,
          forms, and more. Just type your question or pick one below to get started.
        </p>
        <div className="welcome-suggestions">
          {SUGGESTIONS.map((s, i) => (
            <button
              key={i}
              className="suggestion-card"
              onClick={() => onSelectQuestion(s.text)}
            >
              <span className="card-icon">{s.icon}</span>
              {s.text}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

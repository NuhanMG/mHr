import React from 'react'

export default function Header({ onToggleSidebar }) {
  return (
    <div className="chat-header">
      <div className="chat-header-left">
        <button
          className="sidebar-toggle"
          onClick={onToggleSidebar}
          title="Toggle sidebar"
        >
          ☰
        </button>
        <h2>💬 Chat</h2>
        <div className="status-badge">
          <span className="status-dot"></span>
          Online
        </div>
      </div>
    </div>
  )
}

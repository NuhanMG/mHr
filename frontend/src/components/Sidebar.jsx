import React, { useState } from 'react'
import FAQPanel from './FAQPanel'

const QUICK_QUESTIONS = [
  { icon: '📋', text: 'What is the annual leave policy?' },
  { icon: '💰', text: 'How do I apply for salary advance?' },
  { icon: '🏥', text: 'How does staff medical insurance work?' },
  { icon: '✈️', text: 'How do I apply for foreign travel?' },
  { icon: '📝', text: 'Download visa application form' },
  { icon: '📅', text: 'Suggest me good days for 4 day leave' },
]

export default function Sidebar({
  isOpen,
  onSelectQuestion,
  modelProvider,
  modelName,
  lastResponseTime,
  onModelSwitch,
  openaiAvailable,
}) {
  const [activeTab, setActiveTab] = useState('quick')
  const [dropdownOpen, setDropdownOpen] = useState(false)

  const providerLabel = modelProvider === 'openai' ? 'OpenAI' : 'Ollama'
  const displayModel = modelProvider === 'openai' ? 'GPT-4.1 Nano' : 'Qwen 2.5 7B'

  const handleSelect = (provider) => {
    onModelSwitch(provider)
    setDropdownOpen(false)
  }

  return (
    <aside className={`sidebar ${isOpen ? 'open' : ''}`}>
      {/* Brand */}
      <div className="sidebar-header">
        <div className="sidebar-brand">
          <div className="sidebar-brand-icon">🤖</div>
          <div className="sidebar-brand-text">
            <h1>Mobitel HR</h1>
            <p>AI Assistant</p>
          </div>
        </div>
      </div>

      {/* Model Selector Dropdown */}
      <div className="model-selector">
        <label className="model-selector-label">LLM Model</label>
        <div className="model-dropdown-wrapper">
          <button
            className="model-dropdown-toggle"
            onClick={() => setDropdownOpen(!dropdownOpen)}
          >
            <span className="model-dropdown-icon">
              {modelProvider === 'openai' ? '🌐' : '🖥️'}
            </span>
            <span className="model-dropdown-text">
              <span className="model-provider-name">{providerLabel}</span>
              <span className="model-model-name">{displayModel}</span>
            </span>
            <span className={`model-dropdown-arrow ${dropdownOpen ? 'open' : ''}`}>▾</span>
          </button>

          {dropdownOpen && (
            <div className="model-dropdown-menu">
              <button
                className={`model-dropdown-item ${modelProvider === 'ollama' ? 'active' : ''}`}
                onClick={() => handleSelect('ollama')}
              >
                <span className="model-dropdown-icon">🖥️</span>
                <div className="model-dropdown-item-text">
                  <span className="model-item-provider">Ollama (Local)</span>
                  <span className="model-item-model">Qwen 2.5 7B</span>
                </div>
                {modelProvider === 'ollama' && <span className="model-check">✓</span>}
              </button>
              <button
                className={`model-dropdown-item ${modelProvider === 'openai' ? 'active' : ''} ${!openaiAvailable ? 'disabled' : ''}`}
                onClick={() => openaiAvailable && handleSelect('openai')}
                disabled={!openaiAvailable}
              >
                <span className="model-dropdown-icon">🌐</span>
                <div className="model-dropdown-item-text">
                  <span className="model-item-provider">OpenAI (API)</span>
                  <span className="model-item-model">GPT-4.1 Nano</span>
                </div>
                {modelProvider === 'openai' && <span className="model-check">✓</span>}
                {!openaiAvailable && <span className="model-unavailable">No Key</span>}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="sidebar-tabs">
        <button
          className={`sidebar-tab ${activeTab === 'quick' ? 'active' : ''}`}
          onClick={() => setActiveTab('quick')}
        >
          ⚡ Quick Ask
        </button>
        <button
          className={`sidebar-tab ${activeTab === 'faq' ? 'active' : ''}`}
          onClick={() => setActiveTab('faq')}
        >
          📋 FAQs
        </button>
      </div>

      {/* Content */}
      <div className="sidebar-content">
        {activeTab === 'quick' && (
          <div className="quick-actions">
            <h3>Try asking</h3>
            {QUICK_QUESTIONS.map((q, i) => (
              <button
                key={i}
                className="quick-action-btn"
                onClick={() => onSelectQuestion(q.text)}
              >
                <span className="icon">{q.icon}</span>
                {q.text}
              </button>
            ))}
          </div>
        )}

        {activeTab === 'faq' && (
          <FAQPanel onSelectQuestion={onSelectQuestion} />
        )}
      </div>

      {/* Digital Status Screen */}
      <div className="model-status-screen">
        <div className="status-screen-inner">
          <div className="status-screen-header">
            <span className="status-screen-dot"></span>
            <span className="status-screen-title">SYSTEM STATUS</span>
          </div>
          <div className="status-screen-row">
            <span className="status-label">MODEL</span>
            <span className="status-value model-value">{displayModel}</span>
          </div>
          <div className="status-screen-row">
            <span className="status-label">PROVIDER</span>
            <span className={`status-value provider-badge ${modelProvider}`}>
              {providerLabel}
            </span>
          </div>
          <div className="status-screen-row">
            <span className="status-label">LAST RESPONSE</span>
            <span className="status-value time-value">
              {lastResponseTime !== null ? `${lastResponseTime}s` : '--'}
            </span>
          </div>
        </div>
      </div>
    </aside>
  )
}

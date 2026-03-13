import React, { useState, useCallback, useEffect, useRef } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import ChatWindow from './components/ChatWindow'
import ChatInput from './components/ChatInput'
import WelcomeScreen from './components/WelcomeScreen'
import { sendMessage, switchModel, getModelStatus } from './api'

// Generate a unique session ID
function generateSessionId() {
  return 'sess_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36)
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId] = useState(generateSessionId)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Model state
  const [modelProvider, setModelProvider] = useState('ollama')
  const [modelName, setModelName] = useState('qwen2.5:7b')
  const [lastResponseTime, setLastResponseTime] = useState(null)
  const [openaiAvailable, setOpenaiAvailable] = useState(false)

  // Abort controller ref for cancelling in-flight requests
  const abortControllerRef = useRef(null)

  // Fetch initial model status with retry (backend may still be starting)
  useEffect(() => {
    let cancelled = false
    const fetchStatus = (retries = 10) => {
      getModelStatus()
        .then((status) => {
          if (cancelled) return
          setModelProvider(status.provider)
          setModelName(status.model)
          setLastResponseTime(status.last_response_time)
          setOpenaiAvailable(status.openai_available)
        })
        .catch(() => {
          if (!cancelled && retries > 0) {
            setTimeout(() => fetchStatus(retries - 1), 3000)
          }
        })
    }
    fetchStatus()
    return () => { cancelled = true }
  }, [])

  const handleModelSwitch = useCallback(async (provider) => {
    try {
      await switchModel(provider)
      const status = await getModelStatus()
      setModelProvider(status.provider)
      setModelName(status.model)
      setOpenaiAvailable(status.openai_available)
    } catch (err) {
      console.error('Model switch error:', err)
    }
  }, [])

  const handleSend = useCallback(async (text) => {
    if (!text.trim() || isLoading) return

    // Add user message immediately
    const userMsg = { role: 'user', content: text }
    setMessages((prev) => [...prev, userMsg])
    setIsLoading(true)

    // Create a new AbortController for this request
    const controller = new AbortController()
    abortControllerRef.current = controller

    try {
      // Build history (exclude files from the API payload)
      const history = messages.map(({ role, content }) => ({ role, content }))

      const data = await sendMessage(text, history, sessionId, controller.signal)

      // Update model status from response
      if (data.response_time !== null && data.response_time !== undefined) {
        setLastResponseTime(data.response_time)
      }
      if (data.model_provider) {
        setModelProvider(data.model_provider)
      }
      if (data.model_name) {
        setModelName(data.model_name)
      }

      // Add bot response
      const botMsg = {
        role: 'assistant',
        content: data.answer,
        files: data.files || [],
        followUps: data.follow_ups || [],
      }
      setMessages((prev) => [...prev, botMsg])
    } catch (error) {
      // If the request was aborted by the user, show an interrupted message
      if (error.name === 'AbortError') {
        const abortMsg = {
          role: 'assistant',
          content: '⛔ Response generation was stopped. You can retype your question or ask something else.',
          files: [],
          followUps: [],
        }
        setMessages((prev) => [...prev, abortMsg])
      } else {
        console.error('Chat error:', error)
        const errorMsg = {
          role: 'assistant',
          content: `⚠️ ${error.message || 'Something went wrong. Please try again.'}`,
          files: [],
        }
        setMessages((prev) => [...prev, errorMsg])
      }
    } finally {
      abortControllerRef.current = null
      setIsLoading(false)
    }
  }, [messages, isLoading, sessionId])

  const handleStop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
  }, [])

  const handleQuestionSelect = useCallback((question) => {
    handleSend(question)
    // Close sidebar on mobile after selecting
    if (window.innerWidth <= 900) {
      setSidebarOpen(false)
    }
  }, [handleSend])

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev)
  }, [])

  const hasMessages = messages.length > 0

  return (
    <div className="app-layout">
      {/* Mobile overlay */}
      <div
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onSelectQuestion={handleQuestionSelect}
        modelProvider={modelProvider}
        modelName={modelName}
        lastResponseTime={lastResponseTime}
        onModelSwitch={handleModelSwitch}
        openaiAvailable={openaiAvailable}
      />

      {/* Main Chat Area */}
      <main className="main-area">
        <Header onToggleSidebar={toggleSidebar} />

        {hasMessages ? (
          <ChatWindow messages={messages} isLoading={isLoading} onFollowUpClick={handleQuestionSelect} />
        ) : (
          <WelcomeScreen onSelectQuestion={handleQuestionSelect} />
        )}

        <ChatInput onSend={handleSend} onStop={handleStop} isLoading={isLoading} />
      </main>
    </div>
  )
}

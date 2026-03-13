import React, { useState, useEffect } from 'react'
import { fetchFAQs } from '../api'

export default function FAQPanel({ onSelectQuestion }) {
  const [faqData, setFaqData] = useState([])
  const [openCategory, setOpenCategory] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchFAQs()
      .then((data) => {
        setFaqData(data)
        setLoading(false)
      })
      .catch((err) => {
        console.error('Failed to load FAQs:', err)
        setLoading(false)
      })
  }, [])

  const toggleCategory = (idx) => {
    setOpenCategory(openCategory === idx ? null : idx)
  }

  if (loading) {
    return (
      <div className="faq-section">
        <p style={{ color: 'var(--color-text-muted)', fontSize: '13px', padding: '12px 8px' }}>
          Loading FAQs...
        </p>
      </div>
    )
  }

  if (!faqData.length) {
    return (
      <div className="faq-section">
        <p style={{ color: 'var(--color-text-muted)', fontSize: '13px', padding: '12px 8px' }}>
          No FAQs available.
        </p>
      </div>
    )
  }

  return (
    <div className="faq-section">
      {faqData.map((cat, catIdx) => (
        <div className="faq-category" key={catIdx}>
          <button
            className={`faq-category-header ${openCategory === catIdx ? 'open' : ''}`}
            onClick={() => toggleCategory(catIdx)}
          >
            <span>{cat.category}</span>
            <span className={`faq-chevron ${openCategory === catIdx ? 'open' : ''}`}>
              ▼
            </span>
          </button>

          {openCategory === catIdx && (
            <div className="faq-questions">
              {cat.faqs.map((faq, faqIdx) => (
                <button
                  key={faqIdx}
                  className="faq-question-btn"
                  onClick={() => onSelectQuestion(faq.question)}
                >
                  ❓ {faq.question}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

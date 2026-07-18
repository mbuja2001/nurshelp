import React, { useState } from 'react';
import { FiHelpCircle, FiChevronDown, FiChevronUp, FiMail, FiPhone, FiBook } from 'react-icons/fi';
import SidebarLayout from '../SidebarLayout';
import { useUserFromToken } from '../../hooks/useUserFromToken';

const Help = () => {
  const { userName } = useUserFromToken();
  const [expandedFaq, setExpandedFaq] = useState(null);

  const faqs = [
    {
      id: 1,
      question: 'How do I triage a new patient?',
      answer: 'Navigate to the Dashboard and click "New Triage". Follow the guided workflow to assess the patient using the ESI algorithm and clinical indicators.'
    },
    {
      id: 2,
      question: 'How does the AI assist with clinical decisions?',
      answer: 'The ESI Engine analyzes patient data and retrieves relevant clinical evidence to support your triage decisions. Always apply clinical judgment and facility protocols.'
    },
    {
      id: 3,
      question: 'Can I edit a patient record after submission?',
      answer: 'Yes, you can edit submitted records within your authority level. Navigate to the patient record and use the Edit button. All changes are logged for audit purposes.'
    },
    {
      id: 4,
      question: 'How do I access reference cases?',
      answer: 'Go to "Reference Cases Library" in the sidebar. Filter by condition, ESI level, or keyword to find similar clinical scenarios.'
    },
    {
      id: 5,
      question: 'Where can I find my shift schedule?',
      answer: 'Your shift schedule is available under "Staff Schedule" in the sidebar. You can also view it in your Profile.'
    },
  ];

  const toggleFaq = (id) => {
    setExpandedFaq(expandedFaq === id ? null : id);
  };

  return (
    <SidebarLayout username={userName} handleLogout={() => {}}>
      <div style={{ maxWidth: 1000 }}>
        {/* Header */}
        <div style={{ marginBottom: 32 }}>
          <h1 style={{ margin: '0 0 8px 0', fontSize: 28, fontWeight: 700, color: '#0f172a', display: 'flex', alignItems: 'center', gap: 12 }}>
            <FiHelpCircle size={32} style={{ color: '#0369a1' }} />
            Help & Support
          </h1>
          <p style={{ margin: 0, fontSize: 14, color: '#6b7280' }}>Get answers to common questions and find support resources</p>
        </div>

        {/* Contact Section */}
        <div style={{
          backgroundColor: 'white',
          borderRadius: 12,
          padding: 24,
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          border: '1px solid #e5e7eb',
          marginBottom: 24,
        }}>
          <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a' }}>Contact Support</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 16 }}>
            <div style={{ padding: 16, backgroundColor: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                <FiMail size={20} style={{ color: '#0369a1' }} />
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: '#0f172a' }}>Email Support</h3>
              </div>
              <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>support@nursehelp.com</p>
              <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#9ca3af' }}>Response time: 24 hours</p>
            </div>
            <div style={{ padding: 16, backgroundColor: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                <FiPhone size={20} style={{ color: '#0369a1' }} />
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: '#0f172a' }}>Phone Support</h3>
              </div>
              <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>1-800-NURSE-HELP</p>
              <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#9ca3af' }}>Mon-Fri 8am-6pm EST</p>
            </div>
            <div style={{ padding: 16, backgroundColor: '#f9fafb', borderRadius: 8, border: '1px solid #e5e7eb' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                <FiBook size={20} style={{ color: '#0369a1' }} />
                <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: '#0f172a' }}>Documentation</h3>
              </div>
              <p style={{ margin: 0, fontSize: 13, color: '#6b7280' }}>Help Center & Guides</p>
              <p style={{ margin: '8px 0 0 0', fontSize: 12, color: '#9ca3af' }}>Available 24/7</p>
            </div>
          </div>
        </div>

        {/* FAQ Section */}
        <div style={{
          backgroundColor: 'white',
          borderRadius: 12,
          padding: 24,
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          border: '1px solid #e5e7eb',
        }}>
          <h2 style={{ margin: '0 0 20px 0', fontSize: 18, fontWeight: 700, color: '#0f172a' }}>Frequently Asked Questions</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {faqs.map(faq => (
              <div
                key={faq.id}
                style={{
                  border: '1px solid #e5e7eb',
                  borderRadius: 8,
                  overflow: 'hidden',
                  backgroundColor: expandedFaq === faq.id ? '#f9fafb' : 'white',
                  transition: 'all 0.2s',
                }}
              >
                <button
                  onClick={() => toggleFaq(faq.id)}
                  style={{
                    width: '100%',
                    padding: 16,
                    backgroundColor: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    fontWeight: 600,
                    color: '#0f172a',
                    fontSize: 14,
                    textAlign: 'left',
                    transition: 'all 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = '#0369a1';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = '#0f172a';
                  }}
                >
                  {faq.question}
                  {expandedFaq === faq.id ? <FiChevronUp size={20} /> : <FiChevronDown size={20} />}
                </button>
                {expandedFaq === faq.id && (
                  <div style={{
                    padding: '0 16px 16px 16px',
                    borderTop: '1px solid #e5e7eb',
                    fontSize: 14,
                    color: '#6b7280',
                    lineHeight: 1.6,
                  }}>
                    {faq.answer}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </SidebarLayout>
  );
};

export default Help;

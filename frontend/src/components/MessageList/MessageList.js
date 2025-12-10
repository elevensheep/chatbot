import React, { useEffect, useRef } from 'react';
import './MessageList.css';
import MessageItem from '../MessageItem/MessageItem';

export default function MessageList({ messages }) {
  const ref = useRef();
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [messages]);

  return (
    <div className="cb-message-list" ref={ref}>
      {messages.length === 0 ? (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: '#666',
          textAlign: 'center',
          padding: '20px'
        }}>
          <h2 style={{ marginBottom: '10px', color: '#333' }}>👋 환영합니다!</h2>
          <p style={{ fontSize: '16px', lineHeight: '1.6' }}>
            수업 플래너 챗봇입니다.<br />
            궁금한 내용을 입력하시면 도움을 드리겠습니다.
          </p>
        </div>
      ) : (
        messages.map((m) => (
          <MessageItem key={m.id} message={m} />
        ))
      )}
    </div>
  );
}

import React, { useState } from 'react';
import './MessageInput.css';

export default function MessageInput({ onSend, sidebarOpen }) {
  const [text, setText] = useState('');

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  function send() {
    // 줄바꿈을 공백으로 변환하고 앞뒤 공백 제거
    const t = text.replace(/\n/g, ' ').trim();
    if (!t) return;
    onSend(t);
    setText('');
  }

  return (
    <div className={`cb-message-input${sidebarOpen ? ' sidebar-open' : ''}`}>
      <div className="cb-input-wrap">
        <textarea
          placeholder="메세지를 입력하세요..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={2}
        />
        <button className="cb-send" onClick={send} aria-label="전송">➤</button>
      </div>
    </div>
  );
}

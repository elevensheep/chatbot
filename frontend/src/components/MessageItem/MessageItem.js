import React from 'react';
import './MessageItem.css';

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function renderTextWithWordBreaks(text, n = 16) {
  if (!text) return null;
  const words = text.split(/(\s+)/); // split and keep spaces
  const result = [];
  let wordCount = 0;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (!/^\s+$/.test(w)) wordCount++;
    result.push(w);
    if (wordCount > 0 && wordCount % n === 0 && i < words.length - 1) {
      result.push(<br key={i} />);
    }
  }
  return result;
}

export default function MessageItem({ message }) {
  const isUser = message.from === 'user';
  if (!isUser) {
    return (
      <div className="cb-message-row bot">
        <div className="cb-message-col">
          <div className="cb-message-bubble bot">
            <div className="cb-message-text">{renderTextWithWordBreaks(message.text, 16)}</div>
          </div>
          <div className="cb-message-time bot-time">{formatTime(message.ts)}</div>
        </div>
      </div>
    );
  }
  return (
    <div className="cb-message-row user">
      <div className="cb-message-col user-col">
        <div className="cb-message-bubble user">
          <div className="cb-message-text">{renderTextWithWordBreaks(message.text, 16)}</div>
        </div>
        <div className="cb-message-time user-time">{formatTime(message.ts)}</div>
      </div>
    </div>
  );
}

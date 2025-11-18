import React from 'react';
import './MessageItem.css';

function formatTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

// 텍스트를 더 읽기 쉽게 포맷팅하는 함수
function formatMessageText(text) {
  if (!text) return '';
  
  // 문장 단위로 나누어 가독성 향상
  // 마침표, 느낌표, 물음표 뒤에 공백이 있으면 줄바꿈 추가
  let formatted = text
    // 문장 종료 기호 뒤의 공백을 줄바꿈으로 변환 (숫자. 패턴 제외)
    .replace(/([가-힣a-zA-Z])\.\s+/g, '$1.\n')
    .replace(/([!?])\s+/g, '$1\n')
    // "첫 번째", "두 번째", "세 번째" 같은 패턴 뒤에 줄바꿈
    .replace(/((?:첫|두|세|네|다섯|여섯|일곱|여덟|아홉|열)[\s]*번째[^가-힣]*)\s+/g, '$1\n')
    // 연속된 줄바꿈 정리 (최대 2개)
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  
  return formatted;
}

export default function MessageItem({ message }) {
  const isUser = message.from === 'user';
  const displayText = isUser ? message.text : formatMessageText(message.text);
  
  if (!isUser) {
    return (
      <div className="cb-message-row bot">
        <div className="cb-message-col">
          <div className="cb-message-bubble bot">
            <div className="cb-message-text">{displayText}</div>
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
          <div className="cb-message-text">{displayText}</div>
        </div>
        <div className="cb-message-time user-time">{formatTime(message.ts)}</div>
      </div>
    </div>
  );
}

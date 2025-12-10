import React from 'react';
import './Sidebar.css';

export default function Sidebar({ open, conversations = [], currentConversationId, onSelectConversation, onNewChat, onDeleteConversation, onLogout }) {
  const handleDelete = (e, conversationId) => {
    e.stopPropagation(); // 채팅방 선택 이벤트 방지
    if (window.confirm('이 대화를 삭제하시겠습니까?')) {
      onDeleteConversation(conversationId);
    }
  };

  return (
    <div className={`sidebar${open ? ' sidebar--open' : ''}`}> 
      <div className="sidebar__menu">
        {/* 상단: 새 채팅 버튼 */}
        <button className="sidebar__newchat-btn" onClick={onNewChat}>
          새 채팅
        </button>
        
        {/* 채팅 목록 제목 */}
        <div className="sidebar__item sidebar__item--title">채팅 목록</div>
        
        {/* 채팅방 목록 */}
        <div className="sidebar__sessions">
          {conversations.length === 0 ? (
            <div style={{ padding: '10px', color: '#999', fontSize: '14px', textAlign: 'center' }}>
              채팅방이 없습니다
            </div>
          ) : (
            conversations.map((conversation) => {
              const isActive = conversation.id === currentConversationId;
              return (
                <div
                  key={conversation.id}
                  className={`sidebar__session${isActive ? ' sidebar__session--active' : ''}`}
                  onClick={() => onSelectConversation(conversation.id)}
                  title={conversation.title}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        fontWeight: isActive ? 600 : 400
                      }}>
                        {conversation.title || '새 대화'}
                      </div>
                      {conversation.updated_at && (
                        <div style={{
                          fontSize: '12px',
                          color: isActive ? 'rgba(255,255,255,0.8)' : '#999',
                          marginTop: '4px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}>
                          {new Date(conversation.updated_at).toLocaleDateString('ko-KR', {
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                          })}
                        </div>
                      )}
                    </div>
                    <button
                      className="sidebar__delete-btn"
                      onClick={(e) => handleDelete(e, conversation.id)}
                      title="대화 삭제"
                    >
                      ×
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
      
      {/* 하단: 로그아웃 버튼 */}
      <div className="sidebar__footer">
        <div className="sidebar__item" style={{ cursor: 'pointer', color: '#d32f2f', fontWeight: 600 }} onClick={onLogout}>
          로그아웃
        </div>
      </div>
    </div>
  );
}

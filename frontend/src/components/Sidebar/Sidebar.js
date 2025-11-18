import React, { useEffect, useState } from 'react';
import './Sidebar.css';
import { getConversations, createConversation } from '../../utils/api';

export default function Sidebar({ open, sessions = [], currentSessionIdx, onSelectSession, onNewChat, onSelectRoom }) {
  const [rooms, setRooms] = useState([]);
  const [newRoomTitle, setNewRoomTitle] = useState('');

  useEffect(() => {
    let isMounted = true;
    
    async function fetchRooms() {
      try {
        const data = await getConversations();
        if (isMounted) {
          setRooms(data);
        }
      } catch (error) {
        console.error('Failed to load chat rooms:', error);
      }
    }
    fetchRooms();
    
    return () => {
      isMounted = false;
    };
  }, []);

  async function handleCreateRoom() {
    if (!newRoomTitle.trim()) return;
    try {
      const newRoom = await createConversation(newRoomTitle);
      setRooms([newRoom, ...rooms]);
      setNewRoomTitle('');
    } catch (error) {
      console.error('Failed to create chat room:', error);
    }
  }

  return (
    <div className={`sidebar${open ? ' sidebar--open' : ''}`}> 
      <div className="sidebar__menu">
        {/* 상단: 새 채팅/채팅목록/채팅 세션 */}
        <button className="sidebar__newchat-btn" onClick={onNewChat}>
          새 채팅
        </button>
        <div className="sidebar__item sidebar__item--title">채팅 목록</div>
        <div className="sidebar__sessions">
          {sessions
            .filter(session => session.messages.some(m => m.from === 'user'))
            .map((session, idx) => {
              const lastUserMsg = [...session.messages].reverse().find(m => m.from === 'user');
              const preview = lastUserMsg ? lastUserMsg.text.slice(0, 18) : '';
              return (
                <div
                  key={session.id}
                  className={`sidebar__session${idx === currentSessionIdx ? ' sidebar__session--active' : ''}`}
                  onClick={() => onSelectSession(idx)}
                >
                  {preview}
                </div>
              );
            })}
        </div>
      </div>
      {/* 하단: 활동/설정 및 도움말 */}
      <div className="sidebar__footer">
        <div className="sidebar__item">활동</div>
        <div className="sidebar__item">설정 및 도움말</div>
      </div>

      {/* 새로운 대화방 관리 로직 (UI 변경 없음) */}
      <div style={{ display: 'none' }}>
        <input
          type="text"
          placeholder="새 대화방 제목"
          value={newRoomTitle}
          onChange={(e) => setNewRoomTitle(e.target.value)}
        />
        <button onClick={handleCreateRoom}>대화방 생성</button>
        <div>
          {rooms.map((room) => (
            <div key={room.id} onClick={() => onSelectRoom(room.id)}>
              {room.title}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

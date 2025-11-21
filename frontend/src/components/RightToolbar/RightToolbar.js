import React from 'react';
import './RightToolbar.css';


export default function RightToolbar({ onToggle, sidebarOpen, onNewChat, onLoginClick, onLogoutClick, isLoggedIn }) {
  
  return (
    <nav className="rightbar-toolbar-22" aria-label="오른쪽 툴바">
      <div className="rightbar-toolbar-22__top">
        <button className="rightbar-toolbar-22__btn" onClick={onToggle} aria-label="메뉴">
          <img src="/아이콘/세점.png" alt="메뉴" className="rightbar-toolbar-22__icon" />
        </button>
        <button className="rightbar-toolbar-22__btn" onClick={onNewChat} aria-label="새 채팅">
          <img src="/아이콘/연필.png" alt="새 채팅" className="rightbar-toolbar-22__icon" />
        </button>
      </div>
    </nav>
  );
}

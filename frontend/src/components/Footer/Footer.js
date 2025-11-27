import React from 'react';
import './Footer.css';

export default function Footer({ currentPage, onPageChange }) {
  return (
    <div className="cb-footer-hint">
      {/* 교수용 수업페이지 버튼 (임시 비활성화)
      <div className="footer-left">
        <button 
          className="study-plan-btn" 
          onClick={handleStudyPlanClick}
        >
          {currentPage === 'studyplan' ? '채팅으로 돌아가기' : '교수용 수업계획서'}
        </button>
      </div>
      */}
      <div className="footer-center">
        Enter로 전송, Shift + Enter로 줄바꿈
      </div>
    </div>
  );
}

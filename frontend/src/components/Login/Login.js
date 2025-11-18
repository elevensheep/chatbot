import React, { useState } from 'react';
import { authAPI } from '../../utils/api';
import './Login.css';

export default function Login({ onLogin, onSignup, onBack }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    if (!email || !password) {
      setError('이메일과 비밀번호를 입력하세요.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await authAPI.login(email, password);
      
      // 로그인 성공시 토큰을 localStorage에 저장
      // 백엔드 응답 형식: { success: true, data: { access_token: "...", ... } }
      if (response.success && response.data?.access_token) {
        localStorage.setItem('access_token', response.data.access_token);
        localStorage.setItem('authToken', response.data.access_token); // 호환성을 위해 유지
        localStorage.setItem('userEmail', email);
      }
      
      onLogin(email);
    } catch (error) {
      console.error('Login failed:', error);
      setError(error.message || '로그인에 실패했습니다. 이메일과 비밀번호를 확인하세요.');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="login-container">
      {onBack && (
        <button
          type="button"
          className="back-btn"
          onClick={onBack}
        >
          ← 뒤로가기
        </button>
      )}
      <form className="login-form" onSubmit={handleSubmit}>
        <h2>로그인</h2>
        <input
          type="email"
          placeholder="이메일"
          value={email}
          onChange={e => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="비밀번호"
          value={password}
          onChange={e => setPassword(e.target.value)}
        />
        {error && <div className="login-error">{error}</div>}
        <button type="submit" disabled={isLoading}>
          {isLoading ? '로그인 중...' : '로그인'}
        </button>
        <button
          type="button"
          className="login-signup-btn"
          onClick={onSignup}
          style={{ marginTop: 8, background: '#eee', color: '#1976d2' }}
        >
          회원가입
        </button>
      </form>
    </div>
  );
}

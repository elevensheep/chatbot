import React, { useState } from 'react';
import { authAPI } from '../../utils/api';
import './Login.css';

export default function Login({ onLogin, onSignup, onBack, onFindPassword }) {
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
      console.log('로그인 요청 시작:', email);
      console.log('authAPI.login 함수 호출 직전');
      const loginPromise = authAPI.login(email, password);
      console.log('Promise 생성됨:', loginPromise);
      const response = await loginPromise;
      console.log('로그인 응답:', response);

      // 로그인 성공시 토큰을 localStorage에 저장
      // 백엔드는 { success: true, data: { access_token, user } } 형식으로 반환
      const accessToken = response.data?.access_token || response.access_token;
      const userData = response.data?.user || response.user;

      console.log('추출된 토큰:', accessToken);
      console.log('추출된 사용자 정보:', userData);

      if (accessToken) {
        localStorage.setItem('authToken', accessToken);
        localStorage.setItem('access_token', accessToken); // API 호출용
        localStorage.setItem('userEmail', email);

        // 사용자 정보도 저장
        if (userData) {
          localStorage.setItem('user', JSON.stringify(userData));
        }
        console.log('localStorage 저장 완료');
      } else {
        console.error('토큰이 없습니다!');
      }

      console.log('onLogin 호출 전');
      onLogin(email);
      console.log('onLogin 호출 완료');
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
        <button
          type="button"
          className="login-signup-btn"
          onClick={onFindPassword}
          style={{ marginTop: 8, background: '#eee', color: '#1976d2' }}
        >
          비밀번호 찾기
        </button>
      </form>
    </div>
  );
}

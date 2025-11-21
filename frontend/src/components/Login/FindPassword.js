import React, { useState } from 'react';
import { authAPI } from '../../utils/api';
import './Login.css';

export default function FindPassword({ onBack }) {
  const [step, setStep] = useState(1); // 1: 이메일 입력, 2: 인증코드+비밀번호 입력
  const [email, setEmail] = useState('');
  const [token, setToken] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // 1단계: 이메일 입력 → 인증 코드 발송
  async function handleRequestReset(e) {
    e.preventDefault();
    setMessage('');
    setError('');

    if (!email) {
      setError('이메일을 입력하세요.');
      return;
    }

    setIsLoading(true);
    try {
      const response = await authAPI.requestPasswordReset(email);
      setMessage(response.message || '비밀번호 재설정 인증 코드가 이메일로 발송되었습니다.');
      setError('');
      setStep(2); // 다음 단계로 이동
    } catch (error) {
      setError(error.message || '비밀번호 재설정 요청에 실패했습니다.');
      setMessage('');
    } finally {
      setIsLoading(false);
    }
  }

  // 2단계: 인증 코드 + 새 비밀번호 입력 → 비밀번호 변경
  async function handleConfirmReset(e) {
    e.preventDefault();
    setMessage('');
    setError('');

    if (!token || !newPassword || !confirmPassword) {
      setError('모든 필드를 입력하세요.');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('비밀번호가 일치하지 않습니다.');
      return;
    }

    // 비밀번호 유효성 검증
    if (newPassword.length < 8) {
      setError('비밀번호는 최소 8자 이상이어야 합니다.');
      return;
    }

    if (!/[a-zA-Z]/.test(newPassword)) {
      setError('비밀번호는 영문을 포함해야 합니다.');
      return;
    }

    if (!/\d/.test(newPassword)) {
      setError('비밀번호는 숫자를 포함해야 합니다.');
      return;
    }

    if (!/[!@#$%^&*(),.?":{}|<>]/.test(newPassword)) {
      setError('비밀번호는 특수문자를 포함해야 합니다.');
      return;
    }

    setIsLoading(true);
    try {
      console.log('비밀번호 재설정 요청:', { email, token, newPassword });
      const response = await authAPI.confirmPasswordReset(email, token, newPassword);
      console.log('비밀번호 재설정 성공:', response);
      setMessage(response.message || '비밀번호가 성공적으로 변경되었습니다. 로그인 페이지로 돌아가세요.');
      setError('');

      // 3초 후 자동으로 로그인 페이지로 이동
      setTimeout(() => {
        onBack();
      }, 3000);
    } catch (error) {
      console.error('비밀번호 재설정 실패:', error);
      setError(error.message || '비밀번호 변경에 실패했습니다.');
      setMessage('');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="login-container">
      <button type="button" className="back-btn" onClick={onBack}>
        ← 뒤로가기
      </button>

      {step === 1 ? (
        // 1단계: 이메일 입력
        <form className="login-form" onSubmit={handleRequestReset}>
          <h2>비밀번호 찾기</h2>
          <p style={{ fontSize: 14, color: '#666', marginBottom: 16 }}>
            등록된 이메일로 인증 코드를 발송합니다.
          </p>
          <input
            type="email"
            placeholder="이메일 (@bu.ac.kr)"
            value={email}
            onChange={e => setEmail(e.target.value)}
          />
          {error && <div className="login-error" style={{ color: 'red' }}>{error}</div>}
          {message && <div className="login-error" style={{ color: 'green' }}>{message}</div>}
          <button type="submit" disabled={isLoading} style={{ marginTop: 8 }}>
            {isLoading ? '요청 중...' : '인증 코드 발송'}
          </button>
        </form>
      ) : (
        // 2단계: 인증 코드 + 새 비밀번호 입력
        <form className="login-form" onSubmit={handleConfirmReset}>
          <h2>비밀번호 재설정</h2>
          <p style={{ fontSize: 14, color: '#666', marginBottom: 16 }}>
            이메일로 받은 6자리 인증 코드와 새 비밀번호를 입력하세요.
          </p>
          <input
            type="text"
            placeholder="인증 코드 (6자리 숫자)"
            value={token}
            onChange={e => setToken(e.target.value)}
            maxLength={6}
            pattern="[0-9]*"
          />
          <input
            type="password"
            placeholder="새 비밀번호 (영문+숫자+특수문자, 8자 이상)"
            value={newPassword}
            onChange={e => setNewPassword(e.target.value)}
          />
          <input
            type="password"
            placeholder="비밀번호 확인"
            value={confirmPassword}
            onChange={e => setConfirmPassword(e.target.value)}
          />
          {error && <div className="login-error" style={{ color: 'red' }}>{error}</div>}
          {message && <div className="login-error" style={{ color: 'green' }}>{message}</div>}
          <button type="submit" disabled={isLoading} style={{ marginTop: 8 }}>
            {isLoading ? '변경 중...' : '비밀번호 변경'}
          </button>
          <button
            type="button"
            onClick={() => setStep(1)}
            style={{ marginTop: 8, background: '#eee', color: '#1976d2' }}
          >
            이메일 다시 입력
          </button>
        </form>
      )}
    </div>
  );
}

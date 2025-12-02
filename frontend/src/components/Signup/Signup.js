import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { authAPI } from '../../utils/api';
import './Signup.css';

export default function Signup({ onSignup, onBack }) {
  const [emailPrefix, setEmailPrefix] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isEmailVerified, setIsEmailVerified] = useState(false);
  const [verificationCode, setVerificationCode] = useState('');
  const [sentCode, setSentCode] = useState('');
  const [isVerifying, setIsVerifying] = useState(false);

  const fullEmail = emailPrefix ? `${emailPrefix}@bu.ac.kr` : '';

  const location = useLocation();

  // 이메일 링크로 인증된 경우 처리: /signup?verifiedEmail=student@bu.ac.kr
  useEffect(() => {
    try {
      const params = new URLSearchParams(location.search);
      const verifiedEmail = params.get('verifiedEmail');
      if (verifiedEmail && verifiedEmail.endsWith('@bu.ac.kr')) {
        const prefix = verifiedEmail.replace('@bu.ac.kr', '');
        setEmailPrefix(prefix);
        setIsEmailVerified(true);
        // 깔끔하게 쿼리 제거
        const newUrl = window.location.origin + '/signup';
        window.history.replaceState({}, '', newUrl);
      }
    } catch (e) {
      // ignore
    }
  }, [location.search]);

  // 팝업 창에서 이메일 인증 완료 메시지 수신 처리
  useEffect(() => {
    const handleMessage = (event) => {
      // 보안을 위해 origin 확인 (프로토콜 제외하고 도메인만 비교)
      const currentOrigin = window.location.origin;
      const eventOrigin = event.origin;
      
      // 같은 도메인이면 HTTP/HTTPS 모두 허용
      const currentDomain = currentOrigin.replace(/^https?:\/\//, '');
      const eventDomain = eventOrigin.replace(/^https?:\/\//, '');
      
      if (currentDomain !== eventDomain) {
        return;
      }

      if (event.data && event.data.type === 'EMAIL_VERIFIED') {
        const verifiedEmail = event.data.email;
        if (verifiedEmail && verifiedEmail.endsWith('@bu.ac.kr')) {
          const prefix = verifiedEmail.replace('@bu.ac.kr', '');
          setEmailPrefix(prefix);
          setIsEmailVerified(true);
        }
      }
    };

    window.addEventListener('message', handleMessage);
    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, []);

  // 이메일 인증 코드 발송
  async function handleSendVerification() {
    if (!emailPrefix.trim()) {
      setError('이메일을 입력해주세요.');
      return;
    }

    setIsVerifying(true);
    setError('');

    try {
      // 실제 API 호출
      // 서버는 이메일에 6자리 인증번호를 포함하여 전송합니다.
      await authAPI.sendVerificationEmail(fullEmail);
      alert(`${fullEmail}로 인증 메일을 발송했습니다. 메일의 6자리 인증번호를 입력해주세요.`);
      
      // 인증번호 입력 UI 표시를 위해 sentCode 설정 (실제 코드는 이메일에서 확인)
      setSentCode('sent'); // 플래그로 사용
      
      // 개발 모드에서는 디버깅용 로그만 출력
      if (process.env.NODE_ENV === 'development') {
        console.log('개발 모드: 인증 이메일 발송 호출 완료');
      }
    } catch (error) {
      console.error('Verification email failed:', error);
      
      // 백엔드에서 전달된 구체적인 에러 메시지 사용
      const errorMessage = error.message || '인증번호 발송에 실패했습니다. 다시 시도해주세요.';
      setError(errorMessage);
      
      // 개발용 폴백: API 실패 시에도 임시 코드 생성 (409 에러가 아닐 때만)
      if (process.env.NODE_ENV === 'development' && !errorMessage.includes('이미 등록된')) {
        const code = Math.floor(100000 + Math.random() * 900000).toString();
        setSentCode(code);
        alert(`개발 모드: 인증번호 ${code}`);
      }
    } finally {
      setIsVerifying(false);
    }
  }

  // 인증번호 확인
  async function handleVerifyCode() {
    if (!verificationCode.trim()) {
      setError('인증번호를 입력해주세요.');
      return;
    }

    setError('');

    try {
      // 실제 API 호출
      await authAPI.verifyEmail(fullEmail, verificationCode);
      setIsEmailVerified(true);
      alert('이메일 인증이 완료되었습니다.');
    } catch (error) {
      console.error('Email verification failed:', error);
      
      // 개발용 폴백: sentCode와 비교
      if (process.env.NODE_ENV === 'development' && verificationCode === sentCode) {
        setIsEmailVerified(true);
        alert('이메일 인증이 완료되었습니다. (개발 모드)');
      } else {
        setError('인증번호가 올바르지 않습니다.');
      }
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    
    // 유효성 검사
    if (!emailPrefix.trim()) {
      setError('이메일을 입력해주세요.');
      return;
    }
    
    if (!isEmailVerified) {
      setError('이메일 인증을 완료해주세요.');
      return;
    }
    
    if (!name.trim()) {
      setError('이름을 입력해주세요.');
      return;
    }
    
    if (!password.trim()) {
      setError('비밀번호를 입력해주세요.');
      return;
    }
    
    if (!confirm.trim()) {
      setError('비밀번호 확인을 입력해주세요.');
      return;
    }
    
    if (password !== confirm) {
      setError('비밀번호가 일치하지 않습니다.');
      return;
    }
    
    if (password.length < 8) {
      setError('비밀번호는 8자 이상이어야 합니다.');
      return;
    }

    // 비밀번호 유효성 검증 (영문, 숫자, 특수문자)
    if (!/[a-zA-Z]/.test(password)) {
      setError('비밀번호는 영문을 포함해야 합니다.');
      return;
    }

    if (!/\d/.test(password)) {
      setError('비밀번호는 숫자를 포함해야 합니다.');
      return;
    }

    if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
      setError('비밀번호는 특수문자를 포함해야 합니다.');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      // 실제 API 호출 - 객체 형태로 전달
      const response = await authAPI.signup({
        email: fullEmail,
        name: name.trim(),
        password: password
      });
      
      // 회원가입 성공시 자동 로그인 처리
      console.log('Signup successful:', response);
      
      try {
        // 회원가입한 이메일과 비밀번호로 자동 로그인
        const loginResponse = await authAPI.login(fullEmail, password);
        const accessToken = loginResponse.data?.access_token || loginResponse.access_token;
        const refreshToken = loginResponse.data?.refresh_token || loginResponse.refresh_token;
        const userData = loginResponse.data?.user || loginResponse.user;

        if (accessToken) {
          localStorage.setItem('authToken', accessToken);
          localStorage.setItem('access_token', accessToken);
          localStorage.setItem('refresh_token', refreshToken);
          localStorage.setItem('userEmail', fullEmail);
          if (userData) {
            localStorage.setItem('user', JSON.stringify(userData));
          }
          console.log('자동 로그인 완료');
          // 로그인 성공 후 메인 페이지로 이동
          onSignup(fullEmail);
        } else {
          alert('회원가입이 완료되었습니다. 로그인해주세요.');
          onSignup(fullEmail);
        }
      } catch (loginError) {
        console.error('자동 로그인 실패:', loginError);
        alert('회원가입이 완료되었습니다. 로그인해주세요.');
        onSignup(fullEmail);
      }
    } catch (error) {
      console.error('Signup failed:', error);
      
      // 네트워크 오류 처리
      if (error.name === 'NetworkError' || error.message.includes('서버에 연결할 수 없습니다')) {
        setError('서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해주세요.');
      } else if (error.name === 'TimeoutError' || error.message.includes('요청 시간이 초과')) {
        setError('요청 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.');
      } else if (error.response?.data?.message) {
        setError(error.response.data.message);
      } else if (error.message) {
        setError(error.message);
      } else {
        setError('회원가입에 실패했습니다. 다시 시도해주세요.');
      }
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="signup-container">
      {onBack && (
        <button
          type="button"
          className="back-btn"
          onClick={onBack}
        >
          ← 뒤로가기
        </button>
      )}
      <form className="signup-form" onSubmit={handleSubmit}>
        <h2>회원가입</h2>
        
        {/* 이메일 입력 (학교 도메인 고정) */}
        <div className="email-input-group">
          <div className="email-container">
            <input
              type="text"
              placeholder="이메일 앞부분"
              value={emailPrefix}
              onChange={e => setEmailPrefix(e.target.value)}
              className="email-prefix"
              disabled={isEmailVerified}
            />
            <span className="email-suffix">@bu.ac.kr</span>
          </div>
          {!isEmailVerified ? (
            <button
              type="button"
              onClick={handleSendVerification}
              disabled={isVerifying || !emailPrefix.trim()}
              className="verify-btn"
            >
              {isVerifying ? '발송중...' : '인증번호 발송'}
            </button>
          ) : (
            <span className="verified-badge">✓ 인증완료</span>
          )}
        </div>

        {/* 인증번호 입력 (인증 완료 전까지만 표시) */}
        {sentCode && !isEmailVerified && (
          <div className="verification-group">
            <input
              type="text"
              placeholder="인증번호 6자리"
              value={verificationCode}
              onChange={e => {
                // 숫자만 입력 허용
                const value = e.target.value.replace(/[^0-9]/g, '');
                setVerificationCode(value);
              }}
              maxLength="6"
              className="verification-input"
            />
            <button
              type="button"
              onClick={handleVerifyCode}
              disabled={!verificationCode.trim() || verificationCode.length !== 6}
              className="verify-code-btn"
            >
              인증확인
            </button>
          </div>
        )}

        {/* 이름 입력 */}
        <input
          type="text"
          placeholder="이름"
          value={name}
          onChange={e => setName(e.target.value)}
          disabled={!isEmailVerified}
        />

        <input
          type="password"
          placeholder="비밀번호 (8자 이상, 영문+숫자+특수문자)"
          value={password}
          onChange={e => setPassword(e.target.value)}
          disabled={!isEmailVerified}
        />
        <input
          type="password"
          placeholder="비밀번호 확인"
          value={confirm}
          onChange={e => setConfirm(e.target.value)}
          disabled={!isEmailVerified}
        />
        {error && <div className="signup-error">{error}</div>}
        <button type="submit" disabled={isLoading || !isEmailVerified}>
          {isLoading ? '회원가입 중...' : '회원가입'}
        </button>
      </form>
    </div>
  );
}
